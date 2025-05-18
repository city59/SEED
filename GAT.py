import torch
import torch.nn as nn
import torch.nn.functional as F


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha,
                 relation_domain_probs, num_domains, relation_embed_layer):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.num_heads = 1
        self.relation_domain_probs = relation_domain_probs
        self.num_domains = num_domains
        self.relation_embed_layer = relation_embed_layer

        self.layers = nn.ModuleList([
            GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True,
                                relation_domain_probs=self.relation_domain_probs,
                                num_domains=self.num_domains,
                                relation_embed_layer=self.relation_embed_layer
                                )
            for _ in range(self.num_heads)
        ])
        self.out = nn.Linear(nhid * self.num_heads, nhid)

    def forward(self, item_embs, entity_embs, adj):
        x = F.dropout(item_embs, self.dropout, training=self.training)
        y = F.dropout(entity_embs, self.dropout, training=self.training)
        x = self.out(x, y, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        return x

    def forward_relation(self, item_embs, entity_embs, relations_embed, relation_ids, adj_mask):

        gat_layer_outputs = []
        for att_layer in self.layers:
            h_prime_layer = att_layer.forward_relation(item_embs, entity_embs, relations_embed, relation_ids, adj_mask)
            gat_layer_outputs.append(h_prime_layer)

        x_aggregated_neighbors = torch.cat(gat_layer_outputs, dim=1) if self.num_heads > 1 else gat_layer_outputs[0]

        output = self.out(x_aggregated_neighbors + item_embs)
        output = F.relu(output)
        output = F.dropout(output, self.dropout, training=self.training)
        return output


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True,
                 relation_domain_probs=None, num_domains=None, relation_embed_layer=None):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.relation_domain_probs = relation_domain_probs # (total_relations, num_domains)
        self.num_domains = num_domains

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.fc_attention = nn.Linear(in_features * 3, 1)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward_relation(self, item_embs, entity_embs, relations_embed, relation_ids, adj_mask):

        Wh = item_embs.unsqueeze(1).expand_as(entity_embs)

        attention_fc_input = torch.cat([Wh, relations_embed, entity_embs], dim=-1)
        e_scores = self.leakyrelu(self.fc_attention(attention_fc_input).squeeze(-1))

        zero_vec = -9e15 * torch.ones_like(e_scores)
        masked_e_scores = torch.where(adj_mask > 0, e_scores, zero_vec)

        pi_icv = F.softmax(masked_e_scores, dim=1)
        pi_icv = F.dropout(pi_icv, self.dropout, training=self.training)  # (batch_items, num_neighbors)

        base_messages_M = pi_icv.unsqueeze(-1) * relations_embed * entity_embs

        aggregated_item_repr = torch.zeros_like(item_embs)  # (batch_items, in_features)

        if self.num_domains > 0 and self.relation_domain_probs is not None and self.relation_domain_probs.numel() > 0:

            for s_domain_idx in range(self.num_domains):
                flat_relation_ids = relation_ids.view(-1)  # (batch_items * num_neighbors)

                a_ps_flat = torch.zeros_like(flat_relation_ids, dtype=self.relation_domain_probs.dtype,
                                             device=self.relation_domain_probs.device)

                valid_ids_mask = (flat_relation_ids >= 0) & (flat_relation_ids < self.relation_domain_probs.shape[0])

                if torch.any(valid_ids_mask):  # Check if there are any valid IDs to process
                    valid_indices_in_flat = flat_relation_ids[valid_ids_mask]
                    if s_domain_idx < self.relation_domain_probs.shape[1]:
                        a_ps_flat[valid_ids_mask] = self.relation_domain_probs[valid_indices_in_flat, s_domain_idx]

                a_ps_for_edges = a_ps_flat.view_as(relation_ids)  # (batch_items, num_neighbors)

                domain_s_weighted_messages = a_ps_for_edges.unsqueeze(-1) * base_messages_M

                sum_domain_s_messages_per_item = torch.sum(domain_s_weighted_messages,
                                                           dim=1)  # (batch_items, in_features)

                aggregated_item_repr += sum_domain_s_messages_per_item
        else:
            aggregated_item_repr = torch.sum(base_messages_M, dim=1)

        h_prime = aggregated_item_repr  # (batch_items, out_features)

        return h_prime

    def forward(self, item_embs, entity_embs, adj):
        Wh = torch.mm(item_embs,self.W)
        We = torch.matmul(entity_embs, self.W)
        a_input = self._prepare_cat(Wh, We)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout,training=self.training)
        entity_emb_weighted = torch.bmm(attention.unsqueeze(1),entity_embs).squeeze()
        h_prime = entity_emb_weighted + item_embs

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_cat(self, Wh, We):
        Wh = Wh.unsqueeze(1).expand(We.size())
        return torch.cat((Wh, We), dim=-1)

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
