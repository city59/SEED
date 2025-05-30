import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from Mul_Par import args

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class myModel(nn.Module):
    def __init__(self, userNum, itemNum, behavior, behavior_mats): 
        super(myModel, self).__init__()  
        
        self.userNum = userNum
        self.itemNum = itemNum
        self.behavior = behavior
        self.behavior_mats = behavior_mats
        
        self.embedding_dict = self.init_embedding() 
        self.weight_dict = self.init_weight()
        self.gcn = GCN(self.userNum, self.itemNum, self.behavior, self.behavior_mats)


    def init_embedding(self):
        
        embedding_dict = {  
            'user_embedding': None,
            'item_embedding': None,
            'user_embeddings': None,
            'item_embeddings': None,
        }
        return embedding_dict

    def init_weight(self):  
        initializer = nn.init.xavier_uniform_
        
        weight_dict = nn.ParameterDict({
            'w_self_attention_item': nn.Parameter(initializer(torch.empty([args.hidden_dim, args.hidden_dim]))),
            'w_self_attention_user': nn.Parameter(initializer(torch.empty([args.hidden_dim, args.hidden_dim]))),
            'w_self_attention_cat': nn.Parameter(initializer(torch.empty([args.head_num*args.hidden_dim, args.hidden_dim]))),
            'alpha': nn.Parameter(torch.ones(2)),
        })      
        return weight_dict  

    def forward(self):
        user_embed, item_embed, user_embeds, item_embeds = self.gcn()

        return user_embed, item_embed, user_embeds, item_embeds 

    def para_dict_to_tenser(self, para_dict):
        tensors = []
        for beh in para_dict.keys():
            tensors.append(para_dict[beh])
        tensors = torch.stack(tensors, dim=0)

        return tensors.float()

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_parameters(), source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:
            for name, param in self.named_parameters()(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  
                    self.set_param(self, name, param)


class GCN(nn.Module):
    def __init__(self, userNum, itemNum, behavior, behavior_mats):
        super(GCN, self).__init__()  
        self.userNum = userNum
        self.itemNum = itemNum
        self.hidden_dim = args.hidden_dim

        self.behavior = behavior
        self.behavior_mats = behavior_mats

        self.user_embedding, self.item_embedding = self.init_embedding()         
        
        self.alpha, self.i_concatenation_w, self.u_concatenation_w, self.i_input_w, self.u_input_w = self.init_weight()
        
        self.sigmoid = torch.nn.Sigmoid()
        self.act = torch.nn.PReLU()
        self.dropout = torch.nn.Dropout(args.drop_rate)

        self.gnn_layer = eval(args.gnn_layer)
        self.layers = nn.ModuleList()
        for i in range(0, len(self.gnn_layer)):  
            self.layers.append(GCNLayer(args.hidden_dim, args.hidden_dim, self.userNum, self.itemNum, self.behavior, self.behavior_mats))  

    def init_embedding(self):
        user_embedding = torch.nn.Embedding(self.userNum, args.hidden_dim)
        item_embedding = torch.nn.Embedding(self.itemNum, args.hidden_dim)
        nn.init.xavier_uniform_(user_embedding.weight)
        nn.init.xavier_uniform_(item_embedding.weight)

        return user_embedding, item_embedding

    def init_weight(self):
        alpha = nn.Parameter(torch.ones(2))  
        i_concatenation_w = nn.Parameter(torch.Tensor(len(eval(args.gnn_layer))*args.hidden_dim, args.hidden_dim))
        u_concatenation_w = nn.Parameter(torch.Tensor(len(eval(args.gnn_layer))*args.hidden_dim, args.hidden_dim))
        i_input_w = nn.Parameter(torch.Tensor(args.hidden_dim, args.hidden_dim))
        u_input_w = nn.Parameter(torch.Tensor(args.hidden_dim, args.hidden_dim))
        init.xavier_uniform_(i_concatenation_w)
        init.xavier_uniform_(u_concatenation_w)
        init.xavier_uniform_(i_input_w)
        init.xavier_uniform_(u_input_w)

        return alpha, i_concatenation_w, u_concatenation_w, i_input_w, u_input_w

    def forward(self, user_embedding_input=None, item_embedding_input=None):

        user_emb_list = []
        item_emb_list = []

        for i in range(len(self.behavior)):
            if i == 0:
                temp_user_emb_list = []
                temp_item_emb_list = []
                user_embedding = self.user_embedding.weight
                item_embedding = self.item_embedding.weight
            else:
                temp_user_emb_list.clear()
                temp_item_emb_list.clear()

                user_embedding = user_embedding + user_emb_list[i-1]
                item_embedding = item_embedding + item_emb_list[i-1]


            for k, layer in enumerate(self.layers):
                u_emb, i_emb = layer(user_embedding, item_embedding, i)
                temp_user_emb_list.append(u_emb)
                temp_item_emb_list.append(i_emb)
            user_emb_list.append(torch.matmul(torch.cat(temp_user_emb_list, dim=1), self.u_concatenation_w))
            item_emb_list.append(torch.matmul(torch.cat(temp_item_emb_list, dim=1), self.i_concatenation_w))

        u_emb = torch.stack(user_emb_list, dim=2)
        u_emb = torch.mean(u_emb,dim=2)

        i_emb = torch.stack(item_emb_list, dim=2)
        i_emb = torch.mean(i_emb,dim=2)

        return u_emb, i_emb, user_emb_list, item_emb_list


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, userNum, itemNum, behavior, behavior_mats):
        super(GCNLayer, self).__init__()

        self.behavior = behavior
        self.behavior_mats = behavior_mats

        self.userNum = userNum
        self.itemNum = itemNum
        self.node_dropout_flag = args.node_dropout_flag

        self.act = torch.nn.Sigmoid()
        self.i_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.u_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        init.xavier_uniform_(self.i_w)
        init.xavier_uniform_(self.u_w)


    def forward(self, user_embedding, item_embedding, ind_beh):

        u, i = self.create_mb_embed1(user_embedding, item_embedding, ind_beh)

        u_emb = self.act(torch.matmul(u, self.u_w))
        i_emb = self.act(torch.matmul(i, self.i_w))

        return u_emb, i_emb

    def create_mb_embed1(self, user_embedding, item_embedding, ind_beh):

        u = torch.spmm(self.behavior_mats[ind_beh]['A'], item_embedding)
        i = torch.spmm(self.behavior_mats[ind_beh]['AT'], user_embedding)

        u_emb = user_embedding + u
        i_emb = item_embedding + i


        return u_emb, i_emb
