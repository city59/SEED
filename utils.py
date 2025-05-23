from sklearn.mixture import BayesianGaussianMixture

import Kg_Par
import torch
import numpy as np
import scipy.sparse as sp


def get_use(behaviors_data):
    behavior_mats = {}
    behaviors_data = (behaviors_data != 0) * 1

    behavior_mats['A'] = matrix_to_tensor(normalize_adj(behaviors_data))
    behavior_mats['AT'] = matrix_to_tensor(normalize_adj(behaviors_data.T))
    behavior_mats['A_ori'] = None

    return behavior_mats


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    rowsum_diag = sp.diags(np.power(rowsum + 1e-8, -0.5).flatten())

    colsum = np.array(adj.sum(0))
    colsum_diag = sp.diags(np.power(colsum + 1e-8, -0.5).flatten())

    return adj


def matrix_to_tensor(cur_matrix):
    if type(cur_matrix) != sp.coo_matrix:
        cur_matrix = cur_matrix.tocoo()
    indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))
    values = torch.from_numpy(cur_matrix.data)
    shape = torch.Size(cur_matrix.shape)

    return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).cuda()


class BPRLoss:
    def __init__(self, recmodel, opt):
        self.model = recmodel
        self.opt = opt
        self.weight_decay = Kg_Par.config["decay"]

    def compute(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss
        return loss


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class timer:
    from time import time
    TAPE = [-1]
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


def dp_semantic_domains(relations_embeddings, alpha=1.0, max_domains=10, random_state=42):

    if not isinstance(relations_embeddings, np.ndarray):
        if torch.is_tensor(relations_embeddings):
            relations_embeddings = relations_embeddings.cpu().detach().numpy()
        else:
            raise TypeError("relations_embeddings must be a NumPy array or a PyTorch tensor.")

    if relations_embeddings.shape[0] == 0:
        print("Warning: No relations found for DP semantic domains. Returning empty results.")
        return np.array([]).reshape(0, relations_embeddings.shape[1]), np.array([]).reshape(0, max_domains), 0

    if relations_embeddings.shape[0] < max_domains:
        current_max_domains = relations_embeddings.shape[0]
    else:
        current_max_domains = max_domains

    if current_max_domains == 0:
        print("Warning: max_domains (or number of relations) is 0. Returning empty results for DP.")
        return np.array([]).reshape(0, relations_embeddings.shape[1]), np.array([]).reshape(0, max_domains), 0

    dp_model = BayesianGaussianMixture(
        n_components=current_max_domains,
        weight_concentration_prior=alpha,
        covariance_type='full',
        max_iter=1000,
        random_state=random_state,
        n_init=3,
        weight_concentration_prior_type='dirichlet_process',
        tol=1e-3,
    )

    try:
        dp_model.fit(relations_embeddings)
    except ValueError as e:
        print(f"Error during DPGMM fit: {e}. Returning empty results.")
        return np.array([]).reshape(0, relations_embeddings.shape[1]), np.array([]).reshape(0, max_domains), 0

    probs = dp_model.predict_proba(relations_embeddings)

    active_domains_indices = np.where(dp_model.weights_ > 1e-3)[0]

    if len(active_domains_indices) == 0 and probs.shape[1] > 0:
        active_domains_indices = np.array([np.argmax(dp_model.weights_)])

    if len(active_domains_indices) > 0:
        centroids = dp_model.means_[active_domains_indices]
        probs_active = probs[:, active_domains_indices]
        n_domains_found = len(active_domains_indices)
    else:
        if dp_model.means_.shape[0] > 0:
            centroids = dp_model.means_
            probs_active = probs
            n_domains_found = dp_model.means_.shape[0]
        else:
            centroids = np.array([]).reshape(0, relations_embeddings.shape[1])
            probs_active = np.array([]).reshape(relations_embeddings.shape[0], 0)  # (n_samples, 0)
            n_domains_found = 0

    return centroids, probs_active, n_domains_found