import scipy.io as sp
import os
import numpy as np
import torch

def load_data(root_path):
    V = sp.loadmat(os.path.join(root_path, 'VOC2007.mat'))['data']
    L = sp.loadmat(os.path.join(root_path, 'LabelMe.mat'))['data']
    S = sp.loadmat(os.path.join(root_path, 'SUN09.mat'))['data']
    C = sp.loadmat(os.path.join(root_path, 'Caltech101.mat'))['data']
    data = np.array([V[:, :-1], L[:, :-1], S[:, :-1], C[:, :-1]])
    label = np.array([V[:, -1], L[:, -1], S[:, -1], C[:, -1]])
    label = [item - 1 for item in label]
    return data, label, ['V', 'L', 'S', 'C']

def split_train_test(data, label, target_domain,test_split=0.3):
    data_train = []
    data_test = []
    label_train = []
    label_test = []
    for i in range(len(data)):
        if i == target_domain:
            continue
        length = data[i].shape[0]
        test_size = int(test_split * length)
        test_idx = np.random.choice(np.arange(length), test_size, replace=False)
        data_test.append(data[i][test_idx])
        data_train.append(np.delete(data[i], test_idx, axis=0))
        label_test.append(label[i][test_idx])
        label_train.append(np.delete(label[i], test_idx, axis=0))
    return np.array(data_train), np.array(data_test), np.array(label_train), np.array(label_test), data[target_domain], label[target_domain]



def MMD_Loss_func(num_source, sigmas=None):
    if sigmas is None:
        sigmas = [1, 5, 10]
    def loss(e_pred,d_ture):
        cost = 0.0
        for i in range(num_source):
            domain_i = e_pred[d_ture == i]
            for j in range(i+1,num_source):
                domain_j = e_pred[d_ture == j]
                single_res = mmd_two_distribution(domain_i,domain_j,sigmas=sigmas)
                cost += single_res
        return cost
    return loss

def mmd_two_distribution(source, target, sigmas):
    sigmas = torch.tensor(sigmas).cuda()
    xy = rbf_kernel(source, target, sigmas)
    xx = rbf_kernel(source, source, sigmas)
    yy = rbf_kernel(target, target, sigmas)
    return xx + yy - 2 * xy

def rbf_kernel(x, y, sigmas):
    sigmas = sigmas.reshape(sigmas.shape + (1,))
    beta = 1. / (2. * sigmas)
    dist = compute_pairwise_distances(x, y)
    dot = -torch.matmul(beta, torch.reshape(dist, (1, -1)))
    exp = torch.mean(torch.exp(dot))
    return exp

def compute_pairwise_distances(x, y):
    dist = torch.zeros(x.size(0),y.size(0)).cuda()
    for i in range(x.size(0)):
        dist[i,:] = torch.sum(torch.square(x[i].expand(y.shape) - y),dim=1)
    return dist