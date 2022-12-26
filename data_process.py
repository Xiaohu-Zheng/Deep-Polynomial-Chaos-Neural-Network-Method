import math
import torch
import numpy as np
from pyDOE import lhs
from scipy import stats
from torch.utils.data import DataLoader, Dataset


class TensorDataset(Dataset):
    # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
    # 实现将一组Tensor数据对封装成Tensor数据集
    # 能够通过index得到数据集的数据，能够通过len，得到数据集大小

    def __init__(self, data_tensor, label_tensor):
        self.data_tensor = data_tensor
        self.label_tensor = label_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.label_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def probability_fun(x, threshold):
    y = torch.where(x <= threshold)[0]
    p = y.size(0) / x.size(0)
    return p


def lognormal_pram(m0, cov):
    sigma = m0 * cov
    v = sigma ** 2
    mu = math.log(m0**2/math.sqrt(m0**2+v))
    std = math.sqrt(math.log(1+v/m0**2))

    return mu, std


def gumbel_pdf(x, mu=0, beta=1):
    z = (x - mu) / beta
    return torch.exp(-z - torch.exp(-z)) / beta


def lognormal_pdf(x, mu=0, sigma=1):
    z = -(torch.log(x) - mu) ** 2 / (2 * sigma ** 2)
    return 1 / (math.sqrt(2 * math.pi) * x * sigma) * torch.exp(z)


def normal_pdf(x, mu=0, sigma=1):
    z = -(x - mu) ** 2 / (2 * sigma ** 2)
    return 1 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(z)


def rosenbaltt(prob, dis_type, param):
    if dis_type == 'norm':
        X = stats.norm.ppf(prob, param[0], param[1])
    if dis_type == 'lognorm':
        X = stats.lognorm.ppf(prob, param[0], param[1])
    elif dis_type == 'gamma':
        X = stats.gamma.ppf(prob, param[0], param[1])
    elif dis_type == 'exp':
        X = stats.expon.ppf(prob, param)
    elif dis_type == 'uniform':
        xi = stats.uniform.ppf(prob)
        X = xi * (param[1] - param[0]) + param[0]
    elif dis_type == 'gumbel':
        X = stats.gumbel_r.ppf(prob, param[0], param[1])
    elif dis_type == 'beta':
        X = stats.beta.ppf(prob, param[0], param[1])

    return X


def lhd_samping(num, dis_type, param):
    prob = lhs(1, samples=num, criterion='center')
    x = stats.norm(loc=0, scale=1).ppf(prob)
    X = rosenbaltt(prob, dis_type, param)

    return x, X