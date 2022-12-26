import math
import torch

def tube_fun(X):
    theta1 = 5 * math.pi / 180
    theta2 = 10 * math.pi / 180
    t = X[:, 0:1]
    d = X[:, 1:2]
    L1 = X[:, 2:3]
    L2 = X[:, 3:4]
    F1 = X[:, 4:5]
    F2 = X[:, 5:6]
    P = X[:, 6:7]
    T = X[:, 7:8]
    Sy = X[:, 8:9]
    M = F1 * L1 * math.cos(theta1) + F2 * L2 * math.cos(theta2)
    A = math.pi / 4 * (d ** 2 - (d - 2 * t) ** 2)
    c = d / 2
    I = math.pi / 64 * (d ** 4 - (d - 2 * t) ** 4)
    J = 2 * I
    tau_zx = T * d / (2 * J)
    sigma_x = (P + F1 * math.sin(theta1) + F2 * math.sin(theta2)) / A + M * c / I
    sigma_max = torch.sqrt(sigma_x ** 2 + 3 * tau_zx ** 2)
    G = Sy - sigma_max

    return G