import os
import sys
import time
import math
import torch
import pickle
import scipy.io as sio
import seaborn as sns
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

sys.path.append('/mnt/jfs/zhengxiaohu/GitHub/Deep-Polynomial-Chaos-Neural-Network-Method')
import Deep_PCE as dPC
import data_process as dp
from tube_fun import tube_fun
from PCNN import PCNN
from pce_loss import CoefficientPCENNLoss, CalculatePCELoss, CoefficientPCELoss, CoefficientPCELoss_coeff_batch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("Use CPU")
    device = torch.device('cpu')
else:
    print("Use GPU")
# device = torch.device('cpu')
print(os.getpid())

root_path = "/mnt/jfs/zhengxiaohu/GitHub/Deep-Polynomial-Chaos-Neural-Network-Method/"
data_path = '/mnt/jfs/zhengxiaohu/GitHub/Deep-Polynomial-Chaos-Neural-Network-Method/tube_data/'

# Basic parameter
train = True
# train = False
model = 3
pc_dpce='apc'
pc_pcnn='apc'
order = 2
order_pc = 5
dim = 9
lr_c = 0.01 #0.05
x_num = 200
x_coeff_batch_size = 20000
max_epoch = 20000
object_fun = f"PCNN_{order_pc}order_sat_frame" + "_" + pc_dpce + "_" + pc_pcnn + f"_model_{model}"

# Parameters
m_t = 5
sigma_t = 0.1

m_d = 42
sigma_d = 0.5

lb_L1 = 119.75
ub_L1 = 120.25
m_L1 = (lb_L1 + ub_L1) / 2
sigma_L1 = (ub_L1 - lb_L1) / math.sqrt(12)

lb_L2 = 59.75
ub_L2 = 60.25
m_L2 = (lb_L2 + ub_L2) / 2
sigma_L2 = (ub_L2 - lb_L2) / math.sqrt(12)

m_F1 = 3000
sigma_F1 = 300

m_F2 = 3000
sigma_F2 = 300

m_T = 90000
sigma_T= 9000

m_Sy= 220
sigma_Sy = 22

# Gumbel Parameter
m_P = 1.2e4
sigma_P = m_P * 0.1
v_P = sigma_P ** 2
belta_P = math.sqrt(6 * v_P / (math.pi**2))
mu_P = m_P - 0.57721 * belta_P

mean = torch.tensor([[m_t, m_d, m_L1, m_L2, m_F1, m_F2, m_P, m_T, m_Sy]])
std = torch.tensor([[sigma_t, sigma_d, sigma_L1, sigma_L2, sigma_F1, 
                       sigma_F2, sigma_P, sigma_T, sigma_Sy]])

if train:
    # Prepare training data
    data = data_path + 'samples_{}.mat'.format(x_num)
    data = sio.loadmat(data)
    X_train = torch.from_numpy(data['X']).float()
    x_train = (X_train - mean) / std
    y_train = tube_fun(X_train)
    dataset = dp.TensorDataset(x_train, y_train)

    # Loading training data
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=50000,
                                            shuffle=True, num_workers=2)

    # Prepare unlabeled data
    # Determine the number of unlabeled data according to the studied problem.
    num_coeff = int(1e+05)
    data_coeff = data_path + f'samples_{num_coeff}.mat'
    data_coeff = sio.loadmat(data_coeff)
    X_coeff = torch.from_numpy(data_coeff['X']).float()
    x_coeff = (X_coeff - mean) / std

    # Prepare testing data
    num_test = 1000
    data_test = data_path + 'samples_{}.mat'.format(num_test)
    data_test = sio.loadmat(data_test)
    X_test = torch.from_numpy(data_test['X']).float()
    x_test = (X_test - mean) / std
    y_test = tube_fun(X_test)


# Calculate K-order moment
mu_k_rooth = root_path + 'mu_ks/mu_k.mat'
mu_k_temp = sio.loadmat(mu_k_rooth)
mu_k = torch.from_numpy(mu_k_temp['mu_k']).float()
p_orders = dPC.all_orders_univariate_basis_coefficients(mu_k, order_pc)

# Initialize model
order_mat = dPC.order_mat_fun(dim, order)
order_mat_pc = dPC.order_mat_fun(dim, order_pc)
num_c = order_mat.size(0)
c = torch.rand(1, order_mat_pc.size(0)) * 0.1
c[0, 0] = 85.8168

if model == 0:
    # model-0：
    hiden_neurons = [32, 64, 128, 128, 64, 64]
elif model == 1:
    # model-1：
    hiden_neurons = [32, 64, 128, 64, 64]
elif model == 2:
    # model-2：
    hiden_neurons = [32, 64, 128, 64]
elif model == 3:
    # model-3：
    hiden_neurons = [32, 128, 64]
elif model == 4:
    # model-4：
    hiden_neurons = [128, 64]
net_c = PCNN(dim, num_c, hiden_neurons, c)

# Defining optimizer
net_c = net_c.to(device)
criterion = nn.L1Loss()
criterion_pce_loss = CalculatePCELoss(order, order_mat, pc_dpce, p_orders)
criterion_coeff_deep = CoefficientPCELoss()
criterion_coeff = CoefficientPCENNLoss(lam_mean=1, lam_var=1)
optimizer_c = optim.Adam(net_c.parameters(), lr=lr_c)

# Training model
test_acc_best = 0.03
if train:
    print("training on ", device)
    x_coeff = x_coeff.to(device)
    phi_x_coeff = dPC.orthogonal_basis(x_coeff, order_pc, order_mat_pc, pc_pcnn, p_orders, x_coeff_batch_size)

    start_train = time.time()
    for epoch in range(max_epoch):  # loop over the dataset multiple times
        if epoch % 300 == 299:
            optimizer_c.param_groups[0]['lr'] = optimizer_c.param_groups[0]['lr'] * 0.85

        train_l_fea_sum, train_l_c_sum, train_acc_sum, batch_count, start = 0.0, 0.0, 0.0, 0, time.time()
        for i, data in enumerate(dataloader, 0):
            # Obtain input
            x, y = data
            x, y = x.to(device), y.to(device)
            phi_x = dPC.orthogonal_basis(x, order_pc, order_mat_pc, pc_pcnn, p_orders)
            
            # Gradient return to zero
            optimizer_c.zero_grad()

            # net_c forward + backward + optimize
            c_nn, y_pcnn_pre = net_c(x, phi_x)
            loss_y_dpce = criterion_pce_loss(x, y, c_nn)
            loss_y_pcnn = criterion(y_pcnn_pre, y)
            
            c_coeff, y_pcnn_coeff = net_c(x_coeff, phi_x_coeff)
            y_dpce_coeff = dPC.deep_pce_fun(x_coeff, c_coeff, order, order_mat, pc_dpce, p_orders)
            c = net_c.state_dict()['c']
            loss_dpce_coeff = criterion_coeff_deep(y_dpce_coeff, c_coeff)
            loss_diff_coeff = criterion(y_dpce_coeff.detach(), y_pcnn_coeff)
            loss = loss_y_pcnn + loss_y_dpce + loss_diff_coeff + loss_dpce_coeff

            loss.backward()
            optimizer_c.step()
            train_l_c_sum += loss.item()
            batch_count += 1

        x_test, y_test = x_test.to(device), y_test.to(device)
        phi_x_test = dPC.orthogonal_basis(x_test, order_pc, order_mat_pc, pc_pcnn, p_orders)
        c_dpce_test, y_pcnn_test = net_c(x_test, phi_x_test)
        test_acc_dpce = criterion_pce_loss(x_test, y_test, c_dpce_test)
        test_acc_pcnn = criterion(y_pcnn_test, y_test)

        if test_acc_best > test_acc_pcnn:
            test_acc_best = test_acc_pcnn
            if not os.path.exists(root_path + 'trained_models'):
                os.makedirs(root_path + 'trained_models')
            torch.save(net_c.state_dict(), root_path +
                    'trained_models/{}_model_c_{}_{}_temp.pth'.format(object_fun, x_num, max_epoch))


        # writer.add_scalar('loss', train_l_sum / batch_count, epoch + 1)

        print('epoch %d, loss_c=%.6f, test_acc_dpce=%.6f, test_acc_pcnn=%.6f, lr_fea=%.6f, time %.6f sec'
            % (epoch + 1, train_l_c_sum / batch_count, test_acc_dpce, test_acc_pcnn, optimizer_c.param_groups[0]['lr'], time.time() - start))

    print('order={}, dim={}, Trainning over!'.format(order, dim))

    print('Train time:', time.time() - start_train)

    # Save the trained model
    if not os.path.exists(root_path + 'trained_models'):
        os.makedirs(root_path + 'trained_models')
    torch.save(net_c.state_dict(), root_path +
            'trained_models/{}_model_c_{}_{}.pth'.format(object_fun, x_num, max_epoch))

# Model prediction
net_c.load_state_dict(torch.load(
    root_path + 'trained_models/{}_model_c_{}_{}_temp.pth'.format(object_fun, x_num, max_epoch),
    map_location='cuda:0'))
c_train = net_c.state_dict()['c']

c_rest = c_train[0, 1:]
var = 0
for i in range(len(c_rest)):
    var += c_rest[i] ** 2
print(c_train[0, 0], var ** 0.5)

num_pre = int(1e5)
pre_batch_size = 20000
num_batch = math.ceil(num_pre/pre_batch_size)

#-----------------------------------------------------------------------------------
data_pred = data_path + 'samples_{}.mat'.format(num_pre)
data_pred = sio.loadmat(data_pred)
X_pred = torch.from_numpy(data_pred['X']).float()
x_pred = (X_pred - mean) / std
x_pred = torch.cat((x_pred, x_pred), dim=1)
y_grd = tube_fun(X_pred)
#-------------------------------------------------------------------------------------

PCNN = dPC.Predict_PCNN_regression(net_c, order, order_mat, 
                                   order_pc, order_mat_pc, pc_pcnn, pc_dpce, p_orders)
x_pred = x_pred.to(device)
Y_pcnn_pre, Y_dpce_pre = PCNN.prediction(x_pred, num_batch)

# x_pred = x_pred.to(device)
# start = time.time()
# phi_x = dPC.orthogonal_basis(x_pred, order_pc, order_mat_pc, pc_pcnn, p_orders)
# y_pce = torch.sum(phi_x * c_train, dim=1).view(-1, 1)
# print(time.time() - start)

# Mean and standard deviation
mean_by_c = c_train[0, 0]
c_mean_inter = c_train[0, 1:]
std_by_c = ((c_mean_inter ** 2).sum()) ** 0.5
print(mean_by_c, std_by_c)
mean_MC = y_grd.mean()
std_MC = y_grd.std()
mean_dpce = Y_dpce_pre.mean()
std_dpce = Y_dpce_pre.std()
mean_pcnn = Y_pcnn_pre.mean()
std_pcnn = Y_pcnn_pre.std()
print('PCNN prediction mean({} data):'.format(x_num), mean_pcnn.item(), '\n',
      'DPCE prediction mean({} data):'.format(x_num), mean_dpce.item(), '\n',
      'MC sampling mean:', mean_MC.item(), '\n',
      'PCNN prediction standard deviation({} data):'.format(x_num), std_pcnn.item(), '\n',
      'DPCE prediction standard deviation({} data):'.format(x_num), std_dpce.item(), '\n',
      'MC sampling standard deviation:', std_MC.item())

mean_error = abs(Y_pcnn_pre.mean() - y_grd.mean()) / y_grd.mean() * 100
print('Mean error:{}'.format(mean_error))

std_error = abs(Y_pcnn_pre.std() - y_grd.std()) / y_grd.std() * 100
print('PCNN Stddard deviation error:{}'.format(std_error))
std_error_dpce = abs(Y_dpce_pre.std() - y_grd.std()) / y_grd.std() * 100
print('DPCE Stddard deviation error:{}'.format(std_error_dpce))

#------------------------------------------------------------------------------------------
# skewness
ske_mc = (((y_grd - mean_MC) / std_MC) ** 3).mean()
ske_pcnn = (((Y_pcnn_pre - mean_pcnn) / std_pcnn) ** 3).mean()
ske_dpce = (((Y_dpce_pre - mean_dpce) / std_dpce) ** 3).mean()
print('ske_mc:%.8f'%(ske_mc.item()))
print('ske_PCNN:%.8f'%(ske_pcnn.item()))
print('ske_DPCE:%.8f'%(ske_dpce.item()))

ske_error_pcnn = abs(ske_pcnn - ske_mc) / ske_mc * 100
print('PCNN Ske error:{}'.format(ske_error_pcnn))
ske_error_dpce = abs(ske_dpce - ske_mc) / ske_mc * 100
print('DPCE Ske error:{}'.format(ske_error_dpce))

# kurtosis
kur_mc = (((y_grd - mean_MC) / std_MC) ** 4).mean()
kur_pcnn = (((Y_pcnn_pre - mean_pcnn) / std_pcnn) ** 4).mean()
kur_dpce = (((Y_dpce_pre - mean_dpce) / std_dpce) ** 4).mean()
print('kur_mc:%.8f'%(kur_mc.item()))
print('kur_PCNN:%.8f'%(kur_pcnn.item()))
print('kur_DPCE:%.8f'%(kur_dpce.item()))

kur_error_pcnn = abs(kur_pcnn - kur_mc) / kur_mc * 100
print('PCNN Kur error:{}'.format(kur_error_pcnn))
kur_error_dpce = abs(kur_dpce - kur_mc) / kur_mc * 100
print('DPCE Kur error:{}'.format(kur_error_dpce))

#----------------------------------------------------------------------------------------------
rmse_pcnn = torch.sqrt(((y_grd - Y_pcnn_pre.cpu()) ** 2).mean())
rmse_dpce = torch.sqrt(((y_grd - Y_dpce_pre.cpu()) ** 2).mean())
print('RMSE_pcnn:%.8f'%(rmse_pcnn))
print('RMSE_dpce:%.8f'%(rmse_dpce))

mae_pcnn = (torch.absolute(y_grd - Y_pcnn_pre.cpu())).mean()
mae_dpce = (torch.absolute(y_grd - Y_dpce_pre.cpu())).mean()
print('MAE_pcnn:%.8f'%(mae_pcnn))
print('MAE_dpce:%.8f'%(mae_dpce))

mre_pcnn = (torch.absolute((y_grd - Y_pcnn_pre.cpu()) / y_grd)).mean()
mre_dpce = (torch.absolute((y_grd - Y_dpce_pre.cpu()) / y_grd)).mean()
print('MRE_pcnn:%.8f'%(mre_pcnn))
print('MRE_dpce:%.8f'%(mre_dpce))

R2_pcnn = 1 - torch.sum( (y_grd - Y_pcnn_pre.cpu()) ** 2) / torch.sum( (y_grd - mean_MC)**2 )
R2_dpce = 1 - torch.sum( (y_grd - Y_dpce_pre.cpu()) ** 2) / torch.sum( (y_grd - mean_MC)**2 )
print('R2_pcnn:%.8f'%(R2_pcnn))
print('R2_dpce:%.8f'%(R2_dpce))

# e3_pcnn = torch.sqrt(torch.sum((Y_pcnn_pre.cpu() - y_grd)**2) / torch.sum(y_grd**2))
# e3_dpce = torch.sqrt(torch.sum((Y_dpce_pre.cpu() - y_grd)**2) / torch.sum(y_grd**2))
# print('e3_pcnn:%.8f'%(e3_pcnn))
# print('e3_nn:%.8f'%(e3_dpce))

# Failure probability
threshold = 0
prob_mcs = dp.probability_fun(y_grd, threshold)
prob_pcnn = dp.probability_fun(Y_pcnn_pre.cpu(), threshold)
prob_dpce = dp.probability_fun(Y_dpce_pre.cpu(), threshold)
prob_pcnn_error = (prob_pcnn - prob_mcs) / prob_mcs * 100
prob_dpce_error = (prob_dpce - prob_mcs) / prob_mcs * 100
print('prob_MCS:%.8f'%(prob_mcs))
print('prob_PCNN:%.8f'%(prob_pcnn))
print('prob_DPCE:%.8f'%(prob_dpce))
print('prob_pcnn_error:%.4f'%(prob_pcnn_error))
print('prob_dpce_error:%.4f'%(prob_dpce_error))