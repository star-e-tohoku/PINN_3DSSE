"""
PINNs for Forward calculation : simulating SSEs on the HN19 model 
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import h5py
import torch
import torch.autograd as autograd
import torch.nn as nn
#from pyDOE import lhs
import scipy.io
import matplotlib.pyplot as plt
from numpy import format_float_scientific as ffs
import sys
sys.path.append("..")
import time as Time

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=10)
torch.manual_seed(1234)#1234
np.random.seed(1234)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if device == 'cuda':
    print(torch.cuda.get_device_name())

place = "/Result/"
name = "PINN_forward_3DSSE"
case_number = 1 #Save the interim result on caseN folder
print(name)

### Learning Parameter 
max_iter = 20000
t_col_span_day = 10

### import Green Function
fname = "Green_Functions_HN2019.jld2"
f = h5py.File(fname, "r")
K = np.array(f["Fault/KernFF"]).T
f.close()

### import the result of Hirahara and Nishikiori 2019
fname = "2dfault_SSE_HN+.jld2"
f = h5py.File(fname, 'r')
# K = np.array(f["Green_Function"]).T # !!!!!!!
[FullFaultLengthStrike, FullFaultLengthDip, FullFaultDepth, StrikeAngle, DipAngle] = np.array(f["para/fault"])
[CellLengthStrike, CellLengthDip] = np.array(f["para/cell"])
[ShearModulus, PoissonRatio] = np.array(f["para/elastic"])
time0 = np.array(f["ode/x"])
a_ = np.array(f["para/a"])[0]
b_ = np.array(f["para/b"])[0]
bp = np.array(f["para/bp"])[0]
dc_ = np.array(f["para/dc"])[0]
sigma0 = np.array(f["para/sigma"])[0]
v_true0 = np.array(f["ana/v"])
state_true0 = np.array(f["ana/theta"])
f.close()

### Define Fault geometry
FFLS = FullFaultLengthStrike - CellLengthStrike; FFLD = FullFaultLengthDip - CellLengthDip
left_strike = - FFLS / 2  ; right_strike = FFLS / 2
left_dip    = - FFLD / 2  ; right_dip = FFLD / 2
mesh_strike = np.arange(left_strike, right_strike +1, CellLengthStrike)
mesh_dip    = np.arange(left_dip,    right_dip +1,    CellLengthDip   )
N_strike = len(mesh_strike); N_dip = len(mesh_dip)
N_cell =  N_strike * N_dip
patch_radius = 35e3 #35 km


### Define model parameters
year2sec = 365.25 * 24 * 60 * 60
TINY = 1e-20
G = 20  *1e9 #Pa shear modulus
mu = 0.25 #poisson ratio
rho = 2400 #kg/m^3
vs = np.sqrt(G / rho) #m/s
eta = G / (2 * vs)
vpl = 6.5e-2 / year2sec #m/s
statepl = dc_ / vpl #s
#conversion from numpy to Tensor
K_t = torch.from_numpy(K).double().to(device)


###  Setting Collocation Point

# Define target one cycle
center_place = 1474
cycle_start = np.argmin(v_true0[0:8 * 365, center_place])
cycle_end   = np.argmin(v_true0[8 * 365 : 16 * 365, center_place]) + len(v_true0[0:8 * 365, center_place])

t1_iter = cycle_start;    t2_iter = cycle_end    #cycle_start + 365
t1 = t1_iter * 3600 * 24; t2 = t2_iter * 3600 * 24
span = 3600 * 24 * t_col_span_day
v_true     = v_true0[t1_iter : t2_iter+1]
state_true = state_true0[t1_iter : t2_iter+1]
time_true  = time0[t1_iter : t2_iter+1] - time0[t1_iter]
v1 = v_true[0]; state1 = state_true[0]

t = np.arange(t1, t2+1 , span) - t1
x = np.arange(- FullFaultLengthStrike / 2, FullFaultLengthStrike / 2, CellLengthStrike) + CellLengthStrike / 2
y = np.arange(- FullFaultLengthDip / 2, FullFaultLengthDip / 2, CellLengthDip) + CellLengthDip / 2

Nt = len(t); Nxy = len(x) * len(y)
print("Nt = ", Nt, " Nxy = ", Nxy)
x = x.reshape(1, -1); y = y.reshape(1, -1); t = t.reshape(1, -1)

X, T, Y  = np.meshgrid(x, t, y)
txy_test = np.hstack((T.flatten()[:, None], X.flatten()[:, None], Y.flatten()[:, None]))

lx = - FullFaultLengthStrike / 2; rx = FullFaultLengthStrike / 2
ly = - FullFaultLengthDip / 2   ; ry = FullFaultLengthDip / 2
lb = np.array([0, lx, ly]); ub = np.array([t2 - t1, rx, ry])

N_uIni = 3000; N_f = len(txy_test)

def trainingdata(N_uIni, N_f):
    #Initial Condition t = 0
    initial_txy = np.hstack((T[0, :, :].flatten()[:, None], X[0, :, :].flatten()[:, None], Y[0, :, :].flatten()[:, None]))
    initial_u = np.hstack((np.log(v1[:, None] / vpl), np.log(state1[:, None] / statepl)))
    txy_ini_train = initial_txy
    u_ini_train = initial_u

    #Collcation point generation by Latin Hypercube Sampling
    #xyt_f_train = lb + (ub-lb)*lhs(3,N_f)
    txy_f_train = np.copy(txy_test)

    return txy_ini_train, u_ini_train, txy_f_train

txy_ini_train_np, u_ini_train_np, txy_f_train_np = trainingdata(N_uIni, N_f)

### Define the mesh for evaluating stress
tau_x = np.arange(- FullFaultLengthStrike / 2, FullFaultLengthStrike / 2, CellLengthStrike) + CellLengthStrike / 2
tau_y = np.arange(- FullFaultLengthDip / 2, FullFaultLengthDip / 2, CellLengthDip) + CellLengthDip / 2
tau_t = np.copy(t)
tau_x = tau_x.reshape(1, -1); tau_y = tau_y.reshape(1, -1); tau_t = tau_t.reshape(1, -1)

TauX, TauT, TauY  = np.meshgrid(tau_x, tau_t, tau_y)
txy_tau = np.hstack((TauT.flatten()[:, None], TauX.flatten()[:, None], TauY.flatten()[:, None]))
txy_tau = torch.from_numpy(txy_tau).double().to(device)

#conversion from numpy to ensor
txy_ini_train = torch.from_numpy(txy_ini_train_np).double().to(device)
u_ini_train = torch.from_numpy(u_ini_train_np).double().to(device)
txy_f_train =  torch.from_numpy(txy_f_train_np).double().to(device)
txy_test_tensor = torch.from_numpy(txy_test).double().to(device)
f_hat = torch.zeros(txy_f_train.shape[0],1).to(device)


### Define NN
class Sequentialmodel(nn.Module):

    def __init__(self,layers):
        super().__init__()

        'activation function'
        self.activation = nn.Tanh()

        'loss function'
        self.loss_function = nn.MSELoss(reduction ='mean')

        'Initialise neural network as a list using nn.Modulelist'
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])

        self.iter = 0

        self.loss_hist = [] #lossのリスト
        self.lossini_hist = []
        self.lossf_hist = []
        self.lossd_hist = []

        for i in range(len(layers)-1):
            #Xavier Initialization
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)

    #foward computation
    def forward(self,x):

        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)
        u_b = torch.from_numpy(ub).double().to(device)
        l_b = torch.from_numpy(lb).double().to(device)
        #scaling
        x = (x - l_b)/(u_b - l_b)
        #convert to double
        a = x.double()
        for i in range(len(layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)

        return a

    #loss function induced from initial condition
    def loss_IC(self, txy, uini):

        z = txy.clone()
        z.requires_grad = True
        u = self.forward(z)

        loss_u = self.loss_function(u[:, :], uini[:, :])

        return loss_u
    
    def cal_loading_stress(self, txy):
        u = self.forward(txy_tau)
        u_0 = torch.reshape(u[:, 0], (Nt, N_cell)) #v
        velocity = vpl * torch.exp(u_0)
        loading_stress = torch.matmul(K_t, velocity.T)
        loading_stress = torch.reshape(loading_stress, (-1, 1))

        return loading_stress

    #loss function induced from governing equation
    def loss_PDE(self, txy):

        z = txy.clone()
        z.requires_grad = True

        u = self.forward(z)
        u_0 = torch.reshape(u[:, 0],(txy.shape[0],1)) #v
        u_1 = torch.reshape(u[:, 1],(txy.shape[0],1)) #state

        P_t = autograd.grad(u_0,z,torch.ones([txy.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0][:, 0].reshape(-1, 1)
        Q_t = autograd.grad(u_1,z,torch.ones([txy.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0][:, 0].reshape(-1, 1)

        velocity = vpl * torch.exp(u_0)
        state    = statepl * torch.exp(u_1)

        R = torch.sqrt(txy[:, 1] ** 2 + txy[:, 2] ** 2)
        b = torch.where(R < patch_radius, bp, b_)[:, None]
        loading_stress = self.cal_loading_stress(txy)

        eps = 1.0e-30
        p = a_ / (velocity + eps) + eta / sigma0
        r = 1.0 - velocity * state / dc_
        q = loading_stress / sigma0 - b * r / (state + eps)

        f_P = P_t - q / (p * velocity)
        f_Q = Q_t - r / state

        #Lr = f_P ** 2 + f_Q ** 2
        loss_f_P = self.loss_function(f_P, f_hat)
        loss_f_Q = self.loss_function(f_Q, f_hat)

        loss_f = loss_f_P + loss_f_Q #self.loss_function(f,f_hat)
        dv_dt = velocity * P_t; dstate_dt = state * Q_t
        loss_f = loss_f * ((dc_ / vpl) ** 2)

        return loss_f, dv_dt, dstate_dt

    def loss_data(self, txy, p_data):
        z = txy.clone()
        z.requires_grad = True

        u = self.forward(z)
        loss_p_data = self.loss_function(u[:, 0], p_data[:, 0])
        #loss_q_data = self.loss_function(u[:, 1], u_data[:, 1])

        loss_d = loss_p_data

        return loss_d

    #summation of loss functions
    def loss(self,txy_ini_train,u_ini_train, txy_f_train):

        loss_ini = self.loss_IC(txy_ini_train,u_ini_train)
        loss_f, dv_dt, dstate_dt = self.loss_PDE(txy_f_train)
        #loss_d = self.loss_data(txy_data, p_data)

        loss_val = loss_ini + weight*loss_f #+ weight2*loss_d

        return loss_val, loss_ini, loss_f

    def closure(self):

        optimizer.zero_grad()
        loss ,loss_ini, loss_f = self.loss(txy_ini_train,u_ini_train,txy_f_train)
        self.loss_hist.append(loss.item())
        self.lossini_hist.append(loss_ini.item())
        self.lossf_hist.append(loss_f.item())
        #self.lossd_hist.append(loss_d.item())

        loss.backward()

        self.iter += 1

        if self.iter <= 9:
            _ = PINN.test()
            print('step=',self.iter, 'loss=', loss.item(),loss_ini.item(), loss_f.item())

        if self.iter % 100 == 0:
            _ = PINN.test()
            #print('step=',self.iter, ', loss=', loss.item(),loss_ini.item(), loss_f.item())
            print('step=',self.iter, 'loss=', loss.item(),loss_ini.item(), loss_f.item())

        return loss

    def test(self):

        u_pred_tensor = self.forward(txy_test_tensor)
        u_pred = u_pred_tensor.cpu().detach().numpy()
        #u_pred = np.reshape(u_pred,(Nt,8),order='F')

        return u_pred


def optimize(NN, epsilon, max_eval, loss_list, name):
    PINN = NN

    loss_iter = loss_list["total"]
    loss_ini = loss_list["ini"]
    loss_f = loss_list["ode"]
    loss_wf = loss_list["wode"]
    loss_data = loss_list["data"]

    for i in range(max_eval):
        optimizer.step(PINN.closure)

        loss_temp = PINN.loss_hist[-1]
        loss_iter.append(loss_temp)
        loss_ini.append(PINN.lossini_hist[-1]) #loss_ini
        loss_f.append(PINN.lossf_hist[-1]) #loss_ode
        loss_wf.append(PINN.lossf_hist[-1] * weight) #loss * weight
        #loss_data.append(PINN.lossd_hist[-1] * weight)

        if(abs(loss_iter[-1] - loss_iter[-2]) < epsilon):
            break

        if ((i+1) % 500 == 0):
              name0 = place + name + "_Interim"
              save_PINN(PINN, name0, loss_list)
        if ((i+1) % 1000 == 0):
              case_place = f"Case{case_number}/"
              name0 = place + case_place + name + f"_{i+1}"
              save_PINN(PINN, name0, loss_list)
    del loss_iter[0]
    #loss_eval = PINN.loss_hist


def save_PINN(PINN, name, loss_list):
    model = PINN.to(device)
    torch.save(model.state_dict(), name + '.pth')

    f = h5py.File(name + '_loss.h5', 'w')
    #f["hist_lbfgs"] = hist_lbfgs
    f["loss"] = loss_list["total"]
    f["loss_ini"] = loss_list["ini"]
    f["loss_f"] = loss_list["ode"]
    f["loss_wf"] = loss_list["wode"]
    f["loss_data"] = loss_list["data"]
    f.close()

    print("NN is saved")

def load_PINN(PINN, name, loss_list):

    PINN.load_state_dict(torch.load(name + '.pth'))
    f = h5py.File(name + '_loss.h5', 'r')

    loss_list["total"] = np.array(f["loss"])
    loss_list["ini"] = np.array(f["loss_ini"])
    loss_list["ode"] = np.array(f["loss_f"])
    loss_list["wode"] = np.array(f["loss_wf"])
    loss_list["data"] = np.array(f["loss_data"])

    f.close()

    print("NN is loaded")


### Learning
start = Time.time()

'''Fix seed'''
seed = 1234
torch.manual_seed(seed); np.random.seed(seed)

'''Define Neural Network'''
weight = 1; weight2 = 0
layers = np.array([3,20,20,20,20,20,20,20,20,2])
PINN = Sequentialmodel(layers)
#PINN = nn.DataParallel(PINN, device_ids=[0, 1])
PINN.to(device)

'''Optimize parameters'''
params = list(PINN.parameters())

optimizer = torch.optim.LBFGS(params, lr=1.0,
                              max_iter = 1,
                              max_eval = 100,
                              tolerance_grad = 1e-20,
                              tolerance_change = 1e-6 * 1e-6,
                              history_size = 100,
                              line_search_fn = 'strong_wolfe')

loss_list ={"total":[0], "ini":[], "ode":[], "wode":[], "data":[]}

optimize(PINN, 1e-10 * 1e-10, max_iter, loss_list, name)

end = Time.time()
print("Traininig time = ", end - start, " [s]")

save_PINN(PINN, place + name, loss_list)