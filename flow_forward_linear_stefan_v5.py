###
## code adopted from https://github.com/PredictiveIntelligenceLab/DeepStefan
###


# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import copy
import timeit

import matplotlib.pyplot as plt

class Sampler:
    def __init__(self, dim, coords, func, name=None):
        self.dim = dim
        self.coords = coords
        self.func = func
        self.name = name
    
    def sample(self, N):
        x = self.coords[0:1, :] + (self.coords[1:2, :] - self.coords[0:1, :]) \
                                  * np.random.uniform(0, 1, size=(N, self.dim))
        y = self.func(x)
        return x, y

def u_0(x, p_in, p_out, x_in):  #Initial Condition  u0(x)
    p = copy.copy(x[:,0:1]) * 0.
    idx = x[:,0]<x_in
    p[idx,:] = p_in * ((x_in-x[idx,0:1])/x_in)
    p[~idx,:] = p_out
    return p

def g(x, p_in):  #Dirichlet Boundary Condition  g(t)
    p = copy.copy(x[:,1:2])* 0. + p_in
    return p

def u(x):  #u(x,t), if Exact solution exists, otherwise, just return x
    return x[:,0:1]


# heterogeneous field of alfa=K/mu
class local_alfa:
    def __init__(self, H0, An, Bn, c, mu):
        self.H0 = H0
        self.An = An
        self.Bn = Bn
        self.c = c
        self.mu = mu
        self.N = len(self.An)
        
    def getvalue(self, x_lst):
        return [1. / (self.H0 + sum([self.An[i-1] * np.cos(self.c*i*x) + \
                                     self.Bn[i-1] * np.sin(self.c*i*x) \
                                     for i in range(1,self.N+1)])) / self.mu \
                 for x in x_lst]
                 
    def getvalue_tensor(self, x_lst):
        H0 = torch.tensor(self.H0)
        An = torch.tensor(self.An)
        Bn = torch.tensor(self.Bn)
        c = torch.tensor(self.c)
        mu = torch.tensor(self.mu)
        alfa = [1. / (H0 + sum([An[i-1] * np.cos(c*i*x) + \
                                Bn[i-1] * np.sin(c*i*x) \
                               for i in range(1,self.N+1)])) / mu \
                 for x in x_lst]
        return torch.tensor(alfa, dtype=torch.float32)

# semi-analytical solution
class ExactSol:
    def __init__(self, H0, An, Bn, c, s0, G, mu):
        self.H0 = H0  #zero-mode of mu/K
        self.An = An  #Fourier coefficients An*cos
        self.Bn = Bn  #Fourier coefficients Bn*sin
        self.c = c   #angular frequency
        self.s0 = s0 #initial flow front location
        self.G = G  #pressure gradient, p_in-p_out
        self.mu = mu
        
    #g(t; G,mu)
    def g(self,t):
        return self.G/self.mu*t
    
    #F(s; H0,An,Bn,c, s0)
    def F(self,s):
        N = len(self.An)
        return 0.5*self.H0*(s**2-self.s0**2) - sum([self.An[i-1]/(self.c*i)**2 * (np.cos(self.c*i*s) - np.cos(self.c*i*self.s0)) + \
                                                    self.Bn[i-1]/(self.c*i)**2 * (np.sin(self.c*i*s) - np.sin(self.c*i*self.s0)) - \
                                                    self.Bn[i-1]/(self.c*i) * (s-self.s0) for i in range(1,N+1)])
    
    #F(s)-g(t)=0
    def equation(self, s, t):
        return self.F(s) - self.g(t)
    
    #solution of the moving boundary s(t)
    def solve_s(t):
        initial_guess = self.g(t) 
        return fsolve(self.equation, initial_guess, args=(t))
    
    #solution of pressure field p(x,t)  --> to be completed
    def solve_p(x,t):
        pass
        


# the physics-informed neural network
class Stefan1D_direct():
    def __init__(self, ics_sampler, 
                       Dcs_sampler, 
                       res_sampler, 
                       layers_u, 
                       layers_s, 
                       alfa,
                       alfa0,
                       p_out,
                       x_in,
                       tc):
        
        self.p_in = 1 #This is not used in the network
        self.p_out = p_out
        self.alfa = alfa
        self.alfa0 = alfa0
        self.s0 = x_in  #inlet location
        self.vc = vc
        
        # Samplers
        self.ics_sampler = ics_sampler
        self.Dcs_sampler = Dcs_sampler
        self.res_sampler = res_sampler
        
        # Initialize network weights and biases
        self.layers_u = layers_u
        self.weights_u, self.biases_u = self.initialize_NN(layers_u)
        
        self.layers_s = layers_s
        self.weights_s, self.biases_s = self.initialize_NN(layers_s)
        
        self.parameters = self.weights_u + self.biases_u + \
                          self.weights_s + self.biases_s
        
        # Initialize Fourier Feature mapping
        sigma_u = torch.tensor(1./self.res_sampler.coords[1,0])
        sigma_s = torch.tensor(1./self.res_sampler.coords[1,1])
        self.W_u = (torch.randn(2, layers_u[0]//2) * sigma_u)
        self.W_s = (torch.randn(1, layers_s[0]//2) * sigma_s)
        
        #
        self.iter = 0
        self.batch_size = 0
        self.lr = 1e-3
        
        self.timer = 0.
        
        # Adam optimizer
        self.optimizer_adam = torch.optim.Adam(self.parameters, lr=self.lr)
        
        # lbfgs optimizer
        self.optimizer_lbfgs = torch.optim.LBFGS(
            self.parameters,
            max_iter=100,
            history_size=100,
            tolerance_grad=1.0 * np.finfo(float).eps, 
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn='strong_wolfe'
        )
        
        # losses
        self.loss_all = list()
        self.loss_ics_all = list()
        self.loss_bcs_all = list()
        self.loss_Sbc_r_all = list()
        self.loss_pde_all = list()
        
        #
        self.X_ics_batch = []
        self.u_0 = []
        self.X_Dcs_batch = []
        self.u_Dc = []
        self.X_res_batch = []
        
        
    # Xavier initialization
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = torch.tensor(1. / np.sqrt((in_dim + out_dim) / 2.), dtype=torch.float32)
        W = torch.randn(in_dim, out_dim, dtype=torch.float32, requires_grad=True) * xavier_stddev
        return W
        
    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = torch.zeros(1, layers[l + 1], dtype=torch.float32, requires_grad=True)
            weights.append(nn.Parameter(W))
            biases.append(nn.Parameter(b))
        return weights, biases
    
    # Evaluates the forward pass
    def forward_pass_u(self, H):
        num_layers = len(self.layers_u)
        #fourier feature encoding
        H = torch.cat( [torch.sin(torch.matmul(H, self.W_u)),
                        torch.cos(torch.matmul(H, self.W_u))], 1)
        #fully connected layers
        for l in range(0, num_layers - 2):
            W = self.weights_u[l]
            b = self.biases_u[l]
            H = torch.tanh(torch.add(torch.matmul(H, W), b))
        W = self.weights_u[-1]
        b = self.biases_u[-1]
        H = torch.add(torch.matmul(H, W), b)
        return H
        
    def forward_pass_s(self, H):
        num_layers = len(self.layers_s)
        #fourier feature encoding
        H = torch.cat( [torch.sin(torch.matmul(H, self.W_s)),
                        torch.cos(torch.matmul(H, self.W_s))], 1)
        #fully connected layers
        for l in range(0, num_layers - 2):
            W = self.weights_s[l]
            b = self.biases_s[1]
            H = torch.tanh(torch.add(torch.matmul(H, W), b))
        W = self.weights_s[-1]
        b = self.biases_s[-1]
        H = torch.add(torch.matmul(H, W), b)
        return H
        
    #u(x,t)
    def net_u(self, x, t):
        return self.forward_pass_u(torch.cat([x, t], 1))
    
    #s(t)
    def net_s(self, t):
        return self.forward_pass_s(t)
    
    #{s,t} -> s_t
    def net_s_t(self, s, t):
        return torch.autograd.grad(s, t, grad_outputs=torch.ones_like(s), create_graph=True)[0] 
        
    #{u,x} -> u_x
    def net_u_x(self, u, x):
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    # darcy's law + divergence free condition
    def net_r_u(self, x, t):
        u = self.net_u(x, t)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0] 
        u_xx = torch.autograd.grad(self.alfas_r*u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0] 
        return u_xx
    
    # batcher
    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        return torch.tensor(X, dtype=torch.float32, requires_grad=True), \
               torch.tensor(Y, dtype=torch.float32), \
               torch.tensor(self.alfa.getvalue_tensor(X[:,0])/self.alfa0, dtype=torch.float32)
               
    # calculate loss
    def loss(self, batch_size):
        
        if self.iter==0 or self.iter%10==0:
            # Fetch boundary and Neumann mini-batches
            self.X_ics_batch, self.u_0, _ = self.fetch_minibatch(self.ics_sampler, batch_size)
            self.X_Dcs_batch, self.u_Dc, _ = self.fetch_minibatch(self.Dcs_sampler, batch_size)
            
            # Fetch residual mini-batch
            self.X_res_batch, _, self.alfas_r = self.fetch_minibatch(self.res_sampler, batch_size*10)
        
        # extract data
        x_0 = self.X_ics_batch[:, 0:1]
        t_0 = self.X_ics_batch[:, 1:2]
        
        x_Dc = self.X_Dcs_batch[:, 0:1]
        t_Dc = self.X_Dcs_batch[:, 1:2]
        
        x_r = self.X_res_batch[:, 0:1]
        t_r = self.X_res_batch[:, 1:2]
        
        # forward pass
        u_0_pred = self.net_u(x_0, t_0) #u(x,0), IC
        s_0_pred = self.net_s(t_0)      #s(0), IC
        
        u_Dc_pred = self.net_u(x_Dc, t_Dc) #u(0,t), BC
        
        s_pred = self.net_s(t_r)               #s(t), the moving boundary
        s_t_pred = self.net_s_t(s_pred, t_r)    #s_t(t) at the moving boundary
        u_Sbc_pred = self.net_u(s_pred, t_r)   #u(s(t),t) at the moving boundary
        
        alfas = torch.tensor(self.alfa.getvalue_tensor(s_pred[:,0].detach())/self.alfa0, dtype=torch.float32)
        v_Sbc_pred = alfas*self.net_u_x(u_Sbc_pred, s_pred) #velocity at the moving boundary
        r_u_pred = self.net_r_u(x_r, t_r) #* (x_r<=s_pred).float()  #PDE residual, divergence free
        
        # losses
        loss_u_0 = torch.mean((u_0_pred-self.u_0)**2)       #IC, pressure
        loss_s_0 = torch.mean((s_0_pred-self.s0)**2)   #IC, s==x_in at initial condition
        
        loss_Sbc_u = torch.mean((u_Sbc_pred-self.p_out)**2)  #moving boundary, p=p_out
        loss_Sbc_r = torch.mean((s_t_pred*self.vc+v_Sbc_pred)**2)    #moving boundary, -alfa*p_x=s_t
        loss_uDc = torch.mean((u_Dc_pred-self.u_Dc)**2)      #fixed BC, p=p_in
        loss_pde = torch.mean((r_u_pred)**2)    #darcy PDE, divergence free
        
        ## loss for pressure in the un-filled region
        loss_u_unfilled = torch.mean((self.net_u(x_r, t_r)-self.p_out)**2*(x_r>s_pred).float())
        
        # Total loss
        loss_ics = loss_u_0 + loss_s_0
        loss_bcs = loss_uDc
        loss = loss_bcs + loss_ics + loss_pde + loss_Sbc_r + loss_Sbc_u #+ loss_u_unfilled
        
        # record the loss
        self.loss_all.append(loss.item())
        self.loss_bcs_all.append(loss_bcs.item())
        self.loss_ics_all.append(loss_ics.item())
        self.loss_Sbc_r_all.append(loss_Sbc_r.item())
        self.loss_pde_all.append(loss_pde.item())
        
        #
        self.iter += 1
        # Print
        if self.iter % 20 == 0:
            elapsed = timeit.default_timer() - self.timer
            print('It: %d, Loss: %.3e, l_ics: %.3e, l_bcs: %.3e, l_Sbc_r: %.3e, l_pde: %.3e, Time: %.2f' %
                  (self.iter, loss.item(), loss_ics.item(), loss_bcs.item(), loss_Sbc_r.item(), loss_pde.item(), elapsed))
            self.timer = timeit.default_timer()
        
        return loss
    
    
    # trainer with ADAM
    def train_adam(self, nIter=10000, batch_size=128, lr=1e-3):
        # assign the learning rate to ADAM optimizer
        for g in self.optimizer_adam.param_groups:
            g['lr'] = lr
            
        # training loop
        self.timer = timeit.default_timer()
        for it in range(nIter):
            self.optimizer_adam.zero_grad()
            
            # loss
            loss = self.loss(batch_size)
            
            # Optimization step
            loss.backward()
            self.optimizer_adam.step()
        
        
    # trainer with LBFGS
    def closure(self):
        self.optimizer_lbfgs.zero_grad()
        loss = self.loss(self.batch_size)
        loss.backward()
        return loss
        
    def train_lbfgs(self, nIter=10000, batch_size=128):
        self.batch_size = batch_size
        for it in range(nIter):
            self.optimizer_lbfgs.step(self.closure)
        
    # predictor
    def predict_u(self, X_star):
        x_u_star = torch.tensor(X_star[:, 0:1], dtype=torch.float32)
        t_u_star = torch.tensor(X_star[:, 1:2], dtype=torch.float32)
        u_star = self.net_u(x_u_star, t_u_star)
        return u_star.detach().numpy()
        
    def predict_s(self, X_star):
        t_u_star = torch.tensor(X_star[:, 1:2], dtype=torch.float32)
        s_star = self.net_s(t_u_star)
        return s_star.detach().numpy()




# Domain boundaries
xmax = 10       #mm
tmax = 1000    #s

# Inlet / outlet pressure
p_in = 1e0  #inlet pressure, MPa
p_out = 0  #outlet pressure, MPa

# inital flow front location
x_in = 0.1*xmax    #inlet region

# local permeability field (K/mu)
#alfa = 1e-2
#alfa0 = alfa

H0 = 1e2
An = [100,50]
Bn = [0,0]
c = 2*np.pi/10  #angular frequency
mu = 1.

alfa = local_alfa(H0,An,Bn,c,mu)

#calculate the average (numerically)
x_lst = np.linspace(0, xmax, 100000)
alfa0 = sum(alfa.getvalue(x_lst)) / len(x_lst)

plt.figure()
plt.plot(x_lst, alfa.getvalue(x_lst), '*-')
plt.xlabel('x')
plt.ylabel('alfa')


# samplers
ic_coords = np.array([[0.0, 0.0],
                      [1, 0.0]])  #[[x,t]]
Dc_coords = np.array([[0.0, 0.0],
                      [0.0, 1]])  #[[x,t]]
dom_coords = np.array([[0.0, 0.0],
                       [1, 1]]) #[[x,t]]

# characteristic time
vc = xmax**2 / alfa0 / p_in / tmax
print('characteristic velocity: ', vc)

# sample the domain
ics_sampler = Sampler(2, ic_coords, lambda x: u_0(x, 1, p_out/p_in, x_in/xmax), name='Initial Condition')
Dcs_sampler = Sampler(2, Dc_coords, lambda x: g(x, 1), name='Direchlet Boundary Condition')
res_sampler = Sampler(2, dom_coords, lambda x: u(x), name='Forcing')


#
layers_p = [100, 100, 100, 100, 100, 1]  #(x,t) -> p
layers_s = [100, 100, 100, 100, 100, 1]  #t->s

model = Stefan1D_direct(ics_sampler,
                        Dcs_sampler,
                        res_sampler,
                        layers_p,
                        layers_s,
                        alfa,
                        alfa0,
                        p_out/p_in,
                        x_in/xmax,
                        vc)

model.train_adam(nIter=1000, batch_size=1000, lr=1e-3)
model.train_lbfgs(nIter=100, batch_size=1000)


# training curve
plt.figure()
plt.plot(model.loss_all, label='total loss')
plt.title('total loss')
plt.yscale("log")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.figure()
plt.plot(model.loss_ics_all, label='ic loss')
plt.title('ic loss')
plt.yscale("log")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.figure()
plt.plot(model.loss_bcs_all, label='bc loss')
plt.title('bc loss')
plt.yscale("log")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.figure()
plt.plot(model.loss_Sbc_r_all, label='Sbc residual loss')
plt.title('Sbc residual loss')
plt.yscale("log")
plt.xlabel('epoch')
plt.ylabel('loss')


# Test data
nn_x = 300
nn_t = 300
x = np.linspace(0, xmax, nn_x)[:, None]
t = np.linspace(0, tmax, nn_t)[:, None]
X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
 
# PINN prediction
u_pred = model.predict_u(X_star/np.array((xmax, tmax)))
s_pred = model.predict_s(X_star/np.array((xmax, tmax)))

# scale back
u_pred = u_pred * p_in
s_pred = s_pred * xmax


from scipy.interpolate import griddata
U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
S_pred = griddata(X_star, s_pred.flatten(), (X, T), method='cubic')

plt.figure()
plt.pcolor(X, T, U_pred, cmap='jet')
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$t$')
plt.title('Predicted $p(x,t)$')

plt.plot(model.X_Dcs_batch[:,0].detach()*xmax, model.X_Dcs_batch[:,1].detach()*tmax,'xw')
plt.plot(model.X_ics_batch[:,0].detach()*xmax, model.X_ics_batch[:,1].detach()*tmax,'xw')
plt.plot(model.X_res_batch[:,0].detach()*xmax, model.X_res_batch[:,1].detach()*tmax,'xw')

for i in range(nn_t):
    for j in range(nn_x):
        x_ij = np.array([X[i,j], T[i,j]]).reshape(1,2)[0][0]
        if x_ij > S_pred[i,j]:
            U_pred[i,j] = np.nan

plt.figure()
plt.pcolor(X, T, U_pred, cmap='jet')
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$t$')
plt.title('Predicted $p(x,t)$')

fig,ax = plt.subplots()
h1,=ax.plot(X_star[:,1], s_pred, '*-', label='PINN')
solver = ExactSol(H0, An, Bn, c, x_in, p_in-p_out, mu)
s_exact = []
for t in X_star[:,1]:
    s = solver.solve_s(t)[0]
    s_exact.append(s)
s_exact = np.array(s_exact)
h2,=ax.plot(X_star[:,1], s_exact, '.-', label='Exact')
plt.xlabel(r'$t$')
plt.ylabel(r'$s$')
ax.legend(handles=[h1,h2])

fig,ax=plt.subplots()
h1,=ax.plot(U_pred[0,:], label='t=0')
h2,=ax.plot(U_pred[30,:], label='t=0.1 tmax')
h3,=ax.plot(U_pred[150,:], label='t=0.5 tmax')
h4,=ax.plot(U_pred[299,:], label='t=tmax')
plt.xlabel(r'$x$')
plt.ylabel(r'$p$')
ax.legend(handles=[h1,h2,h3,h4])

plt.show()



