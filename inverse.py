import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Network(nn.Module):

    def __init__(self, input, layers, hidden, output):
        super().__init__()
        # activation function
        activation = nn.Tanh
        # input layer
        self.fci = nn.Sequential(*[nn.Linear(input, hidden), activation()])
        # hidden layers
        self.fch = nn.Sequential(*[nn.Sequential(*[nn.Linear(hidden, hidden), activation()]) for i in range(layers-1)])
        # output layer
        self.fco = nn.Linear(hidden, output)

  # forward propagation
    def forward(self, x, t):
        x = torch.cat((x, t), dim = 1)
        x = self.fci(x)
        x = self.fch(x)
        x = self.fco(x)
        return x
  
    def net_c_t(self, c, t):
        return torch.autograd.grad(c, t, grad_outputs=torch.ones_like(t), create_graph=True)[0]
  
    def net_c_r(self, c, r):
        return torch.autograd.grad(c, r, grad_outputs=torch.ones_like(r), create_graph=True)[0]

    def net_c_rr(self, c, t):
        c_r = self.net_c_r(c, t)
        c_rr = torch.autograd.grad(c_r, t, grad_outputs=torch.ones_like(c_r), create_graph=True)[0]
        return c_rr

    def loss_phys(self, r, t):
        # compute the gradients
        c = self.forward(r, t)
        c_r = self.net_c_r(c, r)
        c_t = self.net_c_t(c, t)
        c_rr = self.net_c_rr(c, t)

        D = self.Diffusion_coefficient(c, D_ref)
        
        # loss function
        loss = r**2 * c_t - D*(r/Rs)**2 * c_rr - 2*D*r/Rs * c_r
        loss = loss**2
        return loss
    
    def Diffusion_coefficient(self, c, D_ref):
        # constants
        c_max = 4.665e4
        C_theory = 277.84
        C_practical = 160

        # equations
        SOC = (c_max - c/c_max)/c_max * C_theory/C_practical
        D = D_ref * (1 + 100*SOC**(3/2))
        return D_ref * (1 + 0.5 * c)

    def RMS_loss(self, r, t, c_max):
        c = self.forward(r, t)
        loss = c*c_max - self.numerical_solution(r, t)
        return torch.sqrt(torch.mean(loss**2))
    
    def numerical_solution(self, r, t):
        return data[round(10000*r), round(400*t)]
    
    def total_loss(self, r, t, c_max):
        return self.loss_phys(r, t) + self.RMS_loss(r, t, c_max)
    
    def train(self, r, t, epochs, optimizer):
        Network.constants()
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self.total_loss(r, t, c_max)
            loss.backward()
            optimizer.step()
            
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss}', end='\r')

    def constants(self):
        Rs = 5e-6
        c_max = 4.665e4

    def 

        
        

# parameters  
device = 'cpu'
data = pd.read_csv('MATLAB SOLVER\data.csv')

pinn = Network(2, 4, 64, 1).to(device)

D_ref = torch.nn.Parameter(torch.zeros(1, requires_grad = True).to(device))
optimiser = torch.optim.Adam(list(pinn.parameters()) + [D_ref], lr=0.001)

# training







    
    


