
import numpy as np
import torch
import torch.fft as fft
import socket
import math
from random_fields import GaussianRF
from timeit import default_timer
import argparse
import os


class NavierStokes2d(object):

    def __init__(self, s1, s2, L1=2*math.pi, L2=2*math.pi, device=None, dtype=torch.float64):

        self.s1 = s1
        self.s2 = s2

        self.L1 = L1
        self.L2 = L2

        self.h = 1.0/max(s1, s2)

        #Wavenumbers for first derivatives
        freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),\
                                torch.zeros((1,)),\
                                torch.arange(start=-s1//2 + 1, end=0, step=1)), 0)
        self.k1 = freq_list1.view(-1,1).repeat(1, s2//2 + 1).type(dtype).to(device)


        freq_list2 = torch.cat((torch.arange(start=0, end=s2//2, step=1), torch.zeros((1,))), 0)
        self.k2 = freq_list2.view(1,-1).repeat(s1, 1).type(dtype).to(device)

        #Negative Laplacian
        freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),\
                                torch.arange(start=-s1//2, end=0, step=1)), 0)
        k1 = freq_list1.view(-1,1).repeat(1, s2//2 + 1).type(dtype).to(device)

        freq_list2 = torch.arange(start=0, end=s2//2 + 1, step=1)
        k2 = freq_list2.view(1,-1).repeat(s1, 1).type(dtype).to(device)

        self.G = ((4*math.pi**2)/(L1**2))*k1**2 + ((4*math.pi**2)/(L2**2))*k2**2

        #Inverse of negative Laplacian
        self.inv_lap = self.G.clone()
        self.inv_lap[0,0] = 1.0
        self.inv_lap = 1.0/self.inv_lap

        #Dealiasing mask using 2/3 rule
        self.dealias = (self.k1**2 + self.k2**2 <= 0.6*(0.25*s1**2 + 0.25*s2**2)).type(dtype).to(device)
        #Ensure mean zero
        self.dealias[0,0] = 0.0

    #Compute stream function from vorticity (Fourier space)
    def stream_function(self, w_h, real_space=False):
        #-Lap(psi) = w
        psi_h = self.inv_lap*w_h

        if real_space:
            return fft.irfft2(psi_h, s=(self.s1, self.s2))
        else:
            return psi_h

    #Compute velocity field from stream function (Fourier space)
    def velocity_field(self, stream_f, real_space=True):
        #Velocity field in x-direction = psi_y
        q_h = (2*math.pi/self.L2)*1j*self.k2*stream_f

        #Velocity field in y-direction = -psi_x
        v_h = -(2*math.pi/self.L1)*1j*self.k1*stream_f

        if real_space:
            return fft.irfft2(q_h, s=(self.s1, self.s2)), fft.irfft2(v_h, s=(self.s1, self.s2))
        else:
            return q_h, v_h

    #Compute non-linear term + forcing from given vorticity (Fourier space)
    def nonlinear_term(self, w_h, f_h=None):
        #Physical space vorticity
        w = fft.irfft2(w_h, s=(self.s1, self.s2))

        #Physical space velocity
        q, v = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)

        #Compute non-linear term in Fourier space
        nonlin = -1j*((2*math.pi/self.L1)*self.k1*fft.rfft2(q*w) + (2*math.pi/self.L1)*self.k2*fft.rfft2(v*w))

        #Add forcing function
        if f_h is not None:
            nonlin += f_h

        return nonlin
    
    def time_step(self, q, v, f, Re):
        #Maxixum speed
        max_speed = torch.max(torch.sqrt(q**2 + v**2)).item()

        #Maximum force amplitude
        if f is not None:
            xi = torch.sqrt(torch.max(torch.abs(f))).item()
        else:
            xi = 1.0
        
        #Viscosity
        mu = (1.0/Re)*xi*((self.L1/(2*math.pi))**(3.0/4.0))*(((self.L2/(2*math.pi))**(3.0/4.0)))

        if max_speed == 0:
            return 0.5*(self.h**2)/mu
        
        #Time step based on CFL condition
        return min(0.5*self.h/max_speed, 0.5*(self.h**2)/mu)

    def advance(self, w, f=None, T=1.0, Re=100, adaptive=True, delta_t=1e-3):

        #Rescale Laplacian by Reynolds number
        GG = (1.0/Re)*self.G

        #Move to Fourier space
        w_h = fft.rfft2(w)

        if f is not None:
            f_h = fft.rfft2(f)
        else:
            f_h = None
        
        if adaptive:
            q, v = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)
            delta_t = self.time_step(q, v, f, Re)

        time  = 0.0
        #Advance solution in Fourier space
        while time < T:
            if time + delta_t > T:
                current_delta_t = T - time
            else:
                current_delta_t = delta_t

            #Inner-step of Heun's method
            nonlin1 = self.nonlinear_term(w_h, f_h)
            w_h_tilde = (w_h + current_delta_t*(nonlin1 - 0.5*GG*w_h))/(1.0 + 0.5*current_delta_t*GG)

            #Cranck-Nicholson + Heun update
            nonlin2 = self.nonlinear_term(w_h_tilde, f_h)
            w_h = (w_h + current_delta_t*(0.5*(nonlin1 + nonlin2) - 0.5*GG*w_h))/(1.0 + 0.5*current_delta_t*GG)

            #De-alias
            w_h *= self.dealias

            #Update time
            time += current_delta_t

            #New time step
            if adaptive:
                q, v = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)
                delta_t = self.time_step(q, v, f, Re)
        
        return fft.irfft2(w_h, s=(self.s1, self.s2))
    
    def __call__(self, w, f=None, T=1.0, Re=100, adaptive=True, delta_t=1e-3):
        return self.advance(w, f, T, Re, adaptive, delta_t)

# This script is default for dataset generation
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--re", type=float, default=500.0)
    parser.add_argument("--s", type=int, default=64)
    parser.add_argument("--t", type=int, default=32)
    parser.add_argument("--s_sub", type=int, default=1)
    parser.add_argument("--t_slice", type=float, default=1)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--dt", type=float, default=1e-3)
    opt = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    solver = NavierStokes2d(opt.s,opt.s)

    # Initialize Random Flow field and step forward to develop flow 
    GRF = GaussianRF(2, opt.s, 2 * math.pi, alpha=2.5, tau=7, device=device)
    w = GRF.sample(1)

    # forcing
    y = np.linspace(0, 2*np.pi, opt.s + 1)[:-1]
    x = y
    XX, YY = np.meshgrid(x, y, indexing='ij')
    forcing = -4*np.cos(4*YY)
    
    # Initialise dataset storage 
    # Path
    if socket.gethostname() == 'DESKTOP-157DQSC':
        data_path = r'Z:\PRJ-MLFluids\datasets/'
        data_path = r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\PINO datasets/'
    else:
        data_path = r'/home/cfos3120/datasets/'

    # Base name
    dataset_name = f'kflow_re{int(opt.re)}_b{opt.batches}_t{opt.t}_s{opt.s}_sub{opt.s_sub}'

    ckpt_dir = data_path + '%s/' % dataset_name
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    print(f'Saving results in folder {ckpt_dir}', end='\n')

    # Pytorch dataset
    sol_cartesian = np.zeros((opt.batches, opt.t + 1, opt.s // opt.s_sub, opt.s // opt.s_sub, 2))
    sol_vorticity = np.zeros((opt.batches, opt.t + 1, opt.s // opt.s_sub, opt.s // opt.s_sub, 1))
    sol_info = {'Reynolds Number' : int(opt.re),
                'Solver Domain Size' : int(opt.s),
                'Subsample Divisor' : int(opt.s_sub),
                'Subsample Output' : int(opt.s/opt.s_sub),
                'Solver Time-Step': int(opt.dt),
                'Total Time Between Slices': int(opt.t_slice),
                'Time Slices per Batch' : int(opt.t),
                'Number of Batches' : int(opt.batches)
                }
    
    np.save(ckpt_dir + dataset_name + '_info', sol_info); print('Saved Solver Info')
    np.save(ckpt_dir + dataset_name + '_forcing', forcing); print('Saved Forcing Distribution')

    # Solve for each batch
    for batch in range(opt.batches):

        # Initialise Fluid Field
        w = GRF.sample(1)
        stream_function_h = solver.stream_function(fft.rfft2(w), real_space=False)
        u, v = solver.velocity_field(stream_function_h, real_space=True)
        
        # Store initial conditions
        sol_cartesian[batch, 0, :, :, 0], sol_cartesian[batch, 0, :, :, 1] = u[...,::opt.s_sub, ::opt.s_sub].squeeze(0) , v[...,::opt.s_sub, ::opt.s_sub].squeeze(0)
        sol_vorticity[batch, 0, :, :, 0] = w[...,::opt.s_sub, ::opt.s_sub].squeeze(0)

        # Solve for each timeslice
        for i in range(opt.t):
            print(f'Preparing Batch: {int(batch)} Timestep: {int(i)}', end='\r')
            w = solver.advance(w=w, f=torch.tensor(forcing), T=opt.t_slice, Re=opt.re, adaptive=False, delta_t=opt.dt)

            # Convert to Cartesian Velocity
            stream_function_h = solver.stream_function(fft.rfft2(w), real_space=False)
            u, v = solver.velocity_field(stream_function_h, real_space=True)
            
            # Store current timeslice field
            sol_cartesian[batch, i+1, :, :, 0], sol_cartesian[batch, i+1, :, :, 1] = u[...,::opt.s_sub, ::opt.s_sub].squeeze(0) , v[...,::opt.s_sub, ::opt.s_sub].squeeze(0)
            sol_vorticity[batch, i+1, :, :, 0] = w[...,::opt.s_sub, ::opt.s_sub].squeeze(0)

    np.save(ckpt_dir + '_cartesian', sol_cartesian); print('Saved Cartesian Solution')
    np.save(ckpt_dir + '_vorticity', sol_vorticity); print('Saved Vorticity Solution')
    
    
    