import torch
import torch.nn.functional as F
import numpy as np
from .fdm_stencils import *
class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.abs(x, y)

def loss_selector(loss_type_name):
    if loss_type_name == 'LPLoss':
        return LpLoss()
    elif loss_type_name == 'MSE':
        return torch.nn.MSELoss()
    else: raise(NameError)

def PINO_loss_calculator(args, model_output, loss_function):

    B = model_output.shape[0]
    S = model_output.shape[1]
    T = model_output.shape[3]
    C = model_output.shape[4]
    device = model_output.device

    if args['data']['problem_type'] == 'vorticity_periodic':

        # initialize forcing function to compare to
        x2 = torch.tensor(np.linspace(0, 2*np.pi, S+1)[:-1], dtype=torch.float).reshape(1, S).repeat(S, 1)
        forcing = -4 * (torch.cos(4*(x2))).reshape(1,S,S,1).repeat(B, 1, 1, T-2).to(device)
        
        loss_f, losses_list = Navier_Stokes_Vorticity_RHS(args, model_output, loss_function, forcing)

    elif args['data']['problem_type'] == 'cartesian_periodic':

        # initialize forcing function to compare to
        x2 = torch.tensor(np.linspace(0, 2*np.pi, S+1)[:-1], dtype=torch.float).reshape(1, S).repeat(S, 1)
        x_forcing = -1 * (torch.sin(4*(x2))).reshape(1,S,S,1).repeat(B, 1, 1, T-2).to(device)
        y_forcing = torch.zeros_like(x_forcing)
        
        loss_f, losses_list = Navier_Stokes_Cartesian_RHS(args, model_output, loss_function, x_forcing, y_forcing)
        
    else: raise(NameError)

    return loss_f, losses_list


def Navier_Stokes_Vorticity_RHS(args, model_output, loss_function, forcing):

    nu = 1/args['data']['Re']
    t_interval = args['data']['time_interval']
    
    if args['pino_scheme'] == '2nd Order Periodic':
        RHS,__ = Navier_Stokes_Vorticity_periodic(model_output, nu=nu, t_interval=t_interval, order=2)
    elif args['pino_scheme'] == '4th Order Periodic':
        RHS,__ = Navier_Stokes_Vorticity_periodic(model_output, nu=nu, t_interval=t_interval, order=4)
    elif args['pino_scheme'] == 'Torch.Gradient':
        RHS,__ = Navier_Stokes_Vorticity_torch_gradient(model_output, nu=nu, t_interval=t_interval)
    elif args['pino_scheme'] == 'Spectral':
        RHS,__ = Navier_Stokes_Vorticity_spectral(model_output, nu=nu, t_interval=t_interval)
    else: raise(NameError)

    loss_f = loss_function(RHS, forcing)
    loss_list = {'Vorticity Loss': loss_f.item()}

    return loss_f, loss_list

def Navier_Stokes_Cartesian_RHS(args, model_output, loss_function, x_forcing, y_forcing):

    nu = 1/args['data']['Re']
    t_interval = args['data']['time_interval']
    
    if args['pino_scheme'] == '2nd Order Periodic':
        xRHS, yRHS, cRHS,__ = Navier_Stokes_Cartesian_periodic(model_output, nu=nu, t_interval=t_interval, order=2)
    elif args['pino_scheme'] == '4th Order Periodic':
        xRHS, yRHS, cRHS,__ = Navier_Stokes_Cartesian_periodic(model_output, nu=nu, t_interval=t_interval, order=4)
    #elif args['pino_scheme'] == 'Torch.Gradient':
    #    xRHS, yRHS,__ = Navier_Stokes_Cartesian_torch_gradient(model_output, nu=nu, t_interval=t_interval)
    else: raise(NameError)

    loss_fx = loss_function(xRHS, x_forcing)
    loss_fy = loss_function(yRHS, y_forcing)
    loss_fc = loss_function(cRHS, torch.zeros_like(x_forcing))
    loss_list = {'X Momentum Loss': loss_fx.item(), 'Y Momentum Loss': loss_fy.item(), 'Continuity Loss': loss_fc.item()}

    loss_f = loss_fx + loss_fy + loss_fc # For now just backwards pass X momentum.
     
    return loss_f, loss_list