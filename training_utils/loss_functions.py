import torch
import numpy as np

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

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

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
        return self.rel(x, y)

def FDM_NS_vorticity(w, nu=1/40, t_interval=1.0):
    
    assert w.shape[-1] == 1
    w = w.squeeze(-1)

    batchsize = w.size(0)
    nx = w.size(1)
    ny = w.size(2)
    nt = w.size(3)
    device = w.device
    w = w.reshape(batchsize, nx, ny, nt)

    w_h = torch.fft.fft2(w, dim=[1, 2])
    # Wavenumbers in y-direction
    k_max = nx//2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(N, 1).repeat(1, N).reshape(1,N,N,1)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, N).repeat(N, 1).reshape(1,N,N,1)
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[0, 0, 0, 0] = 1.0
    f_h = w_h / lap

    ux_h = 1j * k_y * f_h
    uy_h = -1j * k_x * f_h
    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    wlap_h = -lap * w_h

    ux = torch.fft.irfft2(ux_h[:, :, :k_max + 1], dim=[1, 2])
    uy = torch.fft.irfft2(uy_h[:, :, :k_max + 1], dim=[1, 2])
    wx = torch.fft.irfft2(wx_h[:, :, :k_max+1], dim=[1,2])
    wy = torch.fft.irfft2(wy_h[:, :, :k_max+1], dim=[1,2])
    wlap = torch.fft.irfft2(wlap_h[:, :, :k_max+1], dim=[1,2])

    dt = t_interval / (nt-1)
    wt = (w[:, :, :, 2:] - w[:, :, :, :-2]) / (2 * dt)

    Du1 = wt + (ux*wx + uy*wy - nu*wlap)[...,1:-1]
    return Du1

def FDM_NS_cartesian(u, nu=1/500, t_interval=1.0):

    assert u.shape[-1] == 2
    assert u.shape[0] > 1 #--> There is a bug with torch.gradient which doesnt handle small batches well

    batchsize = u.size(0)
    nx = u.size(1)
    ny = u.size(2)
    nt = u.size(3)
    device = u.device

    # Assuming uniform periodic spatial grid NOTE: These need to line up with the grid function made for training.
    x = torch.arange(0,1.0,1.0/nx, device=device) #<- keep the domain non-dimensional
    y = torch.arange(0,1.0,1.0/ny, device=device)
    t = torch.arange(0,t_interval,t_interval/(nt-1), device=device)

    # each of these (dV_dx etc.) should come with shape (Batch,x,y,t,Velocity direction)
    dV_dx, dV_dy, dV_dt = torch.gradient(u, spacing =tuple([x, y, t]), dim = [1,2,3])
    dV_dxx = torch.gradient(dV_dx, spacing = tuple([x]), dim = 1)[0]
    dV_dyy = torch.gradient(dV_dy, spacing = tuple([y]), dim = 2)[0]

    eqn_c = dV_dx[...,0] + dV_dy[...,1]
    eqn_mx = nu * (dV_dxx[...,0] + dV_dyy[...,0]) - dV_dt[...,0] - u[...,0]*dV_dx[...,0] - u[...,1]*dV_dy[...,0]
    eqn_my = nu * (dV_dxx[...,1] + dV_dyy[...,1]) - dV_dt[...,1] - u[...,0]*dV_dx[...,1] - u[...,1]*dV_dy[...,1]

    return eqn_c, eqn_mx, eqn_my

def FDM_NS_cartesian_hard(A):
    # This is based on the model producing a vector potential A, instead of output velocity V.
    # Thus gradients of A need to be calculated to enforce mass conservation.

    assert A.shape[-1] == 2
    assert A.shape[0] > 1 #--> There is a bug with torch.gradient which doesnt handle small batches well

    batchsize = A.size(0)
    nx = A.size(1)
    ny = A.size(2)
    nt = A.size(3)
    device = A.device

    # Assuming uniform periodic spatial grid NOTE: These need to line up with the grid function made for training.
    x = torch.arange(0,1.0,1.0/nx, device=device) #<- keep the domain non-dimensional
    y = torch.arange(0,1.0,1.0/ny, device=device)

    # each of these (dV_dx etc.) should come with shape (Batch,x,y,t,Velocity direction)
    dA_dx, dA_dy = torch.gradient(A, spacing =tuple([x, y]), dim = [1,2])

    V_tilde = dA_dx + dA_dy

    return V_tilde

def PINO_loss3d_decider(model_input, model_output, model_val, forcing_type, nu, t_interval):
    
    B = model_output.shape[0]
    S = model_output.shape[1]
    T = model_output.shape[3]
    C = model_output.shape[4]
    device = model_output.device

    # Set Loss function
    lploss = LpLoss(size_average=True)
    zero_tensor = torch.zeros(1)

    # Inital condition is the same across all conditions and test types (3rd index onwards excludes the grid)
    u0 = model_input[:, :, :, 0, 3:] 
    loss_ic = lploss(model_output[:, :, :, 0, :], u0)

    # Select what conditions to use for PDE Loss (vorticity, cartesian, cavity)
    if forcing_type == 'vorticity_periodic_short':
        x2 = torch.tensor(np.linspace(0, 2*np.pi, S+1)[:-1], dtype=torch.float).reshape(1, S).repeat(S, 1)
        forcing = -4 * (torch.cos(4*(x2))).reshape(1,S,S,1).repeat(B, 1, 1, T-2).to(device)
        Dw = FDM_NS_vorticity(model_output, nu=nu, t_interval=t_interval)

        loss_w = lploss(Dw, forcing)
        loss_bc, loss_c, loss_m1, loss_m2 = zero_tensor, zero_tensor, zero_tensor, zero_tensor
        loss_l2 = lploss(model_output, model_val)

    elif forcing_type == 'cartesian_periodic_short':
        pass
        forcing_x = 'enter here'
        forcing_y = 'enter here'
        eqn_c, eqn_mx, eqn_my = FDM_NS_cartesian(model_output, nu=nu, t_interval=t_interval)
        
        loss_c = lploss.abs(eqn_c, torch.zeros_like(eqn_c))
        loss_m1 = lploss.abs(eqn_mx, forcing_x)
        loss_m2 = lploss.abs(eqn_my, forcing_y)
        loss_bc, loss_w = zero_tensor, zero_tensor
        loss_l2 = lploss(model_output, model_val)

    elif forcing_type == 'cavity':
        pass 
        eqn_c, eqn_mx, eqn_my = FDM_NS_cartesian(model_output, nu=nu, t_interval=t_interval)
    
    elif forcing_type == 'cartesian_periodic_short_hard_loss':
        V_tilde = FDM_NS_cartesian(model_output)
        loss_bc, loss_w, loss_c, loss_m1, loss_m2 = zero_tensor, zero_tensor, zero_tensor, zero_tensor, zero_tensor
        loss_l2 = lploss(V_tilde, model_val)

    else: 
        raise(ValueError)


    return loss_l2, loss_ic, loss_bc, loss_w, loss_c, loss_m1, loss_m2
        