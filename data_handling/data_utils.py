import torch
import numpy as np

def get_grid3d(S, T, time_scale=1.0, device='cpu'):
    gridx = torch.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=torch.float, device=device)
    gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T, 1])
    gridy = torch.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=torch.float, device=device)
    gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T, 1])
    gridt = torch.tensor(np.linspace(0, 1 * time_scale, T), dtype=torch.float, device=device)
    gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S, S, 1, 1])
    return gridx, gridy, gridt

class NSLoader(object):
    def __init__(self, datapath, nx, nt, sub=1, sub_t=1, t_interval=1.0):
        '''
        Load data from npy and reshape to (N, X, Y, T, C)
        Args:
            datapath1: path to data
            nx:
            nt:
            sub:
            sub_t:
            t_interval:
        '''

        self.S = nx // sub
        self.T = int(nt * t_interval) // sub_t + 1
        self.time_scale = t_interval
        data1 = np.load(datapath)
        data1 = torch.tensor(data1, dtype=torch.float)[..., ::sub_t, ::sub, ::sub]
        
        # Add channel dimension if it doesnt exist
        if len(data1.shape) == 4:
            data1 = data1.reshape(data1.shape[0],
                                  data1.shape[1],
                                  data1.shape[2],
                                  data1.shape[3],
                                  1)
        self.C = data1.shape[-1]

        if t_interval == 0.5:
            data1 = self.extract(data1)
        part1 = data1.permute(0, 2, 3, 1, 4)
        self.data = part1

    def make_loader(self, n_sample, batch_size, start=0, train=True):
        if train:
            a_data = self.data[start:start + n_sample, :, :, 0, :].reshape(n_sample, self.S, self.S, 1, self.C)
            u_data = self.data[start:start + n_sample, ...].reshape(n_sample, self.S, self.S, self.T, self.C)
        else:
            a_data = self.data[-n_sample:, :, :, 0, :].reshape(n_sample, self.S, self.S, 1, self.C)
            u_data = self.data[-n_sample:, ...].reshape(n_sample, self.S, self.S, self.T, self.C)
        a_data = a_data.reshape(n_sample, self.S, self.S, 1, self.C).repeat([1, 1, 1, self.T, 1])
        gridx, gridy, gridt = get_grid3d(self.S, self.T, time_scale=self.time_scale)
        a_data = torch.cat((gridx.repeat([n_sample, 1, 1, 1, self.C]), gridy.repeat([n_sample, 1, 1, 1, self.C]),
                            gridt.repeat([n_sample, 1, 1, 1, self.C]), a_data), dim=-1)
        dataset = torch.utils.data.TensorDataset(a_data, u_data)
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)
        return self.loader

    @staticmethod
    def extract(data):
        '''
        Extract data with time range 0-0.5, 0.25-0.75, 0.5-1.0, 0.75-1.25,...
        Args:
            data: tensor with size N x 129 x 128 x 128

        Returns:
            output: (4*N-1) x 65 x 128 x 128
        '''
        T = data.shape[1] // 2
        interval = data.shape[1] // 4
        N = data.shape[0]
        new_data = torch.zeros(4 * N - 1, T + 1, data.shape[2], data.shape[3], data.shape[4])
        for i in range(N):
            for j in range(4):
                if i == N - 1 and j == 3:
                    # reach boundary
                    break
                if j != 3:
                    new_data[i * 4 + j] = data[i, interval * j:interval * j + T + 1]
                else:
                    new_data[i * 4 + j, 0: interval] = data[i, interval * j:interval * j + interval]
                    new_data[i * 4 + j, interval: T + 1] = data[i + 1, 0:interval + 1]
        return new_data
    
def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def load_dataset(args):
    
    #Load in Dataset and create `Loader`
    data_config = args['data']
    loader = NSLoader(datapath=data_config['datapath'],
                        nx=data_config['nx'], nt=data_config['nt'],
                        sub=data_config['sub'], sub_t=data_config['sub_t'],
                        t_interval=data_config['time_interval'])

    data_loader = loader.make_loader(data_config['n_sample'],
                                      batch_size=args['batchsize'],
                                      start=data_config['offset'],
                                      train=data_config['shuffle'])

    print(f'Data loaded. Resolution : {loader.data.shape}')
    
    return data_loader