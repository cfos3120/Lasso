import numpy as np
import socket
from argparse import ArgumentParser

class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        self.data = scipy.io.loadmat(self.file_path)
        self.old_mat = True

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

if __name__ == '__main__':
    
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--filename', default = 'sample_NS_fft_Re500_T4000_cartesian.npy', type=str, help='Path to the dataset file')
    parser.add_argument('--batches', default = 2, type=int, help='Number of Batches to sample')
    parser.add_argument('--timesteps', default = 5, type=int, help='Number of Timesteps to sample (second shape index of dataset)')
    parser.add_argument('--space_sub', default = 1, type=int, help='Divsior for Spatial subsampling (third, fourht index of dataset)')
    
    options = parser.parse_args()

    if socket.gethostname() == 'DESKTOP-157DQSC':
        path = r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\PINO datasets'
    else:
        path = '/project/MLFluids'


    file = path + f'/{options.filename}'

    if options.filename == 'cavity.mat':
        reader = MatReader(file)
        data_u = reader.read_field('u')[:options.timesteps, ::space_sub, ::space_sub]
        data_v = reader.read_field('v')[:options.timesteps, ::space_sub, ::space_sub]
        data_v = reader.read_field('p')[:options.timesteps, ::space_sub, ::space_sub]
    else:
        
        dataset = np.load(file)


    print(f'Loaded in Dataset {options.filename} with shape {dataset.shape}')

    if options.filename == 'cavity.npy':
        subsample = dataset[0,:options.timesteps,
                            ::options.space_sub, ::options.space_sub,
                            ...]
    if options.filename == 'cavity.mat':
        
        reader = MatReader(PATH)
        data_u = reader.read_field('u')[T_in:T_in+T*sub_t:sub_t, ::sub_s, ::sub_s].permute(1,2,0)
        data_v = reader.read_field('v')[T_in:T_in+T*sub_t:sub_t, ::sub_s, ::sub_s].permute(1,2,0)
        data_v = reader.read_field('p')[T_in:T_in+T*sub_t:sub_t, ::sub_s, ::sub_s].permute(1,2,0)

        subsample = dataset[0,:options.timesteps,
                            ::options.space_sub, ::options.space_sub,
                            ...]
    else:
        subsample = dataset[:options.batches, 
                            :options.timesteps,
                            ::options.space_sub, ::options.space_sub,
                            ...]

    print('Subsampled to shape {dataset.shape}')

    file_name_to_save = 'subsample_' + options.filename
    np.save(file_name_to_save, subsample)
    print(f'Saved Dataset as {file_name_to_save} to working directory')

