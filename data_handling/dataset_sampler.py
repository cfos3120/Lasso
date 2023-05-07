import numpy as np
import socket
from argparse import ArgumentParser

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
    dataset = np.load(file)
    print('Loaded in Dataset {options.filename} with shape {dataset.shape}')

    subsample = dataset[:options.batches, 
                        :options.timesteps,
                        ::options.space_sub, ::options.space_sub,
                        ...]

    print('Subsampled to shape {dataset.shape}')

    file_name_to_save = 'subsample_' + options.filename
    np.save(subsample, file_name_to_save)
    print('Saved Dataset as {file_name_to_save} to working directory')

