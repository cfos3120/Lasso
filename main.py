import yaml

import torch
from model.FNO_3D import FNO3d
from training_utils.train import model_routine
from training_utils.train_cavity import model_routine_cavity

import os
import socket
from argparse import ArgumentParser

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 1. Parse arguments and load configurations
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, default= os.path.dirname(__file__)+r'/default.yaml', help='Path to the configuration file')
    parser.add_argument('--run_name', type=str, default= 'debug', help='Name of test within configuration file')
    options = parser.parse_args()
    config_file = options.config_path
    run_name = options.run_name
    
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    if run_name in config:
        run_config = config[run_name]
    else: raise ValueError(f'{run_name} is not supported by yaml file')
    
    # Check if it has time dimension or not
    for run_type in ['train','eval','fine']:
            if run_type in run_config:
                if run_config[run_type]['data']['problem_type'] == 'cavity':
                    grid_dims = 2
                else:
                    grid_dims = 3

    # 2. Create model and load in model version
    model = FNO3d(modes1=run_config['model']['modes1'],
                  modes2=run_config['model']['modes2'],
                  modes3=run_config['model']['modes3'],
                  fc_dim=run_config['model']['fc_dim'],
                  layers=run_config['model']['layers'],
                  in_dim=run_config['model']['in_channels'] + grid_dims,
                  out_dim=run_config['model']['out_channels']).to(device)
    
    # 3. Override configuration file if running locally (for debugging purposes)
    if socket.gethostname() == 'DESKTOP-157DQSC':
        #rds_data_path = r'Z:\PRJ-MLFluids\datasets'
        rds_data_path = r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\PINO datasets'
        for run_type in ['train','eval','fine']:
            if run_type in run_config:
                server_datapath = run_config[run_type]['data']['datapath']
                file_name = server_datapath.split('/')[-1]
                run_config[run_type]['data']['datapath'] = rds_data_path + '/sample_' + file_name
                run_config[run_type]['data']['n_sample'] = 2
                if run_type != 'eval':
                    run_config[run_type]['epochs'] = 2
                    run_config[run_type]['data_iter'] = 1
            
                if run_config[run_type]['data']['problem_type'] == 'cavity':
                    grid_dims = 2
                else:
                    grid_dims = 3
    else: pass
    
    # 4. Run scripts based on configurations
    if 'train' in run_config:
        if run_config['train']['data']['problem_type'] == 'cavity':
            model_routine_cavity(run_config['train'], model)
        else:
            model_routine(run_config['train'], model)
        print('Training Complete \n\n')
    
    if 'eval' in run_config:
        model_routine(run_config['eval'], model)
        print('Evaluation Complete \n\n')
    
    if 'fine' in run_config:
        model_routine(run_config['fine'], model)
        print('Fine-tuning Complete \n\n')