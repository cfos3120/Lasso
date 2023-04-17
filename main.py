import yaml

import torch
from model.FNO_3D import FNO3d
from training_utils.optimizers import Adam
from training_utils.loss_functions import LpLoss, PINO_loss3d_decider
from data_handling.data_utils import load_dataset, sample_data
from training_utils.save_checkpoint import save_checkpoint

from timeit import default_timer
import numpy as np
import os
import socket
from argparse import ArgumentParser

def train_fno(args, model):
    
    # 0. Print training configurations
    print('Training Configuration:', args)

    # 1. Load Model Checkpoint
    if 'ckpt' in args:
        ckpt_path = args['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)

    # 2. Training optimizer and learning rate scheduler
    optimizer = Adam(model.parameters(), 
                     betas=(0.9, 0.999),
                     lr=args['base_lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=args['milestones'],
                                                     gamma=args['scheduler_gamma'])
    
    # 3. Load Training Data
    dataset = load_dataset(args)
    dataset = sample_data(dataset)

    # 4. Data and training parameters
    nu = 1 / args['data']['Re']
    t_interval = args['data']['time_interval']
    batch_size = args['batchsize']
    ic_weight = args['ic_loss']
    f_weight = args['f_loss']
    xy_weight = args['xy_loss']
    num_data_iter = args['data_iter']
    epochs = args['epochs']
    problem_type = args['data']['problem_type']

    # 5. Initialize Loss Recording Dictionary
    loss_keys = ["Total Weighted Loss", "LP Loss", "IC Loss", "BC Loss", "Vorticity Loss", "Continuity Loss", "X-Momentum Loss", "Y-Momentum Loss", "Time"]
    loss_dict = {key:[] for key in loss_keys}

    # 6. Train Model with training cases
    for epoch in range(epochs):
        model.train()        
        epoch_start_time = default_timer()

        for _ in range(num_data_iter):
            x, y = next(dataset)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)

            if ic_weight != 0 or f_weight != 0:
                loss_l2, loss_ic, loss_bc, loss_w, loss_c, loss_m1, loss_m2 = PINO_loss3d_decider(model_input = x, 
                                                                                                  model_output = out,
                                                                                                  model_val = y,
                                                                                                  forcing_type = problem_type, 
                                                                                                  nu = nu,
                                                                                                  t_interval = t_interval)
            else:
                zero_tensor = torch.zeros(1).to(device)
                loss_l2 = LpLoss.rel(out,y)
                loss_ic, loss_f, loss_bc = zero_tensor, zero_tensor, zero_tensor
                loss_w, loss_c, loss_m1, loss_m2  = zero_tensor, zero_tensor, zero_tensor, zero_tensor

            loss_f = loss_w + loss_c + loss_m1 + loss_m2
            total_loss = loss_l2 * xy_weight + loss_f * f_weight + loss_ic * ic_weight + loss_bc

            total_loss.backward()
            optimizer.step()

        scheduler.step()
        epoch_end_time = default_timer()

        for key_name, key_value in zip(loss_keys, [total_loss.item(),
                                                    loss_l2.item(),
                                                    loss_ic.item(),
                                                    loss_bc.item(), 
                                                    loss_w.item(), 
                                                    loss_c.item(), 
                                                    loss_m1.item(), 
                                                    loss_m2.item(), 
                                                    epoch_end_time - epoch_start_time]):
            loss_dict[key_name].append(key_value)

    # 7. Save Model Checkpoint and Losses
    save_checkpoint(args["save_dir"], args["save_name"], model, 
                    loss_dict=loss_dict, optimizer=optimizer,
                    input_sample=y, output_sample=out)
    print('Training Complete')

def eval_fno(args, model):
    
    # 0. Print training configurations
    print('Evaluation Configuration:', args)

    # 1. Load Model Checkpoint
    if 'ckpt' in args:
        ckpt_path = args['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)

    # 2. Load Validation Data
    dataset = load_dataset(args)

    # 3. Data and training parameters
    nu = 1 / args['data']['Re']
    t_interval = args['data']['time_interval']
    batch_size = args['batchsize']
    ic_weight = args['ic_loss']
    f_weight = args['f_loss']
    xy_weight = args['xy_loss']
    problem_type = args['data']['problem_type']
    
    myloss = LpLoss(size_average=True)

    # 4. Initialize Loss Recording Dictionary
    loss_keys = ["Total Weighted Loss", "LP Loss", "IC Loss", "BC Loss", "Vorticity Loss", "Continuity Loss", "X-Momentum Loss", "Y-Momentum Loss", "Time"]
    loss_dict = {key:[] for key in loss_keys}

    # 5. Evaluate Model with validation cases
    model.eval()        
    epoch_start_time = default_timer()

    with torch.no_grad():
        for x, y in dataset:
            x, y = x.to(device), y.to(device)
            
            out = model(x)
            
            loss_l2 = myloss(out, y)
            
            if ic_weight != 0 or f_weight != 0:
                loss_l2, loss_ic, loss_bc, loss_w, loss_c, loss_m1, loss_m2 = PINO_loss3d_decider(model_input = x, 
                                                                                                  model_output = out,
                                                                                                  model_val = y,
                                                                                                  forcing_type = problem_type, 
                                                                                                  nu = nu,
                                                                                                  t_interval = t_interval)

            else:
                zero_tensor = torch.zeros(1).to(device)
                loss_l2 = LpLoss.rel(out,y)
                loss_ic, loss_f, loss_bc = zero_tensor, zero_tensor, zero_tensor
                loss_w, loss_c, loss_m1, loss_m2  = zero_tensor, zero_tensor, zero_tensor, zero_tensor

            loss_f = loss_w + loss_c + loss_m1 + loss_m2
            total_loss = loss_l2 * xy_weight + loss_f * f_weight + loss_ic * ic_weight + loss_bc
    epoch_end_time = default_timer()

    for key_name, key_value in zip(loss_keys, [total_loss.item(),
                                                loss_l2.item(),
                                                loss_ic.item(),
                                                loss_bc.item(), 
                                                loss_w.item(), 
                                                loss_c.item(), 
                                                loss_m1.item(), 
                                                loss_m2.item(), 
                                                epoch_end_time - epoch_start_time]):
        loss_dict[key_name].append(key_value)

    # 7. Save Losses
    save_checkpoint(args["save_dir"], args["save_name"], loss_dict=loss_dict,
                    input_sample=y, output_sample=out)
    print('Evaluation Complete')

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

    # 2. Create model and load in model version
    model = FNO3d(modes1=run_config['model']['modes1'],
                  modes2=run_config['model']['modes2'],
                  modes3=run_config['model']['modes3'],
                  fc_dim=run_config['model']['fc_dim'],
                  layers=run_config['model']['layers'],
                  in_dim=run_config['model']['in_channels'] + 3,
                  out_dim=run_config['model']['out_channels']).to(device)
    
    # 3. Override configuration file if running locally (for debugging purposes)
    if socket.gethostname() == 'DESKTOP-157DQSC':
        rds_data_path = r'Z:\PRJ-MLFluids\datasets'
        #rds_data_path = r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\PINO datasets'
        for run_type in ['train','eval','fine']:
            if run_type in run_config:
                server_datapath = run_config[run_type]['data']['datapath']
                file_name = server_datapath.split('/')[-1]
                run_config[run_type]['data']['datapath'] = rds_data_path + '/sample_' + file_name
                run_config[run_type]['data']['n_sample'] = 2
                run_config[run_type]['epochs'] = 1
    else: pass
    
    # 4. Run scripts based on configurations
    if 'train' in run_config:
        train_fno(run_config['train'], model)
    
    if 'eval' in run_config:
        eval_fno(run_config['eval'], model)
    
    if 'fine' in run_config:
        train_fno(run_config['fine'], model)