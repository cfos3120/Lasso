import torch
import torch.nn.functional as F
from training_utils.optimizers import Adam
#from training_utils.loss_functions import LpLoss, loss_selector, PINO_loss3d_decider
from training_utils.loss_functions_2 import LpLoss, loss_selector, PINO_loss_calculator
from data_handling.data_utils import load_dataset, sample_data, total_loss_list
from training_utils.save_checkpoint import save_checkpoint
import yaml

from timeit import default_timer

def model_routine(args, model):
    
    # 0. Print training configurations
    print('Routine Configurations:\n\t',yaml.dump(args).replace("\n","\n\t"))
    device = next(model.parameters()).device

    # 1. Load Model Checkpoint
    if 'ckpt' in args:
        ckpt_path = args['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)

    # 2. Training optimizer and learning rate scheduler
    if 'base_lr' in args:
        optimizer = Adam(model.parameters(), betas=(0.9, 0.999), lr=args['base_lr'])

    if 'milestones' in args:   
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args['milestones'], gamma=args['scheduler_gamma'])
        
    # 3. Load Training Data
    dataset = load_dataset(args)
    dataset = sample_data(dataset)

    # 4. Data and training parameters
    ic_weight = args['ic_loss']
    f_weight = args['f_loss']
    xy_weight = args['xy_loss']
    
    if 'epochs' in args:
        epochs = args['epochs']
        num_data_iter = args['data_iter']
        model.train()
        train = True
    else:
        epochs = 1
        num_data_iter = 1
        model.eval()
        train = False

    # 5. Initialize Loss Function and Recording Dictionary
    loss_function = loss_selector(args['loss_type'])
    total_loss_dictionary = total_loss_list()

    # 6. Run Model Routine (Training or Evaluation)  

    for epoch in range(epochs): 
        epoch_start_time = default_timer()

        for _ in range(num_data_iter):
            x, y = next(dataset)
            x, y = x.to(device), y.to(device)

            if train:
                optimizer.zero_grad()

            x_in = F.pad(x, (0, 0, 0, 5), "constant", 0)
            out = model(x_in)
            out = out[..., :-5, :]

            # 6.1. Calculate L2 Loss 
            loss_l2 = loss_function.rel(x=out,y=y)

            # 6.2. Calculate Initial Condition Loss 
            loss_ic = loss_function.rel(x=out,y=y)

            # 6.3. Calculate PDE Loss (optional)
            if f_weight != 0:
                loss_f, losses_list = PINO_loss_calculator(args=args, model_output=out, loss_function=loss_function)

            else:
                loss_f = torch.zeros(1).to(device)

            total_loss = (loss_l2 * xy_weight) + (loss_f * f_weight) + (loss_ic * ic_weight)

            if train:
                total_loss.backward()
                optimizer.step()

        if train and 'milestones' in args:
            scheduler.step()

        epoch_end_time = default_timer()

        # 6.4. Store Losses to Dictionary
        total_loss_dictionary.update({'Epoch Time': epoch_end_time - epoch_start_time})
        total_loss_dictionary.update({'Total Weighted Loss': total_loss.item()})
        total_loss_dictionary.update({'Computer Vision Loss': loss_l2.item()})  
        total_loss_dictionary.update({'Initial Condition Loss': loss_ic.item()})
        if f_weight != 0: 
            total_loss_dictionary.update(losses_list)

        # 6.5. Save Model Outputs at Each Milestone (NOTE: Only saves Output not model, add model keyword for that)
        if train and epoch+1 in args['milestones']:
            save_checkpoint(args["save_dir"], args["save_name"], output_sample=out)

    # 7. Save Model Checkpoint and Losses
    if train:
        save_checkpoint(args["save_dir"], args["save_name"], model, 
                        loss_dict=total_loss_dictionary.fetch_list(), optimizer=optimizer,
                        input_sample=y, output_sample=out)
    else:
        save_checkpoint(args["save_dir"], args["save_name"], 
                        loss_dict=total_loss_dictionary.fetch_list(),
                        input_sample=y, output_sample=out)
    print('Model Routine Complete')