import torch
import torch.nn.functional as F
from training_utils.optimizers import Adam
from training_utils.loss_functions_2 import LpLoss, loss_selector
from data_handling.data_utils import load_dataset, sample_data, total_loss_list
from training_utils.save_checkpoint import save_checkpoint
import yaml

from timeit import default_timer

def model_routine_cavity(args, model):
    
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
    dataset_train, dataset_test = load_dataset(args)
    #dataset_train, dataset_test = sample_data(dataset_train), sample_data(dataset_test)
    dataset_train = sample_data(dataset_train)

    # 4. Data and training parameters
    xy_weight = args['xy_loss']
    
    epochs = args['epochs']
    num_data_iter = args['data_iter']
    
    # 5. Initialize Loss Function and Recording Dictionary
    loss_function = loss_selector(args['loss_type'])
    total_loss_dictionary = total_loss_list()

    # 6. Run Model Routine (Training)  
    model.train()

    for epoch in range(epochs): 
        print(epoch)
        epoch_start_time = default_timer()

        # Set Model to Train
        #model.train()

        for _ in range(num_data_iter):
            x, y = next(dataset_train)
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            x_in = F.pad(x, (0, 0, 0, 5), "constant", 0)
            out = model(x_in)
            out = out[..., :-5, :]

            # 6.1. Calculate L2 Loss 
            loss_l2 = loss_function(out, y)

            total_loss = (loss_l2 * xy_weight)

            total_loss.backward()
            optimizer.step()

        scheduler.step()
        epoch_end_time = default_timer()

        # 6.4. Store Losses to Dictionary
        total_loss_dictionary.update({'Epoch Time': epoch_end_time - epoch_start_time})
        total_loss_dictionary.update({'Total Weighted Loss': total_loss.item()})
        total_loss_dictionary.update({'Computer Vision Loss': loss_l2.item()})

        # 6.5. Save Model Outputs at Each Milestone (NOTE: Only saves Output not model, add model keyword for that)
        if epoch+1 in args['milestones']:
            save_checkpoint(args["save_dir"], args["save_name"], output_sample=out, epoch=epoch+1)

        # 6.6 Evaluate the model
        # model.eval()
        # loss_l2 = 0
        # count = 0

        # for x, y_eval in dataset_test:
        #     x, y_eval = x.to(device), y_eval.to(device)

        #     x_in = F.pad(x, (0, 0, 0, 5), "constant", 0)
        #     out_eval = model(x_in)
        #     out_eval = out_eval[..., :-5, :]

        #     # 6.7. Calculate L2 Loss 
        #     loss_l2 += loss_function(out_eval, y_eval)
        #     count += 1
        
        # # 6.8 Calculate the Average Loss and Store to Dictionary
        # loss_l2 = loss_l2/count
        # total_loss_dictionary.update({'Average Validation Loss': loss_l2.item()})

    # 7. Save Model Checkpoint and Losses
    save_checkpoint(args["save_dir"], args["save_name"], model, 
                    loss_dict=total_loss_dictionary.fetch_list(), optimizer=optimizer,
                    input_sample=y, output_sample=out)
    
    print('Model Routine Complete')