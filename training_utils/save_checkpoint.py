import torch
import os
import numpy as np

def save_checkpoint(path, name, model=None, loss_dict=None, optimizer=None, input_sample=None, output_sample=None):
    ckpt_dir = 'checkpoints/%s/' % path
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    if model != None:
        try:
            model_state_dict = model.module.state_dict()
        except AttributeError:
            model_state_dict = model.state_dict()

        if optimizer is not None:
            optim_dict = optimizer.state_dict()
        else:
            optim_dict = 0.0

        torch.save({
            'model': model_state_dict,
            'optim': optim_dict
        }, ckpt_dir + name + '.pt')
        print('Checkpoint is saved at %s' % ckpt_dir + name + '.pt')

    if loss_dict != None:
        np.save(ckpt_dir + name + '_losses', loss_dict)
        print("Loss Dictionary Saved in Same Location")

    if input_sample != None:
        input_sample = input_sample.cpu().numpy()
        np.save(ckpt_dir + name + '_input_sample', input_sample)
        print("Input Sample Saved in Same Location")

    if output_sample != None:
        try:
            output_sample = output_sample.detach().cpu().numpy()
        except:
            output_sample = output_sample.numpy()
        np.save(ckpt_dir + name + '_output_sample', output_sample)
        print("Output Sample Saved in Same Location")