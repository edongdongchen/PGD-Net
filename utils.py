import os
import sys
import csv
import time
import numpy as np
from datetime import datetime

import torch

"""
Utils for compressive MR fingerprinting (CS-MRF) in the paper
@inproceedings{chen2020compressive,
	author = {Dongdong Chen and Mike E. Davies and Mohammad Golbabaee},
	title = {Compressive MR Fingerprinting reconstruction with Neural Proximal Gradient iterations},
	booktitle={International Conference on Medical image computing and computer-assisted intervention (MICCAI)},
	year = {2020}
}
"""


def set_gpu(gpu):
    print('Current GPU:{}'.format(gpu))
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark =True
    dtype = torch.cuda.FloatTensor
    return dtype

def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)

def prefix(args):
    return '{}_{}_cuda_{}_sampling_{}_iter_{}_T_{}_x_{}_y_{}_m_{}_{}_{}_bs_{}_lr_{}_wd_{}'.format(
        args.filename, str(time.ctime()).replace(' ', '_'),
        args.cuda, args.sampling, args.epochs, args.time_step,
        args.loss_weight['x'], args.loss_weight['y'],
        args.loss_weight['m'][0], args.loss_weight['m'][1], args.loss_weight['m'][2],
        args.batch_size, args.lr, args.weight_decay)

# --------------------------------
# logger
# --------------------------------
def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

class LOG(object):
    def __init__(self, filepath, filename, field_name=['iter', 'loss_x', 'loss_m', 'loss_y', 'loss_total', 'alpha']):
        self.filepath = filepath
        self.filename = filename
        self.field_name = field_name

        self.logfile, self.logwriter = csv_log(file_name=os.path.join(filepath, filename+'.csv'), field_name=field_name)
        self.logwriter.writeheader()

    def record(self, *args):
        dict = {}
        for i in range(len(self.field_name)):
            dict[self.field_name[i]]=args[i]
        self.logwriter.writerow(dict)

    def close(self):
        self.logfile.close()

    def print(self, msg):
        logT(msg)

def csv_log(file_name, field_name):
    assert file_name is not None
    assert field_name is not None
    logfile = open(file_name, 'w')
    logwriter = csv.DictWriter(logfile, fieldnames=field_name)
    return logfile, logwriter

def logT(*args, **kwargs):
     print(datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

def logger(args):
    logfile, logwriter = csv_log(file_name=os.path.join(args.net_dir, args.net_name+'.csv'), field_name=['iter', 'loss'])
    logwriter.writeheader()
    if args.opt['loss_type']=='mse':
        criterion = torch.nn.MSELoss().cuda()
    if args.opt['loss_type']=='l1':
        criterion = torch.nn.L1Loss().cuda()
    if args.opt['val_dataloader'] is not None:
        val_logfile, val_logwriter = csv_log(file_name=os.path.join(args.net_dir, args.net_name+'_val.csv'), field_name=['iter', 'loss'])
        val_logwriter.writeheader()
        return logfile, logwriter, val_logfile, val_logwriter, criterion
    else:
        return logfile, logwriter, criterion


# --------------------------------
# Convert data type
# --------------------------------
def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()


# --------------------------------
# complex-valued operation
# --------------------------------
def complex_matmul(A, B): # A: (dim1, dim2, 2), B:(N, dim2, dim3, 2)' (a+bj)x(c+dj) = (ac-bd) + (bc+ad)j
    return torch.stack([torch.matmul(A[...,0], B[...,0]) - torch.matmul(A[...,1], B[...,1]),
                        torch.matmul(A[...,1], B[...,0]) + torch.matmul(A[...,0], B[...,1])],dim=-1)

def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.
    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.
    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1).sqrt()
