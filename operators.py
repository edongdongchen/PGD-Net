import torch
import torch.nn as nn
import numpy as np
import scipy.io as scio

from utils import to_tensor, complex_matmul

"""
PyTorch implementation of forward/adjoint operators for compressive sensing MR fingerprinting (CS-MRF) in the paper
@inproceedings{chen2020compressive,
	author = {Dongdong Chen and Mike E. Davies and Mohammad Golbabaee},
	title = {Compressive MR Fingerprinting reconstruction with Neural Proximal Gradient iterations},
	booktitle={International Conference on Medical image computing and computer-assisted intervention (MICCAI)},
	year = {2020}
}
"""

class OperatorBatch(nn.Module):
    def __init__(self, C=10, H=128, W=128, sampling='S', dtype=torch.cuda.FloatTensor):
        super(OperatorBatch, self).__init__()
        self.C, self.H, self.W, self.dtype = C, H, W, dtype

        # subspace dimension reduction
        pca_dic_data = scio.loadmat('./matfile/pytorch_Ramp2D_200reps_guido_trainingset.mat')

        self.V = to_tensor(pca_dic_data['V'])
        self.V_conj = to_tensor(pca_dic_data['V_conj'])
        if dtype is not None:
            self.V, self.V_conj = self.V.type(dtype), self.V_conj.type(dtype)
        assert self.V.shape[1]==self.C, 'Channels Error!'

        # init mask
        mask_data = scio.loadmat('./matfile/train_dataXS11_s.mat')
        if sampling=='C':
            mask = mask_data['samplemask_s']
        if sampling=='S':
            mask = mask_data['samplemask_s']
        self.mask = np.squeeze(np.asarray(mask-1))
        print('mask.shape', self.mask.shape)

    def forward(self, x):
        return self.fwd_helper(x, self.mask, self.H, self.W, self.V)

    def adjoint(self, y, only_real=True):
        return self.adj_helper(y, self.mask, self.H, self.W, self.V_conj, only_real)

    def fwd_helper(self, x, mask, H, W, V):#x:NCHW
        N  = x.shape[0]
        x = torch.stack([x, torch.zeros(x.shape).type(self.dtype)], dim=-1)
        x = torch.fft(x, 2)
        x = x.reshape(N, -1, H*W, 2)
        x = complex_matmul(V, x)
        x = x.reshape(N, -1, W, H, 2)
        x = x.permute(0, 1,3,2,4)
        x = x.reshape(N, -1, 1,2)
        x = x[:,mask, :,:]/np.sqrt(H*W)
        return x

    def adj_helper(self, y, mask, H, W, V_conj, only_real): #y:NCHW2
        N = y.shape[0]
        L = V_conj.shape[1]
        x = torch.zeros(N, L*H*W, 1, 2).type(self.dtype)
        x[:,mask,:,:]=y
        x = x.reshape(N, L, -1, 2)
        x = complex_matmul(V_conj, x)
        x = x.reshape(N, -1, W, H, 2)
        x = x.permute(0, 1, 3, 2, 4)
        x = torch.ifft(x,2)*np.sqrt(H*W)
        return x[...,0] if only_real else x