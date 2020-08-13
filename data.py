from torch.utils.data.dataset import Dataset
import torch
import os
import numpy as np
import scipy.io as scio

class MRFData(Dataset):
    def __init__(self, mod='train', sampling='S'):
        '''
        The data was from a partner company and we are restricted from sharing.
        However, our code can be flexibly transferred or directly used on other customized MRF dataset.
        '''

        # users need to specify their own ground truth source
        # X: MRF images
        # Y: MRF (kspace) measurements
        # M: tissue property maps
        self.scaling = 1
        if mod=='train':
            mat_path = './matfile/train_dataXS11_s.mat'
        if mod=='test':
            mat_path = './matfile/test_dataXS11_s.mat'

        mat_data = scio.loadmat(mat_path)
        X = np.transpose(mat_data['X'], (0,3,1,2))
        Y = np.stack([mat_data['y_s_real'], mat_data['y_s_imag']], axis=-1)

        M = np.transpose(mat_data['MRF_maps'], (0,3,1,2))

        self.x = torch.from_numpy(X)
        self.y = torch.from_numpy(Y)
        self.m = torch.from_numpy(M)

        self.y = torch.from_numpy(Y).unsqueeze(-2)


        print('MRF-{}-dataset:\nCS Fourier y: {},\nMRF image x:{},\nTissue map m: {}'.format(sampling, self.y.shape, self.x.shape, self.m.shape))


    def __getitem__(self, index):
        return self.x[index], self.m[index], self.y[index]

    def __len__(self):
        return len(self.x)


class BlochData(Dataset):
    def __init__(self, mat_path='./matfile/Ramp2D_200reps_guido_trainingset.mat'):
        assert os.path.exists(mat_path)
        mat_data = scio.loadmat(mat_path)
        X = mat_data['X']
        Y = mat_data['Y']
        Y = np.concatenate((Y, np.ones((len(Y),1), dtype=float)), axis=1)
        # MRF image
        self.X =  np.reshape(X, (len(X), 1, 1, 10),'F') # N * H * W * C
        self.X = np.transpose(self.X, (0, 3, 1, 2))  # covert to N * C * H * W
        self.X = torch.from_numpy(self.X)
        # tissue map
        self.M = np.reshape(Y, (len(Y), 1, 1, 3),'F') # N * H * W * C
        self.M = np.transpose(self.M, (0,3,1,2)) # covert to N * C * H * W
        self.M = torch.from_numpy(self.M)

    def __getitem__(self, index):
        x, m = self.X[index], self.M[index]
        return x, m

    def __len__(self):
        return len(self.X)