import numpy as np
import torch
import torch.nn as nn

from utils import get_timestamp, logT
import data

"""
Training code of neural network BLOCH estimator for MR fingerprinting in the paper
@inproceedings{chen2020compressive,
	author = {Dongdong Chen and Mike E. Davies and Mohammad Golbabaee},
	title = {Compressive MR Fingerprinting reconstruction with Neural Proximal Gradient iterations},
	booktitle={International Conference on Medical image computing and computer-assisted intervention (MICCAI)},
	year = {2020}
}
"""

class BlochDecoder(nn.Module):
    def __init__(self, in_channels=2, out_channels=10):
        super(BlochDecoder, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=300, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=300, out_channels=out_channels, kernel_size=1, bias=True)
        )

    def forward(self, map):# map: N x C x H x W
        t1t2, pd = map[:, 0:2, :, :], map[:, 2:3, :, :]
        x = self.Conv(t1t2)
        pd = pd.repeat(1,10,1,1) # Nx1xHxW -> Nx10xHxW
        x = x * pd               # MRF image (scaled by pd)
        return x

def train_bloch(lr=0.01, EPOCH=50, BATCH_SIZE=500, weight_decay=1e-10, dtype=torch.cuda.FloatTensor):
    bloch = BlochDecoder().cuda()
    criterion = torch.nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(bloch.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)

    dataloader = torch.utils.data.DataLoader(dataset=data.BlochData(), batch_size=BATCH_SIZE, shuffle=True)


    bloch.train()
    for iter in range(EPOCH):
        loss_epoch = []
        for x, m in dataloader:
            x, m = x.type(dtype), m.type(dtype)
            x_hat = bloch(m)
            loss = criterion(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.item())
        logT("===> Epoch {}: Loss: {:.10f}".format(iter, np.mean(loss_epoch)))
    filename = './models/miccai_bloch_gen_{}.pt'.format(get_timestamp())
    torch.save(bloch.state_dict(), filename)
    print('Saved Bloch generator to the disk: {}'.format(filename))

def BLOCH():
    """
    return: a pre-trained BLOCH estimator.
    One can directly apply this BLOCH estimator to simulate the BLOCH equation response in practice
    """
    bloch = BlochDecoder()
    bloch.load_state_dict(torch.load('./models/miccai_bloch_gen_200210-180246.pt'))
    bloch.eval()
    return bloch