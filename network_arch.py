import torch
import torch.nn as nn

"""
Neural Network Architecture of PGD-Net for compressive sensing MR fingerprinting (CS-MRF) in the paper
@inproceedings{chen2020compressive,
	author = {Dongdong Chen and Mike E. Davies and Mohammad Golbabaee},
	title = {Compressive MR Fingerprinting reconstruction with Neural Proximal Gradient iterations},
	booktitle={International Conference on Medical image computing and computer-assisted intervention (MICCAI)},
	year = {2020}
}
"""

class ProxNet(torch.nn.Module):
    def __init__(self, args):
        super(ProxNet, self).__init__()
        self.args = args
        self.alpha = torch.autograd.Variable(torch.Tensor(args.initial_alpha).type(args.dtype), requires_grad=True)
        self.transformnet = ResNet(in_channels=10, out_channels=3, nRS = 1, chRS=64, MRFNETch=64)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, HTy, H, HT, bloch):
        x = 0
        m_seq, x_seq = [], []
        for t in range(self.args.time_step):
            a = self.relu(self.alpha[t]) + 1
            s = a*HTy if t == 0 else x - a* (HT(H(x)) - HTy)
            m = self.transformnet(s)
            x = bloch(m)

            m_seq.append(m)
            x_seq.append(x)
        return m_seq, x_seq


class ResNet(nn.Module):
    def __init__(self, in_channels=10, out_channels=3, nRS = 2,chRS=120, MRFNETch=400):
        super(ResNet, self).__init__()
        self.name = 'resnet'
        self.rsb1 = ResidualBlock(in_channels, in_channels, chRS)
        self.rsb2 = ResidualBlock(in_channels, in_channels, chRS)

        self.mrfnet = MRFNET(in_channels,out_channels,MRFNETch)

    def forward(self, x):
        # encoding path
        x1 = self.rsb1(x)
        # or
        # x1 = self.rsb2(x1)
        xout = self.mrfnet(x1)
        return xout

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,chRS, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, chRS, stride)
        self.conv2 = conv3x3(chRS, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

class MRFNET(nn.Module):
    def __init__(self, ch_in=10, ch_out=3, MRFNETch=400):
        super(MRFNET, self).__init__()
        self.name='mrfcnn'
        self.cnn = nn.Sequential(
            nn.Conv2d(ch_in, MRFNETch, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(MRFNETch, MRFNETch, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(MRFNETch, ch_out, kernel_size=1, padding=0),
        )
    def forward(self, x):
        return self.cnn(x)