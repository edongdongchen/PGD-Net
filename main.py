import os
import time
import argparse
import numpy as np

import torch

import data
from bloch import BLOCH
from network_arch import ProxNet
from operators import OperatorBatch
from utils import set_gpu, LOG, logT, check_paths, prefix

'''
PyCharm (Python 3.6.9)
PyTorch 1.3
Windows 10 or Linux
Dongdong Chen (d.chen@ed.ac.uk)
github: https://github.com/echendongdong/PGD-Net

If you have any question, please feel free to contact with me.
Dongdong Chen (e-mail: d.chen@ed.ac.uk)
by Dongdong Chen (01/March/2020)
'''

"""
# --------------------------------------------
Training/Testing code (GPU) of PGD-Net for MR fingerprinting in the paper
@inproceedings{chen2020compressive,
	author = {Dongdong Chen and Mike E. Davies and Mohammad Golbabaee},
	title = {Compressive MR Fingerprinting reconstruction with Neural Proximal Gradient iterations},
	booktitle={International Conference on Medical image computing and computer-assisted intervention (MICCAI)},
	year = {2020}
}
# --------------------------------------------
Note: The data was from a partner company and we are restricted from sharing. 
      Users need to specify their own dataset.
      Our code can be flexibly transferred or directly used on other customized MRF dataset.
# --------------------------------------------
"""


def train_proxnet(args):
    check_paths(args)
    # init GPU configuration
    args.dtype = set_gpu(args.cuda)

    # init seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # define training data
    train_dataset = data.MRFData(mod='train', sampling=args.sampling)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    # init operators (subsampling + subspace dimension reduction + Fourier transformation)
    operator = OperatorBatch(sampling=args.sampling.upper()).cuda()
    H, HT = operator.forward, operator.adjoint
    bloch = BLOCH().cuda()

    # init PGD-Net (proxnet)
    proxnet = ProxNet(args).cuda()

    # init optimizer
    optimizer = torch.optim.Adam([{'params': proxnet.transformnet.parameters(),
                                   'lr': args.lr, 'weight_decay': args.weight_decay},
                                  {'params': proxnet.alpha, 'lr': args.lr2}])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.1)

    # init loss
    mse_loss = torch.nn.MSELoss()#.cuda()

    # init meters
    log = LOG(args.save_model_dir, filename=args.filename, field_name=['iter', 'loss_m', 'loss_x', 'loss_y', 'loss_total', 'alpha'])

    loss_epoch = 0
    loss_m_epoch, loss_x_epoch, loss_y_epoch =0,0,0

    # start PGD-Net training
    print('start training...')
    for e in range(args.epochs):
        proxnet.train()
        loss_m_seq = []
        loss_x_seq = []
        loss_y_seq = []
        loss_total_seq = []

        for x, m, y in train_loader:
            # covert data type (cuda)
            x, m, y = x.type(args.dtype), m.type(args.dtype), y.type(args.dtype)
            # add noise
            noise = args.noise_sigam * torch.randn(y.shape).type(args.dtype)
            HTy = HT(y + noise).type(args.dtype)

            # PGD-Net computation (iteration)
            # output the reconstructions (sequence) of MRF image x and its tissue property map m
            m_seq, x_seq = proxnet(HTy, H, HT, bloch)

            loss_x, loss_y, loss_m = 0,0,0
            for t in range(args.time_step):
                loss_y += mse_loss(H(x_seq[t]), y)/args.time_step
            for i in range(3):
                loss_m += args.loss_weight['m'][i] * mse_loss(m_seq[-1][:,i,:,:], m[:,i,:,:])
            loss_x = mse_loss(x_seq[-1], x)

            # compute loss
            loss_total = loss_m + args.loss_weight['x'] * loss_x + args.loss_weight['y']*loss_y

            # update gradient
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            # update meters
            loss_m_seq.append(loss_m.item())
            loss_x_seq.append(loss_x.item())
            loss_y_seq.append(loss_y.item())
            loss_total_seq.append(loss_total.item())

        # (scheduled) update learning rate
        scheduler.step()

        # print meters
        loss_m_epoch = np.mean(loss_m_seq)
        loss_x_epoch = np.mean(loss_x_seq)
        loss_y_epoch = np.mean(loss_y_seq)
        loss_epoch = np.mean(loss_total_seq)

        log.record(e+1, loss_m_epoch, loss_x_epoch, loss_y_epoch, loss_epoch, proxnet.alpha.detach().cpu().numpy())
        logT("==>Epoch {}\tloss_m: {:.6f}\tloss_x: {:.6f}\tloss_y: {:.6f}\tloss_total: {:.6f}\talpha: {}"
             .format(e + 1, loss_m_epoch, loss_x_epoch, loss_y_epoch, loss_epoch, proxnet.alpha.detach().cpu().numpy()))

        # save checkpoint
        if args.checkpoint_model_dir is not None and (e + 1) % args.checkpoint_interval == 0:
            proxnet.eval()
            ckpt = {
                'epoch': e+1,
                'loss_m': loss_m_epoch,
                'loss_x': loss_x_epoch,
                'loss_y': loss_y_epoch,
                'total_loss': loss_epoch,
                'net_state_dict': proxnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'alpha': proxnet.alpha.detach().cpu().numpy()
            }
            torch.save(ckpt, os.path.join(args.checkpoint_model_dir, 'ckp_epoch_{}.pt'.format(e)))
            proxnet.train()

    # save model
    proxnet.eval()
    state = {
        'epoch':args.epochs,
        'loss_m': loss_m_epoch,
        'loss_x': loss_x_epoch,
        'loss_y': loss_y_epoch,
        'total_loss': loss_epoch,
        'alpha': proxnet.alpha.detach().cpu().numpy(),
        'net_state_dict': proxnet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    save_model_path = os.path.join(args.save_model_dir, log.filename+'.pt')
    torch.save(state, save_model_path)
    print("\nDone, trained model saved at", save_model_path)

def test_proxnet(args):
    def load_proxnet(args):
        ckp = torch.load(args.net_path)
        alpha_learned = ckp['alpha']

        net = ProxNet(args).cuda()
        net.load_state_dict(ckp['net_state_dict'])
        net.alpha = torch.from_numpy(alpha_learned)
        net.eval()
        print('alpha={}'.format(net.alpha))
        return net

    operator = OperatorBatch(sampling=args.sampling.upper()).cuda()
    H, HT = operator.forward, operator.adjoint
    bloch = BLOCH().cuda()

    args.dtype = set_gpu(args.cuda)
    net = load_proxnet(args)
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(dataset=data.MRFData(mod='test', sampling=args.sampling),
                                              batch_size=batch_size, shuffle=False)

    rmse_m, rmse_x, rmse_y = [],[],[]
    rmse_torch = lambda a,b:torch.norm(a-b, 2).detach().cpu().numpy()/torch.norm(b, 2).detach().cpu().numpy()/batch_size

    toc = time.time()
    for x, m, y in test_loader:
        m, y = m.type(args.dtype), y.type(args.dtype)
        HTy = HT(y).type(args.dtype)

        m_seq, x_seq = net(HTy, H, HT, bloch)
        m_hat = m_seq[-1]

        rmse_m.append(rmse_torch(m_hat, m))

    elapsed = time.time() - toc
    print('time: {}'.format(elapsed / 16))
    print('m error mean:{}, max: {}, std:{}'.format(np.mean(rmse_m), np.max(rmse_m), np.std(rmse_m)))


if __name__=='__main__':
    def demo_train():
        args = argparse.ArgumentParser().parse_args()

        args.cuda = 0
        args.seed = 5213
        args.sampling = 'S' # 'spiral'
        args.filename = 'pgd_net'

        args.epochs = 2000
        args.batch_size = 4
        args.noise_sigam = 0.01
        args.weight_decay = 1e-8
        args.checkpoint_interval = 100

        # learning rate for neural network
        args.lr = 1e-3
        # learning rate for alpha
        args.lr2 = .05
        # gamma, lambda, beta
        args.loss_weight = {'x': 0.001, 'y': 0.01, 'm': [1, 20, 2.5]}
        # PGD time step (T)
        args.time_step = 2
        # init alpha
        args.initial_alpha = np.asarray([2] * args.time_step)
        # init path
        args.prefix = prefix(args)
        args.save_model_dir = os.path.join('models', args.prefix)
        args.checkpoint_model_dir = os.path.join('models', args.prefix, 'ckp')
        print(args.prefix)

        # start to train
        train_proxnet(args)

    def demo_test():
        args = argparse.ArgumentParser().parse_args()

        args.cuda = 0
        args.sampling = 'S'
        args.time_step = 2
        args.net_path = 'models/PGD_NET_Spiral_T_2.pt'

        test_proxnet(args)

    # demo_train()
    # demo_test()