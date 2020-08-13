# MICCAI'20 - Proximal Gradient Descent Network (PGD-Net)
This is the PyTorch implementation of below paper
[Compressive MR Fingerprinting reconstruction with Neural Proximal Gradient iterations](https://arxiv.org/pdf/2006.15271.pdf).

By [Dongdong Chen](http://dongdongchen.com), [Mike E. Davies](https://scholar.google.co.uk/citations?user=dwmfR3oAAAAJ&hl=en), [Mohammad Golbabaee](https://mgolbabaee.wordpress.com/).

The University of Edinburgh, The University of Bath.

### Table of Contents
0. [Keywords](#Keywords)
0. [Abstract](#Abstract)
0. [Requirement](#Requirement)
0. [Usage](#Usage)
0. [Citation](#citation)

### Keywords

Magnetic Resonance Fingerprinting (MRF), Physics, Proximal gradient Descent (PGD), Inverse problem, Deep learning.

### Abstract

Consistency of the predictions with respect to the physical forward model is pivotal for reliably solving inverse problems. This consistency is mostly un-controlled in the current end-to-end deep learning methodologies proposed for the Magnetic Resonance Fingerprinting (MRF) problem. To address this, we propose PGD-Net, a learned proximal gradient descent framework that directly incorporates the forward acquisition and Bloch dynamic models within a recurrent learning mechanism. The PGD-Net adopts a compact neural proximal model for de-aliasing and quantitative inference, that can be flexibly trained on scarce MRF training datasets. Our numerical experiments show that the PGD-Net can achieve a superior quantitative inference accuracy, much smaller storage requirement, and a comparable runtime to the recent deep learning MRF baselines, while being much faster than the dictionary matching schemes.

### Requirement
0. PyTorch >=1.0
0. CUDA >=8.5

### Usage
0. check the demo_train() and demo_test() in [main.py](https://github.com/edongdongchen/PGD-Net/blob/master/main.py)
0. the neura network architecture of PGD-Net ('proxnet') is defined in [network_arch.py](https://github.com/edongdongchen/PGD-Net/blob/master/network_arch.py)
0. the forward and adjoint operators are implemented in [operators.py](https://github.com/edongdongchen/PGD-Net/blob/master/operators.py)
0. note: the data was from a partner company and we are restricted from sharing. Users need to specify their own dataset. Our code can be flexibly transferred or directly used on other customized MRF dataset.

### Citation

If you use these models in your research, please cite:

	@inproceedings{chen2020compressive,
		author = {Dongdong Chen and Mike E. Davies and Mohammad Golbabaee},
		title = {Compressive MR Fingerprinting reconstruction with Neural Proximal Gradient iterations},
		booktitle={International Conference on Medical image computing and computer-assisted intervention (MICCAI)},
		year = {2020}
	}
