# MICCAI'20 - Proximal Gradient Descent Network (PGD-Net) 

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
0. [torchkbnufft](https://github.com/mmuckley/torchkbnufft) (optional for nufft)

### Usage

### Citation

If you use these models in your research, please cite:

	@inproceedings{chen2020compressive,
		author = {Dongdong Chen and Mike E. Davies and Mohammad Golbabaee},
		title = {Compressive MR Fingerprinting reconstruction with Neural Proximal Gradient iterations},
		booktitle={International Conference on Medical image computing and computer-assisted intervention (MICCAI)},
		year = {2020}
	}
