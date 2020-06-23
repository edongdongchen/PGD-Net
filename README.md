# Proximal gradient decent network (PGD-Net), MICCAI'2020 paper 

By [Dongdong Chen](http://dongdongchen.com), [Mike E. Davies](https://scholar.google.co.uk/citations?user=dwmfR3oAAAAJ&hl=en), [Mohammad Golbabaee](https://mgolbabaee.wordpress.com/).

The University of Edinburgh, The University of Bath.

### Table of Contents
0. [Abstract](#introduction)
0. [Requirement](#Requirement)
0. [Usage](#Usage)
0. [Citation](#citation)


### Abstract

Consistency of the predictions with respect to the physical forward model is pivotal for reliably solving inverse problems. This consistency is mostly un-controlled in the current end-to-end deep learning methodologies proposed for the Magnetic Resonance Fingerprinting (MRF) problem. To address this, we propose PGD-Net, a learned proximal gradient decent framework that directly incorporates the forward acquisition and Bloch dynamic models within a recurrent learning mechanism. The PGD-Net adopts a compact neural proximal model for de-aliasing and quantitative inference, that can be flexibly trained on scarce MRF training datasets. Our numerical experiments show that the PGD-Net can achieve a superior quantitative inference accuracy, much smaller storage requirement, and a comparable runtime to the recent deep learning MRF baselines, while being much faster than the dictionary matching schemes.

### Usage

### Requirement

### Citation

If you use these models in your research, please cite:

	@inproceedings{chen2020neural,
		author = {Dongdong Chen and Mike E. Davies and Mohammad Golbabaee},
		title = {Compressive MR Fingerprinting reconstruction with Neural Proximal Gradient iterations},
		booktitle={International Conference on Medical image computing and computer-assisted intervention},
		year = {2020}
	}
