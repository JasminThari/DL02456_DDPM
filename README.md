Welcome to the DL02456_DDPM Repository!

This repository is an integral component of the exam project titled "Denoising Diffusion Probabilistic Models: A Comprehensive Replication Study", part of the course "Deep Learning (02456)" at DTU COMPUTE Here, you'll find valuable resources and insights related to our project.

Key Contents:

- `utils.py` --> This file consists of basic functions such as data loading, plotting, and image saving, which are utilized across the other files.
- `module.py` --> This file contains the U-Net model.
- `DDPM.py` --> This is the file contains the DDPM model with the training and sampling algorithms. Here, the `module.py` is used to extract the U-Net model. 
- `DDPM_Notebook` --> This notebook demonstrates the results achieved from the DDPM models obtained by running the `DDPM.py` file at HPC.
- `Calc_FID_2048.py` --> This file calculates the FID score of the trained models using 2048 dimensional feature vector for inceptionV3. 

