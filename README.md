# FPMUP
# 1.Install

Here is the list of libraries you need to install to execute the code:

- python = 3.7.7
- tensorflow = 2.3
- numpy
- scipy
- matplotlib
- opencv-python

# 2.Simulation

In the simulation, run the simulation_demo program.

For the reconstruction without aberration and intensity fluctuation, set z = 0 and mode = 0, then use the function named train2.

For the reconstruction with aberration, set z = 2e-4 or 2e-5, mode = 1, then use the function named train2.

For the reconstruction with both aberration and intensity fluctuation, set z = 2e-4 or 2e-5, mode = 2, then use the function named train2.

# 3.Experiment

In the experiment, run the experiment_demo program.

Choose im1 or im2 for different datasets, then use the function named trainrealdata.

# 4.Data description
In simulation, data west.tiff and cameraman.tif are the ground truth used for phase and amplitude. Data kmat_smi.mat is the wave vector k of each LED.

In experiment, data bloodsmear_red.mat and USAF_red.mat are the measurents from Prof. Guoan Zheng’s group (https://github.com/SmartImagingLabUConn/Fourier-Ptychography). Data kxky.mat is the wave vector k of each LED. And data seq.mat is the sequence of illumination LED in experiment.


