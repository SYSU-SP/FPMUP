from simulation_FPMUP import train2
import cv2
import numpy as np

#seting for training
epoch = 500
z = 2e-5 # defocus distance 2e-4 m
mode = 1 # 0 for no aberration mode,1 for only aberration mode, and 2 for aberration and intensity mode

#load the image
obA = cv2.imread('cameraman.tif', 2)
obA = cv2.resize(obA, (256, 256), interpolation=cv2.INTER_CUBIC).astype(float)
obA = cv2.normalize(obA, None, 0, 1, cv2.NORM_MINMAX)
obP = cv2.imread('west.tiff', 2)
obP = cv2.resize(obP, (256, 256), interpolation=cv2.INTER_CUBIC).astype(float)
obP = cv2.normalize(obP, None, 0, 1.0, cv2.NORM_MINMAX)
obP = obP * np.pi
object = obA * np.exp(obP * 1j)

##### simulation for no aberration, only defocus aberration and defocus aberration + intensity fluctuation
train2(object, z, mode, epoch)

