from experiment_FPMUP import trainrealdata
import numpy as np
import scipy.io as io


'''Here is a demo of the program'''

tmp = io.loadmat('./experiment_dataset/USAF_red.mat')
imSeqLowRes = tmp['imlow_HDR']
im1 = np.zeros((1, 128, 128, 225))
im1[0,:,:,:] = imSeqLowRes[5:133,5:133,:]

tmp = io.loadmat('./experiment_dataset/bloodsmear_red.mat')
imSeqLowRes=tmp['imlow_HDR']
im2 = np.zeros((1, 128, 128, 225))
im2[0,:,:,:]= imSeqLowRes[0:128,0:128,:]

epoch = 500

trainrealdata(im2,epoch) # choose im1 or im2



