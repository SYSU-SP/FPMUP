import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, Activation, BatchNormalization, Dense
import random
# from skimage.metrics import structural_similarity as ssim

# setting for GPU
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
setup_seed(1201)
np.set_printoptions(threshold=np.inf)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# parameter setting
kmatsim = scipy.io.loadmat('kmat_sim.mat') # load the kcenter of each illumination
kmat = kmatsim['kmat_sim']
intena = scipy.io.loadmat('intensity.mat') # the random ratio for non-inform illumination simulation
intena2 = np.squeeze(intena['intena2'])
pi = np.pi
waveLength = 0.63e-6  # waveLength
k0 = 2 * pi / waveLength
spsize = 2.75e-6  # sampling pixel size of the CCD
psize = spsize / 4  # final pixel size of the reconstruction
NA = 0.08 # numerical aperture
arraysize = 15 # LED array number
m = 256  # image size of high resolution object
n = 256
m1 = m / (spsize / psize)
n1 = n / (spsize / psize)  # image size of low-resolution abject
dkx = 2 * pi / (psize * n)
dky = 2 * pi / (psize * m)
cutoffFrequency = NA * k0
kmax = pi / spsize
[kxm, kym] = np.meshgrid(np.linspace(-kmax, kmax, int(n1)), np.linspace(-kmax, kmax, int(m1)))
CTF = ((kxm ** 2 + kym ** 2) < cutoffFrequency ** 2)
kzm = np.sqrt(k0**2-kxm**2-kym**2)
lr = 0.1
tp = 20
# obA = cv2.imread('cameraman.tif', 2)
# obA = cv2.resize(obA, (256, 256), interpolation=cv2.INTER_CUBIC).astype(float)
# obA = cv2.normalize(obA, None, 0, 1, cv2.NORM_MINMAX)
# obP = cv2.imread('west.tiff', 2)
# obP = cv2.resize(obP, (256, 256), interpolation=cv2.INTER_CUBIC).astype(float)
# obP = cv2.normalize(obP, None, 0, 1.0, cv2.NORM_MINMAX)

def datasetmake(object,z,flag):
    pupil = np.exp(1j * z * np.real(kzm)) * np.exp(-np.abs(z) * np.abs(np.imag(kzm))) * CTF
    imSeqLowRes = np.zeros((1, 64, 64, 225))
    objectFT = np.fft.fftshift(np.fft.fft2(object))
    for tt in range(0, arraysize ** 2):
        kyl = kmat[0, tt]
        kyh = kmat[1, tt]
        kxl = kmat[2, tt]
        kxh = kmat[3, tt]
        if flag != 0:
            imSeqLowFT = (m1 / m) ** 2 * objectFT[int(kyl):int(kyh) + 1, int(kxl): int(kxh) + 1] * pupil
        else:
            imSeqLowFT = (m1 / m) ** 2 * objectFT[int(kyl):int(kyh) + 1, int(kxl): int(kxh) + 1] * CTF
        if flag == 2:
            imSeqLowRes[0, :, :, tt] = intena2[tt] * abs(np.fft.ifft2(np.fft.ifftshift(imSeqLowFT)))
        else:
            imSeqLowRes[0, :, :, tt] = abs(np.fft.ifft2(np.fft.ifftshift(imSeqLowFT)))
    dataset = tf.data.Dataset.from_tensor_slices((np.zeros(1),imSeqLowRes))
    BATCH_SIZE = 1
    dataset = dataset.batch(BATCH_SIZE)
    return dataset

def Zernike_aberration(z,x,y):
    r,theta = cv2.cartToPolar(x/cutoffFrequency, y/cutoffFrequency)
    r = tf.cast(r,dtype=float)
    theta = tf.cast(theta,dtype=float)
    out = z[0]+0*z[1]*2*r*tf.cos(theta)-0*z[2]*2*r*tf.sin(theta)-z[3]*tf.sqrt(6.0)*r**2*tf.sin(2*theta) + \
          z[4]*tf.sqrt(3.0)*(2*r**2-1)+ z[5]*tf.sqrt(6.0)*r**2*tf.cos(2*theta)-\
        z[6]*tf.sqrt(8.0)*r**3*tf.sin(3*theta)-z[7]*tf.sqrt(8.0)*(3*r**3-2*r)*tf.sin(theta)+\
        z[8]*(3*r**3-2*r)*tf.cos(theta)+z[9]*tf.sqrt(8.0)*r**3*tf.cos(3*theta)-z[10]*tf.sqrt(10.0)*r**4*tf.sin(4*theta)-\
        z[11]*tf.sqrt(10.0)*(4*r**4-3*r**2)*tf.sin(2*theta)+z[12]*tf.sqrt(5.0)*(6*r**4-6*r**2+1)+z[13]*tf.sqrt(10.0)*(4*r**4-3*r**2)*tf.cos(2*theta)+\
        z[14]*tf.sqrt(10.0)*r**4*tf.cos(4*theta)
    return out*CTF

def Net():
    inputs_a = Input(shape=((1)))
    inputs_b = Input(shape=((1)))
    inputs_c = Input(shape=((1)))
    x1 = Dense(65536, kernel_initializer='ones')(inputs_a)
    x1 = BatchNormalization()(x1)
    x1 = Activation('sigmoid')(x1)
    x3 = tf.reshape(x1, [-1, 256, 256, 1])
    x3 = Conv2D(128*8, 3, padding='same', kernel_initializer='he_normal')(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = Conv2D(1, 3, padding='same', kernel_initializer='he_normal')(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation('sigmoid')(x3)

    x4 = Dense(65536, kernel_initializer='ones')(inputs_b)
    x4 = BatchNormalization()(x4)
    x4 = Activation('sigmoid')(x4)
    x6 = tf.reshape(x4, [-1, 256, 256, 1])
    x6 = Conv2D(128*8, 3, padding='same', kernel_initializer='he_normal')(x6)
    x6 = BatchNormalization()(x6)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(1, 3, padding='same', kernel_initializer='he_normal')(x6)
    x6 = BatchNormalization()(x6)
    x6 = Activation('sigmoid')(x6)

    x11 = Dense(15, kernel_initializer='zeros', name='pupillayer')(inputs_c)
    x11 = BatchNormalization()(x11)
    x11 = Activation('relu')(x11)
    x11 = Dense(30, kernel_initializer='he_normal')(x11)
    x11 = BatchNormalization()(x11)
    x11 = Activation('relu')(x11)
    x11 = Dense(15, kernel_initializer='he_normal')(x11)
    x11 = BatchNormalization()(x11)

    model = Model(inputs=[inputs_a, inputs_b,inputs_c], outputs=[x3,x6,x11])
    return model
def Netintensity():
    inputs_a = Input(shape=((1)))
    x11 = Dense(15, kernel_initializer='ones', name='intensity')(inputs_a)
    x11 = BatchNormalization()(x11)
    x11 = Activation('relu')(x11)
    x11 = Dense(225, kernel_initializer='ones')(x11)
    x11 = BatchNormalization()(x11)
    model = Model(inputs=inputs_a,outputs=x11)
    return model

# model of neural network
generator = Net()
generator2 = Netintensity()


# physics_model for FPM
def physics_model(in_A, in_P, in_Aber, ratio, y_ture,flag):
    loss_sum = 0
    in_A = tf.squeeze(in_A)
    in_P = tf.squeeze(in_P)
    in_Aber = tf.cast(tf.squeeze(in_Aber), dtype=float)
    in_P = in_P * pi
    if flag == 0:
        in_Aber = in_Aber-in_Aber
        ratio = tf.squeeze(ratio-ratio+1)
    elif flag == 1:
        ratio = tf.squeeze(ratio-ratio+1)
    elif flag == 2:
        ratio = tf.squeeze(ratio)
    else:
        print('the input mode is wrong')
        quit()
    object1 = tf.complex(in_A * tf.cos(in_P), in_A * tf.sin(in_P))
    objectFT = tf.signal.fftshift(tf.signal.fft2d(object1))
    pupiltest = Zernike_aberration(in_Aber,kxm,kym)
    pupiltest = tf.complex(tf.cos(pupiltest), tf.sin(pupiltest))
    # ratio = tf.squeeze(ratio)
    for tt in range(0, arraysize ** 2):
        kyl = kmat[0, tt]
        kyh = kmat[1, tt]
        kxl = kmat[2, tt]
        kxh = kmat[3, tt]
        temp = objectFT[int(kyl):int(kyh) + 1, int(kxl): int(kxh) + 1]
        temp_x = tf.math.real(temp)
        temp_y = tf.math.imag(temp)
        pupiltest_x = tf.math.real(pupiltest)
        pupiltest_y = tf.math.imag(pupiltest)
        ims = tf.complex(temp_x * pupiltest_x - temp_y * pupiltest_y, temp_x * pupiltest_y + temp_y * pupiltest_x)
        imSeqLowFT = (m1 / m) ** 2 * ims * CTF
        dddd = tf.abs(tf.signal.ifft2d(tf.signal.ifftshift(imSeqLowFT)))
        ratio2 = ratio[tt]
        loss_sum = loss_sum + tf.reduce_sum((tf.square(ratio2*dddd - tf.cast(y_ture[0,:, :, tt], dtype=float))))
    return loss_sum

exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=100,decay_rate=0.95)
exponential_decay2 = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=100,decay_rate=0.95)
generator_optimizer = tf.keras.optimizers.Adam(exponential_decay)
generator_optimizer2 = tf.keras.optimizers.Adam(exponential_decay2)

@tf.function
def train_step(x_batch, y_batch,flag):
    with tf.GradientTape(persistent=True) as r_tape:
        [objtestA,objtestP,objestAber] = generator([np.ones((1)), np.ones((1)),np.ones((1))], training=True)
        ratio = generator2(np.ones((1)),training = True)
        loss_sum1 = physics_model(objtestA, objtestP, objestAber, ratio, y_batch,flag)

        #regularization
        loss_reg = []
        for p in generator.trainable_variables:
            loss_reg.append(tf.nn.l2_loss(p))
        loss_reg = tf.reduce_sum(tf.stack(loss_reg))
        loss_sum1 = loss_sum1 + 1e-3*loss_reg

    gradients_of_generator = r_tape.gradient(loss_sum1, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    grad2 = r_tape.gradient(loss_sum1,generator2.trainable_variables)
    generator_optimizer2.apply_gradients(zip(grad2, generator2.trainable_variables))
    return loss_sum1, objtestA, objtestP, objestAber,ratio

def train(dataset, epochs,flag):
    time_start = time.time()
    for epoch in range(epochs):
        for x_data, y_data in dataset:
            loss_sum1, objtestA, objtestP, objestAber,ratio4 = train_step(x_data, y_data,flag)
            if (epoch+1) % tp == 0:
                # sA = ssim(obA,np.squeeze(objtestA.numpy()))
                # sP = ssim(obP,np.squeeze(objtestP.numpy()))
                # print('finished percent : %', epoch * 100 / epochs, '    loss is : ', loss_sum1.numpy(),' SSIM : A and P', sA,'    ',sP)
                print('finished percent : %', (epoch+1) * 100 / epochs, '    loss is : ', loss_sum1.numpy())
                showimage(objtestA,objtestP,objestAber,ratio4,epoch,flag)
    time_end = time.time()
    scipy.io.savemat('./datasave/A_recovered.mat',{'A':np.squeeze(objtestA)})
    scipy.io.savemat('./datasave/P_recovered.mat', {'P': np.squeeze(objtestP)})
    intensity_recovered = tran_array((tf.squeeze(ratio4)).numpy())
    scipy.io.savemat('./datasave/intensity_recovered.mat',{'intensity':np.squeeze(intensity_recovered)})
    pupil = Zernike_aberration(tf.cast(tf.squeeze(objestAber), dtype=float), kxm, kym)
    pupil = tf.complex(tf.cos(pupil), tf.sin(pupil))
    # scipy.io.savemat('./datasave/Zernikco.mat',{'Zernikco':objestAber.numpy()})
    scipy.io.savemat('./datasave/pupil_recovered.mat',{'pupil':pupil.numpy()})
    plt.figure(2)
    plt.imshow((tf.squeeze(objtestA)).numpy(), cmap='gray')
    plt.figure(3)
    plt.imshow((tf.squeeze(objtestP)).numpy(), cmap='gray')
    plt.show()
    print('totally cost', time_end - time_start)

def train2(image,z,flag,EPOCHS):
    dataset = datasetmake(image,z,flag)
    train(dataset,EPOCHS,flag)
def tran_array(input):
    p = 0
    out = np.zeros((arraysize,arraysize))
    for i in range(arraysize):
        for j in range(arraysize):
            out[i][j] = input[p]
            p = p+1
    return out

def showimage(A,P,Aber,inten,i,flag):
    plt.figure(1)
    plt.suptitle(i)
    plt.subplot(2,2,1)
    plt.imshow((tf.squeeze(A)).numpy(),cmap='gray')
    plt.title('A')
    plt.subplot(2,2,2)
    plt.imshow((tf.squeeze(P)).numpy(),cmap='gray')
    plt.title('P')
    if flag != 0:
        plt.subplot(2,2,3)
        pupil = Zernike_aberration(tf.cast(tf.squeeze(Aber), dtype=float), kxm, kym)
        pupil = tf.complex(tf.cos(pupil), tf.sin(pupil))
        pupil = tf.math.angle(pupil)
        plt.imshow(pupil/pi,vmin=-1,vmax=1,cmap='hsv')
        plt.title('Aberration')
    if flag == 2:
        plt.subplot(2,2,4)
        plt.imshow(tran_array((tf.squeeze(inten)).numpy()),vmin=0,vmax=1,cmap='gray')
        plt.title('Intensity of LED array')
    plt.tight_layout()
    plt.show()
