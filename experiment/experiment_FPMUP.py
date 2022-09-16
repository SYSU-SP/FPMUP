import os
import time
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input,Activation, BatchNormalization, Dense
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
import random
import cv2

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
setup_seed(10086)
np.set_printoptions(threshold=np.inf)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
pi = np.pi

tmp = io.loadmat('./experiment_dataset/kxky.mat')
kx,ky,kxl2,kxh2,kyl2,kyh2,CTF = tmp['kx'],tmp['ky'],tmp['kxlt'],tmp['kxht'],tmp['kylt'],tmp['kyht'],tmp['CTF']
seq = io.loadmat('./experiment_dataset/seq.mat')
xs,ys = np.squeeze(seq['x']),np.squeeze(seq['y'])
wavelength = 0.63e-6
k0 = 2 * pi / wavelength
spsize = 1.845e-6  # sampling pixel size of the CCD
psize = spsize / 4  # final pixel size of the reconstruction
imagsize1 = 128
imagsize2 = 128
NA = 0.1
arraysize = 15
m = imagsize1*4
n = imagsize2*4
m1 = m / (spsize / psize)
n1 = n / (spsize / psize)  # image size of the final output
kx = kx*k0
ky = ky*k0
dkx = 2 * pi / (psize * n)
dky = 2 * pi / (psize * m)
cutoffFrequency = NA * k0
kmax = pi / spsize
[kxm, kym] = np.meshgrid(np.linspace(-kmax, kmax, int(n1)), np.linspace(-kmax, kmax, int(m1)))
lr = 0.1 # learning rate
et = 50 # imshow the output per et iteration

def makedataset(lowrdata):
    y_data = lowrdata/np.max(lowrdata)
    x_data = np.ones(1)
    dataset = tf.data.Dataset.from_tensor_slices((x_data,y_data))
    dataset = dataset.batch(1)
    return dataset

def Zernike_aberration(z,x,y):
    r,theta = cv2.cartToPolar(x/cutoffFrequency, y/cutoffFrequency)
    r = tf.cast(r,dtype=float)
    theta = tf.cast(theta,dtype=float)
    out = z[0]+1*z[1]*2*r*tf.cos(theta)-1*z[2]*2*r*tf.sin(theta)-z[3]*tf.sqrt(6.0)*r**2*tf.sin(2*theta) + \
        z[4]*tf.sqrt(3.0)*(2*r**2-1)+ z[5]*tf.sqrt(6.0)*r**2*tf.cos(2*theta)-\
        z[6]*tf.sqrt(8.0)*r**3*tf.sin(3*theta)-z[7]*tf.sqrt(8.0)*(3*r**3-2*r)*tf.sin(theta)+\
        z[8]*(3*r**3-2*r)*tf.cos(theta)+z[9]*tf.sqrt(8.0)*r**3*tf.cos(3*theta)-z[10]*tf.sqrt(10.0)*r**4*tf.sin(4*theta)-\
        z[11]*tf.sqrt(10.0)*(4*r**4-3*r**2)*tf.sin(2*theta)+z[12]*tf.sqrt(5.0)*(6*r**4-6*r**2+1)+z[13]*tf.sqrt(10.0)*(4*r**4-3*r**2)*tf.cos(2*theta)+\
        z[14]*tf.sqrt(10.0)*r**4*tf.cos(4*theta)
    return out*CTF
def Net8():
    inputs_a = Input(shape=((1)))
    inputs_b = Input(shape=((1)))
    inputs_c = Input(shape=((1)))

    x1 = Dense(imagsize1*imagsize2*16, kernel_initializer='he_normal',use_bias=True)(inputs_a)
    x1 = BatchNormalization()(x1)
    x1 = Activation('sigmoid')(x1)
    x3 = tf.reshape(x1, [-1, imagsize1*4, imagsize2*4, 1])
    x3 = Conv2D(128*8, 3, padding='same', kernel_initializer='he_normal')(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = Conv2D(1, 3, padding='same', kernel_initializer='he_normal')(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation('sigmoid')(x3)

    x4 = Dense(imagsize1*imagsize2*16, kernel_initializer='he_normal',use_bias=True)(inputs_b)
    x4 = BatchNormalization()(x4)
    x4 = Activation('sigmoid')(x4)
    x6 = tf.reshape(x4, [-1, imagsize1*4, imagsize2*4, 1])
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
    model = Model(inputs=[inputs_a, inputs_b, inputs_c], outputs=[x3, x6, x11])
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
generator = Net8()
generator2 = Netintensity()

def physics_model(in_A, in_P, in_Aber,ratio,y_ture):
    loss_sum = 0
    in_A = tf.squeeze(in_A)
    in_P = tf.squeeze(in_P)
    in_P = in_P * pi
    in_Aber = tf.cast(tf.squeeze(in_Aber), dtype=float)
    object1 = tf.complex(in_A * tf.cos(in_P), in_A * tf.sin(in_P))
    objectFT = tf.signal.fftshift(tf.signal.fft2d(object1))
    pupiltest = Zernike_aberration(in_Aber,kxm,kym)
    pupiltest = tf.complex(tf.cos(pupiltest), tf.sin(pupiltest))
    ratio = tf.squeeze(ratio)
    for tt in range(0, arraysize**2):
        kxl = kxl2[tt,0]
        kxh = kxh2[tt,0]
        kyl = kyl2[tt,0]
        kyh = kyh2[tt,0]
        temp = objectFT[int(kyl)-1:int(kyh) , int(kxl)-1: int(kxh) ]
        temp2 = temp
        temp_x = tf.math.real(temp2)
        temp_y = tf.math.imag(temp2)
        pupiltest_x = tf.math.real(pupiltest)
        pupiltest_y = tf.math.imag(pupiltest)
        ims = tf.complex(temp_x * pupiltest_x - temp_y * pupiltest_y, temp_x * pupiltest_y + temp_y * pupiltest_x)
        imSeqLowFT = (m1 / m) ** 2 * ims * CTF
        dddd = abs(tf.signal.ifft2d(tf.signal.ifftshift(imSeqLowFT)))
        ratio2 = ratio[tt]
        loss_sum = loss_sum + tf.reduce_sum((tf.square(((ratio2)) * dddd - tf.cast(y_ture[0, :, :, tt], dtype=float))))
    loss_sum = loss_sum
    return loss_sum

exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=100,
                                                                   decay_rate=0.95)
generator_optimizer = tf.keras.optimizers.Adam(exponential_decay)
exponential_decay2 = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=100, decay_rate=0.95)
generator_optimizer2 = tf.keras.optimizers.Adam(exponential_decay2)

@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape(persistent=True) as r_tape:
        [objtestA, objtestP, objestAber] = generator([np.ones((1)), np.ones((1)), np.ones((1))], training=True)
        intensity_ratio = generator2(np.ones((1)),training=True)
        loss_sum1 = physics_model(objtestA, objtestP,objestAber, intensity_ratio,y_batch)

        loss_reg = []
        for p in generator.trainable_variables:
            loss_reg.append(tf.nn.l2_loss(p))
        loss_reg = tf.reduce_sum(tf.stack(loss_reg))
        loss_sum1 = loss_sum1 + 1e-2*loss_reg

    gradients_of_generator = r_tape.gradient(loss_sum1, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    gradients_of_generator2 = r_tape.gradient(loss_sum1, generator2.trainable_variables)
    generator_optimizer2.apply_gradients(zip(gradients_of_generator2, generator2.trainable_variables))
    return loss_sum1, objtestA, objtestP, objestAber,intensity_ratio

def train(dataset, epochs):
    time_start = time.time()
    for epoch in range(epochs):
        for x_data, y_data in dataset:
            loss_sum1, objtestA, objtestP,objestAber, intensity = train_step(x_data, y_data)
            if (epoch+1) % et == 0:
                print('finished percent : %', (epoch+1) * 100 / epochs, '    loss is : ', loss_sum1.numpy())
                showimage(objtestA, objtestP, objestAber, intensity, epoch)
    time_end = time.time()
    io.savemat('./datasave/A_recovered.mat', {'A': np.squeeze(objtestA)})
    io.savemat('./datasave/P_recovered.mat', {'P': np.squeeze(objtestP)})
    intensity_recovered = tran_array((tf.squeeze(intensity)).numpy())
    io.savemat('./datasave/intensity_recovered.mat', {'intensity': intensity_recovered})
    pupil = Zernike_aberration(tf.cast(tf.squeeze(objestAber), dtype=float), kxm, kym)
    pupil = tf.complex(tf.cos(pupil), tf.sin(pupil))
    # io.savemat('./datasave/Zernikco.mat', {'Zernikco': objestAber.numpy()})
    io.savemat('./datasave/pupil_recovered.mat', {'pupil': pupil.numpy()})
    plt.figure(2)
    plt.imshow((tf.squeeze(objtestA)).numpy(), cmap='gray')
    plt.figure(3)
    plt.imshow((tf.squeeze(objtestP)).numpy(), cmap='gray')
    plt.show()
    print('totally cost', time_end - time_start)

def trainrealdata(dataset,EPOCHS):
    dataset = makedataset(dataset)
    train(dataset,EPOCHS)

def tran_array(input):
    t = 0
    out = np.zeros((arraysize,arraysize))
    for i in range(arraysize):
        for j in range(arraysize):
            out[int(xs[t])-1][int(ys[t])-1] = input[t]
            t = t + 1
    return out
def showimage(A,P,Aber,inten,i):
    plt.figure(1)
    plt.suptitle(i)
    plt.subplot(2,2,1)
    plt.imshow((tf.squeeze(A)).numpy(),cmap='gray')
    plt.title('A')
    plt.subplot(2,2,2)
    plt.imshow((tf.squeeze(P)).numpy(),cmap='gray')
    plt.title('P')
    plt.subplot(2,2,3)
    pupil = Zernike_aberration(tf.cast(tf.squeeze(Aber), dtype=float), kxm, kym)
    pupil = tf.complex(tf.cos(pupil), tf.sin(pupil))
    pupil = tf.math.angle(pupil)
    plt.imshow(pupil/pi,vmin=-1,vmax=1,cmap='hsv')
    plt.title('Aberration')
    plt.subplot(2,2,4)
    plt.imshow(tran_array((tf.squeeze(inten)).numpy()),vmin=0,vmax=1,cmap='gray')
    plt.title('Intensity of LED array')
    plt.tight_layout()
    plt.show()
