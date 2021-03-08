# -*- coding: utf-8 -*-

# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26},
#    number={7},
#    pages={3142-3155},
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/husqin/DnCNN-keras
# =============================================================================

'''
All methods modified from https://github.com/cszn/DnCNN
'''

import glob
import cv2
import numpy as np
import datetime


#Code modified from DnCNN GitHub: https://github.com/husqin/DnCNN-keras
patch_size, stride = 40, 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128

# Code from DnCNN GitHub: https://github.com/husqin/DnCNN-keras
def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()

def data_aug(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def gen_patches(file_name):
    # read image
    img = cv2.imread(file_name, 0)  # gray scale
    h, w = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h * s), int(w * s)
        img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled - patch_size + 1, stride):
            for j in range(0, w_scaled - patch_size + 1, stride):
                x = img_scaled[i:i + patch_size, j:j + patch_size]
                # patches.append(x)
                # data aug
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)

    return patches

def datagenerator(data_dir='data/Train400', verbose=False):
    file_list = glob.glob(data_dir + '/*.png')  # get name list of all .png files
    # initrialize
    data = []
    # generate patches
    for i in range(len(file_list)):
        patch = gen_patches(file_list[i])
        data.append(patch)
        if verbose:
            print(str(i + 1) + '/' + str(len(file_list)) + ' is done ^_^')
    data = np.array(data, dtype='uint8')
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2], data.shape[3], 1))
    discard_n = len(data) - len(data) // batch_size * batch_size;
    data = np.delete(data, range(discard_n), axis=0)
    print('^_^-training data finished-^_^')
    return data

def train_datagen(data_dir,model_settings,train_settings):
    sigma = model_settings['sigma']
    batch_size = train_settings['batch_size']
    epoch_num = 5
    while(True):
        n_count = 0
        if n_count == 0:
            xs = datagenerator(data_dir)
            assert len(xs)%batch_size ==0, \
            log('make sure the last iteration has a full batchsize, this is important if you use batch normalization!')
            xs = xs.astype('float32')/255.0
            indices = list(range(xs.shape[0]))
            n_count = 1
        for _ in range(epoch_num):
            np.random.shuffle(indices)    # shuffle
            for i in range(0, len(indices), batch_size):
                batch_x = xs[indices[i:i+batch_size]]
                noise =  np.random.normal(0, sigma/255.0, batch_x.shape)    # noise
                #noise =  K.random_normal(ge_batch_y.shape, mean=0, stddev=args.sigma/255.0)
                batch_y = batch_x + noise
                yield [batch_y,batch_x]