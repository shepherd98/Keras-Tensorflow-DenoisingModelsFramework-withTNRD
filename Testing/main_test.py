'''
main_test file

Contains functions for testing the model on the test dataset.

'''


import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.io import imread
import csv, datetime

#For printing out to screen
def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

def writeCsv(filename, row, writetype='wb'):
    """
    Writes a row to a csv file

    Parameters
    ----------
    filename : Name of csv file to write to
    row : Row to write to csv file
    writetype : writetype
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, writetype) as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(row)

#Loads an image using imread. Scales to be in (0,1)
def load_image(filename):
    return np.array(imread(filename), dtype=np.float32) / 255.0

#Converts image to tensor shape
def image_to_tensor(image):
    return image[np.newaxis,:,:,np.newaxis]

#Converts image in tensor shape back to image shape
def image_from_tensor(tensor):
    return tensor[0,:,:,0]


def DataSetTest(funcs,model,image_list, sigma, compare = False):
    """

    Parameters
    ----------
    funcs : Functions to get metrics we wish to be computed such as psnr and ssim
    model : Keras Model to be tested
    image_list : List of images to test model on. Just filenames
    sigma : Noise level
    compare : Dictates whether or not metric the metric compares the output of the model to the clean image or not

    Returns
    -------
    List of metrics based on funcs that compute metrics

    """
    values = []
    for j in range(len(funcs)):
        values.append([])
    for im in image_list:
        if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
            x = load_image(im)
            np.random.seed(seed=0)  # for reproducibility
            y = x + np.random.normal(0, sigma / 255.0, x.shape)  # Add Gaussian noise without clipping
            y = y.astype(np.float32)
            y_ = image_to_tensor(y)
            clean = image_to_tensor(x)
            x_ = model.predict([y_,clean])  # inference
            x_ = image_from_tensor(x_)
            for j, func in enumerate(funcs):
                if compare == True:
                    values[j].append(func(x_,x))
                else:
                    values[j].append(func(x_))
    return values

#Get PSNR between two images
def get_psnr(x,y):
    x = np.clip(x,0,1)
    y = np.clip(y,0,1)
    psnr = peak_signal_noise_ratio(y,x)
    return psnr

#Get SSIM between two images
def get_ssim(x,y):
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)
    ssim = structural_similarity(x,y)
    return ssim


def psnr_ssim_test(model, pipeline, sigma):
    """

    Parameters
    ----------
    model: Keras model we are testing
    pipeline: Pipeline of model. Pipelines in data_generators directory
    sigma: Noise level we are testing model on

    Returns
    -------
    Two vectors with psnr and ssim values for each image in test set
    """
    funcs = [get_psnr, get_ssim]
    values = DataSetTest(funcs, model, pipeline.test_images_list(), sigma,compare=True)
    return values[0], values[1]

#Get max value of image
def get_max_val(np_array):
    return np.max(np_array)
#Get min value of image
def get_min_val(np_array):
    return np.min(np_array)

def range_test(model, pipeline, sigma):
    """

        Parameters
        ----------
        model: Keras model we are testing
        pipeline: Pipeline of model. Pipelines in data_generators directory
        sigma: Noise level we are testing model on

        Returns
        -------
        Two vectors with max and min values for output of model for each image in test set
        """
    funcs = [get_max_val, get_min_val]
    values = DataSetTest(funcs, model, pipeline.test_images_list(), sigma, compare=False)
    return values[0], values[1]
