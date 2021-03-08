'''
main_test file

Contains functions for testing the model on the test dataset.

'''

import os
import numpy as np
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

def ApplyModel_to_DataSet(model,image_list, image_transform, second_input_shape):
    """

    Parameters
    ----------
    model : Keras Model
    image_list : List of filenames of images
    image_transform : Function for transforming image in some way before putting it as input into model. In our case,
        we add noise to the images.
    second_input_shape : The way I declare models I have two inputs so that I can implement the loss function with
    the model in models.py. So, I have to put a second input into the model. Second_input_shapre ensures I have correct
    shape.

    Returns
    -------

    """
    outputs = []
    for im in image_list:
        if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
            x = load_image(im)
            y = image_transform(x)
            y = y.astype(np.float32)
            y_ = image_to_tensor(y)
            desired_output = np.ones(second_input_shape)
            x_ = model.predict([y_, desired_output])  # inference
            x_ = image_from_tensor(x_)
            outputs.append(x_)
    return outputs
#