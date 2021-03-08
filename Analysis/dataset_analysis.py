from Analysis.main_analyze import ApplyModel_to_DataSet
import numpy as np
import os
from PIL import Image
from scipy.io import savemat

'''
Functions for analyzing the model. Currently only has methods for saving model/layer outputs.
'''


def clip_image(img):
    """

    Parameters
    ----------
    img : Image to be scaled

    Returns
    -------
    Clipped image so that image range is (0,1)
    """
    return np.clip(img,0,1)

def scale_image(img):
    """
    Shift by min then scale by max so that min is at zero and max is at 1.

    Parameters
    ----------
    img : Image to be scaled

    Returns
    -------
    Image scaled to (0,1)

    """
    im_min = np.min(img)
    shifted_im = img - im_min
    im_max = np.max(shifted_im)
    return shifted_im/im_max

def save_image(img,name, save_dir, change):
    """

    Parameters
    ----------
    img : np image array
    name : name of image for saving
    save_dir : Dir to save image to
    change : Method of scaling image for saving
    """

    #Convert image to be between (0,255)
    im = Image.fromarray((change(img)*255).astype(np.uint8))
    #Save using PIL
    im.save(os.path.join(save_dir,name + '.png'),format='png')


def DataSetDenoiseImages(model_name, model, save_dir, pipeline, sigma, clean_image = False, save_image_output = False,
                          save_matlab_output = False,save_numpy_output = False, clip = False):
    """
    Applys model to model inputs. Saves model outputs in save_dir

    Parameters
    ----------
    model_name : name of model
    model : Keras Model
    save_dir : Directory to save results to
    pipeline : Pipeline of model_class
    sigma : Noise level of input images
    clean_image : Specify whether or not we want the inputs into the model to actually be clean instead of noisy
    save_image_output : True or False. Save png image of output.
    save_matlab_output : True or False. Save matlab mat file of outputs.
    save_numpy_output : True or False. Save numpy file of outputs.
    clip : True or False. If true, we clip model outputs to be in range (0,1). Else, we scale model outputs to be
        in range (0,1)

    Returns
    -------
    Saves model outputs in save_dir

    """
    #Quick method for adding noise to image
    def image_transform_to_noisy(img):
        if clean_image:
            return img
        else:
            np.random.seed(seed=0)  # for reproducibility
            return img + np.random.normal(0, sigma / 255.0, img.shape)  # Add Gaussian noise without clipping

    outputs = ApplyModel_to_DataSet(model,pipeline.analysis_images_list(),image_transform_to_noisy,(1,1,1,1))

    mat_dict = {}
    if clean_image:
        end_name = 'cleaninput_output'
    else:
        end_name = 'output'
    for i,output in enumerate(outputs):
        if save_numpy_output == True:
            np.save(os.path.join(save_dir, model_name+'_'+'im' + str(i + 1) + '_' + end_name + '.npy'), outputs)
        if save_image_output == True:
            func = clip_image if clip else scale_image
            save_image(output, model_name+'_'+'im' + str(i + 1)+ '_' + end_name, save_dir, func)
        if save_matlab_output == True:
            mat_dict[model_name+'_'+'im' + str(i + 1)+ '_' + end_name] = output
    if save_matlab_output == True:
        savemat(os.path.join(save_dir, model_name+'_raw_data_'+end_name+'.mat'), mat_dict, appendmat=False)

    print('The output images have been saved to ' + save_dir)

