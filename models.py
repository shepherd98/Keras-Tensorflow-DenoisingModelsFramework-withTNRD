'''
Name: Ryan Cecil
File: models.py
Purpose: Contains the different types of models that can be run. To add a new model, all we have to do is define
a new class that inherits from the My_Model class in My_Model.py.

Each model in this file inherits from the main model class found in My_Model.py. By overriding the base methods
in the main My_Model class, we can easily define a new model. In our new model, we define

1. A new functional keras model by overriding the Model(self) method
2. A new model name by overriding the get_model_name(self) method
3. A new directory to store the model and all of its seperate trainings by overriding the get_model_dir(self) method.
4. A new directory to store the model's results and all of its seperate training's results
    by overriding the get_results_dir(self) method.
5. A new format for the model's subdirectories to store the model with different settings by overriding the get_sub_dirs
6. A new summary of the model that can easily be printed out by overriding the model_summary method
7. A new data pipeline for the model by overriding the get_pipeline(self) method. See the data_generators directory
    to get more info on the pipelines.
8. A new test callback for validating the model during training by overriding the get_validation_callback. See Testing.test_callbacks
9. The model settings of the model (e.g. # of filters, stages, kernel_size, etc.) by overriding model_settings(self,args)

Here is an example of a new model (Note: I have not tested this):

class Model_Name(My_Model):
    def get_model_name(self):
        return 'Mymodelthatisgoingtodoamazingthings'

    #This fake model takes input images and applys a single convolution. Loss is sum_squared_loss. See losses.py for more losses
    def Model(self):
        # Input to the model: (batch_size, None, None, channels)
        # None corresponds to any dimension
        input = Input(shape=(None, None, self.settings['channels']), name='Noisy')
        desired_output = Input(shape=(None, None, self.settings['channels']), name='Clean')
        # Note that I am using my settings dictionary to specify parameters
        output = tf.keras.layers.Conv2D(self.settings['filters'],self.settings['kernel_size'],strides=(1, 1),padding="valid")
        model = Model(inputs=[noisy,clean], outputs=x)
        # Add the sum_squared_error_loss
        model.add_loss(sum_squared_error_loss(desired_output,x))
        return model

    def get_model_dir(self):
        model_dir = '/home/cpm/Research/KfTNRD/2021/Denoising_Framework/saved_models/{model_dir}'
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        return model_dir

    def get_results_dir(self):
        results_dir = '/home/cpm/Research/KfTNRD/2021/Denoising_Framework/model_results/{model_results_dir}'
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        return results_dir

    #However we want to differ between different settings of the model when saving them. In this case,
    I would only care about different number of filters, model number, and kernel_size for a CNN:
    def get_sub_dirs(self):
        directory = os.path.join('model_' + str(self.settings['model_number'])+'_filters' + \
                                 str(self.settings['filters']) + '_' + str(self.settings['kernel_size']) + 'x' + str(self.settings[
                                     'kernel_size']))
        submodel_dir = os.path.join(self.model_dir, directory)
        subresults_dir = os.path.join(self.results_dir, directory)
        Path(submodel_dir).mkdir(parents=True, exist_ok=True)
        Path(subresults_dir).mkdir(parents=True, exist_ok=True)
        return submodel_dir,subresults_dir

    def model_summary(self):
        #Maybe just self.model.summary()
        #Or give an explanation:
        print('This is my great new model')

    #Define some pipeline for training. See data_generators for more details
    def get_pipeline(self):
        return Train400Im68Pipeline(self.settings,self.training_settings)


    #Make the validation callback during training be the PSNRSSIMTest found in Testing.test_callbacks
    def get_validation_callback(self):
        return PSNRSSIMTest

    #Make sure to include an optimizer, initial_lr, lr_schedule, batch_size, epoch_steps, epochs, save_every,
    $and test_every in the settings or else the model will fail to train
    def model_settings(self,args):
    settings = {'model_number':None, 'sigma':None, 'stages':None, 'filters':None, 'kernel_size':None, 'channels':None,
                'optimizer': Adam, 'initial_lr': 0.001, 'lr_scheduler': lr_schedule1,
                'batch_size': 128, 'epoch_steps': 200, 'epochs': 100, 'save_every': 10, 'test_every': 10
                }
    return load_settings_from_args(args,settings)
'''

from keras.layers import Subtract, Add
from layers.TNRD import *
from layers.layers import *
from keras import Input, Model
from losses import sum_squared_error_loss
from data_generators.pipelines import Train400Im68Pipeline
from keras.optimizers import Adam
from schedulers import *
from Testing.test_callbacks import *
import os
from My_Model import My_Model

def load_settings_from_args(settings,settings_dict):
    """
    Purpose
    --------------------------
    Takes model settings and inputs into the setting args from our call_model.py file

    Parameters
    ----------
    settings : Settings of the model defined in an args parse object
    settings_dict : Settings of he model defined in a dictionary

    Returns
    -------
    An updated version of the model settings dictionary that was updated with the args parse arguments

    """
    for arg in vars(settings):
        if arg in settings_dict:
            settings_dict[arg] = getattr(settings, arg)
    return settings_dict


class TNRD(My_Model):
    '''
    Custom implementation of the Trainable Nonlinear Reaction diffusion model
    of Chen and Pock: https://arxiv.org/pdf/1508.02848.pdf
    Code for Gaussian Radial Basis Activation Functions compiled from
    '''
    def get_model_name(self):
        return 'TNRD'

    #Define model
    def Model(self):
        # Input to the model: (batch_size, None, None, channels)
        # None corresponds to any dimension
        noisy = Input(shape=(None, None, self.settings['channels']), name='Noisy')
        clean = Input(shape=(None, None, self.settings['channels']), name='Clean')
        x = noisy
        # For each stage of the model
        for i in range(self.settings['stages']):
            u_t_1 = x
            # Get data fidelity term: (u_{t_1} - f)
            df = Subtract(name='DataFidelityLayer' + str(i + 1))([u_t_1, noisy])
            # Scale data fidelity term with lambda weight
            scaled_df = Scalar_Multiply(learn_scalar=True, scalar_init=0.1, name='ScaleFidelity' + str(i + 1))(df)
            # Apply kappa_L: K_L(u_{t_1})
            inference = TNRD_Inference(filters=self.settings['filters'], kernel_size=self.settings['kernel_size'], name='Inference' + str(i + 1))(u_t_1)
            # Add data fidelity term: K_L(u_{t_1}) + \lambda(u_{t_1} - f)
            x = Add(name='AddFidelity' + str(i + 1))([inference, scaled_df])
            # Apply Descent step to finish TNRD stage: u_t_1 - {K_L(u_{t_1}) + \lambda(u_{t_1} - f)}
            x = Subtract(name='Descent' + str(i + 1))([u_t_1, x])
        model = Model(inputs=[noisy,clean], outputs=x)
        # Add the sum_squared_error_loss
        model.add_loss(sum_squared_error_loss(clean,x))
        return model

    #Define where model will be saved
    def get_model_dir(self):
        model_dir = '/home/cpm/Research/KfTNRD/2021/Denoising_Framework/saved_models/TNRD/TNRD'
        return model_dir

    #Define where model results will be stored
    def get_results_dir(self):
        results_dir = '/home/cpm/Research/KfTNRD/2021/Denoising_Framework/model_results/TNRD/TNRD'
        return results_dir

    #Define where model will be saved and results will be stored depending on the different settings of the model
    def get_sub_dirs(self):
        directory = os.path.join('model_' + str(self.settings['model_number']),
                                 'sigma' + str(self.settings['sigma']) + '_stages' + str(self.settings['stages']) + '_filters' + \
                                 str(self.settings['filters']) + '_' + str(self.settings['kernel_size']) + 'x' + str(self.settings[
                                     'kernel_size']))
        submodel_dir = os.path.join(self.model_dir, directory)
        subresults_dir = os.path.join(self.results_dir, directory)
        return submodel_dir,subresults_dir

    #Print out a summary of the model
    def model_summary(self):
        self.model.summary()
        print('This is the Trainable Nonlinear Reaction Diffusion model of Chen and Pock')
        print('https://arxiv.org/pdf/1508.02848.pdf')
        pass

    #Grab a data pipeline from the data_generators directory. In this case, the classic 400 Training images
    #people usually use for denoising, and the im68 dataset for testing
    def get_pipeline(self):
        return Train400Im68Pipeline

    #Define a callback so that the model tests itself at test_every epochs during trainings
    def get_validation_callback(self):
        return PSNRSSIMTest

    #Define the settings of the model, to be updated by the args passed into the call_model.py file
    def model_settings(self,args):
        settings = {'model_number':None, 'sigma':None, 'stages':None, 'filters':None, 'kernel_size':None, 'channels':None,
                    'optimizer': Adam, 'initial_lr': 0.001, 'lr_scheduler': lr_schedule1,
                    'batch_size': 128, 'epoch_steps': 200, 'epochs': 100, 'save_every': 10, 'test_every': 10
                    }
        return load_settings_from_args(args,settings)