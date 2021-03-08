'''
TTA: Train, Test, Analyze

A class for training, testing, and analyzing models that are made for the task of image denoising. The model
passed into the class must be a My_Model.py class. See My_Model.py for more details.

'''


from Training.main_train import Train_Model
from Testing.dataset_tests import DataSetPSNRSSIMTest, DataSetRangeTest
from Analysis.dataset_analysis import DataSetDenoiseImages
import os
from keras import Model
from Analysis.visualization import plot_kernels,plot_functions

#Makes directory if it does not exit
def makedir(save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


def fetch_layer(model, layer_name):
    """

    Parameters
    ----------
    model : A keras model
    layer_name : Layer name we are searching

    Returns
    -------
    If layer exists in model, it returns the layer. Else, throws an exception.

    Yes, I know that keras has model.get_layer(). No, I am not changing this.

    """
    # Check that layer name is in model
    layer_in_model = False
    for layer in model.layers:
        if layer.name == layer_name:
            model_layer = layer
            layer_in_model = True
            break
    if not layer_in_model:
        print('The layer specified is not in the model!')
        exit(0)
    return model_layer

def fetch_intermediate_model(model, layer_name):
    """

    Parameters
    ----------
    model : A keras model
    layer_name : Name of layer that we are stopping the model

    Returns
    -------
    A subset of the inputted keras model. It ouputs a new keras model that
    is the same as the inputted keras model expect the output is at the layer
    specified by layer_name.

    """
    # Check that layer name is in model
    model_layer = fetch_layer(model, layer_name)
    return Model(inputs=model.input,outputs=model_layer.output)



#Train,Test,Analyze
class TTA():

    def __init__(self,model):
        #Keep model_class
        self.model_class=model

        #If the model has not been fully trained, train it
        if self.model_class.settings['epochs'] != self.model_class.last_epoch:
            self.train()

    #Train model. See Training directory for more details
    def train(self):
        Train_Model(self.model_class)
        self.model_class.last_epoch = self.model_class.settings['epochs']


    #All functions from Testing directory
    ###############################################################################

    #For testing the model. Currently, implements a denoising test.
    def test(self):
        save_dir = os.path.join(self.model_class.epochsubresults_dir, 'full_model')
        makedir(save_dir)
        DataSetPSNRSSIMTest(self.model_class.name,self.model_class.model,
                            save_dir,self.model_class.pipeline,
                            self.model_class.settings['sigma'])

    # For testing the range of the outputs of the model model
    def test_output_range(self):
        save_dir = os.path.join(self.model_class.epochsubresults_dir, 'full_model')
        makedir(save_dir)
        DataSetRangeTest(self.model_class.name, self.model_class.model,
                            save_dir, self.model_class.pipeline,
                            self.model_class.settings['sigma'])


    #For testing the range of the outputs of a specified layer
    def test_layer_range(self, layer_name):
        intermediate_model = fetch_intermediate_model(self.model_class.model, layer_name)
        save_dir = os.path.join(self.model_class.epochsubresults_dir, layer_name)
        makedir(save_dir)
        DataSetRangeTest(self.model_class.name, intermediate_model,
                            save_dir, self.model_class.pipeline,
                            self.model_class.settings['sigma'])


    #All functions from Analysis directory
    ################################################################################

    #Takes output of model and saves it to results directory
    def analysis_output(self, clean_image = False, save_image_output=False,
                              save_matlab_output=False, save_numpy_output=False):
        save_dir = os.path.join(self.model_class.epochsubresults_dir,'full_model')
        makedir(save_dir)
        DataSetDenoiseImages(self.model_class.name, self.model_class.model,
                            save_dir, self.model_class.pipeline,
                            self.model_class.settings['sigma'], clean_image=clean_image, save_image_output=save_image_output,
                              save_matlab_output=save_matlab_output, save_numpy_output=save_numpy_output, clip=True)

    # Takes layer output in model and saves it to results directory
    def analysis_layer_output(self, layer_name, clean_image = False, save_image_output=False,
                              save_matlab_output=False, save_numpy_output=False):
        intermediate_model = fetch_intermediate_model(self.model_class.model, layer_name)
        save_dir = os.path.join(self.model_class.epochsubresults_dir,layer_name)
        makedir(save_dir)
        DataSetDenoiseImages(self.model_class.name, intermediate_model,
                            save_dir, self.model_class.pipeline,
                            self.model_class.settings['sigma'], clean_image=clean_image, save_image_output=save_image_output,
                              save_matlab_output=save_matlab_output, save_numpy_output=save_numpy_output)

    #Plot the TNRD radial basis function activations and save them to results dir
    def analysis_plot_activations(self, layer_name, weight_loc):
        save_dir = os.path.join(self.model_class.epochsubresults_dir, layer_name)
        makedir(save_dir)
        layer = fetch_layer(self.model_class.model, layer_name)
        plot_functions(save_dir, layer, weight_loc)

    # Plot the TNRD kernels and save them to results dir
    def analysis_plot_kernels(self, layer_name, weight_loc):
        save_dir = os.path.join(self.model_class.epochsubresults_dir, layer_name)
        makedir(save_dir)
        layer = fetch_layer(self.model_class.model,layer_name)
        plot_kernels(save_dir,layer,weight_loc)

    #Display kernel weights from TNRD layer
    def analysis_display_kernel_weights(self, layer_name, loc):
        layer = fetch_layer(self.model_class.model,layer_name)
        weights = layer.get_weights()
        weight = weights[loc]
        filters = weight.shape[3]
        print('{} Kernels:'.format(layer_name))
        print('/////////////////////////////////////////')
        for i in range(filters):
            print('--------------------{}-----------------------'.format(i))
            print(weight[:,:,0,i])

    #Display activation weights from TNRD layer
    def analysis_display_activation_weights(self, layer_name, loc):
        layer = fetch_layer(self.model_class.model,layer_name)
        weights = layer.get_weights()
        weight = weights[loc]
        filters = weight.shape[0]
        print('{} Activations:'.format(layer_name))
        print('/////////////////////////////////////////')
        for i in range(filters):
            print('--------------------{}-----------------------'.format(i))
            print(weight[i,:])