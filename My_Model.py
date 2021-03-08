'''
Contains My_Model class

My_Model class is the abstract model class for the models. Each model implemented, inherits this class
and changes the following class funcions to make the class unique. We make

1. A new functional keras model by overriding the Model(self) method
2. A new model name by overriding the get_model_name(self) method
3. A new directory to store the model and all of its seperate trainings by overriding the get_model_dir(self) method.
4. A new directory to store the model's results and all of its seperate training's results
    by overriding the get_results_dir(self) method.
5. A new format for the model's subdirectories to store the model with different settings by overriding the get_sub_dirs method
6. A new summary of the model that can easily be printed out by overriding the model_summary method
7. A new data pipeline for the model by overriding the get_pipeline(self) method. See the data_generators directory
    to get more info on the pipelines.
8. A new test callback for validating the model during training by overriding the get_validation_callback(self) method. See Testing.test_callbacks
9. The model settings of the model (e.g. # of filters, stages, kernel_size, etc.) by overriding model_settings(self,args)

You can see the methods that need to be overriden at the bottom of the class, unimplemented.
'''


from pathlib import Path
import os
from tensorflow.keras.models import load_model
import glob, re

def save_model_settings(settings, save_dir):
    """
        Saves the settings of the model into the specified directory. Settings should
        be a dictionary containing the model settings.
        Parameters
        ----------
        save_dir : Directory for settings of the model to be saved.
                The settings are taken from the arguments passed when
                training is called.
        """
    file = os.path.join(save_dir, 'settings.txt')
    with open(file, 'w') as f:
        for setting in settings:
            f.write("%s: " % setting)
            f.write("%s\n" % settings[setting])


def findLastCheckpoint(save_dir):
    """

    Parameters
    ----------
    save_dir : Save directory of our current model

    Returns
    -------
    The last epoch checkpoint the model was saved at

    """
    #Grab files
    file_list = glob.glob(os.path.join(save_dir, 'model_*'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            #Grab epochs from saved models of the form model_epoch
            result = re.findall(".*model_(.*)", file_)
            epochs_exist.append(int(result[0]))
        #Get last epoch
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

#Abstract methods to help keep things neat
class My_Model():
    def __init__(self,settings):
        #Get name of model
        self.name = self.get_model_name()
        #Get model settings
        self.settings = self.model_settings(settings)
        #Get model directory
        self.model_dir = self.get_model_dir()
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        #Grab results directory
        self.results_dir = self.get_results_dir()
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        #Get model/results directory depending on model settings
        self.submodel_dir, self.subresults_dir = self.get_sub_dirs()
        Path(self.submodel_dir).mkdir(parents=True, exist_ok=True)
        Path(self.subresults_dir).mkdir(parents=True, exist_ok=True)
        #Get results directory for specific epoch of the model
        self.epochsubresults_dir = os.path.join(self.subresults_dir,self.name + '_epoch{:03d}'.format(self.settings['epochs']))
        Path(self.epochsubresults_dir).mkdir(parents=True, exist_ok=True)
        #Grab pipeline
        self.pipeline = self.get_pipeline()(self.settings)
        #Load last save model
        self.model,self.last_epoch = self.load_saved_model()
        #Grab our validation callback
        self.tester = self.get_validation_callback()(self.name,self.model,self.pipeline,self.subresults_dir,self.settings)

    def load_saved_model(self):
        """

        Returns
        -------
        Returns new or loaded model

        """
        model_path = os.path.join(self.submodel_dir, 'model' + '_{:03d}'.format(self.settings['epochs']))
        #Check if model has already been trained. i.e. the saved model with the last epoch exists
        if os.path.isfile(model_path):
            model = load_model(model_path)
            return model,self.settings['epochs']
        #Else attempt to find the last checkpoint the model was trained at
        else:
            initial_epoch = findLastCheckpoint(self.submodel_dir)
            if initial_epoch != 0:
                model_path = os.path.join(self.submodel_dir, 'model' + '_{:03d}'.format(initial_epoch))
                model = load_model(model_path)
                return model,initial_epoch
            #If there are no saved versions of the model, return an untrained model
            else:
                save_model_settings(self.settings, self.submodel_dir)
                model = self.Model()
                return model,0

    def get_model_name(self):
        """
        Must implemented in inherited class. Return "model_name"
        """
        pass

    def Model(self):
        """
        Must implemented in inherited class. Return keras_model
        """
        pass

    def get_model_dir(self):
        """
        Must implemented in inherited class. Return directory where all models of this type will be saved
        """
        pass

    def get_results_dir(self):
        """
        Must implemented in inherited class. Return directory where resutls from all models of this type will be saved
        """
        pass

    def get_sub_dirs(self):
        """
        Must implemented in inherited class. Return directory where model/results from this specific model dependant
        on settings will be saved.
        """
        pass

    def model_summary(self):
        """
        Must implemented in inherited class. Displays some summary of the model
        """
        pass

    def get_pipeline(self):
        """
        Must implemented in inherited class. Must return a pipeline from the data_generators directory
        """
        pass

    def get_validation_callback(self):
        """
        Must implemented in inherited class. Must return a callback from Testing.test_callbacks.py
        """
        pass

    def model_settings(self, args):
        """
        Must implemented in inherited class. Must return a dictionary of model settings. For training, these keys must
        be in the dictionary:
        {'optimizer': Adam, 'initial_lr': 0.01, 'lr_scheduler': lr_schedule2,
                    'batch_size': 128, 'epoch_steps': 1000, 'epochs': 100, 'save_every': 10, 'test_every': 10}
        """
        pass
