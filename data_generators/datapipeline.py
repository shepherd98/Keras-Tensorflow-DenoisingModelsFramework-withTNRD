'''
Main data pipeline class. All defined pipelines must inherit from this class.

A new data pipeline must override:
1. get_data_dir(self): must return location of data dir with subdirectories Train, Test, and Analyze in it
2. def keras_generator(self): Generator that generates the data for training. These parameters are passed into
    data generator (data_dir,model_settings,train_settings)
3. test_dataset_name(self): returns name of test dataset
4 train_dataset_name(self): returns name of train dataset
'''

import os

def grab_image_fnames(direct):
    """

    Parameters
    ----------
    direct : directory

    Returns
    -------
    list of filenames of directory that should contain only images

    """
    images = os.listdir(direct)
    images = sorted(images)
    for i in range(len(images)):
        images[i] = os.path.join(direct,images[i])
    return images
'''
Main pipeline class:

init contains calls for storing information about names and directories.
'''
class pipeline():
    def __init__(self,settings):
        #Store settings
        self.settings = settings
        self.data_dir = self.get_data_dir()
        self.test_data_name = self.test_dataset_name()
        self.train_data_name = self.train_dataset_name()
        self.train_data_dir = os.path.join(self.data_dir,'Train')
        self.test_data_dir = os.path.join(self.data_dir,'Test')
        self.analysis_data_dir = os.path.join(self.data_dir,'Analysis')
        self.trainset_name = self.train_dataset_name()
        self.testset_name = self.test_dataset_name()

    def train_images_list(self):
        return grab_image_fnames(self.train_data_dir)

    def test_images_list(self):
        return grab_image_fnames(self.test_data_dir)

    def analysis_images_list(self):
        return grab_image_fnames(self.analysis_data_dir)

    def get_data_dir(self):
        pass
    def keras_generator(self):
        pass
    def test_dataset_name(self):
        pass
    def train_dataset_name(self):
        pass