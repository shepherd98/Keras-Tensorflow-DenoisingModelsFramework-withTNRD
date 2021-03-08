'''
File for storing custom pipelines. New pipelines must inherit from pipeline in datapipelines.py

A new data pipeline must override:
1. get_data_dir(self): must return location of data dir with subdirectories Train, Test, and Analyze in it
2. def keras_generator(self): Generator that generates the data for training. These parameters are passed into
    data generator (data_dir,model_settings,train_settings)
3. test_dataset_name(self): returns name of test dataset
4 train_dataset_name(self): returns name of train dataset
'''

from data_generators.datapipeline import pipeline
from data_generators.dncnnmethods import train_datagen

#Dictates pipeline for Train400Im68 data in data directory. Data generator from dncnnmethods.py
class Train400Im68Pipeline(pipeline):
    def get_data_dir(self):
        return '/home/cpm/Research/KfTNRD/2021/Keras_Models_Training_Testing2/data/Train400Im68'
    def keras_generator(self):
        return train_datagen(self.train_data_dir,self.settings,self.settings)
    def test_dataset_name(self):
        return 'im68'
    def train_dataset_name(self):
        return 'Train400'
