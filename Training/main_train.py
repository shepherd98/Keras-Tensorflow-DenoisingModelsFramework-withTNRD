'''
Ryan Cecil - Duquesne University (2020). Research under Dr. Stacey Levine, Ph.D.

Contains code for training keras models

'''
import datetime
import os
from keras.callbacks import CSVLogger, LearningRateScheduler, ModelCheckpoint
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#For printing out during training
def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

def SaveCurve(file, save_file, col1='epoch', col2='loss'):
    """
    Uses pandas to plot a curve stored in a csv file

    Parameters
    ----------
    file : File name
    save_file : Name of file to save to
    col1 : Name of col1 in csv file, used as x values
    col2 : Name of col2 in csv file, used as y values
    """
    data = pd.read_csv(file)
    pl = data.plot(x=col1, y=col2)
    pl.plot()
    plt.savefig(save_file)


def Train_Model(model_class):
    """
    Trains a model contained in the model_class based on model settings.
    Saves the model at each specified checkpoint, plots loss values, plots test values over course of training.

    Parameters
    ----------
    model_class: model class implemented in models.py
    """

    #Compile with optimizer
    model_class.model.compile(optimizer=model_class.settings['optimizer'](model_class.settings['initial_lr']))

    #Checkpoint callback
    checkpointer = ModelCheckpoint(os.path.join(model_class.submodel_dir, 'model_{epoch:03d}'),
                                   verbose=1, save_weights_only=False, period=model_class.settings['save_every'])

    #Csv file to store losses
    loss_file = os.path.join(model_class.subresults_dir, model_class.name + '_' +
                             model_class.pipeline.trainset_name + 'loss_log.csv')

    #Callback to log losses
    csv_logger = CSVLogger(loss_file, append=True, separator=',')

    #Learning rate scheduler
    lr_scheduler = LearningRateScheduler(model_class.settings['lr_scheduler'])

    #Train the model
    history = model_class.model.fit_generator(model_class.pipeline.keras_generator(),
                                  steps_per_epoch=model_class.settings['epoch_steps'], epochs=model_class.settings['epochs'], verbose=1,
                                  callbacks=[checkpointer, csv_logger, lr_scheduler,model_class.tester], initial_epoch=model_class.last_epoch)

    print('---------------------------')
    print('Finished Training')

    # Plot Training and Test Curves
    SaveCurve(loss_file, os.path.join(model_class.subresults_dir, model_class.name + '_' +
                                      model_class.pipeline.trainset_name + 'LossVals'))
    test_file = os.path.join(model_class.subresults_dir, model_class.name + '_' +
                             model_class.pipeline.testset_name + 'test_training_results.csv')
    SaveCurve(test_file, os.path.join(model_class.subresults_dir, model_class.name + '_' +
                                      model_class.pipeline.testset_name + 'TestPSNRVals'), col1='epoch', col2='psnr')
    SaveCurve(test_file, os.path.join(model_class.subresults_dir, model_class.name + '_' +
                                      model_class.pipeline.testset_name + 'TestSSIMVals'), col1='epoch', col2='ssim')
