'''
Created 10/16/20
Ryan Cecil
Custom callbacks for Keras. Tests the model on the data sets specified at every test_every iterations
'''

from keras.callbacks import Callback
import os
import numpy as np
from Testing.main_test import psnr_ssim_test,log,writeCsv

#Callback for conducting PSNR,SSIM test every test_every epochs
class PSNRSSIMTest(Callback):
    def __init__(self, model_name, model,pipeline,subresults_dir,settings):
        super(PSNRSSIMTest, self).__init__()
        self.model_name = model_name
        self.model = model
        self.pipeline = pipeline
        self.test_epoch = settings['test_every']
        self.sigma = settings['sigma']
        self.results_dir = subresults_dir

        super(PSNRSSIMTest, self).__init__()
    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.test_epoch == 0 and epoch != 0:

            #Get psnr,ssim values
            psnrs, ssims = psnr_ssim_test(self.model, self.pipeline, self.sigma)

            psnr_avg = np.mean(psnrs)
            ssim_avg = np.mean(ssims)

            #Write to csv file
            eval_file = os.path.join(self.results_dir, self.model_name + '_' + self.pipeline.testset_name + 'test_training_results.csv')
            if (epoch+1) == self.test_epoch:
                out = ['epoch','psnr','ssim']
                writeCsv(eval_file, out, 'w')
            out = [epoch+1, psnr_avg, ssim_avg]
            writeCsv(eval_file, out, 'a')
            log('Test Datset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(self.pipeline.testset_name, psnr_avg, ssim_avg))