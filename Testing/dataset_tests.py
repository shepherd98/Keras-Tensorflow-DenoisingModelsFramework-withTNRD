from Testing.main_test import psnr_ssim_test, range_test, writeCsv,log
import numpy as np
import os

def DataSetPSNRSSIMTest(model_name, model, save_dir, pipeline, sigma):
    """
    Test for getting PSNR and SSIM metrics on Test set. Saves resuts in csv file

    Parameters
    ----------
    model_name : Name of model
    model : Keras Model
    save_dir : Directory to save results
    pipeline : Pipeline of model_class. See data_generators for different pipeliness / methods
    sigma : Noise level we are testing model on
    """
    psnrs, ssims = psnr_ssim_test(model,pipeline,sigma)

    psnr_avg = np.mean(psnrs)
    ssim_avg = np.mean(ssims)

    out_list = []
    out_list.append(['im','psnr','ssim'])
    for i in range(len(psnrs)):
        out_list.append([str(i + 1), psnrs[i], ssims[i]])
    out_list.append(['Total Average', psnr_avg, ssim_avg])
    eval_file = os.path.join(save_dir, model_name + '_' + pipeline.testset_name + '_denoiseresults.csv')
    for i in range(len(out_list)):
        if i == 0:
            writeCsv(eval_file, out_list[i], 'w')
        else:
            writeCsv(eval_file, out_list[i], 'a')
    log('Datset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(pipeline.testset_name, psnr_avg, ssim_avg))

def DataSetRangeTest(model_name, model, save_dir, pipeline, sigma):
    """
        Test for getting min and max metrics on Test set for an output in the model. Saves results in csv file

        Parameters
        ----------
        model_name : Name of model
        model : Keras Model
        save_dir : Directory to save results
        pipeline : Pipeline of model_class. See data_generators for different pipeliness / methods
        sigma : Noise level we are testing model on
        """
    maxs, mins = range_test(model,pipeline,sigma)

    ranges = np.abs(np.array(maxs)-np.array(mins))

    range_avg = np.mean(ranges)
    max_avg = np.mean(maxs)
    min_avg = np.mean(mins)

    out_list = []
    out_list.append(['im','range','max', 'min'])
    for i in range(len(maxs)):
        out_list.append([str(i + 1), ranges[i], maxs[i], mins[i]])
    out_list.append(['Total Average', range_avg, max_avg, min_avg])
    out_list = sorted(out_list, key=lambda elem: (elem[0]))
    eval_file = os.path.join(save_dir, model_name + '_' + pipeline.testset_name + '_rangeresults.csv')
    for i in range(len(out_list)):
        if i == 0:
            writeCsv(eval_file, out_list[i], 'w')
        else:
            writeCsv(eval_file, out_list[i], 'a')
    log('Datset: {0:10s}: Range = {1:2.2f}dB, Max = {2:1.2f}, Min = {3:1.2f}'.format(pipeline.testset_name, range_avg, max_avg, min_avg))