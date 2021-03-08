'''
Ryan Cecil - Duquesne University (2020). Research under Dr. Stacey Levine, Ph.D.
Entrypoint for working with models.

Current Models:
/////////////////////////////////////////////////////////////////////////////////////////////////
TNRD: https://arxiv.org/pdf/1508.02848.pdf
////////////////////////////////////////////////////////////////////////////////////////////////////

Current program allows user to:

Load in / train model based on the parameters inputted into the file

Do a specific action such as:

- summary: Display summary of model. Shows layer names, number of parameters, trainable weights
- test: Apply model to test set. Save metric results in model_results
- testoutputrange:  Applys model to Test dataset. Outputs a csv file with the range, max,
        and min values of the output of the model
- testlayerrange:  ests the model on the images in the test directory of the pipeline. Outputs a
        csv file with the range, max, and min values of a layer within the model.
- outputs: Applys model to images in analysis directory. Saves model outputs to appropriate folder in model_results
- layeroutputs: Applys model to images in analysis directory. Saves outputs of layer in model to appropriate folder
    in model_results.
- plotkernels: Plots kernels from specific model layer. Layer and weights loc must be given.
- plotfunctions: Plots functions from specific model layer. Layer and weights loc must be given.
- displaykernels: Shows in terminal kernel weights from specific model layer. Layer and weights loc must be given.
- displayfunctions: Shows in terminal function weights from specific model layer. Layer and weights loc must be given
'''


import argparse
from models import *
from TTA import TTA
from actions import common_choices

parser = argparse.ArgumentParser()

#First Parameters
parser.add_argument('--model_name', default = 'TNRD', type = str, help = 'Name of model')
parser.add_argument('--action', default = 'None', type = str, help = 'Action taken by program after model is loaded and/or trained')
parser.add_argument('--model_number', default = 0, type = int, help = 'Number of model trained')
parser.add_argument('--stages', default = 3, type = int, help = 'number of diffusion process steps')
parser.add_argument('--filters', default = 8, type = int, help = 'number of convolutional filters in each layer')
parser.add_argument('--kernel_size', default = 3, type = int, help = 'size of convolutional kernels')
parser.add_argument('--channels', default = 1, type = int, help = 'number of image channels')
parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--epochs', default = 30, type = int, help = 'number of epochs')
parser.add_argument('--save_every', default = 10, type = int, help = 'test at every test_every epochs')
parser.add_argument('--test_every', default = 10, type = int, help = 'test at every test_every epochs')
#Arguments for use with analysis methods
parser.add_argument('--layer_name', default = 'Inference1', type = str, help = 'Name of layer to grab in model')
parser.add_argument('--weight_loc', default = 0, type = int, help = 'If a specific set of weights in layer are required. Dictates location of the weights in the list'
                                                                  'of weights of the layer')
parser.add_argument('--save_numpy', default = False, type = bool, help = 'Save as numpy output in results directory')
parser.add_argument('--save_matlab', default = False, type = bool, help = 'Save as matlab output in results directory')
parser.add_argument('--save_image', default = False, type = bool, help = 'Save as image output in results directory')
parser.add_argument('--clean_image', default = False, type = bool, help = 'Inputs clean instead of noisy image into model')


args = parser.parse_args()


def getGPU():
    """
    Grabs GPU. Sometimes Tensorflow attempts to use CPU when this is not called on my machine.
    From: https://www.tensorflow.org/guide/gpu
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

if __name__ == "__main__":
    getGPU()
    if args.model_name == 'TNRD':
        model = TNRD(args)
    else:
        raise Exception('None of the current models were specified')

    TTA_class = TTA(model)
    if common_choices(TTA_class, args.action, args):
       exit(0)
    else:
       print("No further action was specified")