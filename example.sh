#!/bin/bash

#Train the model
python3 call_model.py --model_name TNRD --stages 3 --kernel_size 3 --filters 8
#Get summary of model
python3 call_model.py --model_name TNRD --stages 3 --kernel_size 3 --filters 8 --action summary
#Get test results on Test set
python3 call_model.py --model_name TNRD --stages 3 --kernel_size 3 --filters 8 --action test
#Get output range on test set
python3 call_model.py --model_name TNRD --stages 3 --kernel_size 3 --filters 8 --action testoutputrange
#Get output range of layer on test set
python3 call_model.py --model_name TNRD --stages 3 --kernel_size 3 --filters 8 --action testlayerrange --layer_name Inference1
#Save model outputs
python3 call_model.py --model_name TNRD --stages 3 --kernel_size 3 --filters 8 --action outputs --save_numpy True --save_matlab True --save_image True
#Save layer outputs
python3 call_model.py --model_name TNRD --stages 3 --kernel_size 3 --filters 8 --action layeroutputs --layer_name Inference1 --save_numpy True --save_matlab True --save_image True
#Save and plot kernels
python3 call_model.py --model_name TNRD --stages 3 --kernel_size 3 --filters 8 --action plotkernels --layer_name Inference1 --weight_loc 0
#Save and plot activation functions
python3 call_model.py --model_name TNRD --stages 3 --kernel_size 3 --filters 8 --action plotfunctions --layer_name Inference1 --weight_loc 1