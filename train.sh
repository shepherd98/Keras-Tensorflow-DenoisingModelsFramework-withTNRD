#!/bin/bash

#Some example calls to call_model
#python3 call_model.py --model_name TNRD --stages 3 --kernel_size 3 --filters 8
#python3 call_model.py --model_name KfTNRD --stages 3 --dfilters 8 --dkernel_size 3 --kkernel_size 3 --kfilters 8
#python3 call_model.py --model_name KfTNRDF --stages 3 --dfilters 8 --dkernel_size 3 --kkernel_size 3 --kfilters 8
#
#python3 call_model.py --model_name TNRD --stages 5 --kernel_size 5 --filters 24
#python3 call_model.py --model_name KfTNRD --stages 5 --dfilters 24 --dkernel_size 5 --kkernel_size 5 --kfilters 24
#python3 call_model.py --model_name KfTNRDF --stages 5 --dfilters 24 --dkernel_size 5 --kkernel_size 5 --kfilters 24
#
#python3 call_model.py --model_name TNRD --stages 5 --kernel_size 7 --filters 48
#python3 call_model.py --model_name KfTNRD --stages 5 --dfilters 24 --dkernel_size 7 --kkernel_size 7 --kfilters 48
#python3 call_model.py --model_name KfTNRDF --stages 5 --dfilters 24 --dkernel_size 7 --kkernel_size 7 --kfilters 48

#python3 call_model.py --model_name KCfTNRDFCurvature --stages 3 --dfilters 8 --dkernel_size 3 --epochs 80
#python3 call_model.py --model_name KCfTNRDFCurvature --stages 5 --dfilters 8 --dkernel_size 3 --epochs 80
#python3 call_model.py --model_name KCfTNRDFCurvature --stages 7 --dfilters 8 --dkernel_size 3 --epochs 80
#python3 call_model.py --model_name KCfTNRDFCurvature --stages 10 --dfilters 8 --dkernel_size 3 --epochs 80
#python3 call_model.py --model_name KCfTNRDFCurvature --stages 3 --dfilters 16 --dkernel_size 3 --epochs 80
#python3 call_model.py --model_name KCfTNRDFCurvature --stages 3 --dfilters 24 --dkernel_size 3 --epochs 80
#
#python3 call_model.py --model_name KCfTNRDCurvature --stages 3 --dfilters 8 --dkernel_size 3 --epochs 80
#python3 call_model.py --model_name KCfTNRDCurvature --stages 5 --dfilters 8 --dkernel_size 3 --epochs 80
#python3 call_model.py --model_name KCfTNRDCurvature --stages 7 --dfilters 8 --dkernel_size 3 --epochs 80
#python3 call_model.py --model_name KCfTNRDCurvature --stages 10 --dfilters 8 --dkernel_size 3 --epochs 80
#python3 call_model.py --model_name KCfTNRDCurvature --stages 3 --dfilters 16 --dkernel_size 3 --epochs 80
#python3 call_model.py --model_name KCfTNRDCurvature --stages 3 --dfilters 24 --dkernel_size 3 --epochs 80

python3 call_model.py --model_name KCfTNRDFCurvature --stages 5 --dfilters 24 --dkernel_size 3 --epochs 80