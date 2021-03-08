'''
Implements a common_choices function for use with call_model.py. Just checks if the action specified in
call_model.py is an actual action. If so, starts the corresponding action with our TTA class.
'''

def common_choices(TTA, action, args):
    r = True

    #####################################################################################
    if action == 'summary':
        '''To display summary of all layers in the model'''
        TTA.model_class.model_summary()

    #Tesing
    ####################################################################################
    elif action == 'test':
        '''
            Tests the model on the images in the test directory of the pipeline.
            Outputs a csv file with the desired test metrics on all test images
            Results stored in model_results
            '''
        TTA.test()

    elif action == 'testoutputrange':
        '''
            Tests the model on the images in the test directory of the pipeline.
            Outputs a csv file with the range, max, and min values of the output of the model
            Results stored in model_results
            '''
        TTA.test_output_range()

    elif action == 'testlayerrange':
        '''
            Tests the model on the images in the test directory of the pipeline.
            Outputs a csv file with the range, max, and min values of a layer within the model
            Results stored in model_results
            '''
        TTA.test_layer_range(args.layer_name)

    #Analysis
    ##########################################################################################
    elif action == 'outputs':
        '''
            Applys model to images in analysis directory
            Saves model outputs to appropriate folder in model_results
            '''
        TTA.analysis_output(clean_image=args.clean_image, save_numpy_output=args.save_numpy,
                            save_matlab_output=args.save_matlab,save_image_output=args.save_image)

    elif action == 'layeroutputs':
        '''
            Applys model to images in analysis directory
            Saves outputs of layer in model to appropriate folder in model_results
            '''
        TTA.analysis_layer_output(args.layer_name, clean_image=args.clean_image,save_numpy_output=args.save_numpy,
                                  save_matlab_output=args.save_matlab,save_image_output=args.save_image)

    elif action == 'plotkernels':
        '''
                    Plots kernels from specific model layer
                    Layer and weights loc must be given
                    Saves plots to appropriate folder in model_results
                    '''
        TTA.analysis_plot_kernels(args.layer_name,args.weight_loc)

    elif action == 'plotfunctions':
        '''
                    Plots functions from specific model layer
                    Layer and weights loc must be given
                    Saves plots to appropriate folder in model_results
                    '''
        TTA.analysis_plot_activations(args.layer_name,args.weight_loc)

    elif action == 'displaykernels':
        '''
                    Shows in terminal kernel weights from specific model layer
                    Layer and weights loc must be given
                    '''
        TTA.analysis_display_kernel_weights(args.layer_name,args.weight_loc)

    elif action == 'displayfunctions':
        '''
                    Shows in terminal function weights from specific model layer
                    Layer and weights loc must be given
                    '''
        TTA.analysis_display_activation_weights(args.layer_name,args.weight_loc)

    else:
        r = False
    return r