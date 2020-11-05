'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import cv2
import numpy as np
from openvino.inference_engine import IECore

class FaceDetectionModel:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, prob_threshold=0.7):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.device = device
        self.prob_threshold = prob_threshold
        self.core = IECore()
        self.network = self.core.read_network(model=str(model_name), weights=str(os.path.splitext(model_name)[0] + ".bin"))
        self.input = next(iter(self.network.inputs))
        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input].shape
        self.output = next(iter(self.network.outputs))
        self.output_names = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_names].shape
        self.exec_net = self.core.load_network(network=self.network, device_name=self.device,num_requests=1)

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.exec_network = self.core.load_network(self.network, self.device)
        return self.exec_network
        
        if len(unsupported_layers) > 0 and self.device=='CPU':
            print("unsupported layers found:{}".format(unsupported_layers))
            if not self.extensions==None:
                print("Attempting to apply known cpu_extension(s)")
                self.plugin.add_extension(self.extensions, self.device)
                supported_layers = self.plugin.query_network(network = self.network, device_name=self.device)
                unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
                if len(unsupported_layers) > 0:
                    print("Unable to find correct custom layer extensions, please specify correct path the exstension(s)")
                else:
                   print("Check for  viable extensions for these unsupported layers, " + str(unsupported_layers))
                   exit(1)
        
    def predict(self, image, prob_threshold):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        
        img_processed = self.preprocess_input(image.copy())
        outputs = self.exec_net.infer({self.input_name:img_processed})
        coords = self.preprocess_output(outputs, prob_threshold)
        if (len(coords)==0):
            return 0, 0
        coords = coords[0] 
        height=image.shape[0]
        width=image.shape[1]
        coords = coords* np.array([width, height, width, height])
        coords = coords.astype(np.int32)
        
        cropped_face = image[coords[1]:coords[3], coords[0]:coords[2]]
        return cropped_face, coords

    def check_model(self):
        ''

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        image_shape = self.network.inputs[self.input].shape
        image_resized = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        img_processed = np.transpose(np.expand_dims(image_resized,axis=0), (0,3,1,2))
        return img_processed
            

    def preprocess_output(self, outputs, prob_threshold):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        coords =[]
        outs = outputs[self.output_names][0][0]
        for out in outs:
            conf = out[2]
            if conf>prob_threshold:
                x_min=out[3]
                y_min=out[4]
                x_max=out[5]
                y_max=out[6]
                coords.append([x_min,y_min,x_max,y_max])
        return coords

   
