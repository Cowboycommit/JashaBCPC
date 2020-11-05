from input_feeder import InputFeeder
from face_detection import FaceDetectionModel
from facial_landmarks_detection import FacialLandmarksDetectionModel
from gaze_estimation import GazeEstimationModel
from head_pose_estimation import HeadPoseEstimationModel
from mouse_controller import MouseController
from argparse import ArgumentParser
from sys import platform
import logging
import os
import cv2
import time
import line_profiler
profile=line_profiler.LineProfiler()
import atexit
stream = open("results/line_profile_analysis.txt", 'w')
atexit.register(profile.print_stats, stream)

# pre args variable assignment
DEVICE_KINDS = ['CPU', 'GPU', 'FPGA', 'MYRIAD', 'HETERO', 'HDDL']

# Avaliable CPU extensions from Openvino Documentation
if  platform == "linux" or platform == "linux2":
    CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
elif platform == "darwin":
    CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib"
elif platform == "win32":
    CPU_EXTENSION = None
else:
    print("Unsupported layers detected.")
    exit(1)




def build_argparser():
    #Argument_Parser command line arguments.
    #:return: command line arguments.
    parser = ArgumentParser()
    
    parser.add_argument("-fd", "--faceDetectionModel", type=str, required=True,
                        help="Path to a face detection model xml file with a trained model.")
    
    parser.add_argument("-fl", "--FacialLandmarksDetectionModel", type=str, required=True,
                        help="Path to a facial landmarks detection model xml file with a trained model.")
    
    parser.add_argument("-hp", "--HeadPoseEstimationModel", type=str, required=True,
                        help="Path to a head pose estimation model xml file with a trained model")
    
    parser.add_argument("-ge", "--gazeEstimationModel", type=str, required=True,
                        help="Path to a gaze estimation model xml file with a trained model.")
    
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to image or video file or CAM")

    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Chooose a target piece of hardware/device, CPU is default, but GPU/MYRIAD/FPGA/HDDL are avaliable as if you have them locally or you are in the DevCloud")
    
    parser.add_argument("-o", '--output_path', default='/results/', type=str)
    
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None, help="MKLDNN (CPU)-targeted custom layers." "Absolute path to a shared library with the" "kernels impl.")
    
    parser.add_argument("-prob", "--prob_threshold", required=False, type=float, default=0.7)                         

    
    return parser

@profile
def main():

    # Grab command line args
    args = build_argparser().parse_args()
    logger = logging.getLogger()
    inputFilePath = args.input
    inputFeeder = None
    inference_time = None
    
    if inputFilePath.lower()=="cam":
            inputFeeder = InputFeeder("cam")
    else:
        if not os.path.isfile(inputFilePath):
            logger.error("Unable to find specified video file")
            exit(1)
        inputFeeder = InputFeeder("video",inputFilePath)
    #else:
    #	if not os.path.isfile(inputFilePath):
    #       logger.error("Unable to find specified image file")
    #       exit(1)
    #   inputFeeder = InputFeeder("image",inputFilePath)

    # Initialize variables with the input arguments
    modelPathDict = {'FaceDetectionModel': args.faceDetectionModel,
    				 'FacialLandmarksDetectionModel': args.FacialLandmarksDetectionModel, 
    				 'GazeEstimationModel': args.gazeEstimationModel, 
    				 'HeadPoseEstimationModel': args.HeadPoseEstimationModel}
  
    for fileNameKey in modelPathDict.keys():
        if not os.path.isfile(modelPathDict[fileNameKey]):
            logger.error("Unable to find specified "+fileNameKey+" xml file")
            exit(1)
            
    fdm = FaceDetectionModel(modelPathDict['FaceDetectionModel'], args.device, args.cpu_extension)
    flm = FacialLandmarksDetectionModel(modelPathDict['FacialLandmarksDetectionModel'], args.device, args.cpu_extension)
    gem = GazeEstimationModel(modelPathDict['GazeEstimationModel'], args.device, args.cpu_extension)
    hpe = HeadPoseEstimationModel(modelPathDict['HeadPoseEstimationModel'], args.device, args.cpu_extension)
    mc = MouseController('high','fast')
 			
 	  
    inputFeeder.load_data()
       
    # Load Models and generate load times
    
    start_time = time.time()
    fdm.load_model()
    logger.error("Face detection model loaded: time: {:.3f} ms".format((time.time() - start_time) * 1000))
    first_mark = time.time()
    flm.load_model()
    logger.error("Facial landmarks detection model loaded: time: {:.3f} ms".format((time.time() - first_mark) * 1000))
    second_mark = time.time()
    hpe.load_model()
    logger.error("Head pose estimation model loaded: time: {:.3f} ms".format((time.time() - second_mark) * 1000))
    third_mark = time.time()
    gem.load_model()
    logger.error("Gaze estimation model loaded: time: {:.3f} ms".format((time.time() - third_mark) * 1000))
    load_total_time = time.time() - start_time
    logger.error("Total loading time: time: {:.3f} ms".format(load_total_time * 1000))
    logger.error("Required models have been loaded..")
   
    frame_count = 0
    start_inf_time = time.time()
    for ret, frame in inputFeeder.next_batch():
        if not ret:
            break
        frame_count+=1
        if frame_count%5==0:
            cv2.imshow('video',cv2.resize(frame,(600,800)))
    
        key = cv2.waitKey(60)
        croppedFace, face_coords = fdm.predict(frame.copy(), args.prob_threshold)
        if type(croppedFace)==int:
            logger.error("Unable to detect the face.")
            if key==27:
                break
            continue
        if frame_count%5==0:
            mc.move(new_mouse_coord[0],new_mouse_coord[1])

        hp_out = hpe.predict(croppedFace.copy())
        left_eye, right_eye, eye_coords = flm.predict(croppedFace.copy())
        new_mouse_coord, gaze_vector = gem.predict(left_eye, right_eye, hp_out)         
            
    inference_time = round(time.time() - start_inf_time, 1)
    total_frames = int(frame_count)
    fps = int(frame_count) / (inference_time)
    logger.error("count {} seconds".format(frame_count))
    logger.error("total inference time {} seconds".format(inference_time))
    logger.error("total frames {} frames".format(frame_count))
    logger.error("fps {} frame/second".format(fps))
    
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RunReport.txt'), 'w') as R:
    	R.write('Load Time: '+ str(load_total_time) + '\n') 
    	R.write('Inference Time :' + str(inference_time) + '\n')
    	R.write('total frames processed' + str(total_frames) +'\n')
    	R.write('fps: ' + str(fps) + '\n')          
    logger.error("VideoStream ended...")
    cv2.destroyAllWindows()
    inputFeeder.close() 
    atexit.register(profile.print_stats)   

if __name__ == '__main__':
	main() 

