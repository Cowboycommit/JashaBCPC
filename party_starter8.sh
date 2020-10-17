

source ~/intel/openvino_2020.4.287/bin/setupvars.sh

cd ~/Desktop/JashaBCPC

python3 src/main.py -f src/models/intel/face-detection-adas-0001/FP16-INT8/face-detection-adas-0001.xml -fl src/models/intel/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009.xml -hp src/models/intel/head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001.xml -g src/models/intel/gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002.xml -i src/bin/demo.mp4


