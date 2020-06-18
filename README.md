# Object Detection using Yolov3

### Pre-requisites

##### This project is written in Python 3.6.6 using Tensorflow (deep learning), NumPy (numerical computing), OpenCV (computer vision).

      pip install -r requirements.txt


### Downloading official pretrained weights

##### Let's download official weights pretrained on COCO dataset.

      wget -P weights https://pjreddie.com/media/files/yolov3.weights


### Running the model

##### Now you can run the model and find the detections using objectDetection.py. You can change the IoU (Intersection over Union) and confidence thresholds as per the requirements.

##### The detections are saved in the data folder.

