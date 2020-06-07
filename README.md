## Object Detection using Yolov3

#Pre-requisites:
#This project is written in Python 3.6.6 using Tensorflow (deep learning), NumPy (numerical computing), OpenCV (computer vision).

pip install -r requirements.txt

# Downloading official pretrained weights
# Let's download official weights pretrained on COCO dataset.

wget -P weights https://pjreddie.com/media/files/yolov3.weights

# Running the model
# Now you can run the model using objectDetection.py script. You can change the IoU (Intersection over Union) and confidence thresholds as per the requirements.
python detect.py images 0.5 0.5 data/images/dog.jpg data/images/office.jpg
# The detections are saved in the detections folder.
# The outputs are shown below.
