#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required
import numpy as np
import cv2


# In[2]:


# Load YOLO network
network = cv2.dnn.readNetFromDarknet("yolov3.cfg","yolov3.weights")
#Get the labels from coco file
with open("coco.names") as f:
    labels = [line.strip() for line in f]


# In[3]:


# Read the input image
imgBGR = cv2.imread("office.jpg")


# In[4]:


# Get the blob from input image
blob = cv2.dnn.blobFromImage(imgBGR,1/255.0,size=(416,416),swapRB=True,crop=False)


# In[5]:


# Implementing Forward pass
network.setInput(blob)
layers = network.getLayerNames()
#Get the output layers
output = [layers[i[0]-1] for i in network.getUnconnectedOutLayers()]
#Get the outputs from the output layer
output_from_network = network.forward(output)


# In[6]:


# Getting classes,confidences,bounding boxes
classes = []
confidences = []
bounding_boxes = []

height,width = imgBGR.shape[:2]
probability_minimum = 0.5
threshold = 0.3

colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

for result in output_from_network:
    for detected_objects in result:
        scores = detected_objects[5:]
        class_index = np.argmax(scores)
        confidence = scores[class_index]
        if confidence > probability_minimum:
            center_x = int(detected_objects[0] * width)
            center_y = int(detected_objects[1] * height)
            w = int(detected_objects[2] * width)
            h = int(detected_objects[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            bounding_boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            classes.append(class_index)


# In[7]:


# Non-maximum suppression
results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,probability_minimum, threshold)


# In[8]:


# Checking if there is at least one detected object after non-maximum suppression
if len(results) > 0:
    for i in results.flatten():
        x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
        box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

        colour_box_current = colours[classes[i]].tolist()

        cv2.rectangle(imgBGR, (x_min, y_min),(x_min + box_width, y_min + box_height),
                      colour_box_current, 2)

        text_box_current = '{}: {:.4f}'.format(labels[int(classes[i])],
                                               confidences[i])

        cv2.putText(imgBGR, text_box_current, (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)

        
cv2.namedWindow('Objects Detected', cv2.WINDOW_NORMAL)
cv2.imshow('Objects Detected', imgBGR)
cv2.imwrite('DetectedImage.jpg',imgBGR)
cv2.waitKey(0)
cv2.destroyWindow('Objects Detected')

