#!/usr/bin/python

#                          License Agreement
#                         3-clause BSD License
#
#       Copyright (C) 2018, Xperience.AI, all rights reserved.
#
# Third party copyrights are property of their respective owners.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#   * Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#   * Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#   * Neither the names of the copyright holders nor the names of the contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
# This software is provided by the copyright holders and contributors "as is" and
# any express or implied warranties, including, but not limited to, the implied
# warranties of merchantability and fitness for a particular purpose are disclaimed.
# In no event shall copyright holders or contributors be liable for any direct,
# indirect, incidental, special, exemplary, or consequential damages
# (including, but not limited to, procurement of substitute goods or services;
# loss of use, data, or profits; or business interruption) however caused
# and on any theory of liability, whether in contract, strict liability,
# or tort (including negligence or otherwise) arising in any way out of
# the use of this software, even if advised of the possibility of such damage.

# https://github.com/opencv/opencv/tree/4.x/samples

import cv2
import sys


win_name = 'Camera Preview'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

net = cv2.dnn.readNetFromCaffe("deploy.prototxt",
                               "res10_300x300_ssd_iter_140000_fp16.caffemodel")
# Model parameters
in_width = 300
in_height = 300
mean = [104, 117, 123]
conf_threshold = 0.75

frame = cv2.imread("test1.jpg")
frame_height = frame.shape[0]
frame_width = frame.shape[1]

# Create a 4D blob from a frame.
blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB = False, crop = False) #Pre-processing
# Run a model
net.setInput(blob)
detections = net.forward()
print(detections.shape)
face_loc = []
for i in range(detections.shape[2]):
    
    confidence = detections[0, 0, i, 2]
    if confidence > conf_threshold:
        x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
        y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
        x_right_top = int(detections[0, 0, i, 5] * frame_width)
        y_right_top = int(detections[0, 0, i, 6] * frame_height)
        
        
        tup=(x_left_bottom , y_left_bottom , x_right_top , y_right_top)
        face_loc.append(tup)
        print(face_loc)


# source.release()
cv2.destroyWindow(win_name)
