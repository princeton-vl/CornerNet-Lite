#!/usr/bin/env python

import cv2
from core.detectors import CornerNet_Saccade
from core.vis_utils import draw_bboxes

detector = CornerNet_Saccade()
image    = cv2.imread("demo.jpg")

bboxes = detector(image)
image  = draw_bboxes(image, bboxes)
cv2.imwrite("demo_out.jpg", image)
