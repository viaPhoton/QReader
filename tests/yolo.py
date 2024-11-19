# import ultralytics, load the custom segmentation model at '/Users/ursinho/src/qrcodes-fos/qrdet/resources/eits-n-1.0-fp16.onnx'

from ultralytics import YOLO
import cv2
import numpy as np

# Load the image using cv2
c = cv2.imread('tests/d-c001.jpg')
# c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
p = cv2.imread('tests/21-52-47-w34-heartbeat.jpg')
p2 = cv2.imread('tests/p.jpg')
# p = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)

model = YOLO('/Users/ursinho/src/qrcodes-fos/qrdet/resources/eits-n-1.0-fp16.onnx', task='segment')
results = model(p2, save=True, show=True)

# for result in results:
#     # Get the height and width of the original image
#     height, width = result.orig_img.shape[:2]

#     # Create the background
#     background = np.ones((height, width, 3), dtype=np.uint8) * 255
    
#     # Get all predicted masks
#     masks = result.masks.xy

#     # Get the original image
#     orig_img = result.orig_img

#     for mask in masks:
#         mask = mask.astype(int)

#         # Create a mask image
#         mask_img = np.zeros_like(orig_img)

#         # Fill the contour of the mask image in white
#         cv2.fillPoly(mask_img, [mask], (255, 255, 255))

#         # Extract the object from the original image using the mask
#         masked_object = cv2.bitwise_and(orig_img, mask_img)

#         # Copy the masked object to the background image
#         background[mask_img == 255] = masked_object[mask_img == 255]

# # Display the result
# cv2.imshow('Segmented objects', background)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
