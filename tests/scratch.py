import cv2
from qreader import QReader
from pprint import pp

detector = QReader(model_input='/Users/ursinho/src/qrcodes-fos/qrdet/resources/eits-n-1.0-fp32.onnx')
# detector = QReader(model_input='/Users/ursinho/src/qrcodes-fos/qrdet/resources/qrdet-n-fp16.onnx')

# Load the image using cv2
c = cv2.imread('tests/d-c001.jpg')
c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)

p2 = cv2.imread('/Users/ursinho/src/qrcodes-fos/QReader/tests/21-52-47-w34-heartbeat.jpg')
p2 = cv2.cvtColor(p2, cv2.COLOR_BGR2RGB)

p = cv2.imread('tests/p.jpg')
p = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)

# Detect and decode the QRs within the image
QRs = detector.detect_and_decode(image=p, return_detections=True)

# Print the results
for QR in QRs:
    pp(QR)
