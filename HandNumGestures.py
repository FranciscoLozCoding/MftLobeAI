import os

# video streamimng
import cv2

camIP = '00.00.0.000'
username = 'admin'
password = 'admin'
streamURL = 'rtsp://' + username + ':' + password + '@' + camIP + ':554/cam/realmonitor?channel=1&subtype=1'

cam = cv2.VideoCapture(streamURL)

#Import Lobe python library
from lobe import ImageModel
from PIL import Image

model = ImageModel.load('./Hand Gestures TFLite')

cv2.namedWindow("Count Fingers")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("Count Fingers", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

        img = Image.open(img_name)
        result = model.predict(img)
        print(result.prediction)

        # Print all classes
        for label, confidence in result.labels:
          print(f"{label}: {confidence*100}%")
        
        os.remove(img_name)

cam.release()

cv2.destroyAllWindows()
