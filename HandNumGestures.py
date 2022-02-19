# from lobe import ImageModel

#model = ImageModel.load('./Hand Gestures TFLite')

#from PIL import Image
#img = Image.open('./number-five-made-with-hand.jpg')
#result = model.predict(img)


# Print top prediction
#print(result.prediction)

# Print all classes
#for label, confidence in result.labels:
 #   print(f"{label}: {confidence*100}%")

# video streamimng
import cv2

camIP = '10.42.0.104'
username = 'admin'
password = 'admin'
streamURL = 'rtsp://' + username + ':' + password + '@' + camIP + ':554/cam/realmonitor?channel=1&subtype=1'

cam = cv2.VideoCapture(streamURL)

#Import Lobe python library
from lobe import ImageModel
from PIL import Image

model = ImageModel.load('./Hand Gestures TFLite')

cam = cv2.VideoCapture(0)

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

cam.release()

cv2.destroyAllWindows()
