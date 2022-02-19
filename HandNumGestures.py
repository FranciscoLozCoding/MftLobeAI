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


# Import Pi Camera library
from waggle.data.vision import Camera
from time import sleep

#Import Lobe python library
from lobe import ImageModel

username = 'admin'
password = 'admin'
ip_address = '10.42.0.104'
camera = Camera(f'rtsp://{username}:{password}@{ip_address}:554/cam/realmonitor?channel=1&subtype=1')

# Load Lobe TF Lite model
# --> Change model path
model = ImageModel.load('./Hand Gestures TFLite')



if __name__ == '__main__':
    # Start the camera preview, make slightly transparent to see any python output
    #   Note: preview only shows if you have a monitor connected directly to the Pi
    camera.start_preview(alpha=200)
    # Pi Foundation recommends waiting 2s for light adjustment
    sleep(5) 
    # Optional image rotation for camera
    # --> Change or comment out as needed
    camera.rotation = 180
    #Input image file path here
    # --> Change image path as needed
    camera.capture('./image.jpg') 
    #Stop camera
    camera.stop_preview()

    # Run photo through Lobe TF model and get prediction results
    result = model.predict_from_file('./image.jpg')

    print(result.labels)

    sleep(1)
