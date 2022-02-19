from lobe import ImageModel

model = ImageModel.load('./Hand Gestures TFLite')

# OPTION 1: Predict from an image file
result = model.predict_from_file('./number-five-made-with-hand.jpg')

# Print top prediction
print(result.prediction)

# Print all classes
for label, confidence in result.labels:
    print(f"{label}: {confidence*100}%")

# Visualize the heatmap of the prediction on the image 
# this shows where the model was looking to make its prediction.
heatmap = model.visualize(img)
heatmap.show()
