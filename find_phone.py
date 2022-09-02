import cv2, sys
import tensorflow as tf

def findPhone(img_path, model, scale_percent=70):
# This function is to locate a phone in the given image by a model
    img_ = cv2.imread(img_path)
    width = int(img_.shape[1] * scale_percent / 100)
    height = int(img_.shape[0] * scale_percent / 100)
    dim = (width, height)
    # Resize the image 
    img_ = cv2.resize(img_, dim, interpolation = cv2.INTER_AREA)/255
    # Ge the prediction from the  model
    pred = model.predict(tf.expand_dims(img_, axis=0), verbose=False)
    return pred[0]

# Get the image path
img_path = sys.argv[1]
# Get the trained model
model = tf.keras.models.load_model("phone_detection.h5")
# Get the phone's location
x, y = findPhone(img_path, model)
# Print out the location
print(f"{x:.4f} {y:.4f}")
