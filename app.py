from image_capture import image_capture
from image_preprocessor import image_preprocess
import numpy as np
import os
import cv2

img = image_capture()

image_preprocess('opencv_frame_0.png')

image = cv2.imread('final_image.png')
final_image = np.expand_dims(image, axis=0)

# waits for user to press any key
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()

model = keras.models.load_model('my_model.h5')

class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

X_new = "final_image.png"
y_preds.round(2)

y_pred = np.argmax(model.predict(X_new), axis=-1)
print(y_pred)
