import cv2
import numpy as np


def image_preprocess(img_path):

    image = cv2.imread(img_path, 0)
    _, thresholded_image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)

    thresholded_image = cv2. bitwise_not(thresholded_image)
    resized_image = cv2.resize(thresholded_image, (28, 28))
    # final_image = np.expand_dims(resized_image, axis=0)
    # final_image = np.expand_dims(resized_image)

    final_image = resized_image/255.0
    cv2.imwrite('final_image.png', final_image)
    # return final_image

ans = image_preprocess('opencv_frame_0.png')


