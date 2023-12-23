import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from unet import build_unet
from metrics import dice_loss, dice_coef

def load_image(path):
    """ Function to load and preprocess the image """
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict_mask(model, image):
    """ Function to predict the mask of the image """
    pred_mask = model.predict(image)[0]  # First element of batch
    pred_mask = np.squeeze(pred_mask, axis=-1)  # Remove channel dimension
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255  # Threshold and convert to byte range
    return pred_mask

if __name__ == "__main__":
    # Load the model
    with CustomObjectScope({'dice_loss': dice_loss, 'dice_coef': dice_coef}):
        model = tf.keras.models.load_model('/Users/VladPavlovich/Downloads/unetfiles/UNET/files/model.h5')

    # Load the image
    image_path = "/Users/VladPavlovich/Desktop/SingleBrainPNG/Y243.png"
    image = load_image(image_path)

    # Predict the mask
    mask = predict_mask(model, image)

    # Save or display the mask
    save_path = "/Users/VladPavlovich/Desktop/SingleBrainPNG/maskY243.png"
    cv2.imwrite(save_path, mask)

    # Optionally, display the mask
    cv2.imshow("Predicted Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
