import os
import sys
import re
import glob
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from unet import data
from unet import process as P
from unet.model import Metrics, UnetModel, get_model

# Images used for test
images_list = glob.glob('./images/*')

def test_predict():

    for fn in images_list:

        # Check if image metadata is complete
        image_meta = P.get_meta(fn)

        assert any(x is None for x in image_meta.values()) is False

        # Check if UNet model is loaded 
        predictor = get_model(image_meta['Image_mode'])

        assert predictor.count_params() == 10115791

        # Check if data pipeline works properly
        X = data.ImageSequence([fn])

        assert X.shape == (1, 256, 256, 3)

        # Check if the predicted result is in right dimension
        y = predictor.predict(X)

        assert y.shape == X.shape

        # Check if predicted image can be restored to original dimension
        y_resized = P.resize_image(y[0, :, :, 0], image_meta['Image_size'])

        assert y_resized.dtype == 'uint8'
        assert y_resized.shape == (*image_meta['Image_size'], 3)