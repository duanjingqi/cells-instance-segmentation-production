import os
import sys
import re
import argparse
import logging

from unet import data
from unet.model import Metrics, UnetModel, get_model
from unet import process as P

parser = argparse.ArgumentParser(description='')
input_grp = parser.add_mutually_exclusive_group()
input_grp.add_argument('-image', dest='image', help='The path to the image')
input_grp.add_argument('-files', dest='files', help='The txt file containing image pathes, one path per line')
parser.add_argument('-dest', dest='destination', help='The directory to save predicted results')
parser.add_argument('-image_mode', dest='image_mode', choices=['L', 'RGB'], help='Image mode: L for grayscale, RGB for color image', default='L')
parser.add_argument('-log', dest='log', help='The name of log file', default='predict.log')
args = parser.parse_args()

# Reset the logfile 
if os.path.exists(args.log):
    os.remove(args.log)

# Create the destination directory
os.makedirs(args.destination, mode=0o755, exist_ok=True)

# logging instance
logging.basicConfig(
    filename=args.log,
    format='%(asctime)s : %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
)
logging.info('Cell Segger logging starts')

# The directory of UNet models
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'unet/saved_models')


def predict():
    
    # Load model, 'unet_model_8.h5' for grayscale image and 'unet_rgb_1.h5' for color image
    
    predictor = get_model(args.image_mode)

    if predictor is not None:
        logging.info('UNet model is ready')
    else: 
        logging.info('ERROR: failed to load UNet model')
        sys.exit(1)

    # Predict 
    if args.image:

        logging.info('Single image prediction started')
        image_meta = P.get_meta(args.image)

        # Check image mode
        if image_meta['Image_mode'] != args.image_mode:
            logging.info('ERROR: image mode does not match')
            sys.exit(1)

        X = data.ImageSequence([args.image])
        logging.info(f'Input image: {args.image}')
        y = predictor.predict(X)[0,:,:,0]
        y_resized = P.resize_image(y, image_meta['Image_size'])
        y_name = image_meta['Filename'].split('.')[0] + '_masks.png'
        P.write_tofile(
            os.path.join(args.destination, y_name),
            y_resized,
        )
        logging.info(f'Predicted masks: {os.path.join(args.destination, y_name)}')

    elif args.files:

        logging.info('Batch prediction started')
        fn_list = open(args.files, 'r').read().rstrip().split('\n')

        for fn in fn_list:
            
            image_meta = P.get_meta(fn)

            # check image mode
            if image_meta['Image_mode'] != args.image_mode:
                logging.info(f'{image_meta["Filename"]} is skipped because image mode does not match.')
                continue

            X = data.ImageSequence([fn])
            logging.info(f'Input image: {fn}')
            y = predictor.predict(X)[0,:,:,0]
            y_resized = P.resize_image(y, image_meta['Image_size'])
            y_name = image_meta['Filename'].split('.')[0] + '_masks.png'
            P.write_tofile(
                os.path.join(args.destination, y_name),
                y_resized,
            )
            logging.info(f'Predicted masks: {os.path.join(args.destination, y_name)}')

    else: 
        logging.info('No input image. Program exits!')
        sys.exit(1)

if __name__ == '__main__': 
    predict()