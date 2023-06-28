# Configure Python environment
# ---- Basic modules ----
import os
import warnings
warnings.filterwarnings('ignore')

# ---- Tensorflow ----
import tensorflow as tf

# ---- Keras ----
from keras import backend as K
from keras.losses import binary_crossentropy

# ---- segmentation_models ----
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()

# Model metrics
class Metrics(object):

    def __init__(self, smooth: int = 1):
        self.smooth = smooth
    
    def dice_coef(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + self.smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + self.smooth)

    def iou_coef(self, y_true, y_pred):
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
        union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
        iou = K.mean((intersection + self.smooth) / (union + self.smooth), axis=0)
        return iou

    def dice_loss(self, y_true, y_pred):
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = y_true_f * y_pred_f
        score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return 1. - score

    def bce_dice_loss(self, y_true, y_pred):
        return binary_crossentropy(tf.cast(y_true, tf.float32), y_pred) + 0.5 * self.dice_loss(tf.cast(y_true, tf.float32), y_pred)


# Unet model
class UnetModel: 

    def __init__(self, model_path, model_metrics):

        self.model_path = model_path
        self.model_metrics = model_metrics
        self.model = None

        self.load()

    def load(self):
        if self.model == None:

            try: 
                self.model = tf.keras.models.load_model(
                    self.model_path,
                    custom_objects=None,
                    compile=False,
                    options=None
                    )
                
                self.model.compile(
                    optimizer='adam',
                    loss=self.model_metrics.bce_dice_loss,
                    metrics=[self.model_metrics.dice_coef, self.model_metrics.iou_coef, 'accuracy']
                )

            except: 

                self.model = None

        else: 
            
            pass
        
        return self


# Call for the Unet model
def get_model(input_mode: str):

    if input_mode ==  'L':
        model_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'unet_model_8.h5')

    elif input_mode == 'RGB':
        model_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'unet_rgb_1.h5')
    else: 
        raise ValueError('Only "L" or "RGB" accepted for image mode')
    
    metrics = Metrics()
    model = UnetModel(model_path=model_path, model_metrics=metrics).model
    
    return model