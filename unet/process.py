# Import modules
# --------------
import os
import numpy as np 

def ImageIterator(path, suffix): 
    '''Collect the filenames of the specified suffix under the given path
    
    Parameters
    ----------
        path: str
        suffix: str
    
    Yield
    -------
        image_path: str
    '''

    import os
    
    for path, _, names in os.walk(path) : 
        for fn in names : 
            if fn.endswith(suffix) : 

                yield os.path.join(path, fn)


def get_meta(image_path) : 
    '''Retrieve image metadata
    
    Parameters
    ----------
    image_path: str
    
    Returns
    -------
    meta_data: dict
    '''

    from PIL import Image 
    from PIL.ExifTags import TAGS
    
    image = Image.open(image_path)
        
    meta_data = {
        'Filename': os.path.basename(image.filename),
        'Image_size': image.size,
        'Image_height': image.height,
        'Image_width': image.width,
        'Image_format': image.format,
        'Image_mode': image.mode, 
        'Is_animated': getattr(image, 'is_animated', False),
        'Frames': getattr(image, 'n_frames', 1),
    }

    return meta_data


def pad_image(image: np.ndarray, dimension: tuple): 
    '''Downsize image to the given dimension while maintain the aspect ratio and pad the new image
    
    Parameters
    ----------
        image: np.ndarray
        dimension: (width, height) 
        
    Returns
    -------
        new_image: np.ndarray
    '''

    import cv2
  
    ori_dimension = (image.shape[1], image.shape[0])
    ratio = float(max(dimension)/max(ori_dimension))
    # Resize with maintained aspect ratio
    new_size = tuple([int(x*ratio) for x in ori_dimension])
    image = cv2.resize(image, new_size)
    # Padding
    delta_w = dimension[0] - new_size[0]
    delta_h = dimension[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
    
    return image


def extract_annotations(image: np.ndarray, thresholding) : 
    '''Extract annotations from the input image
    
    Parameters
    ----------
        image: np.ndarray
        thresholding: cv2.THRESH_BINARY | cv2.THRESH_BINARY_INV |
        cv2.THRESH_TRUNC | cv2.THRESH_TOZERO | cv2.THRESH_TOZERO_INV
        
    Returns
    -------
        annotations: List[np.ndarray]
    '''

    import cv2
    
    annotations = []
    # Apply threshold to input image
    ret, thresh = cv2.threshold(image, 127, 255, thresholding)
    # Extract the contours from the binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # Loop through hierarchy
    for idx, h in enumerate(hierarchy[0]):

        _annotation = np.zeros(image.shape, dtype=np.uint8)
        # The desired contour of hierarchy[2] == hierarchy[3] == -1 or (hierarchy[2] == -1 and hierarchy[3] > 0)
        if h[2] == h[3] == -1 or (h[2] == -1 and h[3] > 0):
            # Filter out noises whose area usually < 50 
            if cv2.contourArea(contours[idx], oriented=True) >= 50:
                cv2.drawContours(_annotation, contours, idx, 255, -1)
                annotations.append(_annotation)
    
    return annotations
               

def annotation2rle(image, binary=True) : 
    '''Generate run length encoding (RLE) from annotation

    Parameters
    ----------
        image: np.ndarray
        binary: True in default
    
    Returns
    -------
        rle: np.ndarray
    '''

    import cv2

    if not binary:
        image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1] / 255
        
    rle = []
    start = 0 
    length = 0 
    prev = None # Use to check previous pixel
    image = image.flatten() # Flatten the image numpy array
        
    for indice, pixel in enumerate(image) : 
        
        if prev == None : # The first pixel
            prev = pixel 
        
        if pixel == 0 : # Background 
            prev = pixel 
            
            # Record start and length
            if (start > 0) and (start not in rle) :
                rle.extend([start, length])
        
        if pixel == 1 : # Signal
            
            if pixel != prev : # Encounter the first signal pixel
                
                start = indice # Record start indice
                length = 1
                prev = pixel 
                
            else : # The following signal pixel
                
                length += 1
                prev = pixel
                
    return ' '.join([str(x) for x in rle])
  
def rle2annotation(rle: np.ndarray, dimension: tuple, color: int = 1):
    '''Generate image from RLE

    Parameters
    ----------
        rle: np.ndarray
        dimension: tuple, e.g. (height, width)

    Returns
    -------
        image: np.ndarray
    '''
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    #starts -= 1
    ends = starts + lengths
    image = np.zeros(dimension[0] * dimension[1], dtype=np.float32)
    for lo, hi in zip(starts, ends):
        image[lo : hi] = color

    return image.reshape(dimension)

def merge_annotations(annotations, dimension):
    '''Merge all annotations in a single image
    
    Parameters
    ----------
    annotations: List[annotations]
    dimension: tuple, e.g. (height, width)
    
    Returns
    -------
    image: np.ndarray
    '''
    
    image = np.zeros(dimension, dtype=np.float32)
    for _annotation in annotations :

        # Make sure shape of annotations is consistent with the given dimension
        if _annotation.shape[0] == dimension[0] and _annotation.shape[1] == dimension[1]: 
            image = np.maximum(image, _annotation)
        else: 
            raise ValueError('Size of annotation ({}, {}) is inconsistent with the desired dimension ({}, {}).'\
                             .format(_annotation.shape[0], _annotation.shape[1], dimension[0], dimension[1]))
        
    return image


def build_masks(labels, dimension):
    '''Usually mask image'''
    
    mask = np.zeros(dimension, dtype=np.float32)
    for label in labels :
        _mask = rle2annotation(label, dimension, color=1)
        mask = np.maximum(mask, _mask)
        
    return mask


def resize_image(image: np.ndarray, dimension):
    '''Resize the input image to the given dimension
    
    Parameters
    ----------
    image: np.ndarray
    dimension: (width, height)
    
    Returns
    -------
    image: np.ndarray
    '''

    import cv2

    # Restore if image was normalized
    if np.amax(image) <= 1: 
        image = image * 255
    image = image.astype('uint8')
    image = cv2.resize(image, dimension, interpolation=cv2.INTER_AREA)
    image = np.expand_dims(image, axis=-1)

    return image

def write_tofile(fn, image):
    import os
    import cv2

    cv2.imwrite(fn, image)
    
    return os.path.exists(fn)