from skimage.filters import threshold_sauvola, gaussian
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

def sauvola(original_image):
    gray = rgb2gray(original_image)
    smoothened = gaussian(gray, sigma=0.4)
    window_size = 25
    thresh_sauvola = threshold_sauvola(smoothened, window_size=window_size, k=0.2)

    binary_sauvola = smoothened > thresh_sauvola
    return img_as_ubyte((~binary_sauvola) * 255)