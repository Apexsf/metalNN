import cv2
import math
import numpy as np
import os 
os.chdir(os.path.dirname(os.path.abspath(__file__)))
def bilinear_interpolation(image, y, x):
    height = image.shape[0]
    width = image.shape[1]

    x1 = max(min(math.floor(x), width - 1), 0)
    y1 = max(min(math.floor(y), height - 1), 0)
    x2 = max(min(math.ceil(x), width - 1), 0)
    y2 = max(min(math.ceil(y), height - 1), 0)

    a = float(image[y1, x1])
    b = float(image[y2, x1])
    c = float(image[y1, x2])
    d = float(image[y2, x2])

    dx = x - x1
    dy = y - y1

    new_pixel = a * (1 - dx) * (1 - dy)
    new_pixel += b * dy * (1 - dx)
    new_pixel += c * dx * (1 - dy)
    new_pixel += d * dx * dy
    return new_pixel
    # return round(new_pixel)


def resize(image, new_height, new_width):
    new_image = np.zeros((new_height, new_width), image.dtype)  # new_image = [[0 for _ in range(new_width)] for _ in range(new_height)]

    orig_height = image.shape[0]
    orig_width = image.shape[1]

    # Compute center column and center row
    x_orig_center = (orig_width-1) / 2
    y_orig_center = (orig_height-1) / 2

    # Compute center of resized image
    x_scaled_center = (new_width-1) / 2
    y_scaled_center = (new_height-1) / 2

    # Compute the scale in both axes
    scale_x = orig_width / new_width
    scale_y = orig_height / new_height

    for y in range(new_height):
        for x in range(new_width):
            x_ = (x - x_scaled_center) * scale_x + x_orig_center
            y_ = (y - y_scaled_center) * scale_y + y_orig_center

            new_image[y, x] = bilinear_interpolation(image, y_, x_)

    return new_image

# img = cv2.imread('graf.png', cv2.IMREAD_GRAYSCALE)  # http://man.hubwiz.com/docset/OpenCV.docset/Contents/Resources/Documents/db/d70/tutorial_akaze_matching.html
np.random.seed(42)
img = np.random.randint(0,255,(64,64)).astype('float32')
# img = np.random.rand(64,64) 

new_width = 128
new_height = 128

resized_img = resize(img, new_height, new_width)

# Reference for testing
reference_resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

abs_diff = cv2.absdiff(reference_resized_img, resized_img)
print('max abs_diff = ' + str(abs_diff.max())) 
print('mean abs_diff = ' + str(abs_diff.mean()))  # 1 gray level difference

img.flatten().tofile("input.bin")
resized_img.flatten().tofile("out.bin")
print()
# cv2.imshow('img', img)
# cv2.imshow('resized_img', resized_img)
# cv2.imshow('reference_resized_img', reference_resized_img)
# cv2.imshow('abs_diff*10', abs_diff*10)
# cv2.waitKey()
# cv2.destroyAllWindows()