import numpy as np
from keras import backend as K
from keras.preprocessing.image import array_to_img
from PIL import Image


def save_image_grid(images, filename):
    num_images = len(images)
    img_width, img_height = images[0].size
    num_wide = int(np.sqrt(num_images))
    num_heigh = int(np.ceil(num_images / num_wide))
    width = num_wide * img_width
    height = num_heigh * img_height
    output_img = Image.new(images[0].mode, (width, height), 'black')
    for i in range(num_images):
        x = (i % num_wide) * img_width
        y = (i / num_wide) * img_height
        output_img.paste(images[i], (x, y))
    output_img.save(filename)


def save_image_array_grid(samples, filename):
    if K.image_dim_ordering() == 'tf':
        num_samples, img_height, img_width, img_channels = samples.shape
    else:
        num_samples, img_channels, img_height, img_width = samples.shape
    num_wide = int(np.sqrt(num_samples))
    num_heigh = int(np.ceil(num_samples / num_wide))
    width = num_wide * img_width
    height = num_heigh * img_height
    img_mode = {1: 'L', 3: 'RGB'}[img_channels]
    output_img = Image.new(img_mode, (width, height), 'black')
    for i in range(num_samples):
        x = (i % num_wide) * img_width
        y = (i / num_wide) * img_height
        sample_arr = samples[i].clip(0, 255)
        sample_img = array_to_img(sample_arr, scale=False)
        output_img.paste(sample_img, (x, y))
    output_img.save(filename)
