import numpy as np
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
