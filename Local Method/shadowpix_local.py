from PIL import Image
import numpy as np
from sympy import *
import math
import shutil
import os


theta_deg = 60
theta_rad = theta_deg * math.pi / 180
s = cot(theta_rad)

image1_path = './images/albert_ss.jpg'
image2_path = './images/donald_ss.jpg'
image3_path = './images/monalisa_ss.jpg'
output_dir = './outputs/'


def read_image(image_path):
    # ffmpeg -i donald_s.jpg -vf -vf hue=s=0 donald_s.jpg
    # ffmpeg -i donald_s.jpg -vf scale=1920:1080 donald_s.jpg

    image = Image.open(image_path)
    image = np.asarray(image)[:, :, 0:1] / 255

    assert image.shape[-1] == 1
    assert image.max() <= 1
    assert image.min() >= 0

    return image[:, :, 0]


def generate_shadow_pix_local(image1, image2, image3):
    shadowed1 = 1 - image1
    shadowed2 = 1 - image2
    shadowed3 = 1 - image3

    image_shape = image1.shape

    y_casters = np.zeros((image_shape[0] + 1, image_shape[1]))
    x_casters = np.zeros((image_shape[0], image_shape[1] + 1))
    receivers = np.zeros((image_shape[0], image_shape[1]))
    
    for i_col in range(x_casters.shape[1] - 1):
        x_casters[:, i_col + 1] = x_casters[:, i_col] + s * (shadowed2[:, i_col] - shadowed1[:, i_col])

    x_casters = x_casters + (s * shadowed1[:, 0]).reshape(-1, 1)
    x_casters[0, :] -= min(0, np.min(x_casters[0, :]))
    
    c = s * (-shadowed1[:image_shape[0] - 1, :] + shadowed1[1:, :] - shadowed3[1:, :])
    
    for i_row in range(image_shape[0] - 1):
        hc = -x_casters[i_row + 1, :-1] + x_casters[i_row, :-1] + c[i_row, :]
        x_casters[i_row + 1, :] += max(np.max(hc), 0)

    receivers = x_casters[:, : image_shape[1]] - s * shadowed1
    y_casters = receivers + s * shadowed3

    y_casters = y_casters.astype(np.float32)
    x_casters = x_casters.astype(np.float32)
    receivers = receivers.astype(np.float32)

    return receivers, x_casters, y_casters


def export_to_text_file(array, file_name):
    with open(output_dir + file_name, 'a') as file_stream:
        file_stream.write('{')

        for i_row in range(array.shape[0]):
            if i_row != 0:
                file_stream.write(',')

            file_stream.write('{' + np.array2string(array[i_row], separator=',', max_line_width=1000000000000000) + '}')

        file_stream.write('}')


def reconstruct_images(receivers, x_casters, y_casters):
    sim_image1 = np.zeros(receivers.shape)
    sim_image2 = np.zeros(receivers.shape)
    sim_image3 = np.zeros(receivers.shape)
    
    for i_row in range(receivers.shape[0]):
        for i_col in range(receivers.shape[1]):
            h1 = x_casters[i_row, i_col] - receivers[i_row, i_col]
            shadowed_area1 = tan(theta_rad) * h1
            sim_image1[i_row, i_col] = (1 - shadowed_area1) * 255
            assert sim_image1[i_row, i_col] <= 255

            h2 = x_casters[i_row, i_col + 1] - receivers[i_row, i_col]
            shadowed_area2 = tan(theta_rad) * h2
            sim_image2[i_row, i_col] = (1 - shadowed_area2) * 255
            assert sim_image2[i_row, i_col] <= 255

            h3 = y_casters[i_row, i_col] - receivers[i_row, i_col]
            shadowed_area3 = tan(theta_rad) * h3
            sim_image3[i_row, i_col] = (1 - shadowed_area3) * 255
            assert sim_image3[i_row, i_col] <= 255
    
    im1 = Image.fromarray(sim_image1.astype(np.uint8))
    im1.save(output_dir + 'recon_image1.png')

    im2 = Image.fromarray(sim_image2.astype(np.uint8))
    im2.save(output_dir + 'recon_image2.png')

    im3 = Image.fromarray(sim_image3.astype(np.uint8))
    im3.save(output_dir + 'recon_image3.png')


def Generate():
    print('Reading images...')

    image1 = read_image(image1_path);
    image2 = read_image(image2_path);
    image3 = read_image(image3_path);

    image_shape = image1.shape
    assert image2.shape == image_shape
    assert image3.shape == image_shape

    print('Generating Shadowpix (local method)...')

    receivers, x_casters, y_casters = generate_shadow_pix_local(image1, image2, image3)

    print('Writing text files...')

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    
    os.mkdir(output_dir)

    export_to_text_file(receivers, 'receivers.txt')
    export_to_text_file(x_casters, 'x_casters.txt')
    export_to_text_file(y_casters, 'y_casters.txt')
    
    print('Reconstructing images...')

    reconstruct_images(receivers, x_casters, y_casters)

    print('Done.')

if __name__ == '__main__':
    Generate()