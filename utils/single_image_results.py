import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from torch.utils.data import DataLoader

from models.parameter_estimators import EllipseParametersRegressor
from svg_dataset.ellipse_dataloader import SVGDataset, svg_code_to_pil_image
from ellipse_generator.svg_ellipse import Ellipse, EllipseImageHandler, \
    rgb_to_hex, hex_color_to_rgb

PARAMETER_NAMES = ['cx', 'cy', 'rx', 'ry', 'fill', 'opacity', 'angle']


def model_output_dict_to_ellipses(output_dict, num_ellipses_in_image):
    ellipses = []
    for shape in range(num_ellipses_in_image):
        shape_params = {param: output_dict[param][0][shape].item()
                        for param in PARAMETER_NAMES if param != 'fill'}
        shape_params['fill'] = rgb_to_hex(
            output_dict['fill'][0][shape].tolist())

        ellipses.append(Ellipse(**shape_params))
    return ellipses


def ellipses_to_svg_code(ellipses: list):
    svg_lines = []
    for ellipse in ellipses:
        svg_lines.append(ellipse.to_svg_line())
    handler = EllipseImageHandler()
    svg_code = handler.warp_shapes_in_svg_header('\n'.join(svg_lines))
    return svg_code


model = EllipseParametersRegressor(num_shapes_in_image=3, channels=4)
model.load_state_dict(
    torch.load('results/single_image_results/3_ellipses/regression_model.pth'))
rand_outputs = model(torch.rand(1, 4, 256, 256))
ellipses = model_output_dict_to_ellipses(rand_outputs, 3)
rand_svg_code = ellipses_to_svg_code(ellipses)
with open('rand_input_single_image.svg', 'w') as f:
    f.write(rand_svg_code)

dataset = SVGDataset('results/single_image_results/3_ellipses/original_image/')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
image = next(iter(dataloader))['image']
image_output = model(image)
ellipses = model_output_dict_to_ellipses(image_output, 3)
image_svg_code = ellipses_to_svg_code(ellipses)
with open('original_input_single_image.svg', 'w') as f:
    f.write(image_svg_code)

with open(
    'results/single_image_results/3_ellipses/original_image/ORIGINAL_IMAGE.svg',
    'r') as f: original_code = f.read()

original_pil_image = svg_code_to_pil_image(original_code)
gen_from_image = svg_code_to_pil_image(image_svg_code)
gen_from_noise = svg_code_to_pil_image(rand_svg_code)

plt.clf()
plt.title('original image')
plt.imshow(original_pil_image)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig('01_original.png')

plt.clf()
plt.title('generated from original image')
plt.imshow(gen_from_image)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig('02_gen_from_original.png')

plt.clf()
plt.title('generated from random noise image')
plt.imshow(gen_from_noise)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig('03_gen_from_noise.png')

plt.clf()
plt.subplot(3, 1, 1)
plt.title('original image')
plt.imshow(original_pil_image)
plt.xticks([])
plt.yticks([])

plt.subplot(3, 3, 4)
plt.title('generated from original image')
plt.imshow(gen_from_image)
plt.xticks([])
plt.yticks([])
plt.subplot(3, 3, 5)
plt.title('rgb channels difference')
plt.imshow(np.abs(
    np.array(gen_from_image).astype(np.float32) -
    np.array(original_pil_image).astype(np.float32))[..., :-1].mean(axis=-1),
           norm=LogNorm(vmin=0.01, vmax=100))
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.subplot(3, 3, 6)
plt.title('alpha channels (abs) difference')
plt.imshow(np.abs(
    np.array(gen_from_image).astype(np.float32) -
    np.array(original_pil_image).astype(np.float32))[..., -1],
           norm=LogNorm(vmin=0.01, vmax=100))
plt.colorbar()
plt.xticks([])
plt.yticks([])

plt.subplot(3, 3, 4 + 3)
plt.title('generated from noise')
plt.imshow(gen_from_noise)
plt.xticks([])
plt.yticks([])
plt.subplot(3, 3, 5 + 3)
plt.title('rgb channels difference')
plt.imshow(np.abs(
    np.array(gen_from_noise).astype(np.float32) -
    np.array(original_pil_image).astype(np.float32))[..., :-1].mean(axis=-1),
           norm=LogNorm(vmin=0.01, vmax=100))
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.subplot(3, 3, 6 + 3)
plt.title('alpha channels (abs) difference')
plt.imshow(np.abs(
    np.array(gen_from_noise).astype(np.float32) -
    np.array(original_pil_image).astype(np.float32))[..., -1],
           norm=LogNorm(vmin=0.01, vmax=100))
plt.colorbar()
plt.xticks([])
plt.yticks([])
fig = plt.gcf()
fig.set_size_inches((12, 8))
plt.tight_layout()
plt.savefig('all in one.png')
