import os
import matplotlib.pyplot as plt
from ellipse_generator.svg_ellipse import Ellipse
from datasets.ellipse_dataloader import SVGDataset


dataset = SVGDataset(root_dir='image_data/micro-test')
print(f'dataset length: {len(dataset)}')
image = dataset[0]['image']
ellipse_data = dataset[0]['ellipse_data']
image_path = dataset[0]['image_path']

for ellipse_dict in ellipse_data:
    ellipse = Ellipse.from_dict(ellipse_dict)
    print(str(ellipse) + ' ' + ', '.join([
        f"{c}: {str(x)}" for c, x in zip('RGB', ellipse.rgb_color)]))
print('\n\nSVG code:')
with open(image_path, 'r') as f:
    print(f.read())


print(f'image shape: {image.shape}')
plt.suptitle('ellipse image')
plt.subplot(3, 3, 2)
plt.title('image')
plt.imshow(image.permute(1, 2, 0))
plt.xticks([])
plt.yticks([])
plt.subplot(3, 2, 3)
plt.title('R-Channel')
plt.imshow(image[0])
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.subplot(3, 2, 4)
plt.title('G-channel')
plt.imshow(image[1])
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.subplot(3, 2, 5)
plt.title('B-channel')
plt.imshow(image[2])
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.subplot(3, 2, 6)
plt.title('A-channel')
plt.imshow(image[3])
plt.xticks([]); plt.yticks([])
plt.colorbar()
plt.savefig(os.path.join('results', 'ellipse_dataset_demo.png'))
