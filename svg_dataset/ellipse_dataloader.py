import os

from tqdm import tqdm
from io import BytesIO

from PIL import Image
from cairosvg import svg2png

import torch
from torch import tensor
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from ellipse_generator.svg_ellipse import Ellipse


def svg_code_to_pil_image(svg_code, format='RGBA'):
    return Image.open(BytesIO(svg2png(bytestring=svg_code))).convert(format)


class SVGDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.svg_files = os.listdir(root_dir)
        self.paths = []
        self.images = []
        self.ellipse_data = []
        for svg_image_name in tqdm(self.svg_files):
            svg_path = os.path.join(root_dir, svg_image_name)
            self.paths.append(svg_path)
            with open(svg_path) as f:
                svg_code = f.read()
            fg_image = svg_code_to_pil_image(svg_code)
            self.images.append(fg_image)
            # background = Image.new('RGBA', fg_image.size, (255, 255, 255))
            # alpha_composite = Image.alpha_composite(background, fg_image)
            # self.images.append(alpha_composite.convert('RGB'))
            ellipse_data_for_current_image = []
            with open(svg_path, 'r') as f:
                data = f.read()
                for line in data.splitlines():
                    if 'ellipse' in line:
                        ellipse_data_for_current_image.append(
                            Ellipse.from_svg_code(line).todict())

            self.ellipse_data.append(torch.cat([
                self.convert_ellipse_dict_to_tensor(x).unsqueeze(0)
                for x in ellipse_data_for_current_image
            ]))
        self.transform = transforms.Compose([transforms.Resize(256),
                                             transforms.ToTensor(),])

    @staticmethod
    def convert_ellipse_dict_to_tensor(ellipse_dict):
        return tensor([ellipse_dict['cx'], ellipse_dict['cy'],
                       ellipse_dict['rx'], ellipse_dict['ry'],
                       ellipse_dict['rgb_color'][0],
                       ellipse_dict['rgb_color'][1],
                       ellipse_dict['rgb_color'][2],
                       ellipse_dict['opacity'], ellipse_dict['angle'],
                       ])

    @staticmethod
    def convert_ellipse_batch_tensor_to_dict(ellipse_tensor):
        assert len(ellipse_tensor.shape) == 3 and ellipse_tensor.shape[-1] ==\
               9,\
            "The tensor dimensions should be BxN_shapesx9, where 9 is the " \
            "number of ellipse parameters"
        d = {}
        d['cx'] = ellipse_tensor[..., 0]
        d['cy'] = ellipse_tensor[..., 1]
        d['rx'] = ellipse_tensor[..., 2]
        d['ry'] = ellipse_tensor[..., 3]
        d['fill'] = ellipse_tensor[..., 4:4 + 3]
        d['opacity'] = ellipse_tensor[..., 7]
        d['angle'] = ellipse_tensor[..., -1]
        return d

    def __len__(self):
        return len(self.svg_files)

    def __getitem__(self, idx):
        pil_image = self.images[idx]
        ellipse_data = self.ellipse_data[idx]

        return {'image': self.transform(pil_image),
                'ellipse_data': ellipse_data,
                'image_path': self.paths[idx]}
