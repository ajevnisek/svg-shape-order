import json
import os

import torchvision.transforms
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from svg_dataset.ellipse_dataloader import SVGDataset, svg_code_to_pil_image
from ellipse_generator.svg_ellipse import Ellipse, EllipseImageHandler, \
    rgb_to_hex, hex_color_to_rgb
from models.parameter_estimators import EllipseParametersRegressor, N_ellipse_parameters
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from configs.yaml_parser import yaml_parser

PARAMETER_NAMES = ['cx', 'cy', 'rx', 'ry', 'fill', 'opacity', 'angle']

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str,
                    default='configs/single_image_3_ellipses.yaml',
                    help='path to config file.')
args = parser.parse_args()
configurations = yaml_parser(args.config)

RESULTS_DIR = configurations.output_stream.results_dir
os.makedirs(RESULTS_DIR, exist_ok=True)
N_ellipse_in_image = configurations.experiment_parameters.num_ellipses
DATASET_PATH = configurations.dataset_parameters.train.path
PRINT_FREQ = configurations.output_stream.print_freq
NUM_EPOCHS = configurations.train.epochs
TRAIN_BATCH_SIZE = configurations.dataset_parameters.train.batch_size
LEARNING_RATE = configurations.train.learning_rate
PARAMETER_NAME_TO_LOSS = configurations.train.ellipse_param_to_loss_function
PARAMETER_NAME_TO_WEIGHT = configurations.train.ellipse_param_to_loss_weight
LOSS_NAME_TO_LOSS = {'MSELoss': nn.MSELoss(),
                  'L1Loss': nn.L1Loss()}


def custom_loss(predicted_param, gt):
    parameter_name_to_loss = {param_name: LOSS_NAME_TO_LOSS[loss_name]
                              for param_name, loss_name in
                              PARAMETER_NAME_TO_LOSS.items()}
    parameter_name_to_weight = PARAMETER_NAME_TO_WEIGHT
    loss = 0
    loss_dict = {}
    for param_name in predicted_param:
        loss_func = parameter_name_to_loss[param_name]
        loss_for_param = loss_func(predicted_param[param_name],
                                   gt[param_name])
        weight = parameter_name_to_weight[param_name]
        loss_dict[param_name] = loss_for_param.item()
        loss = loss + weight * loss_for_param
    return loss, loss_dict


def svg_code_diff(original_image_code, predicted_image_code, image_name):
    handler = EllipseImageHandler()
    original_ellipses = handler.svg_code_to_ellipses_list(original_image_code)
    predicted_ellipses = handler.svg_code_to_ellipses_list(
        predicted_image_code)

    diff_data = OrderedDict()
    for idx, (gt, pred) in enumerate(zip(original_ellipses,
                                         predicted_ellipses)):
        diff_data[f'Ellipse #{idx:02d}'] = {}
        for param_name in PARAMETER_NAMES:
            if param_name not in ['fill', 'rgb_color']:
                diff_data[f'Ellipse #{idx:02d}'][param_name] = {
                    'gt': f"{gt.todict()[param_name]}",
                    'pred': f"{pred.todict()[param_name]}",
                    'diff': f"{float(gt.todict()[param_name]) - float(pred.todict()[param_name]):.2f}"}
            elif param_name == 'fill':
                rgb_gt = hex_color_to_rgb(gt.todict()[param_name])
                rgb_pred = hex_color_to_rgb(pred.todict()[param_name])
                for color, color_gt, color_pred in zip(['R', 'G', 'B'],
                                                       rgb_gt, rgb_pred):
                    diff_data[f'Ellipse #{idx:02d}'][color] = {
                        'gt': f"{color_gt:.2f}",
                        'pred': f"{color_pred:.2f}",
                        'diff': f"{color_gt - color_pred}"}
    with open(os.path.join(RESULTS_DIR,
                           f'{image_name}_diff_data.json'), 'w') as f:
        json.dump(diff_data, f, indent=2)


def generate_images_from_batch(batch, model, writer, epoch):
    with torch.no_grad():
        images = batch['image']  # Bx4x256x256
        ellipse_data = batch['ellipse_data']  # Bx N_ellipse_in_image x 9
        for image in images:
            outputs = model(image.unsqueeze(0))
            svg_lines = []
            for shape in range(ellipse_data.shape[1]):
                shape_params = {param: outputs[param][0][shape].item()
                                for param in PARAMETER_NAMES if param != 'fill'}
                shape_params['fill'] = rgb_to_hex(outputs['fill'][0][shape].tolist())

                ellipse = Ellipse(**shape_params)
                svg_lines.append(ellipse.to_svg_line())
            handler = EllipseImageHandler()
            svg_code = handler.warp_shapes_in_svg_header('\n'.join(svg_lines))
            with open(os.path.join(RESULTS_DIR, 'NETWORK_OUT.svg'), 'w') as f:
                f.write(svg_code)

            with open(batch['image_path'][0], 'r') as f:
                original_image_code = f.read()
            with open(os.path.join(RESULTS_DIR, 'ORIGINAL_IMAGE.svg'),
                      'w') as f:
                f.write(original_image_code)

            recons_pil_image = svg_code_to_pil_image(svg_code).convert('RGB')
            original_pil_image = svg_code_to_pil_image(original_image_code).convert('RGB')
            recons_torch = torchvision.transforms.ToTensor()(recons_pil_image)
            orig_torch = torchvision.transforms.ToTensor()(original_pil_image)
            images_list = torch.cat([orig_torch.unsqueeze(0),
                                     recons_torch.unsqueeze(0)], axis=0)
            x = vutils.make_grid(images_list, normalize=True, scale_each=True)
            writer.add_image('Orig_Recons', x, epoch)

        svg_code_diff(original_image_code, svg_code,
                      image_name=os.path.basename(batch['image_path'][0])[
                                 :-len('.svg')])


def create_dataset_if_missing(dataset_path, num_ellipses):
    if not os.path.exists(dataset_path):
        from ellipse_generator.ellipse_dataset_generator import create_dataset
        os.makedirs(dataset_path, exist_ok=True)
        create_dataset(dataset_path, 1, num_ellipses)


def main():
    # Create the dataset and dataloader
    create_dataset_if_missing(DATASET_PATH, N_ellipse_in_image)
    dataset = SVGDataset(DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

    # Initialize the model and optimizer
    model = EllipseParametersRegressor(
        channels=4, num_shapes_in_image=N_ellipse_in_image)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Define the logger
    writer = SummaryWriter(os.path.join(RESULTS_DIR,
                                        f'single_image_'
                                        f'{N_ellipse_in_image}_ellipses'))

    # Define the loss function
    criterion = custom_loss

    # Train the model
    pbar = tqdm(range(NUM_EPOCHS))
    for epoch in pbar:
        running_loss = 0.0
        running_loss_for_each_param = {p: 0.0 for p in PARAMETER_NAMES}
        for i, batch in enumerate(dataloader):
            images = batch['image']  # Bx4x256x256
            ellipse_data = batch['ellipse_data']  # Bx N_ellipse_in_image x 9
            gt_parameters = SVGDataset.convert_ellipse_batch_tensor_to_dict(
                ellipse_data)
            optimizer.zero_grad()

            outputs = model(images)

            loss, loss_dict = criterion(outputs, gt_parameters)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            for p in loss_dict:
                running_loss_for_each_param[p] += loss_dict[p]

        writer.add_scalar('loss/total_loss', running_loss / len(dataloader),
                          epoch)
        for p in running_loss_for_each_param:
            writer.add_scalar(f'loss/{p}', running_loss_for_each_param[p],
                              epoch)

        sub_losses_info = ' | '.join([
            f'{p} loss: {running_loss_for_each_param[p] / len(dataloader) :.3f}'
            for p in running_loss_for_each_param])

        pbar.set_description(
            f'Epoch {epoch + 1} | '
            f'Total Loss: {running_loss/ len(dataloader) :.3f} | '
            f'{sub_losses_info}')
        # Create loss figures
        if epoch % PRINT_FREQ == 1:
            generate_images_from_batch(next(iter(dataloader)), model,
                                       writer, epoch)

    # Save the trained model
    torch.save(model.state_dict(),
               os.path.join(RESULTS_DIR, 'regression_model.pth'))
    # export scalar data to JSON for external processing
    writer.export_scalars_to_json(os.path.join(RESULTS_DIR,
                                               "loss_stats.json"))
    writer.close()


if __name__ == '__main__':
    main()
