import os
import json
import argparse
import datetime

from tqdm import tqdm
from shutil import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from configs.yaml_parser import yaml_parser
from svg_dataset.ellipse_dataloader import SVGDataset, svg_code_to_pil_image
from ellipse_generator.svg_ellipse import Ellipse, EllipseImageHandler, \
    rgb_to_hex, hex_color_to_rgb
from models.parameter_estimators import EllipseParametersRegressor


device = 'cuda' if torch.cuda.is_available() else 'cpu'
PARAMETER_NAMES = ['cx', 'cy', 'rx', 'ry', 'fill', 'opacity', 'angle']

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str,
                    default='configs/multiple_images_3_ellipses.yaml',
                    help='path to config file.')
args = parser.parse_args()
configurations = yaml_parser(args.config)

RESULTS_DIR = os.path.join(configurations.output_stream.results_dir,
                           datetime.datetime.now().strftime(
                               '%Y_%m_%d_%H_%M_%S'))
os.makedirs(RESULTS_DIR, exist_ok=True)
copy(args.config, os.path.join(RESULTS_DIR, os.path.basename(args.config)))
TEXTUAL_DIFF_DATA = os.path.join(RESULTS_DIR, 'diff_data')
os.makedirs(TEXTUAL_DIFF_DATA, exist_ok=True)
N_ellipse_in_image = configurations.experiment_parameters.num_ellipses
N_images_train = configurations.dataset_parameters.train.num_images
N_images_test = configurations.dataset_parameters.test.num_images
TRAIN_DATASET_PATH = configurations.dataset_parameters.train.path
TEST_DATASET_PATH = configurations.dataset_parameters.test.path
PRINT_FREQ = configurations.output_stream.print_freq
NUM_EPOCHS = configurations.train.epochs
TRAIN_BATCH_SIZE = configurations.dataset_parameters.train.batch_size
LEARNING_RATE = configurations.train.learning_rate
PARAMETER_NAME_TO_LOSS = configurations.train.ellipse_param_to_loss_function
PARAMETER_NAME_TO_WEIGHT = configurations.train.ellipse_param_to_loss_weight
LOSS_NAME_TO_LOSS = {'MSELoss': nn.MSELoss(), 'L1Loss': nn.L1Loss()}


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
    with open(os.path.join(TEXTUAL_DIFF_DATA,
                           f'{image_name}_diff_data.json'), 'w') as f:
        json.dump(diff_data, f, indent=2)


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


def generate_images_from_batch(batch, model, writer, epoch, is_test=True):
    with torch.no_grad():
        images = batch['image'].to(device)  # Bx4x256x256
        # Bx N_ellipse_in_image x 9
        ellipse_data = batch['ellipse_data'].to(device)
        num_of_ellipses_in_image = ellipse_data.shape[1]
        max_images_to_generate = 3
        for image in images[:max_images_to_generate, ...]:
            outputs = model(image.unsqueeze(0))
            ellipses = model_output_dict_to_ellipses(outputs,
                                                     num_of_ellipses_in_image)
            svg_code = ellipses_to_svg_code(ellipses)

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
            dataset_name = 'test' if is_test else 'train'
            image_name = os.path.basename(batch['image_path'][0])[
                                 :-len('.svg')]
            title = f'Orig_Recons_{dataset_name}_{image_name}'
            writer.add_image(title, x, epoch)

        svg_code_diff(original_image_code, svg_code,
                      image_name=os.path.basename(batch['image_path'][0])[
                                 :-len('.svg')])


def create_dataset_if_missing(dataset_path, num_ellipses,
                              num_images_in_dataset):
    if not os.path.exists(dataset_path):
        from ellipse_generator.ellipse_dataset_generator import create_dataset
        os.makedirs(dataset_path, exist_ok=True)
        create_dataset(dataset_path, num_images_in_dataset, num_ellipses)


def train(model, train_dataloader, criterion, optimizer,):
    running_loss = 0.0
    running_loss_for_each_param = {p: 0.0 for p in PARAMETER_NAMES}
    for i, batch in enumerate(train_dataloader):
        images = batch['image'].to(device)  # Bx4x256x256
        ellipse_data = batch['ellipse_data']  # Bx N_ellipse_in_image x 9
        gt_parameters = SVGDataset.convert_ellipse_batch_tensor_to_dict(
            ellipse_data)
        gt_parameters = {k: v.to(device) for k,v in gt_parameters.items()}
        optimizer.zero_grad()

        outputs = model(images)

        loss, loss_dict = criterion(outputs, gt_parameters)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        for p in loss_dict:
            running_loss_for_each_param[p] += loss_dict[p]
    return model, running_loss, running_loss_for_each_param


def main():
    # Create the dataset and dataloader
    create_dataset_if_missing(TRAIN_DATASET_PATH, N_ellipse_in_image,
                              N_images_train)
    create_dataset_if_missing(TEST_DATASET_PATH, N_ellipse_in_image,
                              N_images_test)
    train_dataset = SVGDataset(TRAIN_DATASET_PATH)
    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE,
                                  shuffle=True)
    val_dataset = SVGDataset(TRAIN_DATASET_PATH)
    val_dataloader = DataLoader(val_dataset, batch_size=TRAIN_BATCH_SIZE,
                                shuffle=False)
    test_dataset = SVGDataset(TEST_DATASET_PATH)
    test_dataloader = DataLoader(test_dataset, batch_size=TRAIN_BATCH_SIZE,
                                 shuffle=False)

    # Initialize the model and optimizer
    model = EllipseParametersRegressor(
        channels=4, num_shapes_in_image=N_ellipse_in_image).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Define the logger
    writer = SummaryWriter(os.path.join(RESULTS_DIR,
                                        f'multiple_images_'
                                        f'{N_ellipse_in_image}_ellipses'))

    # Define the loss function
    criterion = custom_loss

    # Train the model
    pbar = tqdm(range(NUM_EPOCHS))
    for epoch in pbar:
        model.train()
        model, running_loss, running_loss_for_each_param = train(
            model, train_dataloader, criterion, optimizer)
        avg_running_loss = running_loss / len(train_dataloader)
        writer.add_scalar('loss/total_loss', avg_running_loss, epoch)
        for p in running_loss_for_each_param:
            writer.add_scalar(f'loss/{p}', running_loss_for_each_param[p],
                              epoch)

        sub_losses_info = ' | '.join([
            f'{p} loss: {running_loss_for_each_param[p] / len(train_dataloader) :.3f}'
            for p in running_loss_for_each_param])

        pbar.set_description(
            f'Epoch {epoch + 1} | '
            f'Total Loss: {avg_running_loss :.3f} | '
            f'{sub_losses_info}')
        # Create loss figures
        if epoch % PRINT_FREQ == 1:
            model.eval()
            generate_images_from_batch(next(iter(test_dataloader)), model,
                                       writer, epoch, is_test=True)
            generate_images_from_batch(next(iter(val_dataloader)), model,
                                       writer, epoch, is_test=False)

    # Save the trained model
    torch.save(model.state_dict(),
               os.path.join(RESULTS_DIR, 'regression_model.pth'))
    writer.close()


if __name__ == '__main__':
    main()
