import json
import os
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


PARAMETER_NAMES = ['cx', 'cy', 'rx', 'ry', 'fill', 'opacity', 'angle']


def svg_code_diff(original_image_code, predicted_image_code):
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
    with open('results/diff_data.json', 'w') as f:
        json.dump(diff_data, f, indent=2)


def generate_images_from_batch(batch, model, epoch):
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
            recons_pil_image = svg_code_to_pil_image(svg_code)
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.title('original image')
            with open(batch['image_path'][0], 'r') as f:
                original_image_code = f.read()
            plt.imshow(svg_code_to_pil_image(original_image_code))
            plt.subplot(1, 2, 2)
            plt.title('predicted image')
            plt.imshow(recons_pil_image)
            name = os.path.basename(batch['image_path'][0])[:-len('.svg')]
            plt.savefig(f'results/recons/epoch_{epoch}_recons_{name}.png')
            with open('results/network_out.svg', 'w') as f:
                f.write(svg_code)
    svg_code_diff(original_image_code, svg_code)


def main():
    N_ellipse_in_image = 3

    # Create the dataset and dataloader
    dataset = SVGDataset('../data/svg/generated_images/3_ellipses/micro-test')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    test_dataset = SVGDataset('../data/svg/generated_images/3_ellipses/single'
                            '-test')
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Initialize the model and optimizer
    model = EllipseParametersRegressor(
        channels=4, num_shapes_in_image=N_ellipse_in_image)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Define the loss function
    def custom_loss(predicted_param, gt):
        parameter_name_to_loss = {
            'cx': nn.MSELoss(), 'cy': nn.MSELoss(),
            'rx': nn.MSELoss(), 'ry': nn.MSELoss(),
            'fill': nn.L1Loss(), 'opacity': nn.MSELoss(),
            'angle': nn.MSELoss()
        }
        parameter_name_to_weight = {
            'cx': 1.0, 'cy': 1.0,
            'rx': 10.0, 'ry': 10.0,
            'fill': 250.0, 'opacity': 100.0,
            'angle': 30.0}
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

    criterion = custom_loss
    print_freq = 50
    # Train the model
    total_loss_tracker = []
    sub_loss_tracker = []
    pbar = tqdm(range(1000))
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

        sub_losses_info = ' | '.join([
            f'{p} loss: {running_loss_for_each_param[p] / len(dataloader) :.3f}'
            for p in running_loss_for_each_param])

        pbar.set_description(
            f'Epoch {epoch + 1} | '
            f'Total Loss: {running_loss/ len(dataloader) :.3f} | '
            f'{sub_losses_info}')
        total_loss_tracker.append(running_loss/ len(dataloader))
        sub_loss_tracker.append(running_loss_for_each_param)
        # Create loss figures
        if epoch % print_freq == 1:
            generate_images_from_batch(next(iter(test_dataloader)), model, epoch)
            plt.clf()
            plt.title(f'total_loss, epoch: {epoch-1:03d}')
            plt.plot(total_loss_tracker)
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.grid(True)
            plt.savefig(os.path.join('results', 'losses',
                                     f'total_loss_epoch_{epoch-1:03d}.png'))
            plt.clf()
            for p in running_loss_for_each_param:
                plt.clf()
                plt.title(f'{p} loss, epoch: {epoch-1:03d}')
                plt.plot([epoch_dict[p]
                          for epoch_dict in sub_loss_tracker])
                plt.ylabel(f'{p} loss')
                plt.xlabel('epoch')
                plt.grid(True)
                plt.savefig(os.path.join('results', 'losses',
                                         f'{p}_epoch_{epoch-1:03d}.png'))

    # Save the trained model
    torch.save(model.state_dict(), 'results/regression_model.pth')


if __name__ == '__main__':
    main()
