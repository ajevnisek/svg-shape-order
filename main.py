import os.path

from ellipse_generator.ellipse_dataset_generator import create_dataset
from ellipse_generator.svg_ellipse import DemoEllipseInSVG, EllipseImageHandler


if __name__ == '__main__':
    demo = DemoEllipseInSVG()
    demo.create_example_ellipse_image('results/ellipse_example.svg')
    handler = EllipseImageHandler()
    svg_ellipses = handler.create_n_random_ellipses(n=2)
    svg_code = handler.warp_shapes_in_svg_header(svg_ellipses)
    with open('results/two_ellipses.svg', 'w') as f:
        f.write(svg_code)

    svg_ellipses = handler.create_n_random_ellipses(n=5)
    svg_code = handler.warp_shapes_in_svg_header(svg_ellipses)
    with open('results/five_ellipses.svg', 'w') as f:
        f.write(svg_code)
    svg_ellipses = handler.create_n_random_ellipses(n=20)
    svg_code = handler.warp_shapes_in_svg_header(svg_ellipses)
    with open('results/twenty_ellipses.svg', 'w') as f:
        f.write(svg_code)
    N_train = 1e4
    N_test = 200
    for dataset in ['train', 'test']:
        root = os.path.join('..', 'data', 'svg', 'generated_images',
                            '3_ellipses', dataset)
        os.makedirs(root, exist_ok=True)
        create_dataset(root, N_test if dataset == 'test' else N_train, num_ellipses=8)
