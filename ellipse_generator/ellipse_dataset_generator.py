import os

from ellipse_generator.svg_ellipse import EllipseImageHandler


def create_dataset(root_path, num_samples, num_ellipses):
    handler = EllipseImageHandler()
    for sample in range(int(num_samples)):
        svg_ellipses = handler.create_n_random_ellipses(n=num_ellipses)
        svg_code = handler.warp_shapes_in_svg_header(svg_ellipses)
        with open(os.path.join(root_path, f"{sample:04d}.svg"), 'w') as f:
            f.write(svg_code)
