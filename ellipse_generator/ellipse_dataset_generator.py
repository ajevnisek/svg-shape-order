import os
from ellipse_generator.svg_ellipse import EllipseImageHandler


N_train = int(1e4)
N_test = int(1e2)


def create_dataset(root_path, num_samples, num_ellipses):
    handler = EllipseImageHandler()
    for sample in range(int(num_samples)):
        svg_ellipses = handler.create_n_random_ellipses(n=num_ellipses)
        svg_code = handler.warp_shapes_in_svg_header(svg_ellipses)
        with open(os.path.join(root_path, f"{sample:04d}.svg"), 'w') as f:
            f.write(svg_code)


if __name__ == '__main__':
    datasets_to_nof_images = {'train': N_train, 'test': N_test,
                              'micro-train': 100, 'micro-test': 20,
                              'single-train': 1, 'single-test': 1}
    for dataset in datasets_to_nof_images:
        num_samples = datasets_to_nof_images[dataset]
        dataset_root = os.path.join('image_data', dataset, )
        print(f"Creating {dataset} dataset with {num_samples} "
              f"in {dataset_root}...")
        os.makedirs(dataset_root, exist_ok=True)
        create_dataset(dataset_root, num_samples, num_ellipses=3)
