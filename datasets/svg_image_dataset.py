from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import svgutils.compose as sc


class SVGDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.svg_files = list(self.data_dir.glob('*.svg'))
        self.transform = transform

    def __len__(self):
        return len(self.svg_files)

    def __getitem__(self, idx):
        svg_file = self.svg_files[idx]

        # Load the SVG file using svgutils
        svg_image = sc.SVG(svg_file)

        # Convert the SVG image to a PIL image
        png_image = svg_image.to_png()

        # Apply any specified transforms
        if self.transform is not None:
            png_image = self.transform(png_image)

        # Convert the PNG image to a PyTorch tensor
        tensor = transforms.ToTensor()(png_image)

        # Return the tensor and any additional metadata
        return tensor, svg_file.stem


def main():
    # Example usage
    data_dir = 'data/ellipses_dataset/eight_ellipses/test/'
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = SVGDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))
    images, names = batch
    print(images.shape)
    print(names)


if __name__ == "__main__":
    main()
