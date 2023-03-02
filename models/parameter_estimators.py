import torch
from vit_pytorch import ViT


# <ellipse cx="33" cy="167" rx="69" ry="157" fill="#19BA5C" opacity="0.56"  transform="rotate(4 33 167)"/>
N_ellipse_parameters = 2 + 2 + 3 + 1 + 1  # cx,cy=2, rx,ry=2, fill=3, opacity=1, angle=1
N_shapes = 8
ViT_out = N_ellipse_parameters * N_shapes


class EllipseParametersRegressor(torch.nn.Module):
    """Regresses out ellipse parameters from image."""
    def __init__(self, image_size=256, channels=3,
                 num_shapes_in_image=N_shapes, patch_size=32,
                 dim=128, depth=6, heads=16, mlp_dim=64, dropout=0.1,
                 emb_dropout=0.1, num_ellipse_parameters=N_ellipse_parameters):
        """

        :param image_size:
        :param num_classes:
        :param patch_size:
        :param dim:
        :param depth:
        :param heads:
        :param mlp_dim:
        :param dropout:
        :param emb_dropout:
        """
        super().__init__()
        self.num_shapes_in_image = num_shapes_in_image
        self.num_ellipse_parameters = num_ellipse_parameters
        self.image_size = image_size
        self.vit = ViT(
            image_size=image_size,
            channels=channels,
            patch_size=patch_size,
            num_classes=num_shapes_in_image * num_ellipse_parameters,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout

        )
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        parameters = self.vit(x)  # Bx (N_shapes * N_parameters)
        B = parameters.shape[0]
        parameters = parameters.reshape(B, self.num_shapes_in_image,
                                        self.num_ellipse_parameters)
        # cx,cy=2, rx,ry=2, fill=3, opacity=1, angle=1
        cx = self.sigmoid(parameters[..., 0]) * self.image_size
        cy = self.sigmoid(parameters[..., 1]) * self.image_size
        rx = self.sigmoid(parameters[..., 2]) * self.image_size
        ry = self.sigmoid(parameters[..., 3]) * self.image_size
        fill = self.sigmoid(parameters[..., 4:4+3]) * 255
        opacity = self.sigmoid(parameters[..., 7])
        angle = self.sigmoid(parameters[..., -1]) * 180
        return {'cx': cx, 'cy': cy, 'rx': rx, 'ry': ry,
                'fill': fill, 'opacity': opacity, 'angle': angle}


def main():
    image = torch.randn((1, 3, 256, 256))
    ellipse_param_reg = EllipseParametersRegressor()
    parameters = ellipse_param_reg(image)
    print('\n'.join([f"{k}: {v.shape}" for k,v in parameters.items()]))


if __name__ == "__main__":
    main()

