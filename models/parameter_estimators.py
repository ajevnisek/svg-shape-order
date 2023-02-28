import torch
from vit_pytorch import ViT


# <ellipse cx="33" cy="167" rx="69" ry="157" fill="#19BA5C" opacity="0.56"  transform="rotate(4 33 167)"/>
N_ellipse_parameters = 2 + 2 + 3 + 1 + 1  # cx,cy=2, rx,ry=2, fill=3, opacity=1, angle=1
N_shapes = 8
ViT_out = N_ellipse_parameters * N_shapes


class EllipseParametersRegressor(torch.nn.Module):
    """Regresses out ellipse parameters from image."""
    def __init__(self, image_size=256, patch_size=16,
            num_classes=ViT_out,
            dim=128,
            depth=6,
            heads=16,
            mlp_dim=64,
            dropout=0.1,
            emb_dropout=0.1):
        super().__init__()
        self.vit = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=N_ellipse_parameters * N_shapes,
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
        parameters = parameters.reshape(B, N_shapes, N_ellipse_parameters)
        # cx,cy=2, rx,ry=2, fill=3, opacity=1, angle=1
        cx = parameters[..., 0]
        cy = parameters[..., 1]
        rx = parameters[..., 2]
        ry = parameters[..., 3]
        fill = parameters[..., 4:4+3]
        opacity = parameters[..., 7]
        angle = parameters[..., -1]
        return {'cx': cx, 'cy': cy, 'rx': rx, 'ry': ry, 'fill': fill, 'opacity': opacity, 'angle': angle}


def main():
    image = torch.randn((1, 3, 256, 256))
    ellipse_param_reg = EllipseParametersRegressor()
    parameters = ellipse_param_reg(image)
    print('\n'.join([f"{k}: {v.shape}" for k,v in parameters.items()]))


if __name__ == "__main__":
    main()
