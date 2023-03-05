import glob
import contextlib
from PIL import Image


FILENAMES = ('01_original.png', '02_gen_from_original.png',
             '03_gen_from_noise.png')


def filenames_path_to_gif_path(filenames: list = FILENAMES,
                               video_path: str = 'movie.gif'):
    # use exit stack to automatically close opened images
    with contextlib.ExitStack() as stack:
        # lazily load images
        imgs = (stack.enter_context(Image.open(f))
                for f in sorted(filenames))

        # extract  first image from iterator
        img = next(imgs)

        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats
        # .html#gif
        img.save(fp=video_path, format='GIF', append_images=imgs,
                 save_all=True, duration=500, loop=0)

