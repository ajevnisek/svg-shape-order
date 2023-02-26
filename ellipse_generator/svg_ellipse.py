import random
import numpy as np


np.random.seed(123456789)


def generate_randon_color():
    """Generate a randon color in hex format.
    :return: str. Random Hex format color (e.g: #FF0000 which is red).
    """
    return "#" + ''.join([np.random.choice([*'ABCDEF0123456789']) for i in range(6)])


def hex_color_to_rgb(hex_color):
    """
    Convert a hexadecimal color to an RGB int tuple.
    :param hex_color: str. color in hex format (e.g: #FF0000 which is red).
    :return: tuple(int, int, int): RGB tuple of the color (e.g: (255, 0, 0) which is red).
    """
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


class Ellipse:
    def __init__(self, cx, cy, rx, ry, angle, fill, opacity):
        self.cx = cx
        self.cy = cy
        self.rx = rx
        self.ry = ry
        self.angle = angle
        self.fill = fill
        self.opacity = opacity

    @classmethod
    def from_svg_code(cls, svg_line: str):
        """Create Ellipse class from svg line.
        Args:
            svg_line: str. An ellipse SVG line.
        Returns:

        """
        cx = svg_line.split('cx="')[-1].split('"')[0]
        cy = svg_line.split('cy="')[-1].split('"')[0]
        rx = svg_line.split('rx="')[-1].split('"')[0]
        ry = svg_line.split('ry="')[-1].split('"')[0]
        fill = svg_line.split('fill="')[-1].split('"')[0]
        opacity = svg_line.split('opacity="')[-1].split('"')[0]
        angle = svg_line.split('rotate("')[-1].split(' ')[0]
        return cls(cx, cy, rx, ry,angle, fill, opacity)

    def to_svg_line(self):
        svg_line = f'  <ellipse cx="{self.cx}" cy="{self.cy}" rx="{self.rx}" ry="{self.ry}" fill="{self.fill}" ' \
                   f'opacity="{self.opacity:.2f}" '
        svg_line += f' transform="rotate({self.angle} {self.cx} {self.cy})"/>\n'
        return svg_line


class DemoEllipseInSVG:
    @staticmethod
    # Get ellipse parameters from user input
    def create_svg_with_one_ellipse_with_params(cx, cy, rx, ry, angle, fill, opacity, width, height):
        """Create an SVG image with a single ellipse.

        :param cx: x-coordinate of the ellipse center.
        :param cy: y-coordinate of the ellipse center.
        :param rx: the horizontal radius of the ellipse
        :param ry: the vertical radius of the ellipse
        :param angle: the rotation angle of the ellipse in degrees
        :param fill: the fill color of the ellipse (e.g. 'red', '#FF0000')
        :param opacity: float. the opacity of the ellipse (e.g. 0.5).
        :param width: the width of the SVG image
        :param height: the height of the SVG image
        :return: svg_code: str. representing the SVG code.
        """

        # Generate SVG code for ellipse
        min_x = min(-cx + rx, 0)
        min_y = min(-cy + ry, 0)
        svg_code = f'<svg viewBox="{min_x} {min_y} {width + abs(min_x)} {height + abs(min_y)}" xmlns="http://www.w3.org/2000/svg">\n'
        svg_code += f'  <ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{fill}" opacity="{opacity:.2f}" '
        svg_code += f' transform="rotate({angle} {cx} {cy})"/>\n'
        svg_code += f'</svg>'
        return svg_code

    @staticmethod
    def create_example_ellipse_image(path: str):
        """ Create a sample ellipse image and save it to path.
        :param path: str. The path to the SVG example image (e.g: 'results/ellipse_example.svg').
        :return: None
        """
        svg_code = DemoEllipseInSVG.create_svg_with_one_ellipse_with_params(cx=75, cy=25, rx=40, ry=25,angle=60,
                                                                            fill='red', opacity=0.5,
                                                                            width=200, height=200)
        with open(path, 'w') as f:
            f.write(svg_code)


class EllipseImageHandler:
    """Creates and Parses SVG images made of Ellipses."""
    @staticmethod
    def create_n_random_ellipses(n):
        """Create n random ellipses in SVG code.

        :param n: int. The number of SVG ellipses.
        :return: str. SVG code representing all ellipses.
        """
        cx_s = np.random.randint(20, 180, size=(n, ))
        cy_s = np.random.randint(20, 180, size=(n,))
        rx_s = np.random.randint(20, 180, size=(n,))
        ry_s = np.random.randint(20, 180, size=(n,))
        fill_s = np.array([generate_randon_color() for _ in range(n)])
        angle_s = np.random.randint(0, 180, size=(n, ))
        opacity_s = np.random.uniform(0.1, 1, size=(n, ))

        svg_code = ''
        for cx, cy, rx, ry, fill, angle, opacity in zip(cx_s, cy_s, rx_s, ry_s, fill_s, angle_s, opacity_s):
            ellipse = Ellipse(cx, cy, rx, ry, angle, fill, opacity)
            svg_code += ellipse.to_svg_line()
        return svg_code

    @staticmethod
    def create_n_image_like_ellipses(n):
        """Create n ellipses gradually in an "image like" manner and returns an SVG code.

        :param n: int. The number of SVG ellipses.
        :return: str. SVG code representing all ellipses.
        """
        cx_s = np.random.randint(20, 180, size=(n,))
        cx_s.sort()
        cx_s = cx_s[::-1]
        cy_s = np.random.randint(20, 180, size=(n,))
        cy_s.sort()
        cy_s = cy_s[::-1]
        rx_s = np.random.randint(20, 180, size=(n,))
        rx_s.sort()
        rx_s = rx_s[::-1]
        ry_s = np.random.randint(20, 180, size=(n,))
        ry_s.sort()
        ry_s = ry_s[::-1]
        fill_s = np.array([generate_randon_color() for _ in range(n)])
        angle_s = np.random.randint(0, 180, size=(n,))
        opacity_s = np.random.uniform(0.1, 1, size=(n,))

        svg_code = ''
        for cx, cy, rx, ry, fill, angle, opacity in zip(cx_s, cy_s, rx_s, ry_s, fill_s, angle_s, opacity_s):
            ellipse = Ellipse(cx, cy, rx, ry, angle, fill, opacity)
            svg_code += ellipse.to_svg_line()
        return svg_code

    @staticmethod
    def warp_shapes_in_svg_header(svg_code):
        svg_code = f'<svg viewBox="-200 -200 600 600" xmlns="http://www.w3.org/2000/svg">\n' + svg_code
        svg_code += f'</svg>'
        return svg_code

    @staticmethod
    def read_svg_file_and_get_ellipses(svg_file_path):
        with open(svg_file_path, 'r') as f:
            data = f.read()
        ellipse_lines = [line for line in data.splitlines() if 'ellipse' in line]
        ellipses_list = []
        for ellipse_line in ellipse_lines:
            ellipses_list.append(Ellipse.from_svg_code(ellipse_line))
        return ellipses_list
