import imageio
import numpy as np

from oee.utils import plt_utils
from oee.utils.elev_est_api import elev_est_api
import argparse


def visualize(img_paths, elev):
    imgs = [imageio.imread_v2(img_path) for img_path in img_paths]
    plt_utils.image_grid(imgs, 2, 2, label=f"elev={elev}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_paths", type=str, nargs=4, help="image paths",
                        default=["assets/example_data/0_0.png",
                                 "assets/example_data/0_1.png",
                                 "assets/example_data/0_2.png",
                                 "assets/example_data/0_3.png"])
    parser.add_argument("--min_elev", type=float, default=30, help="min elevation")
    parser.add_argument("--max_elev", type=float, default=150, help="max elevation")
    parser.add_argument("--dbg", default=False, action="store_true", help="debug mode")
    parser.add_argument("--K_path", type=str, default=None, help="path to K")
    args = parser.parse_args()

    if args.K_path is not None:
        K = np.loadtxt(args.K_path)
    else:
        K = None

    elev = elev_est_api(args.img_paths, args.min_elev, args.max_elev, K, args.dbg)

    visualize(args.img_paths, elev)


if __name__ == '__main__':
    main()
