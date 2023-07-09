import tqdm
import imageio
import json
import os.path as osp
import os

from oee.utils import plt_utils
from oee.utils.elev_est_api import elev_est_api


def visualize(img_paths, elev):
    imgs = [imageio.imread_v2(img_path) for img_path in img_paths]
    plt_utils.image_grid(imgs, 2, 2, label=f"elev={elev}")


def estimate_elev(root_dir):
    # root_dir = "/home/linghao/Datasets/objaverse-processed/zero12345_img/wild"
    # dataset = "supp_fail"
    # root_dir = "/home/chao/chao/OpenComplete/zero123/zero123/gradio_tmp/"
    # obj_names = sorted(os.listdir(root_dir))
    # results = {}
    # for obj_name in tqdm.tqdm(obj_names):
    img_dir = osp.join(root_dir, "stage2_8")
    img_paths = []
    for i in range(4):
        img_paths.append(f"{img_dir}/0_{i}.png")
    elev = elev_est_api(img_paths)
    # visualize(img_paths, elev)
    # results[obj_name] = elev
    # json.dump(results, open(osp.join(root_dir, f"../{dataset}_elev.json"), "w"), indent=4)
    return elev


# if __name__ == '__main__':
#     main()
