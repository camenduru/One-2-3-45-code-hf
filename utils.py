import os, json
import numpy as np
import base64
# import matplotlib.pyplot as plt
import cv2
from PIL import Image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid
def tensor2img(tensor):
    return Image.fromarray((tensor.detach().cpu().numpy().transpose(1,2,0)*255).astype("uint8"))
def titled_image(img, title="main"):
    # add caption to raw_im
    from PIL import ImageDraw, ImageFont
    titled_image = img.copy()
    draw = ImageDraw.Draw(titled_image)
    import cv2
    font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, size=20)
    draw.text((0, 0), title, fill=(255, 0, 0), font=font)
    # show the drawed image
    return titled_image

def find_image_file(shape_dir):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.svg', '.webp']
    processed_images = ['image_sam.png', 'input_256.png', "input_256_rgba.png"]
    image_files = [file for file in os.listdir(shape_dir) if os.path.splitext(file)[1].lower() in image_extensions and file not in processed_images]
    return image_files[0]

def encode_image(filepath):
    with open(filepath, 'rb') as f:
        image_bytes = f.read()
    encoded = str(base64.b64encode(image_bytes), 'utf-8')
    return "data:image/jpg;base64,"+encoded


# contrast correction, rescale and recenter
def image_preprocess(shape_dir, lower_contrast=True, rescale=True):
    nickname = shape_dir.split("/")[-1]
    img_path = os.path.join(shape_dir, "image_sam.png")
    out_path = os.path.join(shape_dir, "input_256.png")
    out_path_rgba = os.path.join(shape_dir, "input_256_rgba.png")
    image = Image.open(img_path) #[:,90:550]
    # print(image.size)
    image_arr = np.array(image)
    in_w, in_h = image_arr.shape[:2]

    if lower_contrast:
        alpha = 0.8  # Contrast control (1.0-3.0)
        beta =  0   # Brightness control (0-100)
        # Apply the contrast adjustment
        image_arr = cv2.convertScaleAbs(image_arr, alpha=alpha, beta=beta)
        image_arr[image_arr[...,-1]>200, -1] = 255

    ret, mask = cv2.threshold(np.array(image.split()[-1]), 0, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(mask)
    max_size = max(w, h)
    print(nickname, max_size/np.max(image.size))
    ratio = 0.75
    if rescale:
        side_len = int(max_size / ratio)
    else:
        side_len = in_w
    padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
    center = side_len//2
    padded_image[center-h//2:center-h//2+h, center-w//2:center-w//2+w] = image_arr[y:y+h, x:x+w]
    rgba = Image.fromarray(padded_image).resize((256, 256), Image.LANCZOS)
    rgba.save(out_path_rgba)

    rgba_arr = np.array(rgba) / 255.0
    rgb = rgba_arr[...,:3] * rgba_arr[...,-1:] + (1 - rgba_arr[...,-1:])
    rgb = Image.fromarray((rgb * 255).astype(np.uint8))
    rgb.save(out_path)

# contrast correction, rescale and recenter
def image_preprocess_nosave(input_image, lower_contrast=True, rescale=True):

    image_arr = np.array(input_image)
    in_w, in_h = image_arr.shape[:2]

    if lower_contrast:
        alpha = 0.8  # Contrast control (1.0-3.0)
        beta =  0   # Brightness control (0-100)
        # Apply the contrast adjustment
        image_arr = cv2.convertScaleAbs(image_arr, alpha=alpha, beta=beta)
        image_arr[image_arr[...,-1]>200, -1] = 255

    ret, mask = cv2.threshold(np.array(input_image.split()[-1]), 0, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(mask)
    max_size = max(w, h)
    ratio = 0.75
    if rescale:
        side_len = int(max_size / ratio)
    else:
        side_len = in_w
    padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
    center = side_len//2
    padded_image[center-h//2:center-h//2+h, center-w//2:center-w//2+w] = image_arr[y:y+h, x:x+w]
    rgba = Image.fromarray(padded_image).resize((256, 256), Image.LANCZOS)

    rgba_arr = np.array(rgba) / 255.0
    rgb = rgba_arr[...,:3] * rgba_arr[...,-1:] + (1 - rgba_arr[...,-1:])
    return Image.fromarray((rgb * 255).astype(np.uint8))

# pose generation
def calc_pose(phis, thetas, size, radius = 1.2, device='cuda'):
    import torch
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)
    thetas = torch.FloatTensor(thetas).to(device)
    phis = torch.FloatTensor(phis).to(device)
    
    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        -radius * torch.cos(thetas) * torch.sin(phis),
        radius * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = normalize(centers).squeeze(0)
    up_vector = torch.FloatTensor([0, 0, 1]).to(device).unsqueeze(0).repeat(size, 1) 
    right_vector = normalize(torch.cross(up_vector, forward_vector, dim=-1))      
    if right_vector.pow(2).sum() < 0.01:
        right_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)  
    up_vector = normalize(torch.cross(forward_vector, right_vector, dim=-1))     

    poses = torch.eye(4, dtype=torch.float, device=device)[:3].unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers 
    return poses

def get_poses(init_elev):
    mid = init_elev
    deg = 10
    if init_elev <= 75:
        low = init_elev + 30
        # e.g. 30, 60, 20, 40, 30, 30, 50, 70, 50, 50
        
        elevations = np.radians([mid]*4 + [low]*4 + [mid-deg,mid+deg,mid,mid]*4 + [low-deg,low+deg,low,low]*4)
        img_ids = [f"{num}.png" for num in range(8)] + [f"{num}_{view_num}.png" for num in range(8) for view_num in range(4)]
    else:
        
        high = init_elev - 30
        elevations = np.radians([mid]*4 + [high]*4 + [mid-deg,mid+deg,mid,mid]*4 + [high-deg,high+deg,high,high]*4)
        img_ids = [f"{num}.png" for num in list(range(4)) + list(range(8,12))]  + \
                [f"{num}_{view_num}.png" for num in list(range(4)) + list(range(8,12)) for view_num in range(4)]
    overlook_theta = [30+x*90 for x in range(4)]
    eyelevel_theta = [60+x*90 for x in range(4)]
    source_theta_delta = [0, 0, -deg, deg]
    azimuths = np.radians(overlook_theta + eyelevel_theta + \
                            [view_theta + source for view_theta in overlook_theta for source in source_theta_delta] + \
                            [view_theta + source for view_theta in eyelevel_theta for source in source_theta_delta])
    return img_ids, calc_pose(elevations, azimuths, len(azimuths)).cpu().numpy()

# eval_path = "/objaverse-processed/zero12345_img/%s" % dataset
# for shape in os.listdir(eval_path):
#     shape_dir = os.path.join(eval_path, shape)
def gen_poses(shape_dir, pose_est):
    img_ids, input_poses = get_poses(pose_est)
        
    out_dict = {}
    focal = 560/2; h = w = 256
    out_dict['intrinsics'] = [[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]]
    out_dict['near_far'] = [1.2-0.7, 1.2+0.7]
    out_dict['c2ws'] = {}
    for view_id, img_id in enumerate(img_ids):
        pose = input_poses[view_id]
        pose = pose.tolist()
        pose = [pose[0], pose[1], pose[2], [0, 0, 0, 1]]
        out_dict['c2ws'][img_id] = pose
    json_path = os.path.join(shape_dir, 'pose.json')
    with open(json_path, 'w') as f:
        json.dump(out_dict, f, indent=4)
    # break
