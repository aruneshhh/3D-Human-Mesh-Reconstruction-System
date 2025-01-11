import numpy as np
import torch
import os
import cv2
from tqdm import tqdm
from glob import glob
from lib.core.config import parse_args
from lib import get_model
from lib.datasets.detect_dataset import DetectDataset
from lib.models.smpl import SMPL
from lib.yolo import Yolov7

# Function to save vertices and faces as a .obj file
def save_obj(vertices, faces, output_path):
    with open(output_path, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

# Yolo model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo = Yolov7(device=DEVICE, weights='data/pretrain/yolov7-e6e.pt', imgsz=1281)

# ReFit
args = ['--cfg', 'configs/config.yaml']
cfg = parse_args(args)
cfg.DEVICE = DEVICE

model = get_model(cfg).to(DEVICE)
checkpoint = 'data/pretrain/refit_all/checkpoint_best.pth.tar'
state_dict = torch.load(checkpoint, map_location=cfg.DEVICE)
_ = model.load_state_dict(state_dict['model'], strict=False)
_ = model.eval()
print('Loaded checkpoint:', checkpoint)

# SMPL model (used for faces)
smpl = SMPL()
faces = smpl.faces  # SMPL model's face topology

# Example image processing
imgfiles = sorted(glob('data/examples/*'))
for imgname in tqdm(imgfiles):
    img = cv2.imread(imgname)[:, :, ::-1].copy()

    ### --- Detection ---
    with torch.no_grad():
        boxes = yolo(img, conf=0.50, iou=0.45)
        boxes = boxes.cpu().numpy()
        
    db = DetectDataset(img, boxes)
    dataloader = torch.utils.data.DataLoader(db, batch_size=8, shuffle=False, num_workers=0)

    ### --- ReFit ---
    vert_all = []
    for batch in dataloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items() if type(v) == torch.Tensor}
        with torch.no_grad():
            out, preds = model(batch, iters=5)
            s_out = model.smpl.query(out)
            vertices = s_out.vertices

        vert_full = vertices + out['trans_full']
        vert_all.append(vert_full)
        
    vert_all = torch.cat(vert_all).cpu().numpy()

    # Save each detected person's mesh as a separate .obj file
    for i, vertices in enumerate(vert_all):
        output_name = os.path.basename(imgname).replace('.png', f'_person_{i}.obj').replace('.jpg', f'_person_{i}.obj')
        output_path = os.path.join('output_meshes', output_name)

        os.makedirs('output_meshes', exist_ok=True)  # Ensure output directory exists
        save_obj(vertices, faces, output_path)
        print(f"Saved 3D mesh to {output_path}")
