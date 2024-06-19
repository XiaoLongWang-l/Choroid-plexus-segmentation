import nibabel as nib
import numpy as np
from monai import transforms, data
import os
import json
import argparse
from monai.inferers import SlidingWindowInferer
import torch
import torch.nn.parallel
import torch.utils.data.distributed

from models.UXNet_3D.network_backbone import UXNET


parser = argparse.ArgumentParser(description='Swin UNETR segmentation pipeline for BRATS Challenge')

parser.add_argument('--fold', default=0, type=int, help='data fold')
parser.add_argument('--pretrain_model_path', default='./model.pt', type=str, help='pretrained model name')
parser.add_argument('--load_pretrain', action="store_true", help='pretrained model name')
parser.add_argument('--json_list', default='./train_valid.json', type=str, help='dataset json file')
parser.add_argument('--max_epochs', default=300, type=int, help='max number of training epochs')
parser.add_argument('--batch_size', default=2, type=int, help='number of batch size')
parser.add_argument('--sw_batch_size', default=1, type=int, help='number of sliding window batch size')

parser.add_argument('--roi_x', default=128, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=128, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=128, type=int, help='roi size in z direction')

parser.add_argument('--infer_overlap', default=0.25, type=float, help='sliding window inference overlap')

def datafold_read(datalist,
                  basedir,
                  fold=0,
                  key='training'
                  ):

    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr=[]
    val=[]
    for d in json_data:
        if 'fold' in d and d['fold'] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val

args = parser.parse_args()
inf_size = [args.roi_z, args.roi_x, args.roi_y]

window_infer = SlidingWindowInferer(roi_size=inf_size,
                                        sw_batch_size=args.sw_batch_size,
                                        overlap=args.infer_overlap,
                                       )
val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.AddChanneld(keys=["image"]),
            # transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image"]),
            transforms.Transposed(keys=['image'], indices=(0, 3, 1, 2))
        ]
    )
device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
# features = [32, 64, 128, 256]
# features = [48, 96, 192, 384]
# model = UNet(num_classes=1, in_channels=1, fea=features)  
model = UXNET(in_chans=1, out_chans=1, depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384], drop_path_rate=0,
                  layer_scale_init_value=1e-6, spatial_dims=3, )
model.to(device)
checkpoint_path = "logs/CT_0919/UX2/model_final_best.pt"
checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint["state_dict"]

model.load_state_dict(checkpoint['state_dict'])
model.eval()
train_files, validation_files = datafold_read(
    datalist="NMOSD_T1.json", basedir='./', fold=0)
val_ds = data.Dataset(data=validation_files, transform=val_transform)
val_loader = data.DataLoader(val_ds,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=True,
                             )




for idx, batch in enumerate(val_loader):
    file_path = batch["image_meta_dict"]["filename_or_obj"][0]
    filename_without_extension = os.path.splitext(os.path.basename(file_path))[0]

    val_image = batch["image"].to(device)
    with torch.no_grad():
        # Perform sliding window inference
        pred_label = window_infer(val_image, model)

        # Threshold the predictions
        pred_label = (pred_label > 0.5).float().permute(0, 1, 3, 4, 2)


        # Save the predicted label as NIfTI
        pred_label_nifti = nib.Nifti1Image(pred_label.squeeze().cpu().numpy(), np.eye(4))
        # output_filename = f"predicted_label_{filename_without_extension}.nii"
        output_filename = f"{filename_without_extension}.nii"
        if not os.path.exists("masks"):
            os.makedirs("masks")
        output_filepath = os.path.join("masks", output_filename)
        nib.save(pred_label_nifti, output_filepath)