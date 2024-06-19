# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import math
import numpy as np
import torch
from monai import transforms, data

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


def get_loader(args):
    data_dir = args.data_dir
    datalist_json = args.json_list
    train_files, validation_files = datafold_read(
        datalist=datalist_json, basedir=data_dir, fold=args.fold)

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["label","image"]),
            # transforms.ConvertToMultiChannelBasedOnBratsClassesD(keys=["label"]),
            # transforms.CropForegroundd(keys=["image", "label"], source_key="image",
            #                            k_divisible=[args.roi_x,args.roi_y,args.roi_z]),
            # 这里数据形状是？
            transforms.RandSpatialCropd(keys=["image", "label"],
                                        roi_size=[args.roi_x,args.roi_y,args.roi_z],random_size=False),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            # transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            # transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            transforms.ToTensord(keys=["image", "label"],),
            transforms.Transposed(keys=['image', 'label'], indices=(0, 3, 1, 2))
            # transforms.SqueezeDimd(keys=["label"]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["label","image"]),
            # transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
            transforms.Transposed(keys=['image', 'label'], indices=(0, 3, 1, 2))
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            # transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            # transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
            transforms.Transposed(keys=['image', 'label'], indices=(0, 3, 1, 2))
        ]
    )

    train_ds = data.Dataset(data=train_files, transform=train_transform)
    #train_sampler = Sampler(train_ds) if args.distributed else None
    train_sampler=None

    # 返回DataLoader
    train_loader = data.DataLoader(train_ds,
                                   batch_size=args.batch_size,
                                   shuffle=(train_sampler is None),
                                   num_workers=args.workers,
                                   sampler=train_sampler,
                                   pin_memory=True,
                                   )

    val_ds = data.Dataset(data=validation_files, transform=val_transform)

    val_sampler = None
    val_loader = data.DataLoader(val_ds,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=args.workers,
                                 sampler=val_sampler,
                                 pin_memory=True,
                                 )

    loader = [train_loader, val_loader]

    return loader
