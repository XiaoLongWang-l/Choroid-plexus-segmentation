import os
import json

def datafold_read(datalist = './train_valid.json',
                  basedir = 'E:/',
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

train_files, validation_files = datafold_read(fold=0)

if __name__ == '__main__':
    from monai import transforms, data

    trans = transforms.Compose(
        [

            transforms.LoadImaged(keys=["image", "label"]),
            # (1, 4, 240, 240, 155)
            # (1, 240, 240, 155)

            # (1, 512, 512, 28)
            # (1, 512, 512, 28)

            transforms.AddChanneld(keys=['image','label']),

            # (1, 1, 512, 512, 28)
            # (1, 1, 512, 512, 28)

            # transforms.ConvertToMultiChannelBasedOnBratsClassesD(keys=["label"]),
            # (1, 4, 240, 240, 155)
            # (1, 3, 240, 240, 155)

            # transforms.CropForegroundd(keys=["image", "label"], source_key="image",
            #                            k_divisible=[128, 128, 128]),

            # (1, 4, 256, 256, 256)
            # (1, 3, 256, 256, 256)

            # transforms.CropForegroundd(keys=["image", "label"], source_key="image",
            #                            k_divisible=[256, 256, 16]),

            # (1, 1, 512, 512, 32)---为啥会扩增这么多？
            # (1, 1, 512, 512, 32)

            # transforms.RandSpatialCropd(keys=["image", "label"],
            #                             roi_size=[128, 128, 128], random_size=False),

            # (1, 4, 128, 128, 128)
            # (1, 3, 128, 128, 128)

            transforms.RandSpatialCropd(keys=["image", "label"],
                                        roi_size=[256, 256, 16], random_size=False),
            # (1, 1, 256, 256, 16)
            # (1, 1, 256, 256, 16)

            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            # transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            transforms.ToTensord(keys=["image", "label"], ),
            transforms.Transposed(keys=['image','label'],indices=(0,3,1,2))
            # (1, 4, 128, 128, 128)
            # (1, 3, 128, 128, 128)

            # (1, 1, 256, 256, 16)
            # (1, 1, 256, 256, 16)

            # # transforms.SqueezeDimd(keys=["label"]),
        ]
    )

    train_ds = data.Dataset(data=train_files, transform= trans)
    train_loader = data.DataLoader(train_ds,
                                   batch_size=1,
                                   )
    for idx, batch in enumerate(train_loader):

        image = batch['image']
        label = batch['label']
        print(image.shape)
        print(label.shape)
        break

