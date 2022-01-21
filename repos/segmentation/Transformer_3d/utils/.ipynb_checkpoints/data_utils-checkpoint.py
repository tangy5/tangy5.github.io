# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import nibabel as nb
import numpy as np
from monai.transforms import (
    Activations,
    AsChannelFirstd,
    AsDiscrete,
    AddChanneld,
    Compose,
    MapTransform,
    CropForegroundd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    RandRotate90d,
    ToTensord,
    RandAffined,
    RandScaleIntensityd,
    RandAdjustContrastd,
    Resized,
)

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    load_decathlon_properties,
    partition_dataset,
    select_cross_validation_folds,
    SmartCacheDataset,
    Dataset,
    decollate_batch,
    DistributedSampler,
)
from monai.data import CacheDataset,SmartCacheDataset, DataLoader, Dataset, partition_dataset
import torch.distributed as dist

def get_loader(args):
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k = 3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250,b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            ToTensord(keys=["image", "label"]),
        ]
    )

    data_dir ='/dataset/dataset0' # spleen
    
    split_JSON = '/dataset_0.json'
    jsonlist = data_dir + split_JSON

    datalist = load_decathlon_datalist(jsonlist, True, "training", base_dir=data_dir)
    val_files = load_decathlon_datalist(jsonlist, True, "validation", base_dir=data_dir)
    

    
#     train_ds = CacheDataset(data=datalist, transform=train_transforms, cache_num=24, num_workers=8)
#     train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True)
#     train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=8, sampler=train_sampler)
    

#     val_ds = Dataset(data=val_files, transform=val_transforms)
#     val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, shuffle=False)
    
    
    
    
    train_files = partition_dataset(data=datalist, shuffle=True, num_partitions=dist.get_world_size(), even_divisible=True)[dist.get_rank()]
    train_ds = SmartCacheDataset(data=train_files, transform=train_transforms, cache_num=6, replace_rate=1.0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=8)

#     val_files = partition_dataset(data=val_files, shuffle=True, num_partitions=dist.get_world_size(), even_divisible=True)[dist.get_rank()]
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_sampler = DistributedSampler(dataset=val_ds, even_divisible=True, shuffle=False)

    val_loader = DataLoader(val_ds, batch_size=1, num_workers=8, shuffle=False, sampler=val_sampler)

    return train_loader, val_loader
