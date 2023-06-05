# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 16:35:11 2023

@author: Gavin
"""

import os, blosc, cv2

import pandas as pd
import numpy as np

from radio import CTImagesMaskedBatch as CTIMB
from radio.batchflow import FilesIndex, Dataset, Pipeline

batch_size = 1
num_workers = 1

ct_min = -1024
ct_max = 3072

path_to_scans = 'B:/Datasets/LUNA16-custom/data/*.mhd'
path_to_dump = 'B:/Datasets/LUNA16-custom/preprocessed'

nodules = pd.read_csv('B:/Datasets/LUNA16-custom/annotations.csv')

ctx = FilesIndex(path=path_to_scans, no_ext=True)
ctset = Dataset(index=ctx, batch_class=CTIMB)

preprocessing = Pipeline()
preprocessing = preprocessing.load(fmt='raw').fetch_nodules_info(nodules=nodules).create_mask()
    
runner = ctset >> preprocessing

#if not os.path.isdir(f'{path_to_dump}/imgs'):
#    os.mkdir(f'{path_to_dump}/imgs')
#    
#if not os.path.isdir(f'{path_to_dump}/masks'):
#    os.mkdir(f'{path_to_dump}/masks')

nodules_count = 0
for batch in runner.gen_batch(batch_size=batch_size, shuffle=True):
    img = batch._data[0]
    mask = batch._data[1]
    
    filter_idxs = mask.reshape(mask.shape[0], -1).sum(axis=1) > 0
    
    start_idx, = np.where(np.diff(np.concatenate(([0], filter_idxs, [0]))) == 1)
    end_idx, = np.where(np.diff(np.concatenate(([0], filter_idxs, [0]))) == -1)

    filter_idx_regions = [np.zeros_like(filter_idxs) for _ in range(len(start_idx))]
    
    for i, (start, end) in enumerate(zip(start_idx, end_idx)):
            filter_idx_regions[i][start:end + 1] = filter_idxs[start:end + 1]

    img_splits, mask_splits = [], []
    
    for i, region_idxs in enumerate(filter_idx_regions):
        img_splits.append(img[region_idxs, :, :])
        mask_splits.append(mask[region_idxs, :, :])
    
    for img_split, mask_split in zip(img_splits, mask_splits):
        #for i, (img_slide, mask_slide) in enumerate(zip(img_split, mask_split)):
        #    np.save(f'{path_to_dump}/imgs/nodule_{nodules_count}_slide_{i}.npy', img_slide)
        #    np.save(f'{path_to_dump}/masks/nodule_{nodules_count}_slide_{i}.npy', mask_slide)
        img_split = np.rollaxis(img_split, 0, 3)
        img_split = img_split.astype(np.float32)
        img_split[img_split > ct_max] = ct_max
        img_split[img_split < ct_min] = ct_min
        img_split += -ct_min
        img_split /= (ct_max + -ct_min)
        
        mask_split = np.rollaxis(mask_split, 0, 3)
        #mask_split[mask_split > 250] = 1 # In case using 255 instead of 1
        #mask_split[mask_split > 4.5] = 0 # Trachea = 5
        #mask_split[mask_split >= 1] = 1 # Left lung = 3, Right lung = 4
        #mask_split[mask_split != 1] = 0 # Non-Lung/Background
        mask_split = mask_split.astype(np.uint8)
        
        np.savez_compressed(f'{path_to_dump}/nodule_{nodules_count}.npz', img=img_split, mask=mask_split)
        
        nodules_count += 1