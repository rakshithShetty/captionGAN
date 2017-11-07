import json
import os
import random
import scipy.io
import codecs
import numpy as np
from collections import defaultdict


dataset = 'coco'
data_file = 'dataset_newfeat.json'
src_file_Tr = '/triton/ics/project/imagedb/picsom/databases/COCO/download/annotations/instances_train2014.json'
src_file_val = '/triton/ics/project/imagedb/picsom/databases/COCO/download/annotations/sentences_val2014.json'


print 'Initializing data provider for dataset %s...' % (dataset, )

# !assumptions on folder structure
dataset_root = os.path.join('data', dataset)

# load the dataset into memory
dataset_path = os.path.join(dataset_root, data_file)
print 'BasicDataProvider: reading %s' % (dataset_path, )
dB = json.load(open(dataset_path, 'r'))
srcdB_train = json.load(open(src_file_Tr, 'r'))
srcdB_val = json.load(open(src_file_val, 'r'))

trn_idx = 0
val_idx = 0
val_idx_offset = len(srcdB_train['images'])

# group images by their train/val/test split into a dictionary -> list structure
for img in dB['images']:
    if img['split'] == 'train':
       assert img['cocoid'] == srcdB_train['images'][trn_idx]['id'], 'Ids dont match, training'
       img['imgid'] =  trn_idx 
       trn_idx += 1
    else:
       assert img['cocoid'] == srcdB_val['images'][val_idx]['id'], 'Ids dont match, training'
       img['imgid'] = val_idx + val_idx_offset
       val_idx += 1
        
print 'Done with %d %d!! Now writing back dataset ' % (trn_idx, val_idx)
json.dump(dB,open(dataset_path, 'w'))



