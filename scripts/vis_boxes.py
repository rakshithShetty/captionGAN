import os
import numpy as np
import cv2
import argparse
import json
import numpy.random
from numpy.random import randint as randi
from collections import defaultdict


def tileImages(frames,idcs,cent,grid=None,final_size=[150.0,800.0],brd=10,const_bright=0):
    if grid == None:
        grid = [1,len(idcs)]
    widthO  = frames[0].shape[0]+2*brd
    heightO = frames[0].shape[1]+2*brd
    width  = widthO * grid[0];
    height = heightO * grid[1];
    accum_frame = np.zeros((width,height,frames[0].shape[2]),dtype=frames[0].dtype)
    k = 0;
    BLUE = [255,0,0]
    SPL = [0,0,255]
    for j in xrange(grid[1]):
        for i in xrange(grid[0]):
            col = SPL if idcs[k] == cent else BLUE
            accum_frame[i*widthO:(i+1)*widthO,j*heightO:(j+1)*heightO,:] = cv2.copyMakeBorder(frames[idcs[k]]+const_bright,brd,brd,brd,brd,cv2.BORDER_CONSTANT,value=col)            
            k = k+1
    return cv2.resize(accum_frame,None,fx=final_size[1]/height,fy=final_size[0]/width, interpolation = cv2.INTER_CUBIC)

vidId = '210540520'


if __name__ == "__main__":
  
  #cap = cv2.VideoCapture('/triton/ics/project/imagedb/picsom/databases/lsmdc2015/objects/2/3035/230350439.avi')
  #2/3010/230100536.avi
  #1/1051/110510033.avi
  #1/4076/140760125.avi
  #1/1027/110270280.avi
  #1/1026/110260059.avi 
  #1/1051/110510496.avi 
  basepath = '/triton/ics/project/imagedb/picsom/databases/COCO/download/'

  boxAnn = json.load(open(os.path.join(basepath,'annotations/instances_train2014.json'),'r'))

  annById = defaultdict(list)
  for ann in boxAnn['annotations']:
    annById[ann['image_id']].append(ann)
  
  i = 0
  impath = os.path.join(basepath,'images/train2014',boxAnn['images'][i]['file_name'])
  img = cv2.imread(impath,1)
  imgBox = img
  for ann in annById[boxAnn['images'][i]['id']]:
    imgBox = cv2.rectangle(imgBox, (int(ann['bbox'][0]), int(ann['bbox'][1])), 
        (int(ann['bbox'][0] + ann['bbox'][2]), int(ann['bbox'][1] + ann['bbox'][3])),(randi(0,256),randi(0,256),randi(0,256)),3)
  
  cv2.imshow('imBox',imgBox)
  if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()


    

annKeys = set(annById.keys())
for i,img in enumerate(dataset['images']):
    #dataset['images'][i]['bboxAnn'] = []
    if img['split'] != 'train':
        sz = (imgByimgId[img['cocoid']]['width'], imgByimgId[img['cocoid']]['height'])
        dataset['images'][i]['imgSize'] = sz 
        if img['cocoid'] in annKeys:
            for ann in annById[img['cocoid']]:
                x_c = (ann['bbox'][0] + ann['bbox'][2]/2)/sz[0]
                y_c = (ann['bbox'][1] + ann['bbox'][3]/2)/sz[1]
                w = ann['bbox'][2]/sz[0]
                h = ann['bbox'][3]/sz[1]
                dataset['images'][i]['bboxAnn'].append({'bbox':[x_c,y_c,w,h], 'cid':ann['category_id']})

