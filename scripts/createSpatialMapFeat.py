import json
import numpy as np
import argparse
import os


if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  #parser.add_argument(dest='feat_fname',type=str, default='spatMapFeat.bin', help='feature file name')

  parser.add_argument('--grid', dest='grid', type=int, default=8, help='grid size, gxg')
  parser.add_argument('--nObj', dest='nObj', type=int, default=-1, help='number of objects in the Map')
  parser.add_argument('--dataset', dest='dataset', type=str, default='data/coco/datasetBoxAnn.json', help='dataset json File')
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict


  gridSz = params['grid']
  data = json.load(open(params['dataset'],'r'))
  spatMap = np.zeros((len(data['images']),len(data['categories']), gridSz*gridSz))

  gridRectCord = np.zeros((gridSz*gridSz,4)) 

  gridA = (1.0/(gridSz*gridSz))

  catToIdx = {}
  for i,cat in enumerate(data['categories']):
    catToIdx[cat['id']] = i
    

  for i in xrange(gridSz):
    for j in xrange(gridSz):
        gridRectCord[i*gridSz+j][0] = (1.0/gridSz)*i
        gridRectCord[i*gridSz+j][1] = (1.0/gridSz)*j
        gridRectCord[i*gridSz+j][2] = (1.0/gridSz)*(i+1)
        gridRectCord[i*gridSz+j][3] = (1.0/gridSz)*(j+1)

  for img in data['images']:
    for ann in img['bboxAnn']:
        bC = [ann['bbox'][0] - ann['bbox'][2]/2,ann['bbox'][1] - ann['bbox'][3]/2,ann['bbox'][0] + ann['bbox'][2]/2,ann['bbox'][1] + ann['bbox'][3]/2]
        bA = ann['bbox'][2]*ann['bbox'][3]
        cid = catToIdx[ann['cid']]

        for i in xrange(gridRectCord.shape[0]):
            sI = max(0, min(bC[2],gridRectCord[i][2]) - max(bC[0],gridRectCord[i][0]))*max(0, min(bC[3],gridRectCord[i][3]) - max(bC[1],gridRectCord[i][1]))
            sU = gridA + bA - sI
            assert(sU>sI)
            spatMap[img['imgid']][cid][i] += sI/sU

  spatMap.resize((spatMap.shape[0],spatMap.shape[1]*spatMap.shape[2]))
  spatMapSmall = spatMap.astype(np.float16)
  np.save(open('data/coco/spatMapFeat_IOU_8x8_all80_corrected.npy','wb'),spatMapSmall)
  #pickle.dump(spatMap.astype(np.float32),open('data/coco/spatMapFeat_IOU_8x8_all80.bin','wb'))
    
 

 ####
#meanCoOrdvec = np.zeros((len(catToIdx),4))
#catRefCounts = np.zeros((len(catToIdx),))
#for img in data['images']:
#    if img['split']=='train':
#        for ann in img['bboxAnn']:
#            meanCoOrdvec[catToIdx[ann['cid']],:] += np.asarray(ann['bbox'])
#            catRefCounts[catToIdx[ann['cid']]] += 1
#
#for j in xrange(meanCoOrdvec.shape[0]):
#    mC = meanCoOrdvec[j,:]
#    bC = [mC[0] - mC[2]/2,mC[1] - mC[3]/2,mC[0] + mC[2]/2,mC[1] +mC[3]/2]
#    bA = mC[2]*mC[3]
#    cid = catToIdx[ann['cid']]
#    for i in xrange(gridRectCord.shape[0]):
#        sI = max(0, min(bC[2],gridRectCord[i][2]) - max(bC[0],gridRectCord[i][0]))*max(0, min(bC[3],gridRectCord[i][3]) - max(bC[1],gridRectCord[i][1]))
#        sU = gridA + bA - sI
#        assert(sU>sI)
#        meanSpatMap[cid,i] = sI/sU
