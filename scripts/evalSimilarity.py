import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import random
import scipy.io
import codecs
from collections import defaultdict
import sys

#%matplotlib inline
if len(sys.argv) == 1:
    target = '135235570_5698072cd4.jpg'
elif len(sys.argv) == 2:
    target = sys.argv[1]
else:
    targetId = sys.argv[1]

def topN(a,N):
	return np.argsort(a)[::-1][:N]

dataset_root = '../data/flickr8k/'
datafilePath = os.path.join(dataset_root, 'dataset.json')
jsondataset = json.load(open(datafilePath, 'r'))
image_root = os.path.join(dataset_root, 'imgs')

features_path = os.path.join(dataset_root, 'vgg_feats.mat')
print 'BasicDataProvider: reading %s' % (features_path, )
features_struct = scipy.io.loadmat(features_path)
features = features_struct['feats']
cnt = 0
if len(sys.argv) == 2:
    for img in jsondataset['images']:
        if img['filename'] == target:
            targetId = cnt
            break
        cnt += 1

print targetId 
targFeat = features[:,targetId]
scr = targFeat.T.dot(features) 
N = 8
topIdx = topN(scr,N)
figIdx = 1
print topIdx
for n in topIdx:
    fname = os.path.join(image_root,jsondataset['images'][n]['filename'])
    im = plt.imread(fname)
    plt.subplot(4,2,figIdx)
    plt.imshow(np.flipud(im))
    figIdx += 1
    
plt.show()
#labels_df.sort('synset_id')
#predictions_df = pd.DataFrame(np.vstack(df.prediction.values), columns=labels_df['name'])
##print(predictions_df.iloc[0])
#
#
##plt.gray()
##plt.matshow(predictions_df.values)
##plt.xlabel('Classes')
##plt.ylabel('Windows')
##plt.show()
#
#N = 4
#max_s = predictions_df.max(0)
#print type(max_s)
#topIdx = topN(max_s.tolist(),N)
#print max_s[topIdx]
#
#
##print predictions_df[topIdx].argmax()
### Find, print, and display the top detections: person and bicycle.
##
#### Show top predictions for top detection.
##f = pd.Series(df['prediction'].iloc[i], index=labels_df['name'])
##print('Top detection:')
##print(f.order(ascending=False)[:5])
##print('')
##
### Show top predictions for second-best detection.
##f = pd.Series(df['prediction'].iloc[j], index=labels_df['name'])
##print('Second-best detection:')
##print(f.order(ascending=False)[:5])
#
## Show top detection in red, second-best top detection in blue.
##i = predictions_df[max_s[topIdx[0]]].argmax()
#fin = open('_temp2/det_input.txt')
#inputs = [_.strip() for _ in fin.readlines()]
#im = plt.imread(inputs[0])
#width = N;
##colorArray = 
#for n in topIdx:
#	i = predictions_df.ix[::][max_s[n:n+1].index].idxmax().tolist()
#	plt.imshow(np.flipud(im))
#	currentAxis = plt.gca()
#	det = df.iloc[i]
#	coords = (det['xmin'], det['ymin']), det['xmax'] - det['xmin'], det['ymax'] - det['ymin']
#	currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='r', linewidth=width))
#	currentAxis.text(det['xmin'],det['ymin'],max_s[n:n+1].index[0])
#	width -= 1
#	
#
##det = df.iloc[j]
##
##
#plt.show()
