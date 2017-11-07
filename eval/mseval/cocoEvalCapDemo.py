
# coding: utf-8

# In[1]:

# get_ipython().magic(u'matplotlib inline')
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
import skimage.io as io
import pylab
from sys import argv
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

resFileSys = ''
dataTypeSys = ''
dataDirSys = ''

# In[2]:
if len(argv) > 1:
    resFileSys = argv[1]
    if len(argv) > 2:
        dataTypeSys = argv[2] 
    if len(argv) > 3:
        dataDirSys = argv[3]


# set up file names and pathes
dataDir='coco' if dataDirSys == '' else dataDirSys
if dataDir == 'coco':
    imgidStr = 'image_id'
    imgOrVid = 'images'
elif dataDir == 'lsmdc':
    imgidStr = 'video_id'
    imgOrVid = 'images'
else:
    imgidStr = 'video_id'
    imgOrVid = 'videos'

dataType='val2014' if dataTypeSys == '' else dataTypeSys
annFile='./annotations/%s/captions_%s.json'%(dataDir,dataType)
subtypes=['evalImgs', 'eval']
resFile = resFile if resFileSys == '' else resFileSys
algName = resFile.split('_')[-2] 
[evalImgsFile, evalFile]= ['%s/results/captions_%s_%s_%s.json'%(dataDir,dataType,algName,subtype) for subtype in subtypes]

outfile = 'scores/%s/%s'%(dataDir, algName + '.result') 

print imgidStr,imgOrVid
print annFile, evalImgsFile
print resFile
# In[3]:

# create coco object and cocoRes object
coco = COCO(annFile, imgidStr, imgOrVid)
cocoRes = coco.loadRes(resFile)


# In[4]:

# create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes, imgidStr)

# evaluate on a subset of images by setting
# cocoEval.params['image_id'] = cocoRes.getImgIds()
# please remove this line when evaluating the full validation set
cocoEval.params[imgidStr] = cocoRes.getImgIds()

# evaluate results
cocoEval.evaluate()


# In[5]:

fout = open(outfile,'w')
# print output evaluation scores
for metric, score in cocoEval.eval.items():
    print '%s: %.3f'%(metric, score)
    fout.write('%s: %.3f \n'%(metric, score))
fout.close()

# In[6]:

# demo how to use evalImgs to retrieve low score result
#evals = [eva for eva in cocoEval.evalImgs if eva['CIDEr']<30]
#print 'ground truth captions'
#imgId = evals[0]['image_id']
#annIds = coco.getAnnIds(imgIds=imgId)
#anns = coco.loadAnns(annIds)
#coco.showAnns(anns)
#
#print '\n'
#print 'generated caption (CIDEr score %0.1f)'%(evals[0]['CIDEr'])
#annIds = cocoRes.getAnnIds(imgIds=imgId)
#anns = cocoRes.loadAnns(annIds)
#coco.showAnns(anns)
#
#img = coco.loadImgs(imgId)[0]
#I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
#plt.imshow(I)
#plt.axis('off')
#plt.show()
#
#
## In[7]:
#
## plot score histogram
#ciderScores = [eva['CIDEr'] for eva in cocoEval.evalImgs]
#plt.hist(ciderScores)
#plt.title('Histogram of CIDEr Scores', fontsize=20)
#plt.xlabel('CIDEr score', fontsize=20)
#plt.ylabel('result counts', fontsize=20)
#plt.show()
#
#
## In[8]:
#
# save evaluation results to ./results folder
json.dump(cocoEval.evalImgs, open(evalImgsFile, 'w'))
json.dump(cocoEval.eval,     open(evalFile, 'w'))

