from matplotlib.backends.backend_pdf import PdfPages
import json
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import os
from imagernn.data_provider import readFeaturesFromFile
import cv2

def visMaps(inpMap, grdsize, brd=1):
    shp = inpMap.shape
    strX = (shp[0]+brd)
    strY = (shp[1]+brd)
    mX = shp[0]
    mY = shp[1]
    resArry = np.zeros((grdsize[0]*(strX),grdsize[1]*(strY)))
    #first to draw the gridMap
    for i in xrange(1,grdsize[0]):
        resArry[i*strX-brd:i*strX,:] = np.ones((brd,resArry.shape[1]))
    
    for j in xrange(1,grdsize[1]):
        resArry[:,j*strY-brd:j*strY] = np.ones((resArry.shape[0],brd))

    for i in xrange(shp[2]):
        x = i//grdsize[1]
        y = i%grdsize[1]
        # First put the maps in
        resArry[x*strX: x*strX + mX, y*strY:y*strY + mY] = inpMap[:,:,i]
        #print i

    return resArry

####################### PARAMS ##############################################################
ncat = 80;
gridSz = (4,4)
cmap = cm.terrain_r
pGridSz = (8,10)
n_samples = 10
outfile ='plots/spatMapIouVsGauss.pdf' 
basepath = '/projects/databases/coco/download/images/'

####################### CODE ################################################################
data = json.load(open('data/coco/datasetBoxAnn.json','r'))
#spatMapFeat = np.load(open('data/coco/spatMapFeat_IOU_8x8_all80.npy','rb')).T
#spatMapFeat = np.load(open('data/coco/spatMapFeat_IOU_8x8_all80_corrected.npy','rb')).T
spatMapFeat,_ = readFeaturesFromFile('data/coco/fasterRcnn_spatMapFeat4x4IouScaleDet.npy')

#detMapFeat,_ = readFeaturesFromFile('data/coco/linear::c_in14_o6_fc6_d_c::hkm-int2::8x8x80.bin')
detMapFeat,_ = readFeaturesFromFile('data/coco/fasterRcnn_spatMapFeat4x4GaussScaleDet.npy')
randIdces = np.random.randint(0,np.min([spatMapFeat.shape[1],detMapFeat.shape[1],len(data['images'])]),n_samples)

pp = PdfPages(outfile)
fig = plt.figure()
resZeros = visMaps(np.zeros((gridSz[0],gridSz[1],ncat)),pGridSz)
plt.imshow(resZeros,cmap=cmap)
for c in xrange(ncat):
    i = (c//pGridSz[1])*(gridSz[0]+1) + gridSz[0]/3
    j = (c%pGridSz[1])*(gridSz[1]+1) + gridSz[1]/3
    plt.text(j,i,str(c))
    #print i,j 
plt.title('LEGEND: Class locations on the grid')
pp.savefig(fig)

fig = plt.figure()
red_patch = mpatches.Patch(color='red', label='The red data')
catList = ['%d - %s'%(i, ann['name']) for i,ann in enumerate(data['categories'])]
plt.legend([red_patch]*80, catList,ncol=4, fontsize='x-small',markerscale=0.5)
plt.title('LEGEND: Class index to class names')
pp.savefig(fig)

for i in randIdces:
    fig = plt.figure()
    resSpat = visMaps(spatMapFeat[:,i].reshape(ncat,gridSz[0],gridSz[1]).T/np.max(spatMapFeat[:,i]),(pGridSz[0],pGridSz[1]))
    resdet = visMaps(detMapFeat[:,i].reshape(ncat,gridSz[0],gridSz[1]).T/np.max(detMapFeat[:,i]),(pGridSz[0],pGridSz[1]))
    ax = plt.subplot(221);ax.imshow(resSpat,cmap=cmap);ax.set_title('IouMap')
    ax = plt.subplot(222);ax.imshow(resdet,cmap=cmap);ax.set_title('Gauss')
    ax = plt.subplot(223);ax.imshow(plt.imread(os.path.join(basepath,data['images'][i]['filepath'], data['images'][i]['filename'])))
    ax.set_title('idx:%d, cocoid:%d'%(i, data['images'][i]['cocoid']))
    pp.savefig(fig)

plt.close("all")
pp.close()

