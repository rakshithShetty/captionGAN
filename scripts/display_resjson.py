import os
#os.chdir('../')
import os.path as osp
import sys
from collections import OrderedDict

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
main_path = osp.join(this_dir, '..')
add_path(main_path)

import numpy as np
import cv2
import argparse
import json
from random import shuffle
import operator
from os import path as osp


#for imgid in resLDict:
#    if resLDict[imgid]['candidate']['text'] != resHDict[imgid]['candidate']['text']:
#        diffLVH[imgid] = {}
#        diffLVH[imgid]['L'] = resLDict[imgid]['candidate']
#        diffLVH[imgid]['H'] = resLDict[imgid]['candidate']
#len(diffLVH)
#kD = diffLVH.keys()
#diffLVH[kD[0]]
#diffLVH = {}
#for imgid in resLDict:
#    if resLDict[imgid]['candidate']['text'] != resHDict[imgid]['candidate']['text']:
#        diffLVH[imgid] = {}
#        diffLVH[imgid]['L'] = resLDict[imgid]['candidate']
#        diffLVH[imgid]['H'] = resHDict[imgid]['candidate']

def loadResStructs(resFileList, resNames = None, dataset = 'coco'):

  resDicts = OrderedDict()

  for i in xrange(len(resFileList)):
    resL = json.load(open(resFileList[i],'r'))
    resN = resNames[i] if resNames != None else 'res_'+str(i)
    resD = {}
    for img in resL['imgblobs']:
        if dataset == 'coco':
            cocoid = int(img['img_path'].split('_')[-1].split('.')[0])
        elif dataset == 'lsmdc':
            cocoid = int(osp.basename(img['img_path']).split('.')[0])
        else:
            cocoid = int(img['img_path'][5:].split('.')[0])
        resD[cocoid] = img
    resDicts[resN] = resD

  return resDicts

def formatnum(x):
    if type(x) == float:
        return '%.2f'%(x)
    else:
        return '%d'%(x)


def dispResults(resDicts, imidDisp, args):
  resNames = resDicts.keys()

  if args.overridebase != None:
    basepath = args.overridebase
  else:
    basepath = osp.dirname(resDicts[resNames[0]][imidDisp[0]]['img_path'])

  if args.visfeats:
    feats,clsL,featN,cidToidx = loadfeatsandlabels(args)

  font = cv2.FONT_HERSHEY_DUPLEX
  fscale = 0.7
  fsFeat = 0.7
  fTC = 1
  fTF = 1
  lineSpc = 16
  i = 0
  xOff = 0
  ftopk = args.ftopk

  def appendToImg(img, img_appends):
    for iappd in img_appends:
      #print 'Appending shape ', iappd[1].shape, 'to side ', iappd[0]
      aL = [img, iappd[1]] if iappd[0] > 0 else [iappd[1], img]
      img = np.concatenate(aL,axis=np.abs(iappd[0]) - 1)
    return img


  def prepImage(img,imgid):
    img_appends = []
    imshape = list(img.shape)
    n_cands = params['keepN'] - 1 if params['keepN'] !=None else None
    if args.visfeats:
        feat_vis_lines = []
        for iF,f in enumerate(feats):
            feat_vis_lines.append('-----')
            feat_vis_lines.append('%s'%(featN[iF]))
            feat_vis_lines.append('-----')
            topkIdx = np.argsort(f[:,cidToidx[imgid]])[::-1][:ftopk]
            feat_vis_lines.extend(['%s: %.3f'%(clsL[iF][srtidx], f[srtidx,cidToidx[imgid]]) for srtidx in topkIdx])

        mFVs = [xOff,10]
        for txt in feat_vis_lines:
            tS, bs = cv2.getTextSize(txt, font, fsFeat, fTF);
            mFVs[1] = max(tS[0]+20,mFVs[1])
            mFVs[0] += tS[1] + lineSpc
        mFVs[0] += lineSpc
        mFVs[0] = max(imshape[0],mFVs[0])
        if mFVs[0] > img.shape[0]:
            #img = np.concatenate([img, np.zeros((mFVs[0] - imshape[0], imshape[1], imshape[2]), dtype=img.dtype)], axis = 0)
            img_appends.append((+1, np.zeros((mFVs[0] - imshape[0], imshape[1], imshape[2]), dtype=img.dtype)))
            imshape[0] = mFVs[0]

        fVFrm = 255*np.ones((mFVs[0],mFVs[1],imshape[2]),dtype=img.dtype)
        currY = lineSpc + tS[1]
        for txt in feat_vis_lines:
            cv2.putText(fVFrm, txt,(10,currY), font, fsFeat,(0,0,0), fTF,cv2.LINE_AA)
            currY += tS[1] + lineSpc
        #img = np.concatenate([img,fVFrm], axis = 1)
        img_appends.append((+2, fVFrm))
        imshape[1] += fVFrm.shape[1]

    if args.visCaps:
        #mTs = [lineSpc,img.shape[1]]
        mTs = [lineSpc, imshape[1]]
        if args.addId == 1:
            capTexts = ['Video id: %d'%(imgid)]
            textCol = [(100,100,100)]
            tS, bs = cv2.getTextSize(capTexts[-1], font, fscale, fTC);
            mTs[1] = max(tS[0]+xOff,mTs[1])
            mTs[0] += tS[1] + lineSpc
        else:
           capTexts = []
           textCol = []
        for resN in resDicts:
            capTexts.append(resN + ': ' + resDicts[resN][imgid]['candidate']['text'] + '(' + ', '.join([formatnum(resDicts[resN][imgid]['candidate'][shs]) for shs in args.showscore]) +')')
            textCol.append((0,0,0))
            tS, bs = cv2.getTextSize(capTexts[-1], font, fscale, fTC);
            mTs[1] = max(tS[0]+xOff,mTs[1])
            mTs[0] += tS[1] + lineSpc
            if args.dispAllCand:
                for cD in resDicts[resN][imgid]['candidatelist'][:n_cands]:
                    capTexts.append(' '*len(resN) + '  ' + cD['text'] + '(' +', '.join([formatnum(cD[shs]) for shs in args.showscore])+')')
                    textCol.append((0,0,0))
                    tS, bs = cv2.getTextSize(capTexts[-1], font, fscale, fTC);
                    mTs[1] = max(tS[0]+xOff,mTs[1])
                    mTs[0] += tS[1] + lineSpc
                capTexts.append('-'*20)
                tS, bs = cv2.getTextSize(capTexts[-1], font, fscale, fTC);
                mTs[1] = max(tS[0]+xOff,mTs[1])
                mTs[0] += tS[1] + lineSpc
                textCol.append((0,0,0))

        mTs[0] -= lineSpc//2

        capFrm = 255*np.ones((mTs[0],mTs[1],imshape[2]),dtype=img.dtype)
        currY = lineSpc + tS[1]
        for ci,cT in enumerate(capTexts):
            cv2.putText(capFrm, cT,(xOff,currY), font, fscale,textCol[ci], fTC,cv2.LINE_AA)
            currY += tS[1] + lineSpc

        if mTs[1] > imshape[1]:
            #img = np.concatenate([255*np.ones((img.shape[0], (mTs[1] - img.shape[1])//2, img.shape[2]), dtype=img.dtype),
            #        img, 255*np.ones((img.shape[0], (mTs[1] - img.shape[1] + 1)//2, img.shape[2]), dtype=img.dtype)], axis = 1)
            img_appends.append((-2, 255*np.ones((img.shape[0], (mTs[1] - imshape[1])//2, imshape[2]), dtype=img.dtype)))
            img_appends.append((+2, 255*np.ones((img.shape[0], (mTs[1] - imshape[1] + 1)//2 , imshape[2]), dtype=img.dtype)))
            imshape[1] = mTs[1]

        img_appends.append((+1, capFrm))
    return img_appends

  if args.savefile == None:
    imgsavedir = osp.join(args.savedir,'_vs_'.join(resNames))
    if not osp.exists(imgsavedir):
      os.makedirs(imgsavedir)
    while 1:
      imgid = imidDisp[i]
      if args.videos == 0:
        img = cv2.imread(osp.join(basepath,osp.basename(resDicts[resNames[0]][imgid]['img_path'])))
        if args.fixedsize !=[]:
          img = cv2.resize(img, (args.fixedsize[1], args.fixedsize[0]))
      else:
        cap = cv2.VideoCapture(osp.join(basepath,osp.basename(resDicts[resNames[0]][imgid]['img_path'])))
        ret, img = cap.read()
      #print img.shape
      img_appends = prepImage(img,imgid)
      img = appendToImg(img, img_appends)

      if args.videos == 1:
        vid_frames = [img]
        while(cap.isOpened() and ret==True):
          ret, img= cap.read()
          if ret:
            img = appendToImg(img, img_appends)
            vid_frames.append(img)
        print len(vid_frames)
        cap.release()
        ci = 0
        while 1:
            img = vid_frames[ci%len(vid_frames)]
            if args.fixedsize !=[]:
              img = cv2.resize(img, (args.fixedsize[1], args.fixedsize[0]))
            cv2.imshow('frame',img)
            ci += 1
            keyInp = cv2.waitKey(20)
            if keyInp != -1:
              break
      else:
        cv2.imshow('frame',img)
        keyInp = cv2.waitKey(0)

      if keyInp & 0xFF == ord('q'):
          break
      elif keyInp & 0xFF == 81:
          print keyInp & 0xFF
          i = i-1
      elif (keyInp & 0xFF == ord('s')) or args.saveall == 1:
          imgSaveName = osp.join(imgsavedir, str(imgid) +'.png')
          print 'Saving into file: ' + imgSaveName
          cv2.imwrite(imgSaveName,img)
          i += 1
      else:
          i += 1
  else:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pp = PdfPages(args.savefile)
    for imgid in np.random.choice(imidDisp,size=min(args.nSave,len(imidDisp)),replace=False):
      img = plt.imread(osp.join(basepath,osp.basename(resDicts[resNames[0]][imgid]['img_path'])))
      img_appends = prepImage(img,imgid)
      img = appendToImg(img, img_appends)
      fig, ax = plt.subplots(figsize=(12, 12))
      ax.imshow(img, aspect='equal')
      ax.set_title('ImageId = %d '%(imgid))
      pp.savefig(fig)
      plt.close(fig)

    pp.close()

def findIntIds(resDicts, args):
  resNames = resDicts.keys()
  if args.showdiff > 0:
    imidDisp = []
    for imgid in resDicts[resNames[0]].keys():
        allCands = [resDicts[resN][imgid]['candidate']['text'] for resN in resDicts]
        if (allCands.count(allCands[0]) != len(allCands)):
            imidDisp.append(imgid)
    print 'Found differing entries %d/%d, that is %.2f'%(len(imidDisp),len(resDicts[resNames[0]].keys()), 100.0*float(len(imidDisp))/len(resDicts[resNames[0]].keys()))
    shuffle(imidDisp)
  else:
    imidDisp = resDicts[resNames[0]].keys()
    shuffle(imidDisp)

  return imidDisp

def loadfeatsandlabels(args):
  if args.visfeats == 1:
    imgIdLbl = open(args.imgidlabel,'r').read().splitlines()
    cocoIdtoFeatIdx = {}
    for imgL in imgIdLbl:
        cocoIdtoFeatIdx[int(imgL.split()[1][1:-1])] = int(imgL.split()[0][1:])

    # Now load the features:
    params = {}
    f_list = []
    featN = []
    if args.feats != None:
        from imagernn.data_provider import prepare_data, loadArbitraryFeatures
    for i,f in enumerate(args.feats):
        params['feat_file'] = f
        feat, _, feat_idx, _ = loadArbitraryFeatures(params)
        f_list.append(feat)
        featN.append(args.featNames[i] if args.featNames!= None else 'feat_'+str(i))

    cLabls = []
    for l in args.clslabels:
        cL = open(l,'r').read().splitlines()
        cLabls.append(cL)
    return f_list,cLabls,featN,cocoIdtoFeatIdx
  else:
    return [],[], [], []

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument(dest='resFileList', nargs='+',type=str, default=[], help='List of video ids')
  parser.add_argument('--resNames',dest='resNames', nargs='+',type=str, default=None, help='List of video ids')
  parser.add_argument('--showdiff', dest='showdiff', type=int, default=0, help='constant brightness increase')
  parser.add_argument('--refData',dest='refData',type=str, default='dataset.json', help='List of video ids')
  parser.add_argument('--overridebase',dest='overridebase',type=str, default=None, help='List of video ids')
  parser.add_argument('--videos',dest='videos',type=int,default=0,help='showing videos or images')

  # Visualizing feature information
  parser.add_argument('--visfeats', dest='visfeats', type=int, default=0, help='constant brightness increase')
  parser.add_argument('--visCaps', dest='visCaps', type=int, default=1, help='constant brightness increase')
  parser.add_argument('--ftopk', dest='ftopk', type=int, default=5, help='constant brightness increase')
  parser.add_argument('--imgidlabel',dest='imgidlabel',type=str, default='data/coco/labels.txt', help='List of video ids')
  parser.add_argument('--feats',dest='feats', nargs='+',type=str, default=None, help='List of video ids')
  parser.add_argument('--featNames',dest='featNames', nargs='+',type=str, default=None, help='List of video ids')
  parser.add_argument('--clslabels',dest='clslabels', nargs='+',type=str, default=None, help='List of video ids')
  # directory used to save individual images interactively
  parser.add_argument('--savedir', dest='savedir', type=str, default='savedimages', help='directory to save images chosen to save')

  # Show only these ids
  parser.add_argument('--showids',dest='showids', nargs='+',type=int, default=[], help='List of video ids')
  parser.add_argument('--fixedsize',dest='fixedsize', nargs='+',type=int, default=[], help='Fix the size of the image')
  parser.add_argument('--saveall',dest='saveall', type=int, default=0, help='Fix the size of the image')
  parser.add_argument('--addId', dest='addId', type=int, default=0, help='constant brightness increase')

  parser.add_argument('--dispAllCand', dest='dispAllCand', type=int, default=1, help='constant brightness increase')
  parser.add_argument('--keepN',dest='keepN',type=int, default=None, help='List of video ids')

  parser.add_argument('--sortbyres',dest='sortbyres',nargs='+', type=str, default=None, help='List of video ids')
  parser.add_argument('--sortbykey',dest='sortbykey',type=str, default=None, help='List of video ids')
  parser.add_argument('--showscore',dest='showscore',nargs='+',type=str, default=['logprob'], help='List of video ids')

  # Write to a pdf option
  parser.add_argument('--save', dest='savefile', type=str, default=None, help='constant brightness increase')
  parser.add_argument('--nsave', dest='nSave', type=int, default=100, help='constant brightness increase')

  # dataset
  parser.add_argument('--dataset', dest='dataset', type=str, default='coco', help='constant brightness increase')


  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  resDicts = loadResStructs(args.resFileList,args.resNames, args.dataset)
  imidDisp = findIntIds(resDicts, args)
  if args.showids != []:
    imidDisp = list(set(imidDisp) and set(args.showids))
  if args.sortbyres != None:
    if len(args.sortbyres) == 1:
        imidDisp = sorted(imidDisp, key=lambda k: resDicts[args.sortbyres[0]][k][args.sortbykey])
    else:
        imidDisp = sorted(imidDisp, key=lambda k: resDicts[args.sortbyres[0]][k][args.sortbykey] - resDicts[args.sortbyres[1]][k][args.sortbykey])
  dispResults(resDicts, imidDisp, args)

  #resL = json.load(open(args.refData,'r'))
  #resD = {}
  #for img in resL['images']:
  #    cocoid = int(img['img_path'].split('_')[-1].split('.')[0])
  #    resD[cocoid] = img

