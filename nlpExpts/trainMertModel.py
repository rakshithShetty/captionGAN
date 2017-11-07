import argparse
import json
import os
import random
import scipy.io
import codecs
import numpy as np
import cPickle as pickle
from collections import defaultdict
from nltk.tokenize import word_tokenize
import re

def gatherCandidates(params,nSamples=-1,skipSet = set()):
  eval_names = {}
  n_evals = params['n_evals']
  eval_idx = 0
  getSents = lambda x:re.split('\(|\)',x)[0]
  getProb = lambda x:float(re.split('\(|\)',x)[1])
  featMatrixGlobal = np.empty((n_evals,0))
  imgBlobs = []
  rootdir = params['resdir']
  icnt = 0
  lS = len(skipSet)
  for r in os.walk(rootdir):
      if len(r[1]) == 0:
          fcnt = 0
          if lS:
            iid = int(r[0].rsplit('/')[-1].split('.')[0])
            if iid in skipSet:
                continue
          for fl in r[2]:
              if 'eval' in fl:
                 fparts = fl.split('-')
                 mod_name = fparts[-1].split('.')[0]
                 if mod_name not in eval_names.keys():
                     eval_names[mod_name] = eval_idx
                     eval_idx += 1

                 mod_idx = eval_names[mod_name]
                 cands = open(os.path.join(r[0],fl),'r').read().split('#')

                 if fcnt == 0:
                    imgid = fparts[0]
                    nLclCands = len(cands)
                    candSentsLcl = map(getSents,cands)
                    featMatrixLcl = np.empty((n_evals,nLclCands))
                 fcnt += 1
                 featMatrixLcl[mod_idx,:] = map(getProb,cands)
          imgBlobs.append({'imgid':imgid,'cands':candSentsLcl})      
          featMatrixGlobal = np.concatenate([featMatrixGlobal,featMatrixLcl],axis=1)
          if (icnt % 1000 == 1):
            print('%d'%icnt)
          icnt += 1
          if nSamples !=-1 and icnt >= nSamples:
            break

  finalDataset = {}
  finalDataset['imgblobs'] = imgBlobs
  finalDataset['feats'] = featMatrixGlobal
  finalDataset['evaluaters'] = eval_names 

  params['cand_dB'] = 'allCands_dB_%s.p' % (params['fappend'])
  pickle.dump(finalDataset,open(params['cand_dB'],'w'))
  return finalDataset

def gatherCandSrc(params,nSamples=-1,skipSet = set()):
  src_mod_names = {}
  n_evals = params['n_evals']
  finalDataset = pickle.load(open(params['cand_dB'],'r'))
  eval_idx = 0
  getSents = lambda x:re.split('\(|\)',x)[0]
  getProb = lambda x:float(re.split('\(|\)',x)[1])
  featIdx = 0
  
  imgBlobs = finalDataset['imgblobs']
  
  rootdir = params['resdir']
  icnt = 0
  lS = len(skipSet)
  for r in os.walk(rootdir):
      if len(r[1]) == 0:
          if lS:
            iid = int(r[0].rsplit('/')[-1].split('.')[0])
            if iid in skipSet:
                continue
          imgBlobs[featIdx]['src_mods'] = defaultdict(list)
          imgBlobs[featIdx]['beam_sz'] = defaultdict(list)
          currBlobDict = {}
          for i,s in enumerate(imgBlobs[featIdx]['cands']):
            currBlobDict[s.lstrip(' ').rstrip(' ')] = i
          
          for fl in r[2]:
              if 'gene' in fl:
                 fparts = fl.split('-')
                 mod_name = fparts[-1].split('+')[0]
                 beam_size = fparts[-1].split('+')[1].split('.')[0]
                 if mod_name not in src_mod_names:
                     src_mod_names[mod_name] = eval_idx
                     eval_idx += 1

                 mod_idx = src_mod_names[mod_name]
                 cands = open(os.path.join(r[0],fl),'r').read().split('#')

                 imgid = int(fparts[0])
                 if imgid != int(imgBlobs[featIdx]['imgid']):
                    print 'ERROR ids dont match!!'
                    break
                #    return imgBlobs
                 nLclCands = len(cands)
                 candSentsLcl = map(getSents,cands)
                    
                 for s in candSentsLcl:
                    sm = s.lstrip(' ').rstrip(' ')
                    if sm in currBlobDict: 
                        matchIdx = currBlobDict[sm] 
                        imgBlobs[featIdx]['src_mods'][matchIdx].append(mod_idx)
                        imgBlobs[featIdx]['beam_sz'][matchIdx].append(beam_size)
                 
          featIdx += 1
          if (icnt % 1000 == 1):
            print('%d'%icnt)
          icnt += 1
          if nSamples !=-1 and icnt >= nSamples:
            break

  finalDataset['imgblobs'] = imgBlobs
  finalDataset['src_mod_names'] = src_mod_names 

  pickle.dump(finalDataset,open(params['cand_dB'],'w'))
  #return finalDataset


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--resdir', type=str,  default='', help='The root directory from which to gather candidate captions from')
  parser.add_argument('--fappend', type=str, default='', help='str to append to routput files')
  parser.add_argument('--cand_dB', type=str, default='', help='filename of the result struct to save')
  parser.add_argument('--mertDir', type=str, default='./nlpUtils/zmert_v1.50/zmert_ex_coco/', help='filename of the result struct to save')

  
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  
  print 'parsed parameters:'
  print json.dumps(params, indent = 2)
  if params['cand_dB'] == '':
    finalDataset =  gatherCandidates(params)

  feats = finalDataset['feats']
  zrs = np.zeros((1,feats.shape[1]))
  feats = np.concatenate([zrs, feats])
  
  fcnt  = 0
  for img in finalDataset['imgblobs']: 
    for c in img['cands']:
       feats[0,fcnt] = len(word_tokenize(c))
       fcnt += 1

  finalDataset['feats'] = feats
  pickle.dump(finalDataset,open(params['cand_dB'],'w'))
  
  #evaluate_decision(params, com_dataset, eval_array)
  n_ref_objects = []
  for img in finalDataset['imgblob']: 
    for c in img['cands']:
       obj_list = set()
       for w in word_tokenize(c):
         if w in visWordsDict:
            obj_list |= visWordsDict[w]
       n_ref_objects.append(len(obj_list))
  
  
  iblob =  finalDataset['imgblob']
  fcnt = 0
  for i,img in enumerate(iblob):
    nC = iblob[i]['feats'].shape[1]
    n_ref_objects_bmp = np.zeros((8,nC))
    n_ref_objects_bmp[n_ref_objects[fcnt:fcnt+nC],np.arange(nC)] = 1
    iblob[i]['feats'] = np.concatenate([iblob[i]['feats'],n_ref_objects_bmp],axis=0)
    fcnt += nC
 
  finalDataset['imgblob'] = iblob
 
