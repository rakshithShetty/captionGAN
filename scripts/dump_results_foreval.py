import argparse
import json
from collections import defaultdict, OrderedDict
import os.path
import re
import os
import itertools

def buildDbid2Idx(lblfile):
  labels = open(lblfile).read().splitlines()
  dbid2idx = [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]
  cidx = [1, 1, 1, 1]
  for lbl in labels:
      idx,dbStr = lbl.split()
      idx = int(idx[1:])
      splt = int(dbStr[1])
      dbid = dbStr[1:-1]
      if ':' not in dbid:
        dbid2idx[splt][dbid] = cidx[splt]
        cidx[splt] += 1
  
  return dbid2idx


def buildTransDict(dictFile):
  dictVal= open(dictFile,'r').read().splitlines()
  trDict = OrderedDict()
  wSets = {}
  for lns in dictVal:
      if lns[0] == '#':
        wSets[lns.split()[0]] = lns.split()[1:]
      else:
        targ = ' '.join(lns.split()[0].split('_'))
        for wrdSeq in lns.split()[1:]:
          combSets = [wSets.get(wSi,[wSi]) for wSi in wrdSeq.split('_')]
          for oneComb in itertools.product(*combSets):
            key = ' '.join(oneComb)
            trDict[key] = targ
  trDict = OrderedDict((re.escape(k), v) for k, v in trDict.iteritems())
  pattern = re.compile("|".join(trDict.keys()))
  return pattern,trDict 

def main(params):
  
  testRes = json.load(open(params['resFile'],'r'))
  
  if params['algname'] == 'None':
    params['algname'] = params['resFile'].split('_')[-1].rsplit('.',1)[0]        
  outfile = 'captions_%s_%s%s_results.json'%(params['target_split'],params['algname'],params['append'])
  outdir = 'eval/mseval/results/%s'%(params['target_db'])
  if params['target_db'] == 'msr-vtt-server':
    outdir = 'eval/mseval/submission/vttTestSubmissions'
  print 'writing to %s'%(outfile)

  # Build dbid to test dump index
  if params['target_db'] == 'lsmdc2015':
    dbid2idx = buildDbid2Idx(params['labelsFile'])
  
  if params['translate'] == 1:
    pattern,trDict = buildTransDict(params['transdict'])
  
  testResDump = []
  
  for i,img in enumerate(testRes['imgblobs']):
    if params['mc_mode'] == 1:
        txt = img['candidate']['raw']
    elif params['translate'] == 0:
        txt = img['candidate']['text']
    else:
        txt = pattern.sub(lambda m: trDict[re.escape(m.group(0))],img['candidate']['text'])
        testRes['imgblobs'][i]['candidate']['text'] = txt

    if params['target_db'] == 'coco':
        testResDump.append({'caption': txt,'image_id': int(img['img_path'].split('/')[-1].split('_')[-1].split('.')[0])})
    elif params['target_db'] == 'lsmdc2015_picsom':
        testResDump.append({'caption': txt,'video_id': int(img['img_path'].split('/')[-1].split(':')[0])})
    elif params['target_db'] == 'lsmdc2015':
        dbid = os.path.basename(img['img_path']).split('.')[0].zfill(9)
        splt = 2 if params['target_split'] == 'test2015' else 3 if params['target_split'] == 'blindtest2015' else 1
        testResDump.append({'caption': txt,'video_id': dbid2idx[splt][dbid]})
    elif params['target_db'] == 'msr-vtt-server':
        #print "NOT IMPLEMENTED YET"
        testResDump.append({'caption': txt,'video_id': img['img_path'].split('.')[0]})
    elif params['target_db'] == 'msr-vtt-local':
        testResDump.append({'caption': txt,'video_id': img['img_path'].split('.')[0]})
    else:
        raise ValueError('Error: this db is not handled')
  
  if params['target_db'] == 'msr-vtt-server':
    testResDump = {"version":"Version 1.0","result":testResDump}
    testResDump['external_data'] = {"used":"true","details":"We use ILSVRC-2012 for pre-training frame level CNNs, we use Sports-1M dataset to pre-train 3D-CNN"}
  if params['target_db'] == 'lsmdc2015':
      testResDump = sorted(testResDump, key= lambda x:x['video_id'])

  json.dump(testResDump, open(os.path.join(outdir,outfile), 'w'))
  if params['writeback'] != '':
    json.dump(testRes, open(params['writeback'], 'w'))
  

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  parser.add_argument('resFile', type=str, help='the input resultDump')
  parser.add_argument('-td', '--target_db', type=str, default='coco', help='target database to evaluate on')
  parser.add_argument('-ts', '--target_split', type=str, default='val2014', help='target database to evaluate on')
  parser.add_argument('-a', '--algname', type=str, default='None', help='algorithm name to use in output filename')
  parser.add_argument('--append', '--append', type=str, default='', help='algorithm name to use in output filename')
  parser.add_argument('--translate', type=int, default=0, help='Apply language translation via dict')
  parser.add_argument('--transdict', type=str, default='data/lsmdc2015/commons.dict', help='translation dictionary')
  parser.add_argument('--labelsFile',dest='labelsFile', type=str, default='data/lsmdc/labels.txt', help='labels file mapping picsom id to sequence id')
  parser.add_argument('--writeback',dest='writeback', type=str, default='', help='writeback the result struct after translation')
  parser.add_argument('--mc_mode', type=int, default=0, help='Apply language translation via dict')

  
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  main(params)
