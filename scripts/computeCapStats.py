import json
import numpy as np
import argparse
from collections import Counter

def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

def main(params):
 ref = json.load(open(params['refData'],'r'))
 refTrnSentList = [' '.join(sent['tokens']) for img in ref['images'] if img['split'] == 'train' for sent in img['sentences']]
 n_ref_caps = len(refTrnSentList)
 refTrnSentSet = set(refTrnSentList)
 resNstr = '\n'
 for resF in params['resFileList']:
    caps = json.load(open(resF,'r'))
    capsSet = set()
    lenC = []
    vocabSet = set()
    if not params['useRawJson']:
        totCaps = len(caps)
        idstr = 'image_id' if 'image_id' in caps[0] else 'video_id'
        for r in caps:
            capsSet.add(r['caption'])
            lenC.append(len(r['caption'].split()))
            vocabSet.update(r['caption'].split())
    else:
        for img in caps['imgblobs']:
            capsSet.add(img['candidate']['text'])
            lenC.append(len(img['candidate']['text'].split()))
            vocabSet.update(img['candidate']['text'].split())

            capsSet.update([st['text'] for st in img['candidatelist'][:params['keepN']-1]])
            lenC.extend([len(st['text'].split()) for st in img['candidatelist'][:params['keepN']-1]])
            vocabSet.update(*[st['text'].split() for st in img['candidatelist'][:params['keepN']-1]])
        totCaps = params['keepN'] * len(caps['imgblobs'])

    if params['computeChiSq'] != None:
        for i in params['computeChiSq']:
            nGramTrain = Counter()
            for img in ref['images']:
                if img['split'] == 'train':
                    for st in img['sentences']:
                        nGramTrain.update(find_ngrams(st['tokens'],i))

            nGramRes = Counter()
            for img in caps['imgblobs']:
                nGramRes.update(find_ngrams(img['candidate']['text'].split(),i))
                for st in img['candidatelist'][:params['keepN']-1]:
                    nGramRes.update(find_ngrams(st['text'].split(),i))

            trainRatio = float(params['keepN'] * len(caps['imgblobs']))/ float(n_ref_caps)
            score = [ ((float(nGramRes[w]) - np.floor(trainRatio*float(nGramTrain[w])))**2.)/np.floor(trainRatio*float(nGramTrain[w])) for w in nGramTrain if np.floor(trainRatio*float(nGramTrain[w])) > 0.]
            print 'Chi-squared metric for %d-gram is %.3f'%(i, sum(score))


    lenC = np.array(lenC)
#    print '%s'%'\n'.join(list(capsSet))
    print 'RefData %%Uniq sent %.2f'%(float(len(refTrnSentSet))/len(refTrnSentList)*100.0)
    resNstr += '& %.2f & %4d & %.2f & %.2f\n'%(lenC.mean(),len(vocabSet),
                    float(len(capsSet))*100.0/totCaps,
                    float(len(capsSet)-len(refTrnSentSet.intersection(capsSet)))*100.0/totCaps)

    print 'Tot caps %5d, mean len is : %.2f, Vocab is %4d, Uniq sent %%: %.2f, New Sentences : %.2f'%(totCaps,lenC.mean(),len(vocabSet),
                    float(len(capsSet))*100.0/totCaps,
                    float(len(capsSet)-len(refTrnSentSet.intersection(capsSet))))

 print resNstr


if __name__ == "__main__":

 parser = argparse.ArgumentParser()
 parser.add_argument(dest='resFileList', nargs='+',type=str, default=[], help='List of video ids')
 parser.add_argument('--refData',dest='refData',type=str, default='data/coco/dataset.json', help='List of video ids')
 parser.add_argument('--useRawJson',dest='useRawJson',type=int, default=1, help='List of video ids')
 parser.add_argument('--keepN',dest='keepN',type=int, default=5, help='List of video ids')
 parser.add_argument('--computeChiSq',dest='computeChiSq',type=int, nargs='+', default=[1,2,3], help='List of video ids')
 args = parser.parse_args()
 params = vars(args) # convert to ordinary dict

 main(params)
