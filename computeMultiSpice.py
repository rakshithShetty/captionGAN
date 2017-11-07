import theano
import argparse
import numpy as np
import cPickle as pickle
from imagernn.data_provider import getDataProvider, prepare_data, prepare_adv_data
from imagernn.cnn_evaluatorTheano import CnnEvaluator
from imagernn.solver import Solver
from imagernn.imagernn_utils import decodeEvaluator, decodeGenerator, eval_split_theano, eval_prep_refs
#from numbapro import cuda
from imagernn.utils import numpy_floatX, zipp, unzip, preProBuildWordVocab
from collections import defaultdict, OrderedDict
import signal
import json
import os
import os.path as osp
from itertools import permutations
from eval.mseval.pycocoevalcap.spice.spice import Spice
from eval.mseval.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

def main(params):
    tokenizer = PTBTokenizer()
    scorer = Spice(multihyp = 1)
    refsJs = json.load(open(params['refdata'],'r'))
    refsById = defaultdict(list)
    for i,ann in enumerate(refsJs['annotations']):
        refsById[ann['image_id']].append({'image_id':ann['image_id'],'id':i,'caption':ann['caption']})

    refsById = tokenizer.tokenize(refsById)
    n_cands = params['keepN'] - 1 if params['keepN'] !=None else None

    for resF in params['resFileList']:
        caps = json.load(open(resF,'r'))
        capsById = {}
        n=0
        n_cands_per_img = np.zeros((len(caps['imgblobs'])),dtype=np.int32)
        for i, img in enumerate(caps['imgblobs']):
            imgid = int(img['img_path'].split('_')[-1].split('.')[0])
            capsById[imgid] = [{'image_id':imgid, 'caption':img['candidate']['text'], 'id': n}]
            n+=1
            capsById[imgid].extend([{'image_id':imgid, 'caption':cd['text'], 'id': n+j} for j,cd in enumerate(img['candidatelist'][:n_cands])])
            n+=len(capsById[imgid]) -1
            n_cands_per_img[i] = len(capsById[imgid])

        capsById = tokenizer.tokenize(capsById)

        print 'Maximum number of candidates is %d, mean is %.2f'%(n_cands_per_img.max(), n_cands_per_img.mean())

        refToks = {imgid:refsById[imgid] for imgid in capsById if imgid in refsById}
        if len(refToks) < len(capsById):
            capsById = {imgid:capsById[imgid] for imgid in refToks}

        n_refs_perimg = len(refToks[refToks.keys()[0]])

        all_scrs = []
        #met = [[] for i in xrange(len(eval_metric)) if eval_metric[i][:6] != 'lcldiv']
        if params['rev_eval'] == 1:
            tempCont = capsById
            capsById = refToks
            refToks = tempCont

        if params['iterativeEval']:
            npfilename = osp.join('scorelogs',osp.basename(resF).split('.')[0]+'_iterativeSpice_%d'%(params['keepN']))
            if params['refdata'] != '/BS/databases/coco/annotations/captions_val2014.json':
                npfilename += '_visgenome'
            if params['singleHyp']:
                npfilename += '_singlehyp'
            iterIdces = np.arange(n_cands_per_img.max(),dtype=np.int32)
        else:
            iterIdces = [n_cands_per_img.max()-1]

        mean_scr = np.zeros((len(iterIdces)))
        prec = np.zeros((len(iterIdces),len(capsById),7))
        rec = np.zeros((len(iterIdces),len(capsById),7))
        f_scr = np.zeros((len(iterIdces),len(capsById),7))
        for ii,idx in enumerate(iterIdces):
            candsInp = {imgid:[capsById[imgid][min(idx,len(capsById[imgid])-1)]] for imgid in capsById} if params['singleHyp'] else {imgid:capsById[imgid][:idx+1] for imgid in capsById}
            mean_scr[ii], all_scores = scorer.compute_score(refToks, candsInp)
            #Compute mean precision and recalls
            categories = all_scores[0].keys()

            for i,scr in enumerate(all_scores):
                for j, cat in enumerate(categories):
                    f_scr[ii,i,j] = scr[cat]['f']
                    prec[ii,i,j] = scr[cat]['pr']
                    rec[ii,i,j] = scr[cat]['re']
            print 'At idx %d, prec = %.3f, rec= %.3f'%(idx, prec[ii,:,0].mean(), rec[ii,:,0].mean())

        if params['iterativeEval']:
            np.savez(npfilename+'.npz',mean_scr = mean_scr, prec = prec, rec = rec, keys=refToks.keys())
        prec = np.nan_to_num(prec)
        rec = np.nan_to_num(rec)

        print '---------------------\nmean spice is %.3f\n---------------------\n Per category scores are'%(mean_scr[-1])
        for j,cat in enumerate(categories):
            print '%12s: f = %.3f, prec = %.3f, rec= %.3f'%(cat, np.nan_to_num(f_scr[-1,:,j]).mean(), prec[-1,:,j].mean(), rec[-1,:,j].mean()) #~np.isnan(f_scr[-1,:,j])
        #print 'mean fp is %.2f, mean fn %.2f, mean tp is %.2f'%(np.array(fp).mean(), np.array(fn).mean(), np.array(tp).mean())


if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # global setup settings, and checkpoints
  parser.add_argument(dest='resFileList', nargs='+',type=str, default=[], help='List of video ids')
  # Track some metrics during training
  parser.add_argument('--refdata', dest='refdata', type=str, default='/BS/databases/coco/annotations/captions_val2014.json', help='List of video ids')
  parser.add_argument('--rev_eval', dest='rev_eval', type = int, default=0, help='evaluate references against generated sentences')
  parser.add_argument('--keepN',dest='keepN',type=int, default=None, help='List of video ids')
  parser.add_argument('--iterativeEval',dest='iterativeEval',type=int, default=0, help='List of video ids')
  parser.add_argument('--singleHyp',dest='singleHyp',type=int, default=0, help='List of video ids')
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  main(params)

