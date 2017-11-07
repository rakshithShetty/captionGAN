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

def main(params):
    for resF in params['resFileList']:
        caps = json.load(open(resF,'r'))
        dp = getDataProvider(caps['checkpoint_params'])
        trackMetargs = {'eval_metric': params['met_to_track']}
        refToks, scr_info = eval_prep_refs(params['split'], dp, params['met_to_track'])
        trackMetargs['refToks'] = refToks
        trackMetargs['scr_info'] = scr_info
        capsById = {}
        n_cands = params['keepN'] - 1 if params['keepN'] !=None else None
        npfilename = osp.join('scorelogs',osp.basename(resF).split('.')[0]+'_all%s_pairwise_%d'%(params['met_to_track'][0],n_cands+1))
        n=0
        for img in caps['imgblobs']:
            imgid = int(img['img_path'].split('_')[-1].split('.')[0])
            capsById[imgid] = [{'image_id':imgid, 'caption':img['candidate']['text'], 'id': n}]
            n+=1
            capsById[imgid].extend([{'image_id':imgid, 'caption':cd['text'], 'id': n+j} for j,cd in enumerate(img['candidatelist'][:n_cands])])
            if len(capsById[imgid]) < (n_cands+1):
               capsById[imgid].extend([capsById[imgid][-1] for _ in xrange(n_cands+1 - len(capsById[imgid]))])
            n+=len(capsById[imgid]) -1

        n_caps_perimg = len(capsById[capsById.keys()[0]])
        n_refs_perimg = len(refToks[refToks.keys()[0]])
        capsById = trackMetargs['scr_info']['tokenizer'].tokenize(capsById)

        all_scrs = []
        eval_metric = trackMetargs.get('eval_metric','perplex')
        #met = [[] for i in xrange(len(eval_metric)) if eval_metric[i][:6] != 'lcldiv']
        if params['rev_eval'] == 1:
            tempCont = capsById
            capsById = refToks
            refToks = tempCont
            temp_cnt = n_caps_perimg
            n_caps_perimg = n_refs_perimg
            n_refs_perimg = temp_cnt
            npfilename += '_reverse'

        met = np.zeros((len(eval_metric),n_caps_perimg, n_refs_perimg, len(capsById)))

        for j in xrange(n_caps_perimg):
            candToks = {imgid:[capsById[imgid][j]] for imgid in capsById}
            for r in xrange(n_refs_perimg):
                refTokInp = {imgid:refToks[imgid][r:r+1] for imgid in capsById}
                # Now invoke all the scorers and get the scores
                for i,evm in enumerate(eval_metric):
                    score, scores = trackMetargs['scr_info']['scr_fn'][i].compute_score(refTokInp, candToks)
                    met[i,j,r,:] = scores[-1] if type(score) == list else scores

                #print 'evaluated %d sentences and got %s = %f' % (n, evm, met[-1])
        np.savez(npfilename+'.npz',met=met,keys=refTokInp.keys())

        # Compute some specific scores
        mean_max_scr = met[0,:,:,:].max(axis=1).mean()

        if met.shape[1] <= met.shape[2] and met.shape[1] > 1 and params['keepN']<=10:
            perms = np.array([c for c in permutations(xrange(met.shape[2]), met.shape[1])])
            #Compute non-overlapping max-mean

            new_idx = np.concatenate([perms[:,None,:], np.tile(np.arange(met.shape[1])[None,:],[perms.shape[0],1])[:,None,:]],axis=1)

            non_overlapping_scrs = met[0,new_idx[:,0,:], new_idx[:,1,:],:].sum(axis=1).max(axis=0).mean()/float(met.shape[1])
        else:
            non_overlapping_scrs = 0.

        print 'mean %s is %.3f, mean-max is %.3f, non-overlapping mean-max is %.3f'%(eval_metric[0],met.mean(), mean_max_scr, non_overlapping_scrs)


if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # global setup settings, and checkpoints
  parser.add_argument(dest='resFileList', nargs='+',type=str, default=[], help='List of video ids')
  # Track some metrics during training
  parser.add_argument('--metrics_to_track', dest='met_to_track',nargs='+', type=str, default=['spice', 'meteor', 'cider', 'len', 'lcldiv_1', 'lcldiv_2'], help="""Specify the evaluation metric to use on validation. Possible
                                        values are perplex, meteor, cider""")
  parser.add_argument('--rev_eval', dest='rev_eval', type = int, default=0, help='evaluate references against generated sentences')
  parser.add_argument('--keepN',dest='keepN',type=int, default=None, help='List of video ids')
  parser.add_argument('--split',dest='split',type=str, default='test', help='List of video ids')
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  main(params)
