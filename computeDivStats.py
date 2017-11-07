import json
import numpy as np
import argparse
from imagernn.utils import compute_div_n, compute_global_div_n

from eval.mseval.pycocoevalcap.meteor.meteor import Meteor
from eval.mseval.pycocoevalcap.bleu.bleu import Bleu
from eval.mseval.pycocoevalcap.rouge.rouge import Rouge
from eval.mseval.pycocoevalcap.cider.cider import Cider
from eval.mseval.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

def main(params):

 tokenizer = PTBTokenizer()
 for resF in params['resFileList']:
    caps = json.load(open(resF,'r'))
    capsById = {}
    idTocaps = {}
    n_cands = params['keepN'] - 1 if params['keepN'] !=None else None
    n=0
    for i,img in enumerate(caps['imgblobs']):
        imgid = int(img['img_path'].split('_')[-1].split('.')[0])
        capsById[imgid] = [{'image_id':imgid, 'caption':img['candidate']['text'], 'id': n}]
        idTocaps[imgid] = i
        n+=1
        capsById[imgid].extend([{'image_id':imgid, 'caption':cd['text'], 'id': n+j} for j,cd in enumerate(img['candidatelist'][:n_cands])])
        if len(capsById[imgid]) < (n_cands+1):
           capsById[imgid].extend([capsById[imgid][-1] for _ in xrange(n_cands+1 - len(capsById[imgid]))])
        n+=len(capsById[imgid]) -1

    n_caps_perimg = len(capsById[capsById.keys()[0]])
    print n_caps_perimg
    capsById = tokenizer.tokenize(capsById)

    div_1, adiv_1 = compute_div_n(capsById,1)
    div_2, adiv_2 = compute_div_n(capsById,2)

    globdiv_1, _= compute_global_div_n(capsById,1)

    print 'Diversity Statistics are as follows: \n Div1: %.2f, Div2: %.2f, gDiv1: %d\n'%(div_1,div_2, globdiv_1)

    if params['compute_mbleu']:
        scorer = Bleu(4)

        # Run 1 vs rest bleu metrics
        all_scrs = []
        scrperimg = np.zeros((n_caps_perimg, len(capsById)))
        for i in xrange(n_caps_perimg):
            tempRefsById = {}
            candsById = {}
            for k in capsById:
                tempRefsById[k] = capsById[k][:i] + capsById[k][i+1:]
                candsById[k] = [capsById[k][i]]

            score, scores = scorer.compute_score(tempRefsById, candsById)
            all_scrs.append(score)
            scrperimg[i,:] = scores[1]

        all_scrs = np.array(all_scrs)
        if params['writeback']:
            for i,imgid in enumerate(capsById.keys()):
                caps['imgblobs'][idTocaps[imgid]]['mBleu_2'] = scrperimg[:,i].mean()
                caps['imgblobs'][idTocaps[imgid]]['candidate']['mBleu_2'] = scrperimg[0,i]
                for j,st in enumerate(caps['imgblobs'][idTocaps[imgid]]['candidatelist'][:n_cands]):
                    caps['imgblobs'][idTocaps[imgid]]['candidatelist'][j]['mBleu_2'] = scrperimg[1+j,i]
            json.dump(caps,open(resF,'w'))


        print 'Mean mutual Bleu scores on this set is:\nmBLeu_1, mBLeu_2, mBLeu_3, mBLeu_4\n'
        print all_scrs.mean(axis=0)

if __name__ == "__main__":

 parser = argparse.ArgumentParser()
 parser.add_argument(dest='resFileList', nargs='+',type=str, default=[], help='List of video ids')
 parser.add_argument('--mbleu',dest='compute_mbleu',type=int, default=1, help='List of video ids')
 parser.add_argument('--keepN',dest='keepN',type=int, default=None, help='List of video ids')
 parser.add_argument('--writeback',dest='writeback',type=int, default=0, help='List of video ids')
 args = parser.parse_args()
 params = vars(args) # convert to ordinary dict

 main(params)
