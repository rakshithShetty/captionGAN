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
from imagernn.data_provider import getDataProvider
from imagernn.imagernn_utils import decodeGenerator, eval_split, eval_split_theano

from nltk.align.bleu import BLEU
import math

# UTILS needed for BLEU score evaluation      
def BLEUscore(candidate, references, weights):
  p_ns = [BLEU.modified_precision(candidate, references, i) for i, _ in enumerate(weights, start=1)]
  if all([x > 0 for x in p_ns]):
      s = math.fsum(w * math.log(p_n) for w, p_n in zip(weights, p_ns))
      bp = BLEU.brevity_penalty(candidate, references)
      return bp * math.exp(s)
  else: # this is bad
      return 0

def evalCandidate(candidate, references):
  """ 
  candidate is a single list of words, references is a list of lists of words
  written by humans.
  """
  b1 = BLEUscore(candidate, references, [1.0])
  b2 = BLEUscore(candidate, references, [0.5, 0.5])
  b3 = BLEUscore(candidate, references, [1/3.0, 1/3.0, 1/3.0])
  return [b1,b2,b3]

def get_bleu_scores(cands,refs):
    open('eval/output', 'w').write('\n'.join(cands))
    for q in xrange(5):
        open('eval/reference'+`q`, 'w').write('\n'.join([x[q] for x in refs]))
    owd = os.getcwd()
    os.chdir('eval')
    os.system('./multi-bleu.perl reference < output > scr')
    str = open('scr', 'r').read()
    bleus = map(float,str.split('=')[1].split('(')[0].split('/'))
    os.chdir(owd)
    return bleus

def eval_bleu_all_cand(params, com_dataset):
    bleu_array = np.zeros((3,n_imgs*n_sent))
    
    # Also load one of the result structures as the template
    res_struct = json.load(open(com_dataset['members_results'][0],'r'))
    #owd = os.getcwd()
    #os.chdir('eval')

    sid = 0
    for i in xrange(n_imgs):
        img = com_dataset['images'][i]
        refs = [r.values()[0] for r in res_struct['imgblobs'][i]['references']]

        #for q in xrange(5):
        #    open('reference'+`q`, 'w').write('\n'.join([x[q] for x in refs]))
        
        for sent in img['sentences']:
            #os.system('./multi-bleu.perl reference <<<"%s" > scr'%(sent['raw']))
            #str = open('scr', 'r').read()
            #bleus = map(float,str.split('=')[1].split('(')[0].split('/'))
            bleus = evalCandidate(sent['raw'],refs)
            bleu_array[:,sid] = bleus 
            sid +=1
        if ((i) % 500) == 0 :
            print('At %d\r'%i)
    
    #os.chdir(owd)

    return bleu_array 

def evaluate_decision(params, com_dataset, eval_array):
    indx = 0
    n_memb = com_dataset['n_memb']
    n_sent = com_dataset['n_sent']
    n_imgs = len(com_dataset['images'])

    all_references = []
    all_candidates = []
    
    # Also load one of the result structures as the template
    res_struct = json.load(open(com_dataset['members_results'][0],'r'))
    
    scr = eval_array.sum(axis=0)
    
    for i in xrange(n_imgs):
        img = com_dataset['images'][i]
        
        curr_scr = scr[i*n_sent: (i+1)* n_sent]
        best = np.argmax(curr_scr)

        res_struct['imgblobs'][i]['candidate']['logprob'] = curr_scr[best]
        res_struct['imgblobs'][i]['candidate']['text'] = img['sentences'][best]['raw']

        refs = [r.values()[0] for r in res_struct['imgblobs'][i]['references']]

        #calculate bleu of each candidate with reference


        all_references.append(refs)
        all_candidates.append(img['sentences'][best]['raw'])

    print 'writing intermediate files into eval/'
    # invoke the perl script to get BLEU scores
    print 'invoking eval/multi-bleu.perl script...'
    bleus = get_bleu_scores(all_candidates, all_references)
    res_struct['FinalBleu'] = bleus
    print bleus
  
    print 'saving result struct to %s' % (params['result_struct_filename'], )
    json.dump(res_struct, open(params['result_struct_filename'], 'w'))


def hold_comittee_discussion(params, com_dataset):
    
    n_memb = com_dataset['n_memb']
    n_sent = com_dataset['n_sent']
    n_imgs = len(com_dataset['images'])

    eval_array = np.zeros((n_memb,n_imgs*n_sent))
    model_id = 0  
    for mod in com_dataset['members_model']:
        checkpoint = pickle.load(open(mod, 'rb'))
        checkpoint_params = checkpoint['params']
        dataset = checkpoint_params['dataset']
        model_npy = checkpoint['model']

        checkpoint_params['use_theano'] = 1

        if 'image_feat_size' not in  checkpoint_params:
          checkpoint_params['image_feat_size'] = 4096 

        checkpoint_params['data_file'] = params['jsonFname'].rsplit('/')[-1]
        dp = getDataProvider(checkpoint_params)

        ixtoword = checkpoint['ixtoword']

        blob = {} # output blob which we will dump to JSON for visualizing the results
        blob['params'] = params
        blob['checkpoint_params'] = checkpoint_params
        blob['imgblobs'] = []

        # iterate over all images in test set and predict sentences
        BatchGenerator = decodeGenerator(checkpoint_params)

        BatchGenerator.build_eval_other_sent(BatchGenerator.model_th, checkpoint_params,model_npy)

        eval_batch_size = params.get('eval_batch_size',100)
        eval_max_images = params.get('eval_max_images', -1)
        wordtoix = checkpoint['wordtoix']

        split = 'test'
        print 'evaluating %s performance in batches of %d' % (split, eval_batch_size)
        logppl = 0
        logppln = 0
        nsent = 0
        gen_fprop = BatchGenerator.f_eval_other
        blob['params'] = params
        c_id = 0
        for batch in dp.iterImageSentencePairBatch(split = split, max_batch_size = eval_batch_size, max_images = eval_max_images):
          xWd, xId, maskd, lenS = dp.prepare_data(batch,wordtoix)
          eval_array[model_id, c_id:c_id + xWd.shape[1]] = gen_fprop(xWd, xId, maskd)
          c_id += xWd.shape[1]
        
        model_id +=1
    
    # Calculate oracle scores
    bleu_array = eval_bleu_all_cand(params,com_dataset)
    eval_results = {}
    eval_results['logProb_feat'] = eval_array
    eval_results['OracleBleu'] = bleu_array
    #Save the mutual evaluations

    params['comResFname'] = 'committee_evalSc_%s.json' % (params['fappend'])
    com_dataset['com_evaluation'] = params['comResFname']
    pickle.dump(eval_results, open(params['comResFname'], "wb"))
    json.dump(com_dataset,open(params['jsonFname'], 'w'))

    return eval_array


def main(params):
    dataset = 'coco'
    data_file = 'dataset.json'
    
    # !assumptions on folder structure
    dataset_root = os.path.join('data', dataset)
    
    result_list = open(params['struct_list'], 'r').read().splitlines()

    # Load all result files
    result_struct = [json.load(open(res,'r')) for res in result_list]
    
    # load the dataset into memory
    dataset_path = os.path.join(dataset_root, data_file)
    print 'BasicDataProvider: reading %s' % (dataset_path, )
    dB = json.load(open(dataset_path, 'r'))
    
    res_idx = 0

    com_dataset = {}
    com_dataset['dataset'] = 'coco';
    com_dataset['members_results'] = result_list;
    com_dataset['members_model'] = list(set([res['params']['checkpoint_path'] for res in result_struct]));
    com_dataset['images'] = [] 
    com_dataset['n_memb'] = len(com_dataset['members_model'])
    com_dataset['n_sent'] = len(com_dataset['members_results'])

    
    #pick only test images
    # We are doing this circus in order to reuse the data provider class to form nice batches when doing evaluation
    # The data provider expects the database files to be in original "dataset.json" format! 
    # Hence we copy all necessary fields from dataset.json and replace the refernce sentences with the sentences 
    # generated by our models
    for img in dB['images']:
        if img['split'] == 'test':
           # Copy everything!
           com_dataset['images'].append(img)

           # delete reference sentences 
           com_dataset['images'][-1]['sentences'] = []
           for res_st in result_struct:
                #assert img['imgid'] == res_st['imgblobs'][res_idx]['imgid'], 'Ids dont match, Test %d %d'%(res_idx, mod_cnt)
                com_dataset['images'][-1]['sentences'].append( {'img_id': img['imgid'],
                                                                'raw': res_st['imgblobs'][res_idx]['candidate']['text'],
                                                                'sentid':res_st['params']['beam_size'],
                                                                'mid':com_dataset['members_model'].index(res_st['params']['checkpoint_path']),
                                                                'tokens':word_tokenize(res_st['imgblobs'][res_idx]['candidate']['text'])
                                                                })
 
           res_idx += 1
           if res_idx == 5000:
            break;

    print 'Done with %d !Now writing back dataset ' % (res_idx)
    params['jsonFname'] = 'committee_struct_%s.json' % (params['fappend']) 
    params['jsonFname'] = os.path.join(dataset_root, params['jsonFname']) 
    json.dump(com_dataset,open(params['jsonFname'], 'w'))
    return com_dataset, params

    

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('struct_list', type=str, help='the input list of result structures to form committee from')
  parser.add_argument('--fappend', type=str, default='', help='str to append to routput files')
  parser.add_argument('--result_struct_filename', type=str, default='committee_result.json', help='filename of the result struct to save')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print 'parsed parameters:'
  print json.dumps(params, indent = 2)

  com_dataset, params = main(params)
  eval_array = hold_comittee_discussion(params,com_dataset) 
  #evaluate_decision(params, com_dataset, eval_array)
