import argparse
import json
import os
import random
import scipy.io
import codecs
import numpy as np
import cPickle as pickle
import nltk
from collections import defaultdict
from nltk.tokenize import word_tokenize
from imagernn.data_provider import getDataProvider
from imagernn.imagernn_utils import decodeGenerator, eval_split, eval_split_theano

from nltk.align.bleu import BLEU
import progressbar
import math
import operator


def main(params):
    checkpoint_path = params['checkpoint_path']

    print 'loading checkpoint %s' % (checkpoint_path, )
    checkpoint = pickle.load(open(checkpoint_path, 'rb'))
    checkpoint_params = checkpoint['params']
    dp = getDataProvider(checkpoint_params)

	bar = progressbar.ProgressBar(maxval=dp.getSplitSize('train'), \
	    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    
    wordtoix = checkpoint['wordtoix']
    tag_hist = {}
	cnt = 0
    bar.start()
    for sent in dp.iterSentences(split = 'train'):
        tag = nltk.pos_tag(sent['tokens']) 
        
        for w,t in tag:
         if w in tag_hist.keys() and t in tag_hist[w].keys():
             tag_hist[w][t] += 1
         elif w in tag_hist.keys():
             tag_hist[w][t] = 1
         else:
             tag_hist[w] = {}
             tag_hist[w][t] = 1
        
        cnt +=1
        if cnt % 500 == 1:
            bar.update(cnt)


    imp_words = {}
    word_analysis_data = {}
    for w in tag_hist.iterkeys():
        if wordtoix.has_key(w):
            imp_words[w] = {}    
            imp_words[w]['cnt'] = sum(tag_hist[w].values())
            imp_words[w]['tag_hist'] = tag_hist[w]
            imp_words[w]['tag'] = max(tag_hist[w].iteritems(),key=operator.itemgetter(1))[0]

    word_analysis_data['all_tags'] = tag_hist
    word_analysis_data['imp_words'] = imp_words
    
    nn_list= []
    nn_cnts = {}
    nn_cnts['NN'] = 0
    nn_cnts['NNP'] = 0
    nn_cnts['NNPS'] = 0
    nn_cnts['NNS'] = 0
    for w in imp_words.iterkeys():
        if imp_words[w]['tag'][:2] == 'NN':
            nn_list.append(w)
            nn_cnts[imp_words[w]['tag']] +=1


    json.dump(word_analysis_data, open('word_analysis_data_coco.json','w'))


    
    

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('struct_list', type=str, help='the input list of result structures to form committee from')
  parser.add_argument('--fappend', type=str, default='', help='str to append to routput files')
  parser.add_argument('--result_struct_filename', type=str, default='committee_result.json', help='filename of the result struct to save')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print 'parsed parameters:'
  print json.dumps(params, indent = 2)

  main(params)
  #evaluate_decision(params, com_dataset, eval_array)
