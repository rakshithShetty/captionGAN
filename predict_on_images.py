import argparse
import json
import time
import datetime
import numpy as np
import code
import os
import cPickle as pickle
import math
import scipy.io
import ast

from imagernn.solver import Solver
from imagernn.imagernn_utils import decodeGenerator, eval_split
from imagernn.data_provider import prepare_data, loadArbitraryFeatures, prepare_seq_features
from picsom_bin_data import picsom_bin_data
from imagernn.utils import zipp
from imagernn.recurrent_feat_encoder import RecurrentFeatEncoder
from copy import copy

"""
This script is used to predict sentences for arbitrary images
"""

def rescoreProbByLen(preds):
  pred_out = []
  newScores = [pC[0] / len([ix for ix in pC[1] if ix > 0]) for pC in preds]
  srtidx = np.argsort(newScores)

  for s in reversed(srtidx):
    pred_out.append([preds[s][0],preds[s][1], newScores[s]])

  return pred_out

def main(params):

  # load the checkpoint
  checkpoint_path = params['checkpoint_path']
  print 'loading checkpoint %s' % (checkpoint_path, )
  checkpoint = pickle.load(open(checkpoint_path, 'rb'))
  cp_params = checkpoint['params']

  if params['gen_model'] == None:
      model_npy = checkpoint['model'] if 'model' in checkpoint else checkpoint['modelGen']
  else:
      gen_cp = pickle.load(open(params['gen_model'], 'rb'))
      model_npy = gen_cp.get('model',{})

  cp_params['use_theano'] = 1
  if params['dobeamsearch']:
      cp_params['advers_gen'] = 0

  if params['use_label_file'] == 1:
      params['poolmethod'] = cp_params['poolmethod'] if params['poolmethod'] == None else params['poolmethod']
      params['labels'] = cp_params['labels'] if params['labels'] == None else params['labels']
      params['featfromlbl'] = cp_params['featfromlbl'] if params['featfromlbl'] == None else params['featfromlbl']
      params['uselabel'] = cp_params['uselabel'] if params['uselabel'] == None else params['uselabel']
  else:
      params['uselabel'] = 0
  print 'parsed parameters:'
  print json.dumps(params, indent = 2)

  if 'image_feat_size' not in  cp_params:
      cp_params['image_feat_size'] = 4096

  if 'misc' in checkpoint:
    misc = checkpoint['misc']
    ixtoword = misc['ixtoword']
  else:
    misc = {}
    ixtoword = checkpoint['ixtoword']
    misc['wordtoix'] = checkpoint['wordtoix']

  cp_params['softmax_smooth_factor'] = params['softmax_smooth_factor']
  cp_params['softmax_propogate'] = params['softmax_propogate']
  cp_params['computelogprob'] = params['computelogprob']
  cp_params['greedy'] = params['greedy']
  cp_params['gen_input_noise'] = 0

  if cp_params.get('sched_sampling_mode',None) !=None:
      cp_params['sched_sampling_mode'] = None

  # load the tasks.txt file and setupe feature loading
  root_path = params['root_path']

  img_names_list = open(params['imgList'], 'r').read().splitlines()
  auxidxes = []

  img_names = [x.rsplit(',')[0] for x in img_names_list]

  if len(img_names_list[0].split(',',1)) > 1:
    if type(ast.literal_eval(img_names_list[0].split(',',1)[1].strip())) == tuple:
      idxes = [ast.literal_eval(x.split(',',1)[1].strip())[0] for x in img_names_list]
      auxidxes = [ast.literal_eval(x.split(',',1)[1].strip())[1] for x in img_names_list]
    else:
      idxes = [ast.literal_eval(x.split(',',1)[1].strip()) for x in img_names_list]
  else:
    idxes = xrange(len(img_names_list))

  if cp_params.get('swap_aux') == 0 or auxidxes == []:
    features, aux_inp, feat_idx, aux_idx = loadArbitraryFeatures(params, idxes, auxidxes=auxidxes)
  else:
    features, aux_inp, feat_idx, aux_idx = loadArbitraryFeatures(params, auxidxes, auxidxes=idxes)

  ##-------------------------------- Setup the models --------------------------###########
  if cp_params.get('use_encoder_for',0)&1:
    imgFeatEncoder = RecurrentFeatEncoder(cp_params['image_feat_size'], cp_params['word_encoding_size'],
            cp_params, mdl_prefix='img_enc_', features=features.T)

    zipp(model_npy, imgFeatEncoder.model_th)
    (imgenc_use_dropout, imgFeatEnc_inp, xI, updatesLSTMImgFeat) = imgFeatEncoder.build_model(imgFeatEncoder.model_th, cp_params)
  else:
    xI = None
    imgFeatEnc_inp = []

  if cp_params.get('use_encoder_for',0)&2:
    auxFeatEncoder = RecurrentFeatEncoder(cp_params['aux_inp_size'], cp_params['image_encoding_size'], cp_params,
            mdl_prefix='aux_enc_', features=aux_inp.T)
    zipp(model_npy, auxFeatEncoder.model_th)
    (auxenc_use_dropout, auxFeatEnc_inp, xAux, updatesLSTMAuxFeat) = auxFeatEncoder.build_model(auxFeatEncoder.model_th, cp_params)
  else:
    auxFeatEnc_inp = []
    xAux = None

  # Testing to see if diversity can be achieved by weighing words
  if params['word_freq_w'] != None:
      w_freq = json.load(open(params['word_freq_w'],'r'))
      w_logw = np.zeros(len(misc['wordtoix']),dtype=np.float32)
      for w in w_freq:
          if w in misc['wordtoix']:
              w_logw[misc['wordtoix'][w]] = w_freq[w]
      w_logw = w_logw/w_logw[1:].min()
      w_logw[0] = w_logw.max()
      w_logw = -params['word_freq_sc'] * np.log(w_logw)
  else:
      w_logw = None

  BatchGenerator = decodeGenerator(cp_params)
  # Compile and init the theano predictor
  BatchGenerator.prepPredictor(model_npy, cp_params, params['beam_size'], xI, xAux, imgFeatEnc_inp + auxFeatEnc_inp, per_word_logweight=w_logw)
  model = BatchGenerator.model_th
  if params['greedy']:
    BatchGenerator.usegumbel.set_value(0)

  # output blob which we will dump to JSON for visualizing the results
  blob = {}
  blob['params'] = params
  blob['checkpoint_params'] = copy(cp_params)
  if cp_params.get('class_out_factoring',0) == 1:
    blob['checkpoint_params'].pop('ixtoclsinfo')
  blob['imgblobs'] = []


  N = len(img_names)

  # iterate over all images and predict sentences
  print("\nUsing model run for %0.2f epochs with validation perplx at %0.3f\n" % (checkpoint['epoch'], \
    checkpoint['perplexity']))

  kwparams = {}

  jsonFname = 'result_struct_%s.json' % (params['fname_append'] )
  save_file = os.path.join(root_path, jsonFname)

  for n in xrange(N):
    print 'image %d/%d:' % (n, N)

    # encode the image
    D,NN = features.shape
    img = {}
    img['feat'] = features[:, feat_idx[n]].T
    img['img_idx'] = feat_idx[n]
    if cp_params.get('en_aux_inp',0):
        img['aux_inp'] = aux_inp(aux_idx[n]) if aux_inp != [] else np.zeros(cp_params['aux_inp_size'], dtype=np.float32)
        img['aux_idx'] = aux_idx[n] if aux_inp != [] else []
    img['local_file_path'] =img_names[n]
    # perform the work. heavy lifting happens inside
    enc_inp_list = prepare_seq_features([{'image':img}], use_enc_for= cp_params.get('use_encoder_for',0),
          use_shared_mem = cp_params.get('use_shared_mem_enc',0))
    #import pdb;pdb.set_trace()
    Ys, Ax = BatchGenerator.predict([{'image':img}], cp_params, ext_inp = enc_inp_list)

    # build up the output
    img_blob = {}
    img_blob['img_path'] = img['local_file_path']

    # encode the top prediction
    top_predictions = Ys[0] if params['rescoreByLen']==0 else rescoreProbByLen(Ys[0])# take predictions for the first (and only) image we passed in
    top_predictions = sorted(top_predictions,key=lambda aa: aa[0],reverse=True)

    top_prediction = top_predictions[0]  # these are sorted with highest on top
    if cp_params.get('reverse_sentence',0) == 0:
        candidate = ' '.join([ixtoword[int(ix)] for ix in top_prediction[1] if ix > 0]) # ix 0 is the END token, skip that
    else:
        candidate = ' '.join([ixtoword[int(ix)] for ix in reversed(top_prediction[1]) if ix > 0]) # ix 0 is the END token, skip that
    #if candidate == '':
    #    import pdb;pdb.set_trace()
    if params['rescoreByLen']==0:
        print 'PRED: (%f) %s' % (float(top_prediction[0]), candidate)
    else:
        print 'PRED: (%f, %f) %s' % (float(top_prediction[0]), float(top_prediction[2]), candidate)
    img_blob['candidate'] = {'text': candidate, 'logprob': float(top_prediction[0])}

    # Code to save all the other candidates
    candlist = []
    for ci in xrange(len(top_predictions)-1):
        prediction = top_predictions[ci+1] # these are sorted with highest on top
        candidate = ' '.join([ixtoword[int(ix)] for ix in prediction[1] if ix > 0]) # ix 0 is the END token, skip that
        candlist.append({'text': candidate, 'logprob': float(prediction[0])})

    img_blob['candidatelist'] = candlist
    blob['imgblobs'].append(img_blob)
    if (n%5000) == 1:
        print 'writing predictions to %s...' % (save_file, )
        json.dump(blob, open(save_file, 'w'))

  # dump result struct to file
  print 'writing predictions to %s...' % (save_file, )
  json.dump(blob, open(save_file, 'w'))

  # dump output html
  #html = ''
  #for img in blob['imgblobs']:
  #  html += '<img src="%s" height="400"><br>' % (img['img_path'], )
  #  html += '(%f) %s <br><br>' % (img['candidate']['logprob'], img['candidate']['text'])

  #html_file = 'result_%s.html' % (params['fname_append'])
  #html_file = os.path.join(root_path, html_file)
  #print 'writing html result file to %s...' % (html_file, )
  #open(html_file, 'w').write(html)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('checkpoint_path', type=str, help='the input checkpoint')
  parser.add_argument('-g', dest='gen_model', type = str, default=None,help='dataset: flickr8k/flickr30k')
  parser.add_argument('-i', '--imgList', type=str, default='testimgs.txt', help='file with the list of images to process. Either just filenames or in <filename, index> format')
  parser.add_argument('-f', '--feat_file', type=str, default='vgg_feats.mat', help='file with the features. We can rightnow process only .mat format')
  parser.add_argument('-d', '--dest', dest='root_path', default='example_images', type=str, help='folder to store the output files')
  parser.add_argument('-b', '--beam_size', type=int, default=1, help='beam size in inference. 1 indicates greedy per-word max procedure. Good value is approx 20 or so, and more = better.')
  parser.add_argument('--fname_append', type=str, default='', help='str to append to routput files')
  parser.add_argument('--aux_inp_file', dest='aux_inp_file', type=str, default=None, help='Is there any auxillary inputs ? If yes indicate file here')
  parser.add_argument('--softmax_smooth_factor', dest='softmax_smooth_factor', type=float, default=1.0, help='Is there any auxillary inputs ? If yes indicate file here')
  parser.add_argument('--softmax_propogate', dest='softmax_propogate', type=int, default=0, help='Is there any auxillary inputs ? If yes indicate file here')
  parser.add_argument('--rescoreByLen', dest='rescoreByLen', type=int, default=0, help='Is there any auxillary inputs ? If yes indicate file here')
  parser.add_argument('--mat_new_ver', dest='mat_new_ver', type=int, default=1, help='If the .mat feature files are saved with new version (compressed) set this flag to 1')
  # parameters for loading multiple features per video using labels.txt
  parser.add_argument('--greedy', dest='greedy', type=int, default=0, help='Use greedy samples or samples from gumbel')
  parser.add_argument('--computelogprob', dest='computelogprob', type=int, default=0, help='Compute exact logprob')
  parser.add_argument('--keepN', dest='keepN', type=int, default=-1, help='keep top n sentence samples from the model')
  parser.add_argument('--dobeamsearch', dest='dobeamsearch', type=int, default=0, help='Use greedy samples or samples from gumbel')


  parser.add_argument('--use_label_file', dest='use_label_file', type=int, default=1, help='Just use the labels file to get the feature idxes')
  parser.add_argument('--poolmethod', dest='poolmethod', type=str, default=None, help='What pooling to use if multiple features are found')
  parser.add_argument('--uselabel', dest='uselabel', type=int, default=None, help='Just use the labels file to get the feature idxes')
  parser.add_argument('--featfromlbl', dest='featfromlbl', type=str, default=None, help='should we use lables.txt, if yes which feature?'
                  'use space to seperately specify for img and aux')
  parser.add_argument('--labels', dest='labels', type=str, default=None, help='labels.txt file for this dataset')

  # Some params to enable partial feature reading from disk!!
  parser.add_argument('--disk_feature', dest='disk_feature', type=int, default=0, help='Should img or aux features be read from disk')

  parser.add_argument('--word_freq_w', dest='word_freq_w', type = str, default=None, help='re-weigh the word frequencies')
  parser.add_argument('--word_freq_sc', dest='word_freq_sc', type = float, default=0.1, help='re-weigh the word frequencies')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  if params['aux_inp_file'] != None:
    params['en_aux_inp'] = 1
  else:
    params['en_aux_inp'] = 0

  main(params)
