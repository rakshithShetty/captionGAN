import argparse
import json
import time
import datetime
import numpy as np
import code
import socket
import os
import os.path as osp
import theano
from theano import config
import theano.tensor as tensor
import cPickle as pickle
from imagernn.data_provider import getDataProvider, prepare_data, loadArbitraryFeatures, prepare_seq_features
from imagernn.solver import Solver
from imagernn.imagernn_utils import decodeEvaluator 
from imagernn.cnn_evaluatorTheano import CnnEvaluator
from imagernn.utils import numpy_floatX, zipp, unzip, preProBuildWordVocab
from imagernn.recurrent_feat_encoder import RecurrentFeatEncoder
from collections import defaultdict
import ast

def mergeRes(params):
  
  blb = {}
  if len(params['res_list']) == 1:
    model_list_f = open(params['res_list'][0], 'r').read().splitlines()
  else:
    model_list_f = params['res_list']

  mdlNames = []
  mdlLabels = []
  for fnms in model_list_f:
    if fnms[0] =='#':
        continue

    model_name = fnms.split(',')[0]
    model_lbl = fnms.split(',')[-1] 
    mdlNames.append(model_name)
    mdlLabels.append(model_lbl)
    print 'Now loading %s'%model_name
    res = json.load(open(model_name,'r'))

    for r in res['imgblobs']:
        imgid = osp.basename(r['img_path']).split('_')[-1].split('.')[0]
        if blb.get(imgid,[]) == []:
            blb[imgid] = {'candidatelist':[r['candidate']]}
            blb[imgid]['img_path'] = r['img_path']
        else:
            blb[imgid]['candidatelist'].append(r['candidate'])

  resM = {}
  resM['params'] = res['params']
  resM['cp_params'] = res['checkpoint_params']
  resM['imgblobs'] = blb.values()
  resM['mdlNames'] = mdlNames
  resM['lbls'] = mdlLabels
  
  return resM

#######################################################################################################
def main(params):
  checkpoint_path = params['checkpoint_path']
  print 'loading checkpoint %s' % (checkpoint_path, )
  checkpoint = pickle.load(open(checkpoint_path, 'rb'))
  cp_params = checkpoint['params']
  model_npy = checkpoint['model']
  
  # Load the candidates db generated from rnn's
  if params['candDb'] != None:
    candDb = json.load(open(params['candDb'],'r'))
  else:
    candDb = mergeRes(params)

  wordtoix = checkpoint['wordtoix'] if 'wordtoix' in checkpoint else checkpoint['misc']['wordtoix']

  # Read labels and build cocoid to imgid Map
  if params['dataset'] == 'coco':
    lbls = open(params['lblF'],'r').read().splitlines()
    objId2Imgid = {}
    for lb in lbls:
        objId2Imgid[str(int(lb.split()[1][1:-1]))] = int(lb.split()[0][1:])
    features, aux_inp, feat_idx, aux_idx = loadArbitraryFeatures(params, Ellipsis)

  elif params['dataset'] == 'msr-vtt':
    img_names_list = open(params['lblF'], 'r').read().splitlines()
    auxidxes = []
    img_names = [x.rsplit(',')[0] for x in img_names_list]
    objId2Imgid = {imn.split('.')[0]:i for i,imn in enumerate(img_names)}
    if len(img_names_list[0].split(',',1)) > 1:
      if type(ast.literal_eval(img_names_list[0].split(',',1)[1].strip())) == tuple:
        idxes = [ast.literal_eval(x.split(',',1)[1].strip())[0] for x in img_names_list]
        auxidxes = [ast.literal_eval(x.split(',',1)[1].strip())[1] for x in img_names_list]
      else: 
        idxes = [ast.literal_eval(x.split(',',1)[1].strip()) for x in img_names_list]
    else:
      idxes = xrange(len(img_names_list))
    params['poolmethod'] = cp_params['poolmethod'] if params['poolmethod'] == None else params['poolmethod']
    features, aux_inp, feat_idx, aux_idx = loadArbitraryFeatures(params, idxes, auxidxes=auxidxes)

  elif params['dataset'] == 'lsmdc':
    if params['use_label_file'] == 1:
        params['poolmethod'] = cp_params['poolmethod'] if params['poolmethod'] == None else params['poolmethod']
        params['labels'] = cp_params['labels'] if params['labels'] == None else params['labels']
        params['featfromlbl'] = cp_params['featfromlbl'] if params['featfromlbl'] == None else params['featfromlbl']
        params['uselabel'] = cp_params['uselabel'] if params['uselabel'] == None else params['uselabel']
    else:
        params['uselabel'] = 0
    img_names_list = open(params['lblF'], 'r').read().splitlines()
    img_names = [x.rsplit(',')[0] for x in img_names_list]
    idxes = [int(x.rsplit(',')[1]) for x in img_names_list]
    auxidxes = []
    objId2Imgid = {osp.basename(imn).split('.')[0]:i for i,imn in enumerate(img_names)}
    
    #import pdb;pdb.set_trace()
    features, aux_inp, feat_idx, aux_idx = loadArbitraryFeatures(params, idxes, auxidxes=auxidxes)

  if cp_params.get('use_encoder_for',0)&1:
    imgFeatEncoder = RecurrentFeatEncoder(cp_params['image_feat_size'], cp_params['sent_encoding_size'],
            cp_params, mdl_prefix='img_enc_', features=features.T)
    zipp(model_npy, imgFeatEncoder.model_th)
    (imgenc_use_dropout, imgFeatEnc_inp, xI, updatesLSTMImgFeat) = imgFeatEncoder.build_model(imgFeatEncoder.model_th, cp_params)
  else:
    xI = None
    imgFeatEnc_inp = []

  if 'eval_model' not in cp_params:
    cp_params['eval_model'] = params['eval_model'] 
    print 'Using evaluator module: ', cp_params['eval_model']
  

  #find the number of candidates per image and max sentence len
  batch_size = 0
  maxlen = 0
  for i,img in enumerate(candDb['imgblobs']):
    for ids,cand in enumerate(img['candidatelist']):
        tks = cand['text'].split(' ')
        # Also tokenize the candidates
        candDb['imgblobs'][i]['candidatelist'][ids]['tokens'] = tks
        if len(tks) > maxlen:
            maxlen = len(tks)
    if batch_size < len(img['candidatelist']):
        batch_size = len(img['candidatelist'])

  # Get all images to this batch size!
  # HACK!!
  maxlen = 24
  cp_params['maxlen'] = maxlen
 
  cp_params['batch_size'] = batch_size
  print maxlen

  # go over all training sentences and find the vocabulary we want to use, i.e. the words that occur
  # at least word_count_threshold number of times
  
  # This initializes the model parameters and does matrix initializations  
  cp_params['mode'] = 'predict' 
  evalModel = decodeEvaluator(cp_params)
  model = evalModel.model_th
  
  # Define the computational graph for relating the input image features and word indices to the
  # log probability cost funtion. 
  (use_dropout, inp_list_eval,
     f_pred_fns, cost, predTh, modelUpd) = evalModel.build_model(model, cp_params, xI=xI,
                     prior_inp_list = imgFeatEnc_inp)
  
  inp_list = imgFeatEnc_inp + inp_list_eval

  # Add the regularization cost. Since this is specific to trainig and doesn't get included when we 
  # evaluate the cost on test or validation data, we leave it here outside the model definition

  # Now let's build a gradient computation graph and rmsprop update mechanism
  # calculate how many iterations we need, One epoch is considered once going through all the sentences and not images
  # Hence in case of coco/flickr this will 5* no of images
  ## Initialize the model parameters from the checkpoint file if we are resuming training
  model = modelUpd if cp_params['eval_model']=='cnn' else model 
  zipp(model_npy,model)
  print("\nPredicting using model %s, run for %0.2f epochs with validation perplx at %0.3f\n" % (checkpoint_path, checkpoint['epoch'], \
    checkpoint['perplexity']))
  
  pos_samp = np.arange(1,dtype=np.int32) if cp_params['eval_model']=='cnn' else []
  

  #Disable using dropout in training 
  use_dropout.set_value(0.)
  if cp_params.get('use_encoder_for',0)&1:
      imgenc_use_dropout.set_value(0.)
  N = len(candDb['imgblobs'])
  stats = np.zeros((batch_size))
  #################### Main Loop ############################################
  for i,img in enumerate(candDb['imgblobs']):
    # fetch a batch of data
    print 'image %d/%d  \r' % (i, N),
    batch = []
    cbatch_len  = len(img['candidatelist'])
    objid = osp.basename(img['img_path']).split('_')[-1].split('.')[0]
    if params['dataset'] == 'coco':
        objid = str(int(objid))
              
    for s in img['candidatelist']:
        batch.append({'sentence':s, 'image':{'feat':features[:, feat_idx[objId2Imgid[objid]]].T, 'img_idx':feat_idx[objId2Imgid[objid]]}})
        if params['aux_inp_file'] != None:
            batch[-1]['aux_inp'] = aux_inp[:, aux_idx[objId2Imgid[objid]]].T

    if cbatch_len < batch_size and (cp_params['eval_model']=='cnn'):
        for z in xrange(batch_size - cbatch_len):
            batch.append({'sentence':img['candidatelist'][-1]})


    enc_inp_list = prepare_seq_features(batch, use_enc_for= cp_params.get('use_encoder_for',0),
          use_shared_mem = cp_params.get('use_shared_mem_enc',0),pos_samp=pos_samp)
    eval_inp_list, lenS = prepare_data(batch, wordtoix, maxlen=maxlen, pos_samp=pos_samp, prep_for=cp_params['eval_model'], use_enc_for= cp_params.get('use_encoder_for',0))

    real_inp_list = enc_inp_list + eval_inp_list
    
    #import pdb;pdb.set_trace()
    # evaluate cost, gradient and perform parameter update
    scrs = np.squeeze(f_pred_fns[1](*real_inp_list))
    scrs = scrs[:cbatch_len] # + scrs[:,cbatch_len:].sum()/cbatch_len
    for si,s in enumerate(img['candidatelist']):
        candDb['imgblobs'][i]['candidatelist'][si]['logprob'] = float(scrs[si])
        candDb['imgblobs'][i]['candidatelist'][si].pop('tokens')
    bestcand = scrs.argmax()
    stats[bestcand] += 1.0
    candDb['imgblobs'][i]['candidate'] = candDb['imgblobs'][i]['candidatelist'][bestcand]
    srtidx = np.argsort(scrs)[::-1]
    candDb['imgblobs'][i]['candsort'] = list(srtidx)
    # print training statistics

  print ""
  jsonFname = '%s_reranked_%s.json' % (cp_params['eval_model'],params['fname_append'])
  save_file = os.path.join(params['root_path'], jsonFname)
  json.dump(candDb, open(save_file, 'w'))
  print 'Written to file %s'%save_file
  print 'Final stats are:'
  print stats*100.0/N


if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # global setup settings, and checkpoints
  parser.add_argument('checkpoint_path', type=str, help='the input checkpoint of cnn evaluator')
  parser.add_argument('-c','--candDb', type=str, default=None, help='the candidate result file')
  parser.add_argument('--res_list',dest='res_list', nargs='+',type=str, default=[], help='List of candidates to merge')
  parser.add_argument('-f', '--feat_file', type=str, default='vgg_feats.mat', help='file with the features. We can rightnow process only .mat format')
  parser.add_argument('--aux_inp_file', dest='aux_inp_file', type=str, default=None, help='Is there any auxillary inputs ? If yes indicate file here')
  parser.add_argument('-d', '--dest', dest='root_path', default='example_images', type=str, help='folder to store the output files')
  parser.add_argument('-l', '--lblF', type=str, default='data/coco/labels.txt', help='file with the features. We can rightnow process only .mat format')
  parser.add_argument('--fname_append', type=str, default='', help='str to append to routput files')
  parser.add_argument('--dataset', dest='dataset', type=str, default='coco', help='Which dataset do these belong to')
  parser.add_argument('--eval_model', dest='eval_model', type=str, default='cnn', help='Which dataset do these belong to')

  parser.add_argument('--poolmethod', dest='poolmethod', type=str, default=None, help='What pooling to use if multiple features are found')
  parser.add_argument('--use_label_file', dest='use_label_file', type=int, default=0, help='Just use the labels file to get the feature idxes')
  parser.add_argument('--uselabel', dest='uselabel', type=int, default=None, help='Just use the labels file to get the feature idxes')
  parser.add_argument('--featfromlbl', dest='featfromlbl', type=str, default=None, help='should we use lables.txt, if yes which feature?'
                  'use space to seperately specify for img and aux')
  parser.add_argument('--labels', dest='labels', type=str, default=None, help='labels.txt file for this dataset')
  # Some parameters about image features used
  # model parameters
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  print 'parsed parameters:'
  print json.dumps(params, indent = 2)
  config.mode = 'FAST_RUN'
  #config.profile = True
  #config.allow_gc = False
  main(params)
