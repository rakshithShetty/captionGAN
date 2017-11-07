import argparse
import json
import time
import datetime
import numpy as np
import code
import socket
import os
import theano
from theano import config
import theano.tensor as tensor
import cPickle as pickle
from imagernn.data_provider import getDataProvider, prepare_data, prepare_seq_features
from imagernn.solver import Solver
from imagernn.imagernn_utils import decodeGenerator, eval_split_theano, eval_prep_refs
#from numbapro import cuda
from imagernn.utils import numpy_floatX, zipp, unzip, preProBuildWordVocab
from imagernn.recurrent_feat_encoder import RecurrentFeatEncoder
from imagernn.attention_network import AttentionNetwork
from collections import defaultdict
import operator

def main(params):
  batch_size = params['batch_size']
  word_count_threshold = params['word_count_threshold']
  max_epochs = params['max_epochs']
  host = socket.gethostname() # get computer hostname

  #--------------------------------- Init data provider and load data+features #---------------------------------#
  # fetch the data provider
  dp = getDataProvider(params)

  params['aux_inp_size'] = params['featenc_hidden_size'] * params['n_encgt_sent'] if params['encode_gt_sentences'] else dp.aux_inp_size
  params['featenc_hidden_size'] = params['featenc_hidden_size'] if params['encode_gt_sentences'] else params['aux_inp_size']

  params['image_feat_size'] = dp.img_feat_size
  print 'Image feature size is %d, and aux input size is %d'%(params['image_feat_size'],params['aux_inp_size'])

  #--------------------------------- Preprocess sentences and build Vocabulary #---------------------------------#
  misc = {} # stores various misc items that need to be passed around the framework
  # go over all training sentences and find the vocabulary we want to use, i.e. the words that occur
  # at least word_count_threshold number of times
  if params['checkpoint_file_name'] == 'None':
    if params['class_out_factoring'] == 0:
        misc['wordtoix'], misc['ixtoword'], bias_init_vector = preProBuildWordVocab(dp.iterSentences('train'),
                                      word_count_threshold)
    else:
        [misc['wordtoix'], misc['classes']], [misc['ixtoword'],  misc['clstotree'], misc['ixtoclsinfo']], [bias_init_vector,
            bias_init_inter_class] = preProBuildWordVocab(dp.iterSentences('train'), word_count_threshold, params)
        params['nClasses'] = bias_init_inter_class.shape[0]
        params['ixtoclsinfo'] = misc['ixtoclsinfo']
  else:
      misc = checkpoint_init['misc']
      params['nClasses'] =  checkpoint_init['params']['nClasses']
      if 'ixtoclsinfo' in misc:
        params['ixtoclsinfo'] = misc['ixtoclsinfo']


  params['vocabulary_size'] = len(misc['wordtoix'])
  params['output_size'] = len(misc['ixtoword']) # these should match though
  print len(misc['wordtoix']),len(misc['ixtoword'])

  #------------------------------ Initialize the solver/generator and build forward path #-----------------------#
  # Initialize the optimizer
  solver = Solver(params['solver'])
  # This initializes the model parameters and does matrix initializations
  lstmGenerator = decodeGenerator(params)
  model, misc['update'], misc['regularize'] = (lstmGenerator.model_th, lstmGenerator.update_list, lstmGenerator.regularize)

  # force overwrite here. The bias to the softmax is initialized to reflect word frequencies
  # This is a bit of a hack
  if params['checkpoint_file_name'] == 'None':
    model['bd'].set_value(bias_init_vector.astype(config.floatX))
    if params['class_out_factoring'] == 1:
      model['bdCls'].set_value(bias_init_inter_class.astype(config.floatX))

  #----------------- If we are using feature encoders -----------------------
  # This mode can now also be used for encoding GT sentences.
  if params['use_encoder_for']&1:
    if params['encode_gt_sentences']:
        xI = tensor.zeros((batch_size,params['image_encoding_size']))
        imgFeatEnc_inp = []
    else:
        imgFeatEncoder = RecurrentFeatEncoder(params['image_feat_size'], params['word_encoding_size'],
                params, mdl_prefix='img_enc_', features=dp.features.T)
        mdlLen = len(model.keys())
        model.update(imgFeatEncoder.model_th)
        assert(len(model.keys()) == (mdlLen+len(imgFeatEncoder.model_th.keys())))
        misc['update'].extend(imgFeatEncoder.update_list)
        misc['regularize'].extend(imgFeatEncoder.regularize)
        (imgenc_use_dropout, imgFeatEnc_inp, xI, updatesLSTMImgFeat) = imgFeatEncoder.build_model(model, params)
  else:
    xI = None
    imgFeatEnc_inp = []

  if params['use_encoder_for']&2:
    aux_enc_inp = model['Wemb'] if params['encode_gt_sentences'] else dp.aux_inputs.T
    hid_size = params['featenc_hidden_size']
    auxFeatEncoder = RecurrentFeatEncoder(hid_size, params['image_encoding_size'], params,
            mdl_prefix='aux_enc_', features=aux_enc_inp)
    mdlLen = len(model.keys())
    model.update(auxFeatEncoder.model_th)
    assert(len(model.keys()) == (mdlLen+len(auxFeatEncoder.model_th.keys())))
    misc['update'].extend(auxFeatEncoder.update_list)
    misc['regularize'].extend(auxFeatEncoder.regularize)
    (auxenc_use_dropout, auxFeatEnc_inp, xAux, updatesLSTMAuxFeat) = auxFeatEncoder.build_model(model, params)

    if params['encode_gt_sentences']:
        # Reshape it size(batch_size, n_gt, hidden_size)
        xAux = xAux.reshape((-1,params['n_encgt_sent'],params['featenc_hidden_size']))
        # Convert it to size (batch_size, n_gt*hidden_size
        xAux = xAux.flatten(2)

  else:
    auxFeatEnc_inp = []
    xAux = None

  #--------------------------------- Initialize the Attention Network #-------------------------------#
  if params['use_attn'] != None:
    attnModel = AttentionNetwork(params['image_feat_size'], params['hidden_size'], params, mdl_prefix='attn_mlp_')
    mdlLen = len(model.keys())
    model.update(attnModel.model_th)
    assert(len(model.keys()) == (mdlLen+len(attnModel.model_th.keys())))
    misc['update'].extend(attnModel.update_list)
    misc['regularize'].extend(attnModel.regularize)
    attn_nw_func = attnModel.build_model
  else:
    attn_nw_func = None

  #--------------------------------- Build the language model graph #---------------------------------#
  # Define the computational graph for relating the input image features and word indices to the
  # log probability cost funtion.
  (use_dropout, inp_list_gen,
     f_pred_prob, cost, predTh, updatesLSTM) = lstmGenerator.build_model(model, params, xI, xAux, attn_nw = attn_nw_func)


  inp_list = imgFeatEnc_inp + auxFeatEnc_inp + inp_list_gen
  #--------------------------------- Cost function and gradient computations setup #---------------------------------#
  costGrad = cost[0]
  # Add class uncertainity to final cost
  #if params['class_out_factoring'] == 1:
  #  costGrad += cost[2]
  # Add the regularization cost. Since this is specific to trainig and doesn't get included when we
  # evaluate the cost on test or validation data, we leave it here outside the model definition
  if params['regc'] > 0.:
      reg_cost = theano.shared(numpy_floatX(0.), name='reg_c')
      reg_c = tensor.as_tensor_variable(numpy_floatX(params['regc']), name='reg_c')
      reg_cost = 0.
      for p in misc['regularize']:
        reg_cost += (model[p] ** 2).sum()
        reg_cost *= 0.5 * reg_c
      costGrad += (reg_cost /params['batch_size'])

  # Compile an evaluation function.. Doesn't include gradients
  # To be used for validation set evaluation
  f_eval= theano.function(inp_list, cost, name='f_eval')

  # Now let's build a gradient computation graph and rmsprop update mechanism
  grads = tensor.grad(costGrad, wrt=model.values())
  lr = tensor.scalar(name='lr',dtype=config.floatX)
  f_grad_shared, f_update, zg, rg, ud = solver.build_solver_model(lr, model, grads,
                                      inp_list, cost, params)

  print 'model init done.'
  print 'model has keys: ' + ', '.join(model.keys())
  #print 'updating: ' + ', '.join( '%s [%dx%d]' % (k, model[k].shape[0], model[k].shape[1]) for k in misc['update'])
  #print 'updating: ' + ', '.join( '%s [%dx%d]' % (k, model[k].shape[0], model[k].shape[1]) for k in misc['regularize'])
  #print 'number of learnable parameters total: %d' % (sum(model[k].shape[0] * model[k].shape[1] for k in misc['update']), )


  #-------------------------------- Intialize the prediction path if needed by evaluator ----------------------------#
  evalKwargs = {'eval_metric': params['eval_metric'], 'f_gen': lstmGenerator.predict,
                'beamsize': params['eval_beamsize']}
  if params['eval_metric'] != 'perplex':
    lstmGenerator.prepPredictor(None, params, params['eval_beamsize'])
    refToks, scr_info = eval_prep_refs('val', dp, params['eval_metric'])
    evalKwargs['refToks'] = refToks
    evalKwargs['scr_info'] = scr_info
    valMetOp = operator.gt
  else:
    valMetOp = operator.lt

  if params['met_to_track'] != []:
    trackMetargs = {'eval_metric': params['met_to_track'], 'f_gen': lstmGenerator.predict,
                  'beamsize': params['eval_beamsize']}
    lstmGenerator.prepPredictor(None, params, params['eval_beamsize'])
    refToks, scr_info = eval_prep_refs('val', dp, params['met_to_track'])
    trackMetargs['refToks'] = refToks
    trackMetargs['scr_info'] = scr_info


  #--------------------------------- Iterations and Logging intializations ------------------------------------------#
  # calculate how many iterations we need, One epoch is considered once going through all the sentences and not images
  # Hence in case of coco/flickr this will 5* no of images
  num_sentences_total = dp.getSplitSize('train', ofwhat = 'sentences')
  num_iters_one_epoch = num_sentences_total / batch_size
  max_iters = max_epochs * num_iters_one_epoch
  eval_period_in_epochs = params['eval_period']
  eval_period_in_iters = max(1, int(num_iters_one_epoch * eval_period_in_epochs))
  top_val_sc = -1
  smooth_train_ppl2 = len(misc['ixtoword']) # initially size of dictionary of confusion
  val_sc = len(misc['ixtoword'])
  last_status_write_time = 0 # for writing worker job status reports
  json_worker_status = {}
  #json_worker_status['params'] = params
  json_worker_status['history'] = []
  len_hist = defaultdict(int)

  #Initialize Tracking the perplexity of train and val, with iters.
  train_perplex = []
  val_perplex = []
  trackSc_array = []

  #-------------------------------------- Load previously saved model ------------------------------------------------#
  #- Initialize the model parameters from the checkpoint file if we are resuming training
  if params['checkpoint_file_name'] != 'None':
    zipp(model_init_from,model)
    if params['restore_grads'] == 1:
        zipp(rg_init,rg)
    #Copy trackers from previous checkpoint
    if 'trackers' in checkpoint_init:
        train_perplex = checkpoint_init['trackers']['train_perplex']
        val_perplex = checkpoint_init['trackers']['val_perplex']
        trackSc_array = checkpoint_init['trackers'].get('trackScores',[])
    print("""\nContinuing training from previous model\n. Already run for %0.2f epochs with
            validation perplx at %0.3f\n""" % (checkpoint_init['epoch'], checkpoint_init['perplexity']))

  #--------------------------------------  MAIN LOOP ----------------------------------------------------------------#
  for it in xrange(max_iters):
      t0 = time.time()
      # Enable using dropout in training
      use_dropout.set_value(float(params['use_dropout']))
      if params['use_encoder_for']&1:
        imgenc_use_dropout.set_value(float(params['use_dropout']))
      if params['use_encoder_for']&2:
        auxenc_use_dropout.set_value(float(params['use_dropout']))

      epoch = it * 1.0 / num_iters_one_epoch
      #-------------------------------------- Prepare batch-------------------------------------------#
      # fetch a batch of data
      if params['sample_by_len'] == 0:
          batch = [dp.sampleImageSentencePair() for i in xrange(batch_size)]
      else:
          batch,l = dp.getRandBatchByLen(batch_size)
          len_hist[l] += 1

      enc_inp_list = prepare_seq_features( batch, use_enc_for= params['use_encoder_for'], maxlen =  params['maxlen'],
              use_shared_mem = params['use_shared_mem_enc'], enc_gt_sent = params['encode_gt_sentences'],
              n_enc_sent = params['n_encgt_sent'], wordtoix = misc['wordtoix'])

      if params['use_pos_tag'] != 'None':
          gen_inp_list, lenS = prepare_data(batch, misc['wordtoix'], params['maxlen'],
                      sentTagMap,misc['ixtoword'], rev_sents=params['reverse_sentence'],
                      use_enc_for= params['use_encoder_for'],
                      use_unk_token = params['use_unk_token'])
      else:
          gen_inp_list, lenS = prepare_data(batch, misc['wordtoix'], params['maxlen'],
                      rev_sents=params['reverse_sentence'],
                      use_enc_for= params['use_encoder_for'],
                      use_unk_token = params['use_unk_token'])

      if params['sched_sampling_mode'] !=None:
          gen_inp_list.append(epoch)

      real_inp_list = enc_inp_list + gen_inp_list

      #import ipdb; ipdb.set_trace()
      #---------------------------------- Compute cost and apply gradients ---------------------------#
      # evaluate cost, gradient and perform parameter update
      cost = f_grad_shared(*real_inp_list)
      f_update(params['learning_rate'])
      dt = time.time() - t0

      # print training statistics
      train_ppl2 = (2**(cost[1]/lenS)) #step_struct['stats']['ppl2']
      # smooth exponentially decaying moving average
      smooth_train_ppl2 = 0.99 * smooth_train_ppl2 + 0.01 * train_ppl2
      if it == 0: smooth_train_ppl2 = train_ppl2 # start out where we start out

      total_cost = cost[0]
      if it == 0: smooth_cost = total_cost # start out where we start out
      smooth_cost = 0.99 * smooth_cost + 0.01 * total_cost

      #print '%d/%d batch done in %.3fs. at epoch %.2f. loss cost = %f, reg cost = %f, ppl2 = %.2f (smooth %.2f)' \
      #      % (it, max_iters, dt, epoch, cost['loss_cost'], cost['reg_cost'], \
      #         train_ppl2, smooth_train_ppl2)

      #---------------------------------- Write a report into a json file ---------------------------#
      tnow = time.time()
      if tnow > last_status_write_time + 60*1: # every now and then lets write a report
        print '%d/%d batch done in %.3fs. at epoch %.2f. Cost now is %.3f and pplx is %.3f' \
                % (it, max_iters, dt, epoch, smooth_cost, smooth_train_ppl2)
        last_status_write_time = tnow
        jstatus = {}
        jstatus['time'] = datetime.datetime.now().isoformat()
        jstatus['iter'] = (it, max_iters)
        jstatus['epoch'] = (epoch, max_epochs)
        jstatus['time_per_batch'] = dt
        jstatus['smooth_train_ppl2'] = smooth_train_ppl2
        jstatus['val_sc'] = val_sc # just write the last available one
        jstatus['val_metric'] = params['eval_metric'] # just write the last available one
        jstatus['train_ppl2'] = train_ppl2
        #if params['class_out_factoring'] == 1:
        #  jstatus['class_cost'] = float(cost[2])
        json_worker_status['history'].append(jstatus)
        status_file = os.path.join(params['worker_status_output_directory'], host + '_status.json')
        #import pdb; pdb.set_trace()
        try:
          json.dump(json_worker_status, open(status_file, 'w'))
        except Exception, e: # todo be more clever here
          print 'tried to write worker status into %s but got error:' % (status_file, )
          print e

      #--------------------------------- VALIDATION ---------------------------#
      #- perform perplexity evaluation on the validation set and save a model checkpoint if it's good
      is_last_iter = (it+1) == max_iters
      if (((it+1) % eval_period_in_iters) == 0 and it < max_iters - 5) or is_last_iter:
        # Disable using dropout in validation
        use_dropout.set_value(0.)
        if params['use_encoder_for'] & 1:
          imgenc_use_dropout.set_value(0.)
        if params['use_encoder_for'] & 2:
          auxenc_use_dropout.set_value(0.)

        # perform the evaluation on VAL set
        val_sc = eval_split_theano('val', dp, model, params, misc, f_eval, **evalKwargs)
        val_sc = val_sc[0]
        val_perplex.append((it,val_sc))
        train_perplex.append((it,smooth_train_ppl2))

        if params['met_to_track'] != []:
            track_sc = eval_split_theano('val', dp, model, params, misc, f_eval, **trackMetargs)
            trackSc_array.append((it,{evm:track_sc[i] for i,evm in enumerate(params['met_to_track'])}))

        if epoch - params['lr_decay_st_epoch'] >= 0:
          params['learning_rate'] = params['learning_rate'] * params['lr_decay']
          params['lr_decay_st_epoch'] += 1

        print 'validation %s = %f, lr = %f' % (params['eval_metric'], val_sc, params['learning_rate'])
        #if params['sample_by_len'] == 1:
        #  print len_hist


        #----------------------------- SAVE THE MODEL -------------------#
        write_checkpoint_ppl_threshold = params['write_checkpoint_ppl_threshold']
        if valMetOp(val_sc, top_val_sc) or top_val_sc < 0:
          if valMetOp(val_sc, write_checkpoint_ppl_threshold) or write_checkpoint_ppl_threshold < 0:
            # if we beat a previous record or if this is the first time
            # AND we also beat the user-defined threshold or it doesnt exist
            top_val_sc = val_sc
            filename = 'model_checkpoint_%s_%s_%s_%s%.2f.p' % (params['dataset'], host,
                        params['fappend'],params['eval_metric'][:3], val_sc)
            filepath = os.path.join(params['checkpoint_output_directory'], filename)
            model_npy = unzip(model)
            rgrads_npy = unzip(rg)
            checkpoint = {}
            checkpoint['it'] = it
            checkpoint['epoch'] = epoch
            checkpoint['model'] = model_npy
            checkpoint['rgrads'] = rgrads_npy
            checkpoint['params'] = params
            checkpoint['perplexity'] = val_sc
            checkpoint['misc'] = misc
            checkpoint['trackers'] = {'train_perplex':train_perplex,
                                      'val_perplex':val_perplex,
                                      'trackScores':trackSc_array}
            try:
              pickle.dump(checkpoint, open(filepath, "wb"))
              print 'saved checkpoint in %s' % (filepath, )
            except Exception, e: # todo be more clever here
              print 'tried to write checkpoint into %s but got error: ' % (filepath, )
              print e

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # global setup settings, and checkpoints
  parser.add_argument('--use_theano', dest='use_theano', default=1, help='Should we use thano and gpu!?. Actually dont try with value 0 :-|')

  parser.add_argument('-d', '--dataset', dest='dataset', default='coco', help='dataset: flickr8k/flickr30k')
  parser.add_argument('--fappend', dest='fappend', type=str, default='baseline', help='append this string to checkpoint filenames')
  parser.add_argument('-o', '--checkpoint_output_directory', dest='checkpoint_output_directory', type=str, default='cv/', help='output directory to write checkpoints to')
  parser.add_argument('--worker_status_output_directory', dest='worker_status_output_directory', type=str, default='status/', help='directory to write worker status JSON blobs to')
  parser.add_argument('--write_checkpoint_ppl_threshold', dest='write_checkpoint_ppl_threshold', type=float, default=-1, help='ppl threshold above which we dont bother writing a checkpoint to save space')
  parser.add_argument('--continue_training', dest='checkpoint_file_name', type=str, default='None', help='checkpoint file from which to resume training')
  parser.add_argument('--restore_grads', dest='restore_grads', type=int, default=1, help='restore grads from checkpoint or not')
  parser.add_argument('--use_pos_tag', dest='use_pos_tag', type=str, default='None', help='use_pos_tag')

  # Some parameters about image features used
  parser.add_argument('--feature_file', dest='feature_file', type=str, default='vgg_feats.mat', help='Which file should we use for read the CNN features')
  parser.add_argument('--data_file', dest='data_file', type=str, default='dataset.json', help='Which dataset file shpuld we use')
  parser.add_argument('--mat_new_ver', dest='mat_new_ver', type=int, default=1, help='If the .mat feature files are saved with new version (compressed) set this flag to 1')
  parser.add_argument('--aux_inp_file', dest='aux_inp_file', type=str, default='None', help='Is there any auxillary inputs ? If yes indicate file here')
  parser.add_argument('--swap_AuxFeat', dest='swap_aux', type=int, default=1, help='Feed image features through auxillary input!')

  # parameters for loading multiple features per video using labels.txt
  parser.add_argument('--labelsFile', dest='labels', type=str, default='labels.txt', help='labels.txt file for this dataset')
  parser.add_argument('--featfromlbl', dest='featfromlbl', type=str, default='ALL ALL', help='should we use lables.txt, if yes which feature?'
                  'use + sign to seperately specify for img and aux')
  parser.add_argument('--poolmethod', dest='poolmethod', type=str, default='max', help='What pooling to use if multiple features are found')
  parser.add_argument('--uselabel', dest='uselabel', type=int, default=3, help='which features should use labels.txt, img/aux or both, 0 - None, 1 - img, 2 - aux, 3 - both')

  parser.add_argument('--use_partial_sent',dest='use_partial_sent', nargs='+',type=int, default=[], help='Indexes of sentences to use')
  parser.add_argument('--use_train_subset',dest='use_train_subset', nargs='+',type=int, default=[], help='Indexes of training subsets to use')

  # Parameters to enable class based factorization of lstm output
  parser.add_argument('--class_out_factoring', dest='class_out_factoring', type=int, default=0, help='Enable Class based output factorization in generator')
  parser.add_argument('--nClasses', dest='nClasses', type=int, default=200, help='Number of classes to use')
  parser.add_argument('--class_inp_file', dest='class_inp_file', type=str, default=None, help='If clustering is already done, provide the inp file')
  parser.add_argument('--clust_tool_dir', dest='clust_tool_dir', type=str, default=None, help='Needed if clustering needs to be triggered now.'
                                                                        ' Support currently is only for Percy Liang\'s Brown clustering implementation')
  parser.add_argument('--cls_zmean', dest='cls_zmean', type=int, default=0, help='cls_zmean to use')
  parser.add_argument('--cls_diff_layer', dest='cls_diff_layer', type=int, default=0, help='cls_diff_layer')
  # model parameters
  parser.add_argument('--image_encoding_size', dest='image_encoding_size', type=int, default=512, help='size of the image encoding')
  parser.add_argument('--word_encoding_size', dest='word_encoding_size', type=int, default=512, help='size of word encoding')
  parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=512, help='size of hidden layer in generator RNNs')
  parser.add_argument('--hidden_depth', dest='hidden_depth', type=int, default=1, help='depth of hidden layer in generator RNNs')
  parser.add_argument('--en_residual_conn', dest='en_residual_conn', type=int, default=0, help='depth of hidden layer in generator RNNs')
  parser.add_argument('--generator', dest='generator', type=str, default='lstm', help='generator to use')
  parser.add_argument('-c', '--regc', dest='regc', type=float, default=1e-8, help='regularization strength')
  parser.add_argument('--tanhC_version', dest='tanhC_version', type=int, default=0, help='use tanh version of LSTM?')

  # optimization parameters
  parser.add_argument('-m', '--max_epochs', dest='max_epochs', type=int, default=10, help='number of epochs to train for')
  parser.add_argument('--solver', dest='solver', type=str, default='rmsprop', help='solver types supported: rmsprop')
  parser.add_argument('--decay_rate', dest='decay_rate', type=float, default=0.999, help='decay rate for adadelta/rmsprop')
  parser.add_argument('--smooth_eps', dest='smooth_eps', type=float, default=1e-8, help='epsilon smoothing for rmsprop/adagrad/adadelta')
  parser.add_argument('-l', '--learning_rate', dest='learning_rate', type=float, default=1e-3, help='solver learning rate')
  parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=100, help='batch size')
  parser.add_argument('--grad_clip', dest='grad_clip', type=float, default=5, help='clip gradients (normalized by batch size)? elementwise. if positive, at what threshold?')
  parser.add_argument('--use_dropout', dest='use_dropout', type=int, default=1, help='enable or disable dropout')
  parser.add_argument('--drop_prob_encoder', dest='drop_prob_encoder', type=float, default=0.5, help='what dropout to apply right after the encoder to an RNN/LSTM')
  parser.add_argument('--drop_prob_decoder', dest='drop_prob_decoder', type=float, default=0.5, help='what dropout to apply right before the decoder in an RNN/LSTM')
  parser.add_argument('--drop_prob_aux', dest='drop_prob_aux', type=float, default=0.5, help='what dropout to apply for the auxillary inputs to lstm')

  # refere paper: "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks" http://arxiv.org/abs/1506.03099
  parser.add_argument('--sched_sampling_mode', dest='sched_sampling_mode', type=str, default=None, help='should we implement scheduled sampling during training')
  parser.add_argument('--sched_sampling_const', dest='sched_sampling_const', type=float, default=1.0, help='scheduling constant, exact nature depends on the mode')
  parser.add_argument('--sslin_slope', dest='sslin_slope', type=np.float, default=1.0, help='slope of decay in linear scheduling')
  parser.add_argument('--sslin_min', dest='sslin_min', type=np.float, default=1.0, help='min truth constant in linear scheduling')

  parser.add_argument('--sample_by_len', dest='sample_by_len', type=int, default=0, help='enable sampling by length of sentece to speed up training')
  parser.add_argument('--maxlen', dest='maxlen', type=int, default=None, help='enable sampling by length of sentece to speed up training')

  parser.add_argument('--lr_decay', dest='lr_decay', type=float, default=1.0, help='decay factor for learning rate, applied every epoch')
  parser.add_argument('--lr_decay_st_epoch', dest='lr_decay_st_epoch', type=float, default=100.0, help='from which epoch should the lr decay start')

  # data preprocessing parameters
  parser.add_argument('--word_count_threshold', dest='word_count_threshold', type=int, default=5, help="""if a word occurs less than this number
                                                  of times in training data, it is discarded""")
  parser.add_argument('--use_unk_token', dest='use_unk_token', type=int, default=0, help=' replace unknows with UNK token')
  parser.add_argument('--reverse_sentence', dest='reverse_sentence', type=int, default=0, help='Should we reverse the sentences when feeding it to the RNN?')
  parser.add_argument('--use_video_feat', dest='use_video_feat', type=int, default=0, help='Use video features for training')

  # evaluation parameters
  parser.add_argument('-p', '--eval_period', dest='eval_period', type=float, default=1.0, help='in units of epochs, how often do we evaluate on val set?')
  parser.add_argument('--eval_batch_size', dest='eval_batch_size', type=int, default=100, help="""for faster validation performance evaluation, what batch
                                        size to use on val img/sentences?""")
  parser.add_argument('--eval_max_images', dest='eval_max_images', type=int, default=-1, help='for efficiency we can use a smaller number of images to get validation error')
  parser.add_argument('--eval_metric', dest='eval_metric', type=str, default='perplex', help="""Specify the evaluation metric to use on validation. Possible
                                        values are perplex, meteor, cider""")
  parser.add_argument('--metrics_to_track', dest='met_to_track',nargs='+', type=str, default=[], help="""Specify the evaluation metric to use on validation. Possible
                                        values are perplex, meteor, cider""")
  parser.add_argument('--eval_beamsize', dest='eval_beamsize', type=int, default=1, help='what beamsize to use in validation when using AEMs')

  # parameters controlling use of external data (For eg. lsmdc uses COCO Nearest neighbhors)
  parser.add_argument('--ext_data_file', dest='ext_data_file', type=str, default=None, help='Which external dataset file shpuld we use')
  parser.add_argument('--ed_sample_prob', dest='ed_sample_prob', type=np.float, default=0.0, help='probability with which to sample external data')
  parser.add_argument('--ed_feature_file', dest='ed_feature_file', type=str, default=None, help='Which file should we use for read the CNN features')
  parser.add_argument('--ed_aux_inp_file', dest='ed_aux_inp_file', type=str, default=None, help='Is there any auxillary inputs ? If yes indicate file here')

  # parameters to use a feature encoding recurrent network
  parser.add_argument('--feat_encoder', dest='feat_encoder', type=str, default=None, help='Which encoder should we use')
  parser.add_argument('--use_encoder_for', dest='use_encoder_for', type=int, default=0, help='Is it for image feat or aux input')
  parser.add_argument('--use_shared_mem_enc', dest='use_shared_mem_enc', type=int, default=1, help='Is it for image feat or aux input')
  parser.add_argument('--featenc_hidden_size', dest='featenc_hidden_size', type=int, default=512, help='Should img or aux features be read from disk')
  # Implement option to encode GT sentences and use it as features to the generator
  parser.add_argument('--encode_gt_sentences', dest='encode_gt_sentences', type=int, default=0, help='Should img or aux features be read from disk')
  parser.add_argument('--n_encgt_sent', dest='n_encgt_sent', type=int, default=5, help='Should img or aux features be read from disk')

  # parameters to use an attention mechanism.
  parser.add_argument('--use_attention_network', dest='use_attn', type=str, default=None, help='Which encoder should we use')
  parser.add_argument('--attn_hidden_config', dest='attn_hidden_config',nargs='+', type=int, default=[100], help='Is it for image feat or aux input')

  # Some params to enable partial feature reading from disk!!
  parser.add_argument('--disk_feature', dest='disk_feature', type=int, default=0, help='Should img or aux features be read from disk')

  # Implementing gumbel softmax
  parser.add_argument('--use_gumbel_mse', dest='use_gumbel_mse', type=int, default=0, help='Should img or aux features be read from disk')
  parser.add_argument('--gumbel_temp_init', dest='gumbel_temp_init', type=np.float, default=0, help='Should img or aux features be read from disk')



  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  if params['checkpoint_file_name'] != 'None':
    checkpoint_init = pickle.load(open(params['checkpoint_file_name'], 'rb'))
    model_init_from = checkpoint_init['model']
    rg_init = checkpoint_init.get('rgrads',[])

  if params['aux_inp_file'] != 'None' or params['encode_gt_sentences']:
    params['en_aux_inp'] = 1
  else:
    params['en_aux_inp'] = 0

  if params['use_pos_tag'] != 'None':
    sentTagMap = pickle.load(open(params['use_pos_tag'],'r'))
  print 'parsed parameters:'
  print json.dumps(params, indent = 2)
  config.mode = 'FAST_RUN'
  #config.allow_gc = False
  #config.exception_verbosity = 'high'
  main(params)
