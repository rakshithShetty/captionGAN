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
from theano.ifelse import ifelse
import theano.tensor as tensor
import cPickle as pickle
from imagernn.data_provider import getDataProvider, prepare_data, prepare_adv_data, prepare_seq_features
from imagernn.cnn_evaluatorTheano import CnnEvaluator
from imagernn.recurrent_feat_encoder import RecurrentFeatEncoder
from imagernn.solver import Solver
from imagernn.imagernn_utils import decodeEvaluator, decodeGenerator, eval_split_theano, eval_prep_refs
from imagernn.utils import numpy_floatX, zipp, unzip, preProBuildWordVocab
from collections import defaultdict, OrderedDict
import signal
import sys
import matplotlib.pyplot as plt
host = socket.gethostname() # get computer hostname

def eval_gen_samps(gen_func, dp, params, misc, rev_eval, **kwargs):
    ngs = params['n_gen_samples']
    candToks = defaultdict(list)
    eval_metric = kwargs.get('eval_metric','perplex')
    refToks = kwargs.get('refToks',None)
    n = 0
    gts = {}
    for batch in dp.iterImageBatch(split='val', max_batch_size = 100):
        if not params['encode_gt_sentences']:
            xI = np.row_stack([x['feat'] for x in batch])
            inp = [xI]
            if params['en_aux_inp']:
                xAux = np.row_stack([x['aux_inp'] for x in batch])
                inp.append(xAux)
        else:
            batchInp = [{'image':x}for x in batch]
            inp = prepare_seq_features( batchInp, use_enc_for= params['use_encoder_for'], maxlen =  params['maxlen'],
                    use_shared_mem = params['use_shared_mem_enc'], enc_gt_sent = params['encode_gt_sentences'],
                    n_enc_sent = params['n_encgt_sent'], wordtoix = misc['wordtoix'])
            import pdb;pdb.set_trace()

        g_out = gen_func(*inp)
        g_len = g_out[-1]
        g_out = g_out[1].swapaxes(0,1)
        g_out = g_out.reshape(len(batch), ngs, g_out.shape[1])
        for i in xrange(len(batch)):
            imgid = batch[i][dp.idstr]
            gts[imgid] = refToks[imgid]
            for j in xrange(ngs):
                candidate = ' '.join([misc['ixtoword'][gid] for gid in g_out[i,j,:g_len[i*ngs+j]] if gid > 0 ])
                candToks[imgid].append({'image_id':imgid,'caption':candidate,'id':n})
            n += 1

    candToks = kwargs['scr_info']['tokenizer'].tokenize(candToks)
    met = [[] for i in xrange(len(eval_metric)) if eval_metric[i][:6] != 'lcldiv']
    i_lcl_div = []

    if rev_eval == 1:
        candSrc = gts
        gts = candToks
    else:
        candSrc = candToks

    for j in xrange(ngs):
        candInp = {imgid:[candSrc[imgid][j]] for imgid in candSrc}
        # Now invoke all the scorers and get the scores
        for i,evm in enumerate(eval_metric):
            if evm[:6] == "lcldiv":
                if i not in i_lcl_div:
                    i_lcl_div.append(i)
            else:
                score, scores = kwargs['scr_info']['scr_fn'][i].compute_score(gts, candInp)
                if type(kwargs['scr_info']['scr_name'][i]) == list:
                    met[i].append(scores[-1])
                else:
                    met[i].append(scores)

    met = np.array(met)
    met_maxmean = met.max(axis=1).mean(axis=-1)
    met_mean = met.mean(axis=1).mean(axis=-1)
    met_minmean = met.min(axis=1).mean(axis=-1)

    if len(i_lcl_div) > 0:
        for i in i_lcl_div:
            div_out,_ = kwargs['scr_info']['scr_fn'][i].compute_score(gts, candToks)
            met_mean = np.concatenate([met_mean, [div_out]])
            met_maxmean = np.concatenate([met_maxmean, [div_out]])
            met_minmean = np.concatenate([met_minmean, [div_out]])

    print '---------------------Generator Eval ---------------------------'
    for i,evm in enumerate(eval_metric):
        print 'evaluated %d sentences and got %s = %f' % (n, evm, met_mean[i])

    return met_maxmean, met_mean, met_minmean

def disp_some_gen_samps(gen_func, dp, params, misc, n_samp = 5, fixed = [1, 15,  23, 66, 216]):
    if fixed == None:
        batch = dp.sampAdversBatch(n_samp, n_sent=params['n_gen_samples'], split= 'val', probs = [1.0, 0.0, 0.0])
        cnn_inps = prepare_adv_data(batch,misc['wordtoix'], maxlen = params['maxlen'], prep_for=params['eval_model'])
        enc_inp_list = prepare_seq_features( batch, use_enc_for= params['use_encoder_for'], maxlen =  params['maxlen'],
                use_shared_mem = params['use_shared_mem_enc'], enc_gt_sent = params['encode_gt_sentences'],
                n_enc_sent = params['n_encgt_sent'], wordtoix = misc['wordtoix'])
        g_out = gen_func(*(cnn_inps[1+ (params['eval_model']=='lstm_eval'):2+ (params['eval_model']=='lstm_eval')+params['en_aux_inp']] + enc_inp_list))
    else:
        batch = [dp._getImage(dp.split['val'][fidx]) for fidx in fixed]
        xI = np.row_stack([x['feat'] for x in batch])
        inp = [xI]
        if params['en_aux_inp']:
            xAux = np.row_stack([x['aux_inp'] for x in batch])
            inp.append(xAux)
        g_out = gen_func(*inp)
    g_len = g_out[-1]
    g_out = g_out[1].swapaxes(0,1)
    g_out = g_out.reshape(n_samp, params['n_gen_samples'], g_out.shape[1])
    print '--------------------------------Visualizing some generated text------------------------'
    for i in xrange(n_samp):
        #print 'Gen text for img %d with caption: "%s"'%(batch[i]['image']['cocoid'], batch[i]['image']['sentences'][0]['raw'])
        for j in xrange(params['n_gen_samples']):
            print '%s (GT: %s)'%(' '.join([misc['ixtoword'][gid] for gid in g_out[i,j,:g_len[i*params['n_gen_samples']+j]] if gid in misc['ixtoword'] ]),
                                        batch[i]['sentences'][j]['raw'])

    print '---------------------------------------------------------------------------------------'

    return 0

def eval_discrm_gen(split, dp, params, gen_fprop, misc, n_eval=None):
    n_eval = len(dp.split[split]) if n_eval == None else n_eval
    n_iter = (n_eval-1) // params['eval_batch_size'] + 1
    correct = 0.
    n_total = 0.
    g_correct = 0.
    mean_p = 0.
    mean_n = 0.
    mean_g = 0.
    for i in xrange(n_iter):
        batch = dp.sampAdversBatch(params['eval_batch_size'], n_sent=params['n_gen_samples'], split= 'val', probs = [0.6, 0.4, 0.0])
        cnn_inps = prepare_adv_data(batch,misc['wordtoix'],maxlen = params['maxlen'], prep_for=params['eval_model'])
        p_out = gen_fprop(*cnn_inps[:-1])
        y = cnn_inps[-1] if params['t_eval_only'] else np.concatenate([cnn_inps[-1],np.zeros(cnn_inps[-1].shape)])
        correct += ((p_out[0].flatten()>0.) == y).sum()
        if params['t_eval_only']!=1:
            g_correct += (p_out[0].flatten()[cnn_inps[-1].shape[0]:]>0.).sum()
            mean_g += (p_out[0][cnn_inps[-1].shape[0]:]).mean()
        mean_p += (p_out[0][:cnn_inps[-1].shape[0]]*cnn_inps[-1]).sum()/(1e-8+cnn_inps[-1].sum())
        mean_n += (p_out[0][:cnn_inps[-1].shape[0]]*(1-cnn_inps[-1])).sum()/(1e-8 + (1 - cnn_inps[-1]).sum())
        n_total += y.shape[0]

    acc = correct/n_total * 100.0
    mean_p = mean_p/(n_iter)
    mean_n = mean_n/(n_iter)
    mean_g = mean_g/(n_iter)
    g_acc = g_correct*2.0/n_total * 100.0
    print 'evaluated the discriminator. Current disc accuracy is %.2f gen acc is %.2f'%(acc, g_acc)
    print 'Mean scores are pos: %.2f neg: %.2f gen: %.2f'%(mean_p, mean_n, mean_g)

    return acc, g_acc

def dumpCheckpoint(filename, params, modelGen, modelEval, misc, it, val_ppl2):
  filepath = os.path.join(params['checkpoint_output_directory'], filename)
  model_npy_gen = unzip(modelGen)
  model_npy_eval = unzip(modelEval)
  checkpoint = {}
  checkpoint['epoch'] = it
  checkpoint['modelGen'] = model_npy_gen
  checkpoint['modelEval'] = model_npy_eval
  checkpoint['params'] = params
  checkpoint['perplexity'] = val_ppl2
  checkpoint['misc'] = misc
  try:
    pickle.dump(checkpoint, open(filepath, "wb"))
    print 'saved checkpoint in %s' % (filepath, )
  except Exception, e: # todo be more clever here
    print 'tried to write checkpoint into %s but got error: ' % (filepath, )
    print e

def main(params):
  batch_size = params['batch_size']
  word_count_threshold = params['word_count_threshold']
  max_epochs = params['max_epochs']

  # fetch the data provider
  dp = getDataProvider(params)

  # Initialize the optimizer
  solver = Solver(params['solver'])

  params['aux_inp_size'] = dp.aux_inp_size
  params['image_feat_size'] = dp.img_feat_size

  print 'Image feature size is %d, and aux input size is %d'%(params['image_feat_size'],params['aux_inp_size'])

  misc = {} # stores various misc items that need to be passed around the framework

  if params['checkpoint_file_name'] == 'None':
    # go over all training sentences and find the vocabulary we want to use, i.e. the words that occur
    # at least word_count_threshold number of times
    misc['wordtoix'], misc['ixtoword'], bias_init_vector = preProBuildWordVocab(dp.iterSentences('train'),
                                      word_count_threshold)
  else:
    # Load Vocabulary from the checkpoint
    misc = checkpoint_init['misc']

  params['vocabulary_size'] = len(misc['wordtoix'])
  params['output_size'] = len(misc['ixtoword']) # these should match though

  # This initializes the generator model parameters and does matrix initializations
  if params['t_eval_only'] == 0:
    generator = decodeGenerator(params)
    # Build the computational graph

    if params['use_encoder_for']&2:
      aux_enc_inp = generator.model_th['Wemb'] if params['encode_gt_sentences'] else dp.aux_inputs.T
      hid_size = params['featenc_hidden_size']
      auxFeatEncoder = RecurrentFeatEncoder(hid_size, params['image_encoding_size'], params,
              mdl_prefix='aux_enc_', features=aux_enc_inp)
      mdlLen = len(generator.model_th.keys())
      generator.model_th.update(auxFeatEncoder.model_th)
      assert(len(generator.model_th.keys()) == (mdlLen+len(auxFeatEncoder.model_th.keys())))
      (auxenc_use_dropout, auxFeatEnc_inp, xAux, updatesLSTMAuxFeat) = auxFeatEncoder.build_model(generator.model_th, params)

      if params['encode_gt_sentences']:
          # Reshape it size(batch_size, n_gt, hidden_size)
          xAux = xAux.reshape((-1,params['n_encgt_sent'],params['featenc_hidden_size']))
          # Convert it to size (batch_size, n_gt*hidden_size
          xAux = xAux.flatten(2)
          xI = tensor.zeros((batch_size,params['image_encoding_size']))
          imgFeatEnc_inp = []
    else:
      auxFeatEnc_inp = []
      imgFeatEnc_inp = []
      xAux = None
      xI = None


    (gen_inp_list, predLogProb, predIdx, predCand, gen_out, updatesLstm, seq_lengths) = generator.build_prediction_model(
                                              generator.model_th, params, xI = xI, xAux = xAux)
    gen_inp_list = imgFeatEnc_inp + auxFeatEnc_inp + gen_inp_list
    gen_out = gen_out.reshape([gen_out.shape[0], -1, params['n_gen_samples'], params['vocabulary_size']])
    #convert updates lstm to a tuple, this is to help merge it with grad updates
    updatesLstm = [(k, v) for k, v in updatesLstm.iteritems()]
    f_gen_only = theano.function(gen_inp_list, [predLogProb, predIdx, gen_out, seq_lengths], name='f_pred', updates=updatesLstm)

    modelGen = generator.model_th
    upListGen = generator.update_list

    if params['use_mle_train']:
        (use_dropout_genTF, inp_list_genTF,
           _, cost_genTF, _, updatesLSTM_genTF) = generator.build_model(generator.model_th, params)
        f_eval_genTF = theano.function( inp_list_genTF, cost_genTF, name='f_eval')
        grads_genTF = tensor.grad(cost_genTF[0], wrt=modelGen.values(), add_names=True)
        lr_genTF = tensor.scalar(name='lr',dtype=config.floatX)
        f_grad_genTF, f_update_genTF, zg_genTF, rg_genTF, ud_genTF = solver.build_solver_model(lr_genTF, modelGen,
                                                grads_genTF, inp_list_genTF, cost_genTF, params)
  else:
    modelGen = []
    updatesLstm = []

  if params['met_to_track'] != []:
    trackMetargs = {'eval_metric': params['met_to_track']}
    refToks, scr_info = eval_prep_refs('val', dp, params['met_to_track'])
    trackMetargs['refToks'] = refToks
    trackMetargs['scr_info'] = scr_info

  # Initialize the evalator model
  if params['share_Wemb']:
     evaluator = decodeEvaluator(params, modelGen['Wemb'])
  else:
     evaluator = decodeEvaluator(params)
  modelEval = evaluator.model_th

  if params['t_eval_only'] == 0:
    # Build the evaluator graph to evaluate reference and generated captions
    if params.get('upd_eval_ref',0):
        (refeval_inp_list, ref_f_pred_fns, ref_costs, ref_predTh, ref_modelEval) = evaluator.build_advers_eval(modelEval, params)
    (eval_inp_list, f_pred_fns, costs, predTh, modelEval) = evaluator.build_advers_eval(modelEval, params,
                                                             gen_inp_list, gen_out, updatesLstm, seq_lengths)
  else:
    # Build the evaluator graph to evaluate only reference captions
    (eval_inp_list, f_pred_fns, costs, predTh, modelEval) = evaluator.build_advers_eval(modelEval, params)

  # force overwrite here. The bias to the softmax is initialized to reflect word frequencies
  if params['t_eval_only'] == 0:# and 0:
    if params['checkpoint_file_name'] == 'None':
      modelGen['bd'].set_value(bias_init_vector.astype(config.floatX))
      if params.get('class_out_factoring',0) == 1:
        modelGen['bdCls'].set_value(bias_init_inter_class.astype(config.floatX))

  comb_inp_list = eval_inp_list
  if params['t_eval_only'] == 0:
    for inp in gen_inp_list:
      if inp not in comb_inp_list:
          comb_inp_list.append(inp)

  # Compile an evaluation function.. Doesn't include gradients
  # To be used for validation set evaluation or debug purposes
  if params['t_eval_only'] == 0:
      f_eval= theano.function(comb_inp_list, costs[:1], name='f_eval', updates=updatesLstm)
  else:
      f_eval= theano.function(comb_inp_list, costs[:1], name='f_eval')


  if params['share_Wemb']:
    modelEval.pop('Wemb')
  if params['fix_Wemb']:
    upListGen.remove('Wemb')

  #-------------------------------------------------------------------------------------------------------------------------
  # Now let's build a gradient computation graph and update mechanism
  #-------------------------------------------------------------------------------------------------------------------------
  # First compute gradient on the evaluator params w.r.t cost
  if params.get('upd_eval_ref',0):
    gradsEval_ref = tensor.grad(ref_costs[0], wrt=modelEval.values(),add_names=True)
  gradsEval = tensor.grad(costs[0], wrt=modelEval.values(),add_names=True)

  # Update functions for the evaluator
  lrEval = tensor.scalar(name='lrEval',dtype=config.floatX)
  if params.get('upd_eval_ref',0):
    f_grad_comp_eval_ref, f_param_update_eval_ref, _, _, _= solver.build_solver_model(lrEval, modelEval, gradsEval_ref,
                                      refeval_inp_list, ref_costs[0], params, w_clip=params['eval_w_clip'])
  f_grad_comp_eval, f_param_update_eval, zg_eval, rg_eval, ud_eval= solver.build_solver_model(lrEval, modelEval, gradsEval,
                                      comb_inp_list, costs[:1], params, updatesLstm, w_clip=params['eval_w_clip'])

  # Now compute gradient on the generator params w.r.t the cost
  if params['t_eval_only'] == 0:
    gradsGen = tensor.grad(costs[1], wrt=modelGen.values(), add_names=True)
    lrGen = tensor.scalar(name='lrGen',dtype=config.floatX)
    # Update functions for the generator
    f_grad_comp_gen, f_param_update_gen, zg_gen, rg_gen, ud_gen = solver.build_solver_model(lrGen, modelGen, gradsGen,
                                        comb_inp_list[:(len(comb_inp_list)-1+params['gen_feature_matching'])], costs[1], params, updatesLstm)

  #-------------------------------------------------------------------------------------------------------------------------
  # If we want to track some metrics during the training, initialize stuff for that now
  #-------------------------------------------------------------------------------------------------------------------------
  print 'model init done.'
  if params['t_eval_only'] == 0:
    print 'Gen model has keys: ' + ', '.join(modelGen.keys())
  print 'Eval model has keys: ' + ', '.join(modelEval.keys())

  # calculate how many iterations we need, One epoch is considered once going through all the sentences and not images
  # Hence in case of coco/flickr this will 5* no of images
  num_sentences_total = dp.getSplitSize('train', ofwhat = 'images')
  num_iters_one_epoch = num_sentences_total / batch_size
  max_iters = max_epochs * num_iters_one_epoch
  skip_first = 20
  iters_eval= 5
  iters_gen = 1

  cost_eval_iter = []
  cost_gen_iter = []
  trackSc_array = []

  eval_period_in_epochs = params['eval_period']
  eval_period_in_iters = max(1, int(num_iters_one_epoch * eval_period_in_epochs))
  top_val_ppl2 = -1
  smooth_train_ppl2 = 0.5 # initially size of dictionary of confusion
  smooth_train_cost = 0.0 # initially size of dictionary of confusion
  smooth_train_cost_gen = 1.0 # initially size of dictionary of confusion
  val_ppl2 = len(misc['ixtoword'])
  last_status_write_time = 0 # for writing worker job status reports
  json_worker_status = {}
  json_worker_status['params'] = params
  json_worker_status['history'] = []
  write_checkpoint_ppl_threshold = params['write_checkpoint_ppl_threshold']
  iter_out_file = os.path.join('logs','advmodel_checkpoint_%s_%s_%s_log.npz' % (params['dataset'], host, params['fappend']))

  len_hist = defaultdict(int)
  t_print_sec =30
  ## Initialize the model parameters from the checkpoint file if we are resuming training
  if params['checkpoint_file_name'] != 'None':
    if params['t_eval_only'] !=1:
        print '\n Now initing gen Model:'
        zipp(model_init_gen_from,modelGen)
    if 'trackers' in checkpoint_init:
        trackSc_array = checkpoint_init['trackers'].get('trackScores',[])
    print '\n Now initing Eval Model:'
    zipp(model_init_eval_from,modelEval)
    #zipp(rg_init,rgGen)
    print("\nContinuing training from previous model\n. Already run for %0.2f epochs with validation perplx at %0.3f\n" % (checkpoint_init['epoch'], \
      checkpoint_init['perplexity']))

  ##############################################################
  # Define signal handler to catch ctl-c or kills so that we can save the model trained till that point
  def signal_handler(signal, frame):
    print('You pressed Ctrl+C! Saving Checkpoint Now before exiting!')
    filename = 'advmodel_checkpoint_%s_%s_%s_%.2f_INT.p' % (params['dataset'], host, params['fappend'], val_ppl2)
    dumpCheckpoint(filename, params, modelGen, modelEval, misc, it, val_ppl2)
    sys.exit(0)
  #signal.signal(signal.SIGINT, signal_handler)
  ##############################################################

  #In testing disable sampling and use the greedy approach!?
  generator.usegumbel.set_value(1)
  if params['met_to_track'] != []:
      tsc_max, tsc_mean, tsc_min = eval_gen_samps(f_gen_only, dp, params, misc, params['rev_eval'], **trackMetargs)
      trackSc_array.append((0,{evm+'_max':tsc_max[i] for i,evm in enumerate(params['met_to_track'])}))
      trackSc_array[-1][1].update({evm+'_mean':tsc_mean[i] for i,evm in enumerate(params['met_to_track'])})
      trackSc_array[-1][1].update({evm+'_min':tsc_min[i] for i,evm in enumerate(params['met_to_track'])})

  disp_some_gen_samps(f_gen_only, dp, params, misc, n_samp = 5)
  evaluator.use_noise.set_value(1.)
  eval_acc, gen_acc = eval_discrm_gen('val', dp, params, f_pred_fns[0], misc)
  # Re-enable sampling
  generator.usegumbel.set_value(1)

  np.savez(iter_out_file, eval_cost=np.array(cost_eval_iter), gen_cost=np.array(cost_gen_iter), tracksc = np.array(trackSc_array))
  smooth_train_cost = 0.0

  print '###################### NOW BEGINNING TRAINING #################################'

  for it in xrange(max_iters):
    t0 = time.time()
    # Enable using dropout in training
    evaluator.use_noise.set_value(1.)
    dt = 0.
    it2 = 0
    while eval_acc <= 60. or gen_acc >= 45. or it2<iters_eval*skip_first:
        # fetch a batch of data
        t1 = time.time()

        s_probs = [0.6, 0.4, 0.0] if params['eval_loss'] == 'contrastive' else [1.0, 0.0, 0.0]
        batch = dp.sampAdversBatch(batch_size, n_sent=params['n_gen_samples'], probs = s_probs)
        cnn_inps = prepare_adv_data(batch,misc['wordtoix'],maxlen = params['maxlen'], prep_for=params['eval_model'])

        enc_inp_list = prepare_seq_features( batch, use_enc_for= params['use_encoder_for'], maxlen =  params['maxlen'],
                use_shared_mem = params['use_shared_mem_enc'], enc_gt_sent = params['encode_gt_sentences'],
                n_enc_sent = params['n_encgt_sent'], wordtoix = misc['wordtoix'])
        eval_cost = f_grad_comp_eval(*(cnn_inps+enc_inp_list))

        if np.isnan(eval_cost[0]):
            import pdb;pdb.set_trace()
        f_param_update_eval(params['learning_rate_eval'])

        # Track training statistics
        smooth_train_cost =  0.99 * smooth_train_cost + 0.01 * eval_cost[0] if it > 0 else eval_cost[0]
        dt2 = time.time() - t1
        if it2%500 == 499:
            gb =  0. #modelGen['gumb_temp'].get_value() if params['use_gumbel_mse'] == 1 else 0
            print 'Iter %d/%d Eval Only Iter %d/%d, done. in %.3fs. Eval Cost is %.6f' % (it, max_iters, it2,
                    iters_eval*skip_first, dt2, smooth_train_cost)
        if it2%100 == 99:
            eval_acc, gen_acc = eval_discrm_gen('val', dp, params, f_pred_fns[0], misc, n_eval = 500)
        it2 += 1

    evaluator.use_noise.set_value(1.)

    if it >= 0:
        skip_first = 1
    if it >=100:
        skip_first = 1
    if it%1000 == 999:
        skip_first = 1

    s_probs = [1.0, 0.0, 0.0] if params['eval_loss'] == 'contrastive' else [1.0, 0.0, 0.0]
    batch = dp.sampAdversBatch(batch_size, n_sent=params['n_gen_samples'], probs = s_probs)
    cnn_inps = prepare_adv_data(batch,misc['wordtoix'],maxlen = params['maxlen'], prep_for=params['eval_model'])
    enc_inp_list = prepare_seq_features( batch, use_enc_for= params['use_encoder_for'], maxlen =  params['maxlen'],
            use_shared_mem = params['use_shared_mem_enc'], enc_gt_sent = params['encode_gt_sentences'],
            n_enc_sent = params['n_encgt_sent'], wordtoix = misc['wordtoix'])

    gen_cost = f_grad_comp_gen(*(cnn_inps[:(len(cnn_inps)-1+params['gen_feature_matching'])]+enc_inp_list))
    f_param_update_gen(params['learning_rate_gen'])

    if params['use_mle_train']:
        generator.usegumbel.set_value(0)
        batch,l = dp.getRandBatchByLen(batch_size)
        gen_inp_list, lenS = prepare_data( batch, misc['wordtoix'], params['maxlen'])
        cost_genMLE = f_grad_genTF(*gen_inp_list)
        f_update_genTF(np.float32(params['learning_rate_gen']/50.0))
        generator.usegumbel.set_value(1)


    dt = time.time() - t0
    # print training statistics
    smooth_train_cost_gen = gen_cost if it == 0 else 0.99 * smooth_train_cost_gen + 0.01 * gen_cost

    tnow = time.time()
    if tnow > last_status_write_time + t_print_sec*1: # every now and then lets write a report
        gb = 0. #modelGen['gumb_temp'].get_value() if params['use_gumbel_mse'] == 1 else 0
        print 'Iter %d/%d done. in %.3fs. Eval Cost is %.6f, Gen Cost is %.6f, temp: %.4f' % (it, max_iters, dt, \
        	smooth_train_cost, smooth_train_cost_gen, gb)
        last_status_write_time = tnow

    cost_eval_iter.append(smooth_train_cost)
    cost_gen_iter.append(smooth_train_cost_gen)

    if it%500==499:
        # Run the generator on the validation set and compute some metrics
        generator.usegumbel.set_value(1)
        if params['met_to_track'] != []:
            #In testing set the temperature to very low, so that it is equivalent to Greed samples
            tsc_max, tsc_mean, tsc_min  = eval_gen_samps(f_gen_only, dp, params, misc, params['rev_eval'], **trackMetargs)
            trackSc_array.append((it,{evm+'_max':tsc_max[i] for i,evm in enumerate(params['met_to_track'])}))
            trackSc_array[-1][1].update({evm+'_mean':tsc_mean[i] for i,evm in enumerate(params['met_to_track'])})
            trackSc_array[-1][1].update({evm+'_min':tsc_min[i] for i,evm in enumerate(params['met_to_track'])})

        disp_some_gen_samps(f_gen_only, dp, params, misc, n_samp = 5)
        generator.usegumbel.set_value(1)
        # if we beat a previous record or if this is the first time
        # AND we also beat the user-defined threshold or it doesnt exist
        top_val_ppl2 = gen_acc
    if it%500 == 499:
        eval_acc, gen_acc = eval_discrm_gen('val', dp, params, f_pred_fns[0], misc, n_eval = 500)
    if it%1000==999:
        filename = 'advmodel_checkpoint_%s_%s_%s_%d_%.2f_genacc.p' % (params['dataset'], host, params['fappend'],it, gen_acc)
        dumpCheckpoint(filename, params, modelGen, modelEval, misc, it, gen_acc)
    if it%500==499:
        np.savez(iter_out_file, eval_cost=np.array(cost_eval_iter), gen_cost=np.array(cost_gen_iter), tracksc = np.array(trackSc_array))

  # AND we also beat the user-defined threshold or it doesnt exist
  filename = 'advmodel_checkpoint_%s_%s_%s_%d_%.2f_GenDone.p' % (params['dataset'], host, params['fappend'],it, g_acc)
  dumpCheckpoint(filename, params, modelGen, modelEval, misc, it, g_acc)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # global setup settings, and checkpoints
  parser.add_argument('-d', '--dataset', dest='dataset', default='coco', help='dataset: flickr8k/flickr30k/coco')
  parser.add_argument('--fappend', dest='fappend', type=str, default='baseline', help='append this string to checkpoint filenames')
  parser.add_argument('-o', '--checkpoint_output_directory', dest='checkpoint_output_directory', type=str, default='cv/', help='output directory to write checkpoints to')
  parser.add_argument('--write_checkpoint_ppl_threshold', dest='write_checkpoint_ppl_threshold', type=float, default=-1, help='ppl threshold above which we dont bother writing a checkpoint to save space')
  parser.add_argument('--continue_training', dest='checkpoint_file_name', type=str, default='None', help='checkpoint file from which to resume training')
  parser.add_argument('--eval_init', dest='eval_init', type=str, default='None', help='checkpoint file from which to resume training')

  # Some parameters about image features used
  parser.add_argument('--feature_file', dest='feature_file', type=str, default='vgg_feats.mat', help='Which file should we use for read the CNN features')
  parser.add_argument('--data_file', dest='data_file', type=str, default='dataset.json', help='Which dataset file shpuld we use')
  parser.add_argument('--mat_new_ver', dest='mat_new_ver', type=int, default=-1, help='If the .mat feature files are saved with new version (compressed) set this flag to 1')
  parser.add_argument('--aux_inp_file', dest='aux_inp_file', type=str, default='None', help='Is there any auxillary inputs ? If yes indicate file here')
  parser.add_argument('--swap_AuxFeat', dest='swap_aux', type=int, default=1, help='Feed image features through auxillary input!')
  parser.add_argument('--advers_gen', dest='advers_gen', type=str, default=1, help='Should we use adverserial generator!')
  parser.add_argument('--eval_model', dest='eval_model', type=str, default='lstm_eval', help='which evaluator model to use type: cnn/lstm_eval')
  parser.add_argument('--eval_feature', dest='eval_feature', type=str, default='image_feat', help='Should the evaluator use image feature or aux feature')

  # model parameters
  parser.add_argument('--image_encoding_size', dest='image_encoding_size', type=int, default=512, help='size of the image encoding')
  parser.add_argument('--word_encoding_size', dest='word_encoding_size', type=int, default=512, help='size of word encoding')
  parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=512, help='size of hidden layer in generator RNNs')
  parser.add_argument('--hidden_depth', dest='hidden_depth', type=int, default=3, help='depth of hidden layer in generator RNNs')
  parser.add_argument('--en_residual_conn', dest='en_residual_conn', type=int, default=1, help='add residual connections in LSTM')
  parser.add_argument('--generator', dest='generator', type=str, default='lstm', help='generator to use')
  parser.add_argument('-c', '--regc', dest='regc', type=float, default=0., help='regularization strength')
  parser.add_argument('--n_gen_samples', type=int, default=1, help='Number of samples fed to evaluator in one go. 5 is a good value for COCO captioning')

  # Parameters for CNN evaluator
  parser.add_argument('--sent_encoding_size', dest='sent_encoding_size', type=int, default=400, help='size of sentence encoding layer on top of CNN')
  parser.add_argument('--n_fmaps', dest='n_fmaps_psz', type=int, default=100, help='number of cnn feature maps per filter height')
  parser.add_argument('--filter_hs', dest='filter_hs', metavar='N', type=int, nargs='+',default =[3,4,5], help='list fo filter heights to use in CNN')
  parser.add_argument('--conv_non_linear', dest='conv_non_linear', type=str, default='relu', help='nonlinearity type: tanh/relu')
  parser.add_argument('--maxlen', dest='maxlen', type=int, default=15, help='size of sentence encoding layer on top of CNN')
  parser.add_argument('--merge_dim', dest='merge_dim', type=int, default=50, help='size of sentence encoding layer on top of CNN')

  # Regarding word embedding sharing and such
  parser.add_argument('--share_Wemb', dest='share_Wemb', type=int, default=0, help='If 1, share Wemb b/w eval and gen models')
  parser.add_argument('--fix_Wemb', dest='fix_Wemb', type=int, default=0, help='If 1, gen model doesnt learn Wemb and keeps it fixed')

  # optimization parameters
  parser.add_argument('-m', '--max_epochs', dest='max_epochs', type=int, default=10, help='number of epochs to train for')
  parser.add_argument('--solver', dest='solver', type=str, default='rmsprop', help='solver type: vanilla/adagrad/adadelta/rmsprop')
  parser.add_argument('--decay_rate', dest='decay_rate', type=float, default=0.999, help='decay rate for adadelta/rmsprop')
  parser.add_argument('--smooth_eps', dest='smooth_eps', type=float, default=1e-8, help='epsilon smoothing for rmsprop/adagrad/adadelta')
  parser.add_argument('-lg', '--learning_rate_gen', dest='learning_rate_gen', type=float, default=1e-6, help='solver learning rate')
  parser.add_argument('-ld', '--learning_rate_eval', dest='learning_rate_eval', type=float, default=1e-5, help='solver learning rate')

  parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=1, help='batch size')
  parser.add_argument('-cb', '--eval_batch_size', dest='eval_batch_size', type=int, default=50, help='Batch size for eval descriminative network, 1 means it only gets a positive reference\
                                                                                                     and a generated sample, n implies it also gets n-1 negative references ')
  parser.add_argument('--rand_negs', dest='rand_negs', type=int, default=0, help='How many hard negetives obtianed by random permutations of positive to use to train eval n/w')
  parser.add_argument('--sample_by_len', dest='sample_by_len', type=int, default=1, help='enable sampling by length of sentece to speed up training')
  parser.add_argument('--grad_clip', dest='grad_clip', type=float, default=1.0, help='clip gradients (normalized by batch size)? elementwise. if positive, at what threshold?')
  parser.add_argument('--use_dropout', dest='use_dropout', type=int, default=1.0, help='what dropout to apply right after the encoder to an RNN/LSTM')
  parser.add_argument('--drop_prob_encoder', dest='drop_prob_encoder', type=np.float32, default=0.3, help='what dropout to apply right after the encoder to an RNN/LSTM')
  parser.add_argument('--drop_prob_decoder', dest='drop_prob_decoder', type=np.float32, default=0.3, help='what dropout to apply right before the decoder in an RNN/LSTM')
  parser.add_argument('--drop_prob_eval', dest='drop_prob_eval', type=np.float32, default=0.3, help='what dropout to apply right before the decoder in an RNN/LSTM')
  parser.add_argument('--drop_prob_aux', dest='drop_prob_aux', type=np.float32, default=0.3, help='what dropout to apply for the auxillary inputs to lstm')

  parser.add_argument('--lr_decay', dest='lr_decay', type=float, default=1.0, help='decay factor for learning rate, applied every epoch')
  parser.add_argument('--lr_decay_st_epoch', dest='lr_decay_st_epoch', type=float, default=100.0, help='from which epoch should the lr decay start')

  # data preprocessing parameters
  parser.add_argument('--word_count_threshold', dest='word_count_threshold', type=int, default=5, help='if a word occurs less than this number of times in training data, it is discarded')

  # evaluation parameters
  parser.add_argument('-p', '--eval_period', dest='eval_period', type=float, default=1.0, help='in units of epochs, how often do we evaluate on val set?')
  parser.add_argument('--eval_max_images', dest='eval_max_images', type=int, default=-1, help='for efficiency we can use a smaller number of images to get validation error')
  parser.add_argument('--softmax_smooth_factor', dest='softmax_smooth_factor', type=float, default=3.0, help='Is there any auxillary inputs ? If yes indicate file here')

  # Implementing gumbel softmax
  parser.add_argument('--use_gumbel_mse', dest='use_gumbel_mse', type=int, default=0, help='use gumbel approximation to obtain soft samples')
  parser.add_argument('--gumbel_temp_init', dest='gumbel_temp_init', type=np.float, default=0.5, help='temperature value for gumbel')
  parser.add_argument('--use_gumbel_hard', dest='use_gumbel_hard', type=int, default=1, help='Use ST version of the gumbel approximation')
  parser.add_argument('--gen_use_rand_init', dest='gen_use_rand_init', type = int, default=0, help='use noise to intialize the hidden state of the generator')
  parser.add_argument('--gen_feature_matching', dest='gen_feature_matching', type = int, default=1, help='Use feature matching loss, recommended 1')
  parser.add_argument('--gen_input_noise', dest='gen_input_noise', type = int, default=1, help='Provide noise as an additional input to the generator')
  parser.add_argument('--gen_input_noise_dim', dest='gen_input_noise_dim', type = int, default=50, help='Dimension of the noise vector')

  # parameters to use a feature encoding recurrent network
  # Irrelavant for GAN training, but the framework to prepare features needs these defaults
  parser.add_argument('--feat_encoder', dest='feat_encoder', type=str, default=None, help='Which encoder should we use')
  parser.add_argument('--use_encoder_for', dest='use_encoder_for', type=int, default=0, help='Is it for image feat or aux input')
  parser.add_argument('--use_shared_mem_enc', dest='use_shared_mem_enc', type=int, default=1, help='Use shared memory for encoder')
  parser.add_argument('--featenc_hidden_size', dest='featenc_hidden_size', type=int, default=512, help='Should img or aux features be read from disk')

  # Implement option to encode GT sentences and use it as features to the generator
  parser.add_argument('--encode_gt_sentences', dest='encode_gt_sentences', type=int, default=0, help='encode gt sentences and provide as input to generator')
  parser.add_argument('--n_encgt_sent', dest='n_encgt_sent', type=int, default=5, help='how many gt sentences are we encoding')

  # Use MLE loss additionally
  parser.add_argument('--use_mle_train', dest='use_mle_train', type=int, default=0, help='Additionally run an mle training iteration every step?')

  # parameters for loading multiple features per video using labels.txt
  parser.add_argument('--labelsFile', dest='labels', type=str, default='labels.txt', help='labels.txt file for this dataset')
  parser.add_argument('--featfromlbl', dest='featfromlbl', type=str, default='ALL ALL', help='should we use lables.txt, if yes which feature?'
                  'use + sign to seperately specify for img and aux')
  parser.add_argument('--poolmethod', dest='poolmethod', type=str, default='max', help='What pooling to use if multiple features are found')
  parser.add_argument('--uselabel', dest='uselabel', type=int, default=3, help='which features should use labels.txt, img/aux or both, 0 - None, 1 - img, 2 - aux, 3 - both')
  # Some params to enable partial feature reading from disk!!
  parser.add_argument('--disk_feature', dest='disk_feature', type=int, default=0, help='Should img or aux features be read from disk')

  parser.add_argument('--train_evaluator_only', dest='t_eval_only', type=int, default=0, help='Train only the evaluator network')
  parser.add_argument('--eval_loss', dest='eval_loss', type=str, default='contrastive', help='Which loss type for the evaluator network')
  parser.add_argument('--eval_w_clip', dest='eval_w_clip', type=float, default=None, help='Clip weigths of evluator, this is to implement wasserstien loss')

  # Track some metrics during training
  parser.add_argument('--metrics_to_track', dest='met_to_track',nargs='+', type=str, default=[], help="""Specify the evaluation metric to use on validation. Possible
                                        values are perplex, meteor, cider""")
  parser.add_argument('--rev_eval', dest='rev_eval', type = int, default=0, help='evaluate references against generated sentences')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  if params['checkpoint_file_name'] != 'None':
    checkpoint_init = pickle.load(open(params['checkpoint_file_name'], 'rb'))
    if 'model' in checkpoint_init:
        model_init_gen_from = checkpoint_init.get('model',{})
    else:
        model_init_gen_from = checkpoint_init.get('modelGen',{})
    model_init_eval_from = checkpoint_init.get('modelEval',{})
    rg_init = checkpoint_init.get('rgrads',[])

  if params['eval_init'] != 'None':
    checkpoint_init = pickle.load(open(params['eval_init'], 'rb'))
    model_init_eval_from = checkpoint_init.get('modelEval',{})


  if params['aux_inp_file'] != 'None' or params['encode_gt_sentences']:
    params['en_aux_inp'] = 1
  else:
    params['en_aux_inp'] = 0

  params['eval_batch_size'] += params['rand_negs']

  print 'parsed parameters:'
  print json.dumps(params, indent = 2)
  config.mode = 'FAST_RUN'
  #config.allow_gc = False

  main(params)
