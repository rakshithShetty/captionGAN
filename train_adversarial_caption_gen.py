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
from imagernn.data_provider import getDataProvider, prepare_data, prepare_adv_data
from imagernn.cnn_evaluatorTheano import CnnEvaluator
from imagernn.solver import Solver
from imagernn.imagernn_utils import decodeEvaluator, decodeGenerator, eval_split_theano
#from numbapro import cuda
from imagernn.utils import numpy_floatX, zipp, unzip, preProBuildWordVocab
from collections import defaultdict, OrderedDict
import signal
import sys
import matplotlib.pyplot as plt
host = socket.gethostname() # get computer hostname


def disp_some_gen_samps(gen_func, dp, params, misc, n_samp = 5):
    batch = dp.sampAdversBatch(n_samp, n_sent=params['n_gen_samples'], probs = [1.0, 0.0, 0.0])
    cnn_inps = prepare_adv_data(batch,misc['wordtoix'], maxlen = params['maxlen'], prep_for=params['eval_model'])
    g_out = gen_func(cnn_inps[1+ (params['eval_model']=='lstm_eval')])
    g_len = g_out[-1]
    g_out = g_out[1].swapaxes(0,1)
    g_out = g_out.reshape(n_samp, params['n_gen_samples'], g_out.shape[1])
    print '--------------------------------Visualizing some generated text------------------------'
    for i in xrange(n_samp):
        print 'Gen text for img %d with caption: "%s"'%(batch[i]['image']['cocoid'], batch[i]['image']['sentences'][0]['raw'])
        for j in xrange(params['n_gen_samples']):
            print '%d :  %s'%(j, ' '.join([misc['ixtoword'][gid] for gid in g_out[i,j,:g_len[i*params['n_gen_samples']+j]] if gid in misc['ixtoword'] ]))

    print '---------------------------------------------------------------------------------------'

    return 0

def eval_discrm_gen(split, dp, params, gen_fprop, misc, n_eval=None):
    n_eval = len(dp.split[split])
    n_iter = n_eval // params['eval_batch_size'] + 1
    correct = 0.
    n_total = 0.
    g_correct = 0.
    mean_p = 0.
    mean_n = 0.
    mean_g = 0.
    for i in xrange(n_iter):
        batch = dp.sampAdversBatch(params['eval_batch_size'], n_sent=params['n_gen_samples'], probs = [0.4, 0.3, 0.3])
        cnn_inps = prepare_adv_data(batch,misc['wordtoix'],maxlen = params['maxlen'], prep_for=params['eval_model'])
        p_out = gen_fprop(*cnn_inps[:-1])
        y = cnn_inps[-1] if params['t_eval_only'] else np.concatenate([cnn_inps[-1],np.zeros(cnn_inps[-1].shape)])
        correct += ((p_out[0].flatten()>0.) == y).sum()
        if params['t_eval_only']!=1:
            g_correct += ((p_out[0].flatten()[cnn_inps[-1].shape[0]:]+0.5)//1.0 == 1.0).sum()
            mean_g += (p_out[0][cnn_inps[-1].shape[0]:]).mean()
        mean_p += (p_out[0][:cnn_inps[-1].shape[0]]*cnn_inps[-1]).mean()
        mean_n += (p_out[0][:cnn_inps[-1].shape[0]]*(1-cnn_inps[-1])).mean()
        n_total += y.shape[0]

    acc = correct/n_total * 100.0
    mean_p = mean_p/(n_iter-1)
    mean_n = mean_n/(n_iter-1)
    mean_g = mean_g/(n_iter-1)
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

  # go over all training sentences and find the vocabulary we want to use, i.e. the words that occur
  # at least word_count_threshold number of times
  if params['checkpoint_file_name'] == 'None':
        misc['wordtoix'], misc['ixtoword'], bias_init_vector = preProBuildWordVocab(dp.iterSentences('train'),
                                      word_count_threshold)
  else:
      misc = checkpoint_init['misc']
      #params['nClasses'] =  checkpoint_init['params']['nClasses']
      #if 'ixtoclsinfo' in misc:
      #  params['ixtoclsinfo'] = misc['ixtoclsinfo']smooth_train_cost


  params['vocabulary_size'] = len(misc['wordtoix'])
  params['output_size'] = len(misc['ixtoword']) # these should match though

  # This initializes the generator model parameters and does matrix initializations
  if params['t_eval_only'] == 0:
    generator = decodeGenerator(params)
    # Build the computational graph
    (gen_inp_list, predLogProb, predIdx, predCand, gen_out, updatesLstm, seq_lengths) = generator.build_prediction_model(
                                              generator.model_th, params)
    gen_out = gen_out.swapaxes(0,1)
    gen_out = gen_out.reshape([-1, params['n_gen_samples'], gen_out.shape[1], params['vocabulary_size']])
    #convert updates lstm to a tuple, this is to help merge it with grad updates
    updatesLstm = [(k, v) for k, v in updatesLstm.iteritems()]
    f_gen_only = theano.function(gen_inp_list, [predLogProb, predIdx, gen_out, seq_lengths], name='f_pred', updates=updatesLstm)

    modelGen = generator.model_th
    upListGen = generator.update_list
  else:
    modelGen = []
    updatesLstm = []

  if params['share_Wemb']:
     evaluator = decodeEvaluator(params, modelGen['Wemb'])
  else:
     evaluator = decodeEvaluator(params)
  modelEval = evaluator.model_th
  # Define the computational graph for relating the input image features and word indices to the
  # log probability cost funtion.

  if params['t_eval_only'] == 0:
    #This variable is used hold the reference caption inputs (Batch x n_samp x len x Vocab)
    (eval_inp_list, f_pred_fns, costs, predTh, modelEval) = evaluator.build_advers_eval(modelEval, params,
                                                            gen_inp_list, gen_out, updatesLstm, seq_lengths)
  else:
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
  # To be used for validation set evaluation
  if params['t_eval_only'] == 0:
    f_eval= theano.function(comb_inp_list, costs, name='f_eval', updates=updatesLstm)
  else:
    f_eval= theano.function(comb_inp_list, costs[0], name='f_eval')

  # Now let's build a gradient computation graph and rmsprop update mechanism
  if params['share_Wemb']:
    modelEval.pop('Wemb')
  if params['fix_Wemb']:
    upListGen.remove('Wemb')


  #modelGenUpD =  OrderedDict()
  #for k in upListGen:
  # modelGenUpD[k] = modelGen[k]
  gradsEval = tensor.grad(costs[0], wrt=modelEval.values(),add_names=True)

  lrEval = tensor.scalar(name='lrEval',dtype=config.floatX)
  f_grad_comp_eval, f_param_update_eval, zg_eval, rg_eval, ud_eval= solver.build_solver_model(lrEval, modelEval, gradsEval,
                                      comb_inp_list, costs[0], params, updatesLstm, w_clip=params['eval_w_clip'])

  if params['t_eval_only'] == 0:
    gradsGen = tensor.grad(costs[1], wrt=modelGen.values(), add_names=True)
    lrGen = tensor.scalar(name='lrGen',dtype=config.floatX)
    f_grad_comp_gen, f_param_update_gen, zg_gen, rg_gen, ud_gen = solver.build_solver_model(lrGen, modelGen, gradsGen,
                                        comb_inp_list[:-1], costs[1], params, updatesLstm)

  print 'model init done.'
  if params['t_eval_only'] == 0:
    print 'Gen model has keys: ' + ', '.join(modelGen.keys())
  print 'Eval model has keys: ' + ', '.join(modelEval.keys())

  # calculate how many iterations we need, One epoch is considered once going through all the sentences and not images
  # Hence in case of coco/flickr this will 5* no of images
  num_sentences_total = dp.getSplitSize('train', ofwhat = 'images')
  num_iters_one_epoch = num_sentences_total / batch_size
  max_iters = max_epochs * num_iters_one_epoch
  skip_first = 1
  iters_eval= 2500
  iters_gen = 1000

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

  len_hist = defaultdict(int)
  t_print_sec =30
  ## Initialize the model parameters from the checkpoint file if we are resuming training
  if params['checkpoint_file_name'] != 'None':
    if params['t_eval_only'] !=1:
        zipp(model_init_gen_from,modelGen)
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
  if params['t_eval_only']==0:
    disp_some_gen_samps(f_gen_only, dp, params, misc, n_samp = 5)
  eval_discrm_gen('val', dp, params, f_pred_fns[0], misc)
  for it in xrange(max_epochs):
    # Enable using dropout in training
    #use_dropout_eval.set_value(1.)
    evaluator.use_noise.set_value(1.)
    it2 = 0
    if params['t_eval_only'] != 1:
        smooth_train_cost = 0.0 # initially size of dictionary of confusion
    while it2 < iters_eval*skip_first:
    #for it2 in xrange(iters_eval*skip_first):
        t0 = time.time()
        # fetch a batch of data
        if params.get('merge_dim',-1)==-1:
          s_probs = [0.6, 0.0, 0.4] if params['eval_loss'] == 'contrastive' else [0.5, 0.4, 0.1]
        else:
          s_probs = [0.4, 0.4, 0.2] if params['n_gen_samples'] > 1 else [0.5, 0.5, 0.0]
        batch = dp.sampAdversBatch(batch_size, n_sent=params['n_gen_samples'], probs = s_probs)
        cnn_inps = prepare_adv_data(batch,misc['wordtoix'],maxlen = params['maxlen'], prep_for=params['eval_model'])

        #import pdb;pdb.set_trace()
        #if params.get('contrastive_loss',1):
        #    cnn_inps = cnn_inps[:2]

        # evaluate cost, gradient and perform parameter update
        cost = f_grad_comp_eval(*cnn_inps)
        if np.isnan(cost):
            import pdb;pdb.set_trace()

        f_param_update_eval(params['learning_rate_eval'])
        dt = time.time() - t0
        # Track training statistics
        if it == 0: smooth_train_cost = cost
        else: smooth_train_cost = 0.99 * smooth_train_cost + 0.01 * cost

        tnow = time.time()
        if tnow > last_status_write_time + t_print_sec*1: # every now and then lets write a report
          print 'Eval Cnn in epoch %d: %d/%d sample done in %.3fs. Cost now is %.6f' % (it, it2, iters_eval, dt, \
	    	smooth_train_cost)
          last_status_write_time = tnow
        it2 +=1
    #print 'Done training the descriminative model for now. Switching to Genereative model'
    print 'Eval N/W in epoch %d: Cost now is %.3f' % (it, smooth_train_cost)
    #evaluator.use_noise.set_value(0.)

    if it >= 0:
        skip_first = 1

    if params['t_eval_only'] or it%5==1:
        disc_acc, gen_acc = eval_discrm_gen('val', dp, params, f_pred_fns[0], misc)
        if disc_acc > top_val_ppl2 or top_val_ppl2 < 0:
          if disc_acc > write_checkpoint_ppl_threshold or write_checkpoint_ppl_threshold < 0:
            # if we beat a previous record or if this is the first time
            # AND we also beat the user-defined threshold or it doesnt exist
            top_val_ppl2 = disc_acc
            filename = 'advmodel_checkpoint_%s_%s_%s_%d_%.2f_EVOnly.p' % (params['dataset'], host, params['fappend'],it, disc_acc)
            dumpCheckpoint(filename, params, modelGen, modelEval, misc, it, disc_acc)

    #if it == 0:
    #    import pdb; pdb.set_trace()

    # Disable Cnn dropout while training gen network
    if params['t_eval_only'] == 0:
     # use_dropout_eval.set_value(0.)
      it2 = 0
      smooth_train_cost_gen = 1.0
      while (smooth_train_cost_gen >-0.5 and 0) or it2 < iters_gen:
          #for it2 in xrange(iters_gen):
          t0 = time.time()
          # fetch a batch of data
          if params.get('merge_dim',-1)==-1:
            s_probs = [1.0, 0.0, 0.0]
          else:
            s_probs = [0.5, 0.4, 0.1]
          batch = dp.sampAdversBatch(batch_size, n_sent=params['n_gen_samples'], probs = s_probs)
          cnn_inps = prepare_adv_data(batch,misc['wordtoix'],maxlen = params['maxlen'], prep_for=params['eval_model'])
          #import pdb; pdb.set_trace()

          # evaluate cost, gradient and perform parameter update
          #if any([np.isnan(modelGen[m].get_value()).any() for m in modelGen]):
          #    print 'Somebodys NAN!!!'
          #    break;
          #import pdb;pdb.set_trace()
          #asd = f_gen_only(real_inp_list[2],real_inp_list[3])

          #print it2,asd[-1].shape, real_inp_list[0].shape

          #if asd[-1].shape[0] > real_inp_list[0].shape[0]:
          #   import pdb; pdb.set_trace()

          cost = f_grad_comp_gen(*cnn_inps[:-1])
          #print it2,cost

          #if any([np.isnan(zg_gen[i].get_value()).any() for i in xrange(len(zg_gen))]):
          #    print 'Somebody zg is NAN!!!'
          #    break;
          #if any([np.isnan(rg_gen[i].get_value()).any() for i in xrange(len(rg_gen))]) or any([(rg_gen[i].get_value()<0).any() for i in xrange(len(rg_gen))]):
          #    print 'Somebody rg is NAN!!!'
          #    break;

          f_param_update_gen(params['learning_rate_gen'])
          dt = time.time() - t0
          # print training statistics
          if it2 == 0: smooth_train_cost_gen = cost
          else: smooth_train_cost_gen = 0.99 * smooth_train_cost_gen + 0.01 * cost

          #if smooth_train_cost_gen < 0.:
          #    import pdb;pdb.set_trace()

          tnow = time.time()
          if tnow > last_status_write_time + t_print_sec*1: # every now and then lets write a report
              print 'Gen Lstm in epoch %d: %d/%d sample done in %.3fs. Cost now is %.8f, temp: %.4f' % (it, it2, iters_gen, dt, \
              	smooth_train_cost_gen, modelGen['gumb_temp'].get_value())
              last_status_write_time = tnow
          it2 +=1

    if params['t_eval_only'] == 0:
        #print 'Done training the generative model for now. Switching to Genereative model. Final Stats are:'
        print 'Gen Lstm in epoch %d: Cost now is %.3f' % (it, smooth_train_cost_gen)

    if it%1==0 and (params['t_eval_only']==0):
        disp_some_gen_samps(f_gen_only, dp, params, misc, n_samp = 5)

    ## perform perplexity evaluation on the validation set and save a model checkpoint if it's good
    if params['t_eval_only'] == 0 and 0:
        is_last_iter = (it+1) == max_iters
        is_last_iter = 1
        if (((it+1) % eval_period_in_iters) == 0 and it < max_iters - 5) or is_last_iter:
          # Disable using dropout in validation
         # use_dropout.set_value(0.)

         # val_ppl2 = eval_split_theano('val', dp, model, params, misc,f_eval) # perform the evaluation on VAL set

          d_acc, g_acc = eval_discrm_gen('val', dp, params, f_pred_fns[0], misc)
          if g_acc > top_val_ppl2 or top_val_ppl2 < 0:
            if g_acc > write_checkpoint_ppl_threshold or write_checkpoint_ppl_threshold < 0:
              # if we beat a previous record or if this is the first time
              # AND we also beat the user-defined threshold or it doesnt exist
              #top_val_ppl2 = val_ppl2
              filename = 'advmodel_checkpoint_%s_%s_%s_%d_%.2f_GenDone.p' % (params['dataset'], host, params['fappend'],it, g_acc)
              dumpCheckpoint(filename, params, modelGen, modelEval, misc, it, g_acc)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # global setup settings, and checkpoints
  parser.add_argument('-d', '--dataset', dest='dataset', default='coco', help='dataset: flickr8k/flickr30k')
  parser.add_argument('--fappend', dest='fappend', type=str, default='baseline', help='append this string to checkpoint filenames')
  parser.add_argument('-o', '--checkpoint_output_directory', dest='checkpoint_output_directory', type=str, default='cv/', help='output directory to write checkpoints to')
  parser.add_argument('--worker_status_output_directory', dest='worker_status_output_directory', type=str, default='status/', help='directory to write worker status JSON blobs to')
  parser.add_argument('--write_checkpoint_ppl_threshold', dest='write_checkpoint_ppl_threshold', type=float, default=-1, help='ppl threshold above which we dont bother writing a checkpoint to save space')
  parser.add_argument('--continue_training', dest='checkpoint_file_name', type=str, default='None', help='checkpoint file from which to resume training')
  parser.add_argument('--use_pos_tag', dest='use_pos_tag', type=str, default='None', help='use_pos_tag')

  # Some parameters about image features used
  parser.add_argument('--feature_file', dest='feature_file', type=str, default='vgg_feats.mat', help='Which file should we use for read the CNN features')
  parser.add_argument('--image_feat_size', dest='image_feat_size', type=int, default=4096, help='size of the input image features')
  parser.add_argument('--data_file', dest='data_file', type=str, default='dataset.json', help='Which dataset file shpuld we use')
  parser.add_argument('--mat_new_ver', dest='mat_new_ver', type=int, default=-1, help='If the .mat feature files are saved with new version (compressed) set this flag to 1')
  parser.add_argument('--aux_inp_file', dest='aux_inp_file', type=str, default='None', help='Is there any auxillary inputs ? If yes indicate file here')
  parser.add_argument('--swap_AuxFeat', dest='swap_aux', type=int, default=1, help='Feed image features through auxillary input!')
  parser.add_argument('--advers_gen', dest='advers_gen', type=str, default=1, help='Should we use adverserial generator!')
  parser.add_argument('--eval_model', dest='eval_model', type=str, default='cnn', help='which evaluator model to use type: cnn/lstm_eval')

  # model parameters
  parser.add_argument('--image_encoding_size', dest='image_encoding_size', type=int, default=512, help='size of the image encoding')
  parser.add_argument('--word_encoding_size', dest='word_encoding_size', type=int, default=512, help='size of word encoding')
  parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=512, help='size of hidden layer in generator RNNs')
  parser.add_argument('--hidden_depth', dest='hidden_depth', type=int, default=1, help='depth of hidden layer in generator RNNs')
  parser.add_argument('--generator', dest='generator', type=str, default='lstm', help='generator to use')
  parser.add_argument('-c', '--regc', dest='regc', type=float, default=0., help='regularization strength')
  parser.add_argument('--tanhC_version', dest='tanhC_version', type=int, default=0, help='use tanh version of LSTM?')
  parser.add_argument('--n_gen_samples', type=int, default=1, help='beam size in inference. 1 indicates greedy per-word max procedure. Good value is approx 20 or so, and more = better.')

  parser.add_argument('--sent_encoding_size', dest='sent_encoding_size', type=int, default=300, help='size of sentence encoding layer on top of CNN')
  parser.add_argument('--n_fmaps', dest='n_fmaps_psz', type=int, default=100, help='number of cnn feature maps per filter height')
  parser.add_argument('--filter_hs', dest='filter_hs', metavar='N', type=int, nargs='+',default =[3,4,5], help='list fo filter heights to use in CNN')
  parser.add_argument('--conv_non_linear', dest='conv_non_linear', type=str, default='relu', help='nonlinearity type: tanh/relu')
  parser.add_argument('--maxlen', dest='maxlen', type=int, default=15, help='size of sentence encoding layer on top of CNN')
  parser.add_argument('--merge_dim', dest='merge_dim', type=int, default=-1, help='size of sentence encoding layer on top of CNN')

  # Regarding word embedding sharing and such
  parser.add_argument('--share_Wemb', dest='share_Wemb', type=int, default=0, help='If 1, share Wemb b/w eval and gen models')
  parser.add_argument('--fix_Wemb', dest='fix_Wemb', type=int, default=0, help='If 1, gen model doesnt learn Wemb and keeps it fixed')

  # optimization parameters
  parser.add_argument('-m', '--max_epochs', dest='max_epochs', type=int, default=10, help='number of epochs to train for')
  parser.add_argument('--solver', dest='solver', type=str, default='rmsprop', help='solver type: vanilla/adagrad/adadelta/rmsprop')
  parser.add_argument('--decay_rate', dest='decay_rate', type=float, default=0.999, help='decay rate for adadelta/rmsprop')
  parser.add_argument('--smooth_eps', dest='smooth_eps', type=float, default=1e-8, help='epsilon smoothing for rmsprop/adagrad/adadelta')
  parser.add_argument('-lg', '--learning_rate_gen', dest='learning_rate_gen', type=float, default=1e-4, help='solver learning rate')
  parser.add_argument('-ld', '--learning_rate_eval', dest='learning_rate_eval', type=float, default=1e-4, help='solver learning rate')

  parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=1, help='batch size')
  parser.add_argument('-cb', '--eval_batch_size', dest='eval_batch_size', type=int, default=2, help='Batch size for eval descriminative network, 1 means it only gets a positive reference\
                                                                                                     and a generated sample, n implies it also gets n-1 negative references ')
  parser.add_argument('--rand_negs', dest='rand_negs', type=int, default=0, help='How many hard negetives obtianed by random permutations of positive to use to train eval n/w')
  parser.add_argument('--sample_by_len', dest='sample_by_len', type=int, default=1, help='enable sampling by length of sentece to speed up training')
  parser.add_argument('--grad_clip', dest='grad_clip', type=float, default=10.0, help='clip gradients (normalized by batch size)? elementwise. if positive, at what threshold?')
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
  parser.add_argument('--use_gumbel_mse', dest='use_gumbel_mse', type=int, default=0, help='Should img or aux features be read from disk')

  # parameters for loading multiple features per video using labels.txt
  parser.add_argument('--labelsFile', dest='labels', type=str, default='labels.txt', help='labels.txt file for this dataset')
  parser.add_argument('--featfromlbl', dest='featfromlbl', type=str, default='ALL ALL', help='should we use lables.txt, if yes which feature?'
                  'use + sign to seperately specify for img and aux')
  parser.add_argument('--poolmethod', dest='poolmethod', type=str, default='max', help='What pooling to use if multiple features are found')
  parser.add_argument('--uselabel', dest='uselabel', type=int, default=3, help='which features should use labels.txt, img/aux or both, 0 - None, 1 - img, 2 - aux, 3 - both')
  # Some params to enable partial feature reading from disk!!
  parser.add_argument('--disk_feature', dest='disk_feature', type=int, default=3, help='Should img or aux features be read from disk')

  parser.add_argument('--train_evaluator_only', dest='t_eval_only', type=int, default=0, help='which features should use labels.txt, img/aux or both, 0 - None, 1 - img, 2 - aux, 3 - both')
  parser.add_argument('--eval_loss', dest='eval_loss', type=str, default='contrastive', help='which features should use labels.txt, img/aux or both, 0 - None, 1 - img, 2 - aux, 3 - both')
  parser.add_argument('--eval_w_clip', dest='eval_w_clip', type=float, default=None, help='which features should use labels.txt, img/aux or both, 0 - None, 1 - img, 2 - aux, 3 - both')

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
    ##TODO: GET RID OF THEEEESSSEEEEEE !!!!!!!!!!!!!!!!
    #for k in checkpoint_init['params']:
    #  params[k] = checkpoint_init['params'][k]

  if params['aux_inp_file'] != 'None':
    params['en_aux_inp'] = 1
  else:
    params['en_aux_inp'] = 0

  params['eval_batch_size'] += params['rand_negs']

  if params['use_pos_tag'] != 'None':
    sentTagMap = pickle.load(open(params['use_pos_tag'],'r'))
  print 'parsed parameters:'
  print json.dumps(params, indent = 2)
  config.mode = 'FAST_RUN'
  config.allow_gc = False

  main(params)
