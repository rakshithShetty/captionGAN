import argparse
import numpy as np
import cPickle as pickle
from imagernn.data_provider import getDataProvider, prepare_data, prepare_adv_data
from imagernn.cnn_evaluatorTheano import CnnEvaluator
from imagernn.solver import Solver
from imagernn.imagernn_utils import decodeEvaluator, decodeGenerator, eval_split_theano
#from numbapro import cuda
from imagernn.utils import numpy_floatX, zipp, unzip, preProBuildWordVocab
from collections import defaultdict, OrderedDict
import signal

def eval_discrm_gen(split, dp, params, gen_fprop, misc, n_eval=None, probs=[1.0,0.0,0.0]):
    n_eval = len(dp.split[split])
    n_iter = (n_eval-1) // params['eval_batch_size'] + 1
    correct = 0.
    n_total = 0.
    g_correct = 0.
    mean_p = 0.
    mean_n = 0.
    mean_g = 0.
    for i in xrange(n_iter):
        batch = dp.sampAdversBatch(params['eval_batch_size'], n_sent=params['n_gen_samples'], probs = probs)
        cnn_inps = prepare_adv_data(batch,misc['wordtoix'],maxlen = params['maxlen'], prep_for=params['eval_model'])
        if params['t_eval_only'] == 0:
            p_out = gen_fprop(*[cnn_inps[0], cnn_inps[1], cnn_inps[3]])
        else:
            p_out = gen_fprop(*cnn_inps[:-1])
        y = cnn_inps[-1]
        correct += ((p_out[0].flatten()>0.) == y).sum()

        mean_p += (p_out[0][:cnn_inps[-1].shape[0]]*cnn_inps[-1]).mean()
        mean_n += (p_out[0][:cnn_inps[-1].shape[0]]*(1-cnn_inps[-1])).mean()
        n_total += y.shape[0]

    acc = correct/n_total * 100.0
    mean_p = mean_p/(n_iter)
    mean_n = mean_n/(n_iter)
    print 'evaluated the discriminator. Current disc accuracy is %.2f'%(acc)
    print 'Mean scores are pos: %.2f neg: %.2f'%(mean_p, mean_n)

    return acc

def main(cur_params):
  # fetch the data provider
  for i, cpf in enumerate(cur_params['checkpoints']):
    checkpoint = pickle.load(open(cpf, 'rb'))
    if 'model' in checkpoint:
        model_init_gen_from = checkpoint.get('model',{})
    else:
        model_init_gen_from = checkpoint.get('modelGen',{})
    model_init_eval_from = checkpoint.get('modelEval',{})
    params = checkpoint['params']

    # Load data provider and copy misc
    if i == 0:
        dp = getDataProvider(params)
        evaluator = decodeEvaluator(params)
        modelEval = evaluator.model_th
        (eval_inp_list, f_pred_fns, costs, predTh, modelEval) = evaluator.build_advers_eval(modelEval, params)

    misc = checkpoint['misc']

    zipp(model_init_eval_from, modelEval)
    evaluator.use_noise.set_value(1.)

    print '----------------------- Running model %s  -------------------------------'%(cpf.split('_')[-3])
    print 'Evaluating GT 5 vs Negative samples from GT'
    eval_discrm_gen('val', dp, params, f_pred_fns[0], misc, probs = [0.5, 0.5, 0.0])
    print '-------------------------------------------------------------------------'
    print 'Evaluating GT vs repeated GT'
    eval_discrm_gen('val', dp, params, f_pred_fns[0], misc, probs = [0.5, 0.0, 0.5])
    print '-------------------------------------------------------------------------'

  # Initialize the optimizer

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # global setup settings, and checkpoints
  parser.add_argument('-c', dest='checkpoints', type = str, nargs='+', default=[], help='dataset: flickr8k/flickr30k')

  args = parser.parse_args()
  cur_params = vars(args) # convert to ordinary dict
  main(cur_params)
