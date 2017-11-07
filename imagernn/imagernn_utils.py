#from imagernn.generic_batch_generator import GenericBatchGenerator
from imagernn.lstm_generatorTheano import LSTMGenerator
from imagernn.bidir_lstm_generatorTheano import BiLSTMGenerator
from imagernn.lstm_evaluatorTheano import LSTMEvaluator
from imagernn.cnn_evaluatorTheano import CnnEvaluator
from imagernn.data_provider import prepare_data, prepare_seq_features
from imagernn.utils import compute_global_div_n, compute_div_n
import numpy as np
from collections import defaultdict
from eval.mseval.pycocoevalcap.meteor.meteor import Meteor
from eval.mseval.pycocoevalcap.spice.spice import Spice
from eval.mseval.pycocoevalcap.bleu.bleu import Bleu
from eval.mseval.pycocoevalcap.rouge.rouge import Rouge
from eval.mseval.pycocoevalcap.cider.cider import Cider
from eval.mseval.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

def decodeGenerator(params):
  """
  in the future we may want to have different classes
  and options for them. For now there is this one generator
  implemented and simply returned here.
  """
  if params.get('generator','lstm') == 'lstm':
    return LSTMGenerator(params)
  elif params.get('generator','lstm') == 'bilstm':
    return BiLSTMGenerator(params)
  else:
    return 0 #GenericBatchGenerator


def decodeEvaluator(params, Wemb = None):
  """
  For now there are two evaluator models
  implemented and returned here.
  """
  print 'Initializing Model %s'%(params['eval_model'])
  if params['eval_model'] == 'lstm_eval':
    return LSTMEvaluator(params)
  elif params['eval_model'] == 'cnn':
    return CnnEvaluator(params, Wemb = None)
  elif params['eval_model'] == 'bilstm_gen':
    return BiLSTMGenerator(params)
  else:
    raise ValueError('ERROR: %s --> Unsupported Model'%(params['eval_model']))
    return 0 #GenericBatchGenerator

def eval_split(split, dp, model, params, misc, **kwargs):
  """ evaluate performance on a given split """
  # allow kwargs to override what is inside params
  eval_batch_size = kwargs.get('eval_batch_size', params.get('eval_batch_size',100))
  eval_max_images = kwargs.get('eval_max_images', params.get('eval_max_images', -1))
  BatchGenerator = decodeGenerator(params)
  wordtoix = misc['wordtoix']

  print 'evaluating %s performance in batches of %d' % (split, eval_batch_size)
  logppl = 0
  logppln = 0
  nsent = 0
  for batch in dp.iterImageSentencePairBatch(split = split, max_batch_size = eval_batch_size, max_images = eval_max_images):
    Ys, gen_caches = BatchGenerator.forward(batch, model, params, misc, predict_mode = True)

    for i,pair in enumerate(batch):
      gtix = [ wordtoix[w] for w in pair['sentence']['tokens'] if w in wordtoix ]
      gtix.append(0) # we expect END token at the end
      Y = Ys[i]
      maxes = np.amax(Y, axis=1, keepdims=True)
      e = np.exp(Y - maxes) # for numerical stability shift into good numerical range
      P = e / np.sum(e, axis=1, keepdims=True)
      logppl += - np.sum(np.log2(1e-20 + P[range(len(gtix)),gtix])) # also accumulate log2 perplexities
      logppln += len(gtix)
      nsent += 1

  ppl2 = 2 ** (logppl / logppln)
  print 'evaluated %d sentences and got perplexity = %f' % (nsent, ppl2)
  return ppl2 # return the perplexity

class lenComputer:
# ========================================================================================

  def compute_score(self, gts, candToks):
      lenArr = np.array([len(cS.split()) for k in candToks for cS in candToks[k]])
      return lenArr.mean(), lenArr

class divComputer:
# ========================================================================================
  def __init__(self, n):
    self.div_n = n

  def compute_score(self, gts, candToks):
    div_sc, div_sc2= compute_global_div_n(candToks, self.div_n)
    return div_sc, div_sc2

class lcldivComputer:
# ========================================================================================
  def __init__(self, n):
    self.div_n = n

  def compute_score(self, gts, candToks):
    div_sc, div_sc2 = compute_div_n(candToks, self.div_n)
    return div_sc, div_sc2


def eval_prep_refs(split, dp, eval_metric):
  refsById = defaultdict(list)
  for s in dp.iterSentences(split=split):
    refsById[s['cocoid']].append({'image_id':s['cocoid'],'id':s['sentid'],'caption':s['raw']})

  tokenizer = PTBTokenizer()
  refsById = tokenizer.tokenize(refsById)
  if type(eval_metric) != list:
    eval_metric = [eval_metric]

  scorer = []
  scorer_name = []
  for evm in eval_metric:
    if evm == 'meteor':
      scorer.append(Meteor())
      scorer_name.append("METEOR")
    elif evm == 'spice':
      scorer.append(Spice())
      scorer_name.append("Spice")
    elif evm == 'cider':
      scorer.append(Cider())
      scorer_name.append("CIDEr")
    elif evm == 'rouge':
      scorer.append(Rouge())
      scorer_name.append("ROUGE_L")
    elif evm[:4] == 'bleu':
      bn = int(evm.split('_')[1])
      scorer.append(Bleu(bn))
      scorer_name_temp = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]
      scorer_name.append(scorer_name_temp[:bn])
    elif evm == 'len':
      scorer.append(lenComputer())
      scorer_name.append("Mean_Len")
    elif evm[:3] == 'div':
      dn = int(evm.split('_')[1])
      scorer.append(divComputer(dn))
      scorer_name.append("Global_Div_"+str(dn))
    elif evm[:6] == 'lcldiv':
      dn = int(evm.split('_')[1])
      scorer.append(lcldivComputer(dn))
      scorer_name.append("Local_Div_"+str(dn))
    else:
      raise ValueError('ERROR: %s --> Unsupported eval metric'%(evm))

  return refsById,{'scr_fn':scorer, 'scr_name':scorer_name, 'tokenizer':tokenizer}


def eval_split_theano(split, dp, model, params, misc, gen_fprop, **kwargs):
  """ evaluate performance on a given split """
  # allow kwargs to override what is inside params
  eval_batch_size = kwargs.get('eval_batch_size', params.get('eval_batch_size',100))
  eval_max_images = kwargs.get('eval_max_images', params.get('eval_max_images', -1))
  eval_metric = kwargs.get('eval_metric','perplex')

  wordtoix = misc['wordtoix']

  print 'evaluating %s performance in batches of %d' % (split, eval_batch_size)
  logppl = 0
  logppln = 0
  nsent = 0
  if eval_metric == 'perplex':
    for batch in dp.iterImageSentencePairBatch(split = split, max_batch_size = eval_batch_size, max_images = eval_max_images):
      enc_inp_list = prepare_seq_features( batch, use_enc_for= params.get('use_encoder_for',0), maxlen =  params['maxlen'],
              use_shared_mem = params.get('use_shared_mem_enc',0), enc_gt_sent = params.get('encode_gt_sentences',0),
              n_enc_sent = params.get('n_encgt_sent',0), wordtoix = misc['wordtoix'])
      gen_inp_list, lenS = prepare_data(batch,wordtoix, rev_sents=params.get('reverse_sentence',0)
                      ,use_enc_for= params.get('use_encoder_for',0), use_unk_token = params.get('use_unk_token',0))

      if params.get('sched_sampling_mode',None) != None:
          gen_inp_list.append(0.0)
          # This is making sure we don't sample from prediction path for evaluation

      inp_list = enc_inp_list + gen_inp_list
      cost = gen_fprop(*inp_list)
      logppl += cost[1]
      logppln += lenS
      nsent += eval_batch_size

    ppl2 = 2 ** (logppl / logppln)
    print 'evaluated %d sentences and got perplexity = %f' % (nsent, ppl2)
    met = [ppl2]
  else:
    if type(eval_metric) != list:
      eval_metric = [eval_metric]
    refToks = kwargs.get('refToks',None)
    candToks = {}
    beamsize = kwargs.get('beamsize',1)
    predFn = kwargs.get('f_gen')
    ixtoword = misc['ixtoword']
    n = 0
    gts = {}
    for img in dp.iterImages(split = split,shuffle=True, max_images = eval_max_images):
        imgid = img['cocoid']
        gts[imgid] = refToks[imgid]
        Ys = predFn([{'image':img}], params,beam_size=beamsize,ext_inp = [])
        # Only considering the top candidate in Ys, which is [0]
        # ix 0 is the END token, skip that)
        if params.get('reverse_sentence',0) == 0:
            candidate = ' '.join([ixtoword[int(ix)] for ix in Ys[0][0][1] if ix > 0])
        else:
            candidate = ' '.join([ixtoword[int(ix)] for ix in reversed(Ys[0][0][1]) if ix > 0])

        candToks[imgid] = [{'image_id':imgid,'caption':candidate,'id':n}]
        n += 1
        print '\rn=',n,

    # Now tokenize the candidates before calling the eval
    candToks = kwargs['scr_info']['tokenizer'].tokenize(candToks)

    # Now invoke all the scorers and get the scores
    met = []
    for i,evm in enumerate(eval_metric):
        score, scores = kwargs['scr_info']['scr_fn'][i].compute_score(gts, candToks)
        if type(kwargs['scr_info']['scr_name'][i]) == list:
            met.append(score[-1])
        else:
            met.append(score)
        print 'evaluated %d sentences and got %s = %f' % (n, evm, met[-1])

  return met# return the perplexity
