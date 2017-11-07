import theano
import argparse
import numpy as np
import cPickle as pickle
from imagernn.data_provider import getDataProvider, prepare_data, prepare_adv_data
from imagernn.imagernn_utils import decodeGenerator, eval_split_theano, eval_prep_refs
from imagernn.utils import numpy_floatX, zipp, unzip, preProBuildWordVocab
from imagernn.data_provider import getDataProvider, prepare_data, prepare_seq_features
import os
import os.path as osp

def main(scriptparams):
    checkpoint = pickle.load(open(scriptparams['checkpoint'], 'rb'))
    npfilename = osp.join('scorelogs',osp.basename(scriptparams['checkpoint']).split('.')[0]+'_logprob%s'%(scriptparams['split']))
    misc = checkpoint['misc']

    # fetch the data provider
    params = checkpoint['params']
    params['use_gumbel_mse'] = 0
    params['maxlen'] = scriptparams['maxlen']

    dp = getDataProvider(params)
    model_init_gen_from = checkpoint.get('model',{}) if 'model' in checkpoint else checkpoint['modelGen']

    lstmGenerator = decodeGenerator(params)
    model, misc['update'], misc['regularize'] = (lstmGenerator.model_th, lstmGenerator.update_list, lstmGenerator.regularize)


    if params.get('use_encoder_for',0)&1:
      if params.get('encode_gt_sentences',0):
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

    if params.get('use_encoder_for',0)&2:
      aux_enc_inp = model['Wemb'] if params.get('encode_gt_sentences',0) else dp.aux_inputs.T
      hid_size = params['featenc_hidden_size']
      auxFeatEncoder = RecurrentFeatEncoder(hid_size, params['image_encoding_size'], params,
              mdl_prefix='aux_enc_', features=aux_enc_inp)
      mdlLen = len(model.keys())
      model.update(auxFeatEncoder.model_th)
      assert(len(model.keys()) == (mdlLen+len(auxFeatEncoder.model_th.keys())))
      misc['update'].extend(auxFeatEncoder.update_list)
      misc['regularize'].extend(auxFeatEncoder.regularize)
      (auxenc_use_dropout, auxFeatEnc_inp, xAux, updatesLSTMAuxFeat) = auxFeatEncoder.build_model(model, params)

      if params.get('encode_gt_sentences',0):
          # Reshape it size(batch_size, n_gt, hidden_size)
          xAux = xAux.reshape((-1,params['n_encgt_sent'],params['featenc_hidden_size']))
          # Convert it to size (batch_size, n_gt*hidden_size
          xAux = xAux.flatten(2)
    else:
      auxFeatEnc_inp = []
      xAux = None

    attn_nw_func = None

    (use_dropout, inp_list_gen,
       f_pred_prob, cost, predTh, updatesLSTM) = lstmGenerator.build_model(model, params, xI, xAux, attn_nw = attn_nw_func)
    inp_list = imgFeatEnc_inp + auxFeatEnc_inp + inp_list_gen


    f_eval= theano.function(inp_list, cost, name='f_eval')
    #--------------------------------- Cost function and gradient computations setup #---------------------------------#

    zipp(model_init_gen_from,model)
    # perform the evaluation on VAL set
    #val_sc = eval_split_theano(scriptparams['split'], dp, model, params, misc, f_eval)
    logppl = []
    logppln = []
    imgids = []
    nsent = 0

    for batch in dp.iterImageSentencePairBatch(split = scriptparams['split'], max_batch_size = 1, max_images = -1):
      enc_inp_list = prepare_seq_features( batch, use_enc_for= params.get('use_encoder_for',0), maxlen =  params['maxlen'],
              use_shared_mem = params.get('use_shared_mem_enc',0), enc_gt_sent = params.get('encode_gt_sentences',0),
              n_enc_sent = params.get('n_encgt_sent',0), wordtoix = misc['wordtoix'])
      gen_inp_list, lenS = prepare_data(batch, misc['wordtoix'], rev_sents=params.get('reverse_sentence',0)
                      ,use_enc_for= params.get('use_encoder_for',0), use_unk_token = params.get('use_unk_token',0))

      inp_list = enc_inp_list + gen_inp_list
      cost = f_eval(*inp_list)
      logppl.append(cost[1])
      logppln.append(lenS)
      imgids.append(str(batch[0]['image']['cocoid']) + '_' + str(batch[0]['sentidx']))
      nsent += 1

    perplex = 2 ** (np.array(logppl) / np.array(logppln))
    np.savez(npfilename, pplx = perplex, keys = np.array(imgids))

    #ppl2 = 2 ** (logppl / logppln)
    #print 'evaluated %d sentences and got perplexity = %f' % (nsent, ppl2)
    #met = [ppl2]

    print  2 ** (np.array(logppl).sum() / np.array(logppln).sum())

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # global setup settings, and checkpoints
  parser.add_argument('-c', dest='checkpoint', type = str, default=None, help='dataset: flickr8k/flickr30k')
  # Track some metrics during training
  parser.add_argument('-s', dest='split', type = str, default='test', help='dataset: flickr8k/flickr30k')
  parser.add_argument('--maxlen', dest='maxlen', type = int, default=100, help='dataset: flickr8k/flickr30k')

  args = parser.parse_args()
  cur_params = vars(args) # convert to ordinary dict
  main(cur_params)
