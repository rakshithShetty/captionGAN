import numpy as np
import code
import theano
from theano import config
import theano.tensor as T
from theano.ifelse import ifelse
from collections import OrderedDict
import time
from imagernn.utils import _p
from imagernn.utils import *
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from copy import copy


class DiscrimLayer:
  def __init__(self, n_inp_dim = 100, n_kers=100, n_dim_per_kernal = 5, prefix='discrim',init=0.00005):
    self.name = prefix
    self.n_kernals = n_kers
    model = OrderedDict()
    model[self.name+'_W'] = initwThNd([n_inp_dim, n_kers, n_dim_per_kernal], init)
    model[self.name+'_b'] = np.zeros(n_kers).astype(config.floatX)

    self.model_th = init_tparams(model)
    del model

  def par(self, n):
      return self.model_th[self.name+'_'+n]

  def fwd(self, x, y =None):
      # x is assumed to be batch * n_samp * enc_size
      act = T.tensordot(x, self.par('W'), [[x.ndim-1],[0]])
      if y == None:
        abs_dif = (T.sum(abs(act.dimshuffle(0,1,2,3,'x') - act.dimshuffle(0,'x',2,3,1)),axis=3)
                + 1e-6 * T.eye(x.shape[1]).dimshuffle('x',0,'x',1))
        f = T.sum(T.exp(-abs_dif),axis=-1)
      else:
        act_y = T.tensordot(y, self.par('W'), [[y.ndim-1],[0]])
        abs_dif = (T.sum(abs(act - act_y.dimshuffle(0,'x',1,2)),axis=-1) + 1e-8)
        f = T.exp(-abs_dif)

      f += self.par('b').dimshuffle('x','x',0)
      # f will be of size batch * n_samp * ker_size
      return f

class LSTMEvaluator:
  """
  A multimodal long short-term memory (LSTM) generator
  """
# ========================================================================================
  def __init__(self, params, Wemb=None):

    word_encoding_size = params.get('word_encoding_size', 128)
    self.word_encoding_size = word_encoding_size
    aux_inp_size = params.get('aux_inp_size', -1)
    sent_encoding_size = params.get('sent_encoding_size',-1)# size of CNN vectors hardcoded here
    self.sent_enc_size = sent_encoding_size
    self.eval_feature = params.get('eval_feature','image_feat')

    # Output state is the sentence encoding
    output_size = sent_encoding_size

    self.en_residual_conn = params.get('eval_en_residual_conn',0)
    hidden_size = params.get('eval_hidden_size', 512)
    self.hidden_size = hidden_size
    hidden_depth = params.get('eval_hidden_depth', 1)
    self.hidden_depth = hidden_depth

    generator = params.get('generator', 'lstm')
    vocabulary_size = params.get('vocabulary_size',-1)
    image_feat_size = params.get(self.eval_feature + '_size',-1)# size of CNN vectors hardcoded here

    model = OrderedDict()
    # Recurrent weights: take x_t, h_{t-1}, and bias unit
    # and produce the 3 gates and the input to cell signal
    img_encoding_size = self.sent_enc_size if(params.get('multimodal_lstm',0) == 0) else word_encoding_size
    model['WIemb'] = initwTh(image_feat_size, self.sent_enc_size, inittype='xavier') # image encoder
    #model['b_Img'] = np.zeros((img_encoding_size)).astype(config.floatX)

    if Wemb == None:
        model['Wemb'] = initwTh(vocabulary_size, word_encoding_size) # word encoder

    model['lstm_W_hid'] = initwTh(hidden_size, 4 * hidden_size)
    model['lstm_W_inp'] = initwTh(word_encoding_size, 4 * hidden_size)

    for i in xrange(1,hidden_depth):
        model['lstm_W_hid_'+str(i)] = initwTh(hidden_size, 4 * hidden_size)
        model['lstm_W_inp_'+str(i)] = initwTh(hidden_size, 4 * hidden_size)

    model['lstm_b'] = np.zeros((4 * hidden_size,)).astype(config.floatX)
    # Decoder weights (e.g. mapping to vocabulary)
    if(params.get('multimodal_lstm',0) == 0):
        model['Wd'] = initwTh(hidden_size, output_size)# decoder
        #model['Wd'] = np.concatenate([initwTh(hidden_size, output_size)[None,:,:] for _ in xrange(params.get('n_gen_sample',1))],axis=0) # decoder
        #model['bd'] = np.zeros((output_size,)).astype(config.floatX)

    if params['advers_gen'] and params.get('merge_dim',-1) >0:
        # Add a discrim layers and a softmax in the end
        # First disrcim layer measures distances between the input sentences
        n_distances = 0
        n_kers = params['merge_dim']
        if params.get('n_gen_samples',1) > 1:
            self.sent_discrim = DiscrimLayer(self.sent_enc_size, n_kers= n_kers, prefix='sent_discrim', init=0.00005)
            #n_distances += params.get('n_gen_samples',1)* self.sent_discrim.n_kernals
        #This disrcim layer measures distance between the input sentences and the image
        self.img_discrim = DiscrimLayer(self.sent_enc_size, n_kers= n_kers, prefix='img_discrim', init=0.00005)
        n_distances += params.get('n_gen_samples',1) * self.img_discrim.n_kernals
        n_distances = n_distances * 2 if params.get('n_gen_samples',1) > 1 else n_distances

        # Finally we have n_samp x n_ker distances from sent_discrim and
        # n_samp x n_ker distances from img_discrim

        # we add a dense layer on top to compute one score for all this.
        model['W_disttoprob'] = initwTh(n_distances, 2,inittype='xavier') # word encoder


    self.model_th = init_tparams(model)
    if params['advers_gen'] and params.get('merge_dim',-1) >0:
        if params.get('n_gen_samples',1) > 1:
            self.model_th.update(self.sent_discrim.model_th)
        self.model_th.update(self.img_discrim.model_th)

# ========================================================================================
 # BUILD LSTM forward propogation model
  def build_model(self, tparams, optionsInp):
    trng = RandomStreams(1234)
    options = copy(optionsInp)
    if 'en_aux_inp' in options:
        options.pop('en_aux_inp')
    # Used for dropout.
    self.use_noise = theano.shared(numpy_floatX(0.))

    xW = T.matrix('xW', dtype='int64')
    mask = T.vector('mask', dtype='int64')

    n_Rwords= xW.shape[0]
    n_samples = xW.shape[1]

    embW = tparams['Wemb'][xW.flatten()].reshape([n_Rwords,
                                                n_samples,
                                                options['word_encoding_size']])
    xI = T.matrix('xI', dtype=config.floatX)

    if options.get('multimodal_lstm',0) == 1:
        embImg = T.dot(xI, tparams['WIemb']) + tparams['b_Img']
        embImg = T.shape_padleft(T.extra_ops.repeat(embImg,n_samples,axis=0),n_ones=1)
        emb = T.concatenate([embImg, embW], axis=0)
    else:
        emb = embW

    #This is implementation of input dropout !!
    if options['use_dropout']:
        emb = dropout_layer(emb, self.use_noise, trng, options['drop_prob_encoder'], shp = emb.shape)

    # This implements core lstm
    rval, updatesLSTM = basic_lstm_layer(tparams, emb, [], self.use_noise, options, prefix='lstm')

    if options['use_dropout']:
        p = dropout_layer(sliceT(rval[0][mask + options.get('multimodal_lstm',0),T.arange(mask.shape[0]),:],options.get('hidden_depth',1)-1,
            options['hidden_size']), self.use_noise, trng, options['drop_prob_decoder'], (n_samples,options['hidden_size']))
    else:
        p = sliceT(rval[0][mask + options.get('multimodal_lstm',0),T.arange(mask.shape[0]),:],options.get('hidden_depth',1)-1,options['hidden_size'])


    if options.get('multimodal_lstm',0) == 0:
        sent_emb = (T.dot(p,tparams['Wd']) + tparams['bd'])
        probMatch, sim_score = multimodal_cosine_sim_softmax(xI, sent_emb, tparams, options.get('sim_smooth_factor',1.0))
    else:
        sent_emb = T.sum(p,axis=1).T #(T.dot(p,tparams['Wd'])).T
        sim_score = sent_emb #T.maximum(0.0, sent_emb) #T.tanh(sent_emb)
        smooth_factor = T.as_tensor_variable(numpy_floatX(options.get('sim_smooth_factor',1.0)), name='sm_f')
        probMatch = T.nnet.softmax(sim_score*smooth_factor)

    inp_list = [xW, mask, xI]

    if options.get('mode','batchtrain') == 'batchtrain':
        # In train mode we compare a batch of images against each others captions.
        batch_size = options['batch_size']
        cost = -(T.log(probMatch.diagonal()).sum())/batch_size
    else:
        # In predict mode we compare multiple captions against a single image
        posSamp = T.ivector('posSamp')
        batch_size = posSamp.shape[0]
        cost = -(T.log(probMatch[0,posSamp]).sum())/batch_size
        inp_list.append(posSamp)

    f_pred_sim_prob = theano.function(inp_list[:3], probMatch, name='f_pred_sim_prob')
    f_pred_sim_scr = theano.function(inp_list[:3], sim_score, name='f_pred_sim_scr')
    if options.get('multimodal_lstm',0) == 1:
        f_sent_emb = theano.function([inp_list[0],inp_list[2]], [rval[0],emb], name='f_sent_emb')
    else:
        f_sent_emb = theano.function([inp_list[0]], [rval[0],emb], name='f_sent_emb')

    return self.use_noise, inp_list, [f_pred_sim_prob, f_pred_sim_scr, f_sent_emb, updatesLSTM], cost, sim_score, tparams

# ========================================================================================
 # BUILD LSTM forward propogation eval model with ability to take direct Wemb inputs from gen model
  def build_advers_eval(self, tparams, options, gen_inp_list = [], gen_out=None, genUpdates = None, genLens = None):
    trng = RandomStreams(1234)
    # Used for dropout.
    self.use_noise = theano.shared(numpy_floatX(0.))

    xW = T.tensor3('xW', dtype='int64')
    samp_lens = tensor.vector('sL', dtype='int64')
    inp_list = [xW, samp_lens]
    # Swap axes, lstm needs time axis to be 0th

    n_ref_words= xW.shape[0]
    n_batch_samps = xW.shape[1]
    n_samp = xW.shape[2]

    #zero_guy = T.alloc(numpy_floatX(0.),1,self.word_encoding_size)
    Word_Vecs = tparams['Wemb']#T.concatenate([zero_guy, tparams['Wemb']],axis=0)
    embWRef = Word_Vecs[xW.flatten()].reshape([n_ref_words,
                                               n_batch_samps,
                                               n_samp,
                                               options['word_encoding_size']])

    if gen_out != None:
        # TODO: Temporary hack, until support for multi-sentence eval is added
        n_gen_samps = gen_out.shape[1]
        n_gen_words = gen_out.shape[0]
        gen_emb = T.dot(gen_out, Word_Vecs)
        n_ends = n_ref_words - n_gen_words
        # Zero padding to get right size
        z_shape = list(gen_emb.shape)
        #assert_op = T.opt.Assert()
        z_shape[0] = n_ends#assert_op(n_ends,n_ends>=0)

        emb_gen = ifelse(tensor.gt(n_ref_words, n_gen_words), T.concatenate([gen_emb, T.zeros(z_shape)], axis=0),
                                                              gen_emb[:n_ref_words,:,:,:])
        #emb_gen = ifelse(n_ref_words > n_gen_words,T.concatenate([gen_emb,
        #                  T.tile(tparams['Wemb'][[0]],[n_batch_samps*n_ends,1]).reshape([n_ends,n_batch_samps,-1])], axis=0),
        #                                                       gen_emb[:n_ref_words,:,:])
        # Concatenate by the batch axis
        emb = T.concatenate([embWRef, emb_gen], axis=1)
    else:
        emb = embWRef

    if gen_out == None or options['encode_gt_sentences']:
        xImg = T.matrix('xI', dtype=config.floatX)
        inp_list.append(xImg)
    else:
        xImg = gen_inp_list[0] if options.get('eval_feature','image_feat') == 'image_feat' else gen_inp_list[1]


    #This is implementation of input dropout !!
    if options['use_dropout']:
        emb = dropout_layer(emb, self.use_noise, trng, options['drop_prob_encoder'], shp = emb.shape)

    # This implements core lstm
    #rval, updatesLSTM = basic_lstm_layer(tparams, emb, [], use_noise, options, prefix='lstm')
    #############################################################################################################################
    # This implements core lstm
    #Before passing this to lstm, reshape to have only 3 dimensions
    emb = emb.reshape((emb.shape[0],emb.shape[1]*emb.shape[2],emb.shape[3]))

    rval, updatesLSTM = self.lstm_enc_layer(tparams, emb, prefix='lstm')
    #############################################################################################################################
    # This implements core reverse lstm
    #if self.encoder == 'bilstm':
    #    rev_rval, rev_updatesLSTM = basic_lstm_layer(tparams, emb[::-1,:,:], prefix=self.mp+'rev_lstm')

    # NOTE1: we are using the last hidden state!
    if gen_out == None:
        p = sliceT(rval[0][samp_lens-1,T.arange(n_batch_samps*n_samp),:],self.hidden_depth,self.hidden_size)
        p = p.reshape((n_batch_samps, n_samp, self.hidden_size))
    else:
        p = sliceT(rval[0][T.concatenate([samp_lens-1, genLens-1]),T.arange(n_batch_samps*n_samp+genLens.shape[0]),:],self.hidden_depth,self.hidden_size)
        p = p.reshape((n_batch_samps*2, n_samp, self.hidden_size))

    sent_emb = T.dot(p, tparams['Wd'])

    if options['use_dropout']:
        sent_emb= dropout_layer(sent_emb, self.use_noise, trng, options['drop_prob_encoder'], shp = sent_emb.shape)

    #Compute Image embedding:
    embImg = T.dot(xImg, tparams['WIemb'])# + tparams['b_Img']
    if options['use_dropout']:
        embImg = dropout_layer(embImg, self.use_noise, trng, options['drop_prob_encoder'], shp = embImg.shape)

    if options.get('merge_dim', -1) ==-1:
        sent_emb = sent_emb.sum(axis=1)
        m_img =  l2norm(embImg)
        m_sent = l2norm(sent_emb)
        scores = T.dot(m_img, m_sent.T)
        p_out = scores.diagonal()
        if options.get('eval_loss','contrastive')=='contrastive':
            probs = T.nnet.softmax(scores*2.0)
            probs_real = probs.diagonal()
            probs_real_neg = 1. - probs_real
        if gen_out !=None:
            p_out = T.concatenate([p_out, scores[:,n_batch_samps:].diagonal()])
            if options.get('eval_loss','contrastive')=='contrastive':
                probs_gen = probs[:,n_batch_samps:].diagonal()
                probs_gen_neg = 1. - probs[:,n_batch_samps:].diagonal()
    else:
        #Compute learned distance between image and sentence
        if gen_out !=None:
            embImg = T.tile(embImg,[2,1])
        dist_im2sents = self.img_discrim.fwd(sent_emb, embImg)
        if options['use_dropout'] and 0:
            dist_im2sents = dropout_layer(dist_im2sents, self.use_noise, trng, options['drop_prob_encoder'], shp = dist_im2sents.shape)

        next_layer_inp = dist_im2sents.flatten(2)
        m_sent = sent_emb
        #Compute learned distances between the sentences
        if options.get('n_gen_samples', 1) > 1:
            dist_sents = self.sent_discrim.fwd(sent_emb)
            #Collapse it into batch x (n_samp * n_ker) shape
            dist_sents = dist_sents.flatten(2)/np.float32(options.get('n_gen_samples',1))
            if options['use_dropout'] and 0:
                dist_sents = dropout_layer(dist_sents, self.use_noise, trng, options['drop_prob_encoder'], shp = dist_sents.shape)
            next_layer_inp = T.concatenate([next_layer_inp, dist_sents],axis=-1)
            #next_layer_inp = next_layer_inp * dist_sents
            m_sent = dist_sents

        m_img = dist_im2sents

        #Collapse it into batch x (n_samp * n_ker) shape
        scores = T.dot(next_layer_inp,tparams['W_disttoprob'])
        probs = T.nnet.softmax(scores)
        probs_real = probs[:n_batch_samps,0]
        probs_real_neg = probs[:n_batch_samps,1]
        probs_gen = probs[n_batch_samps:,0]
        probs_gen_neg = probs[n_batch_samps:,1]
        p_out = probs[:,0]*2.0 - 1.

    if gen_out !=None:
        for inp in gen_inp_list:
          if inp not in inp_list:
              inp_list.append(inp)
        print inp_list

    xTarg = T.fvector('targ')
    inp_list.append(xTarg)
    #import pdb;pdb.set_trace()
    if options.get('eval_loss','contrastive')=='contrastive':
        print 'USING contrastive loss'
        #costEval, ic_s, ic_i = self.contrastive_loss(m_img, m_sent)
        ceval_pos = -(T.log(probs_real+1e-8)*xTarg).sum()/(xTarg.sum()+1e-8)
        costEval = np.float32(2.0) * ceval_pos if gen_out != None else ceval_pos

        if options.get('merge_dim',-1) !=-1:
            ceval_neg =  -(T.log(probs_real_neg + 1e-8)*(1.-xTarg)).sum()/((1.-xTarg).sum()+1e-8)
            costEval = costEval + ceval_neg
        costs = [costEval]
        if gen_out !=None:
            costGen = -(T.log(probs_gen + 1e-8).mean())
            if options.get('gen_feature_matching',1):
                    costGen += T.sqr((next_layer_inp[:n_batch_samps,:]*xTarg[:,None]).sum(axis=0)/(xTarg.sum()+1e-8) - next_layer_inp[n_batch_samps:,:].mean(axis=0)).mean()
                    #costGen += T.sqr((sent_emb[:n_batch_samps,:,:]*xTarg[:,None,None]).sum(axis=0).sum(axis=0)/(options.get('n_gen_samples',1)*(xTarg.sum()+1e-8))
                    #                - sent_emb[n_batch_samps:,:,:].mean(axis=0).mean(axis=0)).mean()
            # Also minimize the probability assigned to the generated fake samples
            if options.get('merge_dim',-1) !=-1:
                ceval_gen =  - (T.log(probs_gen_neg+1e-8).mean())
                costEval = costEval + ceval_gen
            costs.append(costGen)
        costs[0] = costEval
        costs.extend([ceval_pos])
        ic_s = probs
        ic_i = probs
    elif options.get('eval_loss','contrastive')=='wass':
        print 'USING wass loss'
        costEval = (scores[:n_batch_samps, 0]*xTarg).mean() - (scores[:n_batch_samps,0]*(1.-xTarg)).mean()
        costs = [costEval]
        if gen_out !=None:
            costGen = -(scores[n_batch_samps:,0]).mean()
            costEval += costGen
            costs.append(costGen)
        costEval = -costEval
        costs[0] = costEval
        ic_s = costEval
        ic_i = costEval

    #regularize
    if options.get('regc',0.) > 0.:
        self.reg_cost = theano.shared(numpy_floatX(0.), name='reg_c')
        reg_c = T.as_tensor_variable(numpy_floatX(options['regc']), name='reg_c')
        for p in self.regularize:
          self.reg_cost = self.reg_cost+(self.model_th[p] ** 2).sum()
          self.reg_cost *= 0.5 * reg_c
        costEval += (self.reg_cost /options['batch_size'])
        costs[0] = costEval


    f_pred_cost = theano.function(inp_list, costs, name='f_pred_sim_scr', updates=genUpdates)

    f_pred_sim_prob = theano.function(inp_list[:-1], [p_out], name='f_pred_sim_prob', updates=genUpdates)
    #f_pred_sim_prob = theano.function(inp_list, [p_out, sent_emb, m_img, m_sent, embW, ic_s, ic_i, self.reg_cost], name='f_pred_sim_prob')
    if gen_out == None:
        f_sent_emb = theano.function(inp_list[:-1], [m_sent, m_img, scores, probs, sent_emb, embImg], name='f_sent_emb', updates=genUpdates)
    else:
        f_sent_emb = theano.function(inp_list[:-1], [m_sent, m_img, scores, n_batch_samps, n_ref_words, n_gen_words, n_ends], name='f_sent_emb', updates=genUpdates)

    return inp_list, [f_pred_sim_prob, f_pred_cost, f_sent_emb], costs, p_out, tparams

  def lstm_enc_layer(self, tparams, state_below, prefix='lstm'):
    nsteps = state_below.shape[0]
    h_depth = self.hidden_depth
    h_sz = self.hidden_size

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    def _step(x_in, h_, c_):
        preact = tensor.dot(sliceT(h_, 0, h_sz), tparams[_p(prefix, 'W_hid')])
        preact += x_in

        #  preact += tparams[_p(prefix, 'b')]
        h = [[]]*h_depth
        c = [[]]*h_depth
        outp = [[]]*h_depth

        for di in xrange(h_depth):
            i = tensor.nnet.sigmoid(sliceT(preact, 0, h_sz))
            f = tensor.nnet.sigmoid(sliceT(preact, 1, h_sz))
            o = tensor.nnet.sigmoid(sliceT(preact, 2, h_sz))
            c[di] = tensor.tanh(sliceT(preact, 3, h_sz))
            c[di] = f * sliceT(c_, di, h_sz) + i * c[di]
            h[di] = o * tensor.tanh(c[di])
            outp[di] = h[di]
            if self.en_residual_conn:
                if (di > 0):
                  outp[di] += outp[di-1]
                  print "Connecting residual at %d"%(di)
            if di < (h_depth - 1):
                preact = tensor.dot(sliceT(h_, di+1, h_sz), tparams[_p(prefix, ('W_hid_' + str(di+1)))]) + \
                        tensor.dot(outp[di], tparams[_p(prefix, ('W_inp_' + str(di+1)))])


        c_out = tensor.concatenate(c,axis=1)
        h_out = tensor.concatenate(h+[outp[-1]],axis=1)

        return h_out, c_out

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W_inp')]) + tparams[_p(prefix, 'b')])

    rval, updates = theano.scan(_step,
                                sequences=[state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           (h_depth+1)*h_sz),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           h_depth*h_sz),
                                              ],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval, updates
