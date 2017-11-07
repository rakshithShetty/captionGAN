import numpy as np
import code
import theano
from theano import config
import theano.tensor as tensor
from theano.ifelse import ifelse
from collections import OrderedDict
import time
from imagernn.utils import zipp, initwTh, numpy_floatX, _p, sliceT, basic_lstm_layer, dropout_layer
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class BiLSTMGenerator:
  """
  A multimodal long short-term memory (LSTM) generator
  """
# ========================================================================================
  def __init__(self, params):

    image_encoding_size = params.get('image_encoding_size', 128)
    word_encoding_size = params.get('word_encoding_size', 128)
    aux_inp_size = params.get('aux_inp_size', -1)

    hidden_size = params.get('hidden_size', 128)
    hidden_depth = params.get('hidden_depth', 1)
    generator = params.get('generator', 'lstm')
    vocabulary_size = params.get('vocabulary_size',-1)
    output_size = params.get('output_size',-1)
    image_feat_size = params.get('image_feat_size',-1)# size of CNN vectors hardcoded here

    model = OrderedDict()
    # Recurrent weights: take x_t, h_{t-1}, and bias unit
    # and produce the 3 gates and the input to cell signal
    model['WIemb'] = initwTh(image_feat_size, word_encoding_size) # image encoder
    model['b_Img'] = np.zeros((word_encoding_size)).astype(config.floatX)
    model['Wemb'] = initwTh(vocabulary_size, word_encoding_size) # word encoder

    model['lstm_W_hid'] = initwTh(hidden_size, 4 * hidden_size)
    model['lstm_W_inp'] = initwTh(word_encoding_size, 4 * hidden_size)

    model['rev_lstm_W_hid'] = initwTh(hidden_size, 4 * hidden_size)
    model['rev_lstm_W_inp'] = initwTh(word_encoding_size, 4 * hidden_size)

    for i in xrange(1,hidden_depth):
        model['lstm_W_hid_'+str(i)] = initwTh(hidden_size, 4 * hidden_size)
        model['lstm_W_inp_'+str(i)] = initwTh(hidden_size, 4 * hidden_size)
        model['rev_lstm_W_hid_'+str(i)] = initwTh(hidden_size, 4 * hidden_size)
        model['rev_lstm_W_inp_'+str(i)] = initwTh(hidden_size, 4 * hidden_size)

    model['lstm_b'] = np.zeros((4 * hidden_size,)).astype(config.floatX)
    model['rev_lstm_b'] = np.zeros((4 * hidden_size,)).astype(config.floatX)
    # Decoder weights (e.g. mapping to vocabulary)

    if params.get('class_out_factoring',0) == 0:
        model['Wd'] = initwTh(hidden_size, output_size) # decoder
        model['bd'] = np.zeros((output_size,)).astype(config.floatX)
    else:
        clsinfo = params['ixtoclsinfo']
        max_cls_size = np.max(clsinfo[:,1] - clsinfo[:,0])
        self.max_cls_size = max_cls_size
        Wd = np.zeros((params['hidden_size'],params['nClasses'], max_cls_size),dtype=config.floatX)
        model['bd'] = np.zeros((1,params['nClasses'], max_cls_size),dtype=config.floatX)
        for cix in xrange(clsinfo.shape[0]):
            Wd[:,cix,:clsinfo[cix,1]-clsinfo[cix,0]] = initwTh(params['hidden_size'],clsinfo[cix,1] - clsinfo[cix,0])
        model['Wd'] = Wd

    update_list = ['lstm_W_hid', 'lstm_W_inp', 'lstm_b', 'Wd', 'bd', 'WIemb', 'b_Img', 'Wemb']
    update_list.extend(['rev_lstm_W_hid', 'rev_lstm_W_inp', 'rev_lstm_b'])
    self.regularize = ['lstm_W_hid', 'lstm_W_inp', 'Wd', 'WIemb', 'Wemb']
    self.regularize.extend(['rev_lstm_W_hid', 'rev_lstm_W_inp'])

    if params.get('class_out_factoring',0) == 1:
        model['WdCls'] = initwTh(hidden_size, params['nClasses']) # decoder
        model['bdCls'] = np.zeros((params['nClasses'],)).astype(config.floatX)
        update_list.extend(['WdCls', 'bdCls'])
        self.regularize.extend(['WdCls'])

    for i in xrange(1,hidden_depth):
        update_list.append('lstm_W_hid_'+str(i))
        update_list.append('rev_lstm_W_hid_'+str(i))
        self.regularize.append('lstm_W_inp_'+str(i))
        self.regularize.append('rev_lstm_W_inp_'+str(i))

    if params.get('en_aux_inp',0):
        if params.get('swap_aux',0) == 1:
            model['WIemb_aux'] = initwTh(aux_inp_size, image_encoding_size) # image encoder
            model['b_Img_aux'] = np.zeros((image_encoding_size)).astype(config.floatX)
            model['lstm_W_aux'] = initwTh(image_encoding_size, 4 * hidden_size, 0.00005)
            model['rev_lstm_W_aux'] = initwTh(image_encoding_size, 4 * hidden_size, 0.00005)
            update_list.append('WIemb_aux')
            self.regularize.append('WIemb_aux')
            update_list.append('b_Img_aux')
        else:
            model['lstm_W_aux'] = initwTh(aux_inp_size, 4 * hidden_size, 0.001)
            model['rev_lstm_W_aux'] = initwTh(aux_inp_size, 4 * hidden_size, 0.001)
        update_list.append('lstm_W_aux')
        self.regularize.append('lstm_W_aux')
        update_list.append('rev_lstm_W_aux')
        self.regularize.append('rev_lstm_W_aux')

    self.model_th = self.init_tparams(model)
    self.update_list = update_list

# ========================================================================================
  def init_tparams(self,params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

# ========================================================================================
 # BUILD LSTM forward propogation model
  def build_model(self, tparams, options):
    trng = RandomStreams(1234)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    xW = tensor.matrix('xW', dtype='int64')

    mask = tensor.matrix('mask', dtype=config.floatX)
    n_timesteps = xW.shape[0]
    n_samples = xW.shape[1]

    embW = tparams['Wemb'][xW.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['word_encoding_size']])
    
    embW_rev = tparams['Wemb'][xW[::-1,:].flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['word_encoding_size']])
    xI = tensor.matrix('xI', dtype=config.floatX)
    xAux = tensor.matrix('xAux', dtype=config.floatX)

    if options.get('swap_aux',0):
       xAuxEmb = tensor.dot(xAux,tparams['WIemb_aux']) + tparams['b_Img_aux']
    else:
       xAuxEmb = xAux


    embImg = (tensor.dot(xI, tparams['WIemb']) + tparams['b_Img']).reshape([1,n_samples,options['image_encoding_size']]);
    emb = tensor.concatenate([embImg, embW], axis=0)

    emb_rev = tensor.set_subtensor(embW_rev[mask[::-1,:].argmax(axis=0)-1,tensor.arange(n_samples),:],embImg[0,:,:])

    #This is implementation of input dropout !!
    if options['use_dropout']:
        emb = dropout_layer(emb, use_noise, trng, options['drop_prob_encoder'], shp = emb.shape)
        if options.get('en_aux_inp',0):
            xAuxEmb = dropout_layer(xAuxEmb, use_noise, trng, options['drop_prob_aux'], shp = xAuxEmb.shape)

    #############################################################################################################################
    # This implements core lstm
    rval, updatesLSTM = basic_lstm_layer(tparams, emb[:n_timesteps,:,:], xAuxEmb, use_noise, options,
                                         prefix='lstm', sched_prob_mask = [])
    #############################################################################################################################
    # This implements core reverse lstm
    rev_rval, rev_updatesLSTM = basic_lstm_layer(tparams, emb_rev[:n_timesteps,:,:], xAuxEmb, use_noise, options,
                                         prefix='rev_lstm', sched_prob_mask = [])
    #############################################################################################################################


    # NOTE1: we are leaving out the first prediction, which was made for the image and is meaningless.
    if options['use_dropout']:
        # XXX : Size given to dropout is missing one dimension. This keeps the dropped units consistent across time!?.
        # ###   Is this a good bug ?
        p = dropout_layer(sliceT(rval[0][1:,:,:],options.get('hidden_depth',1),options['hidden_size']), use_noise, trng,
            options['drop_prob_decoder'], (n_samples,options['hidden_size']))
        rev_p = dropout_layer(sliceT(rev_rval[0][:,:,:],options.get('hidden_depth',1),options['hidden_size']), use_noise, trng,
            options['drop_prob_decoder'], (n_samples,options['hidden_size']))
    else:
        p = sliceT(rval[0][1:,:,:],options.get('hidden_depth',1),options['hidden_size'])
        rev_p = sliceT(rev_rval[0][:,:,:],options.get('hidden_depth',1),options['hidden_size'])

    n_out_samps = (n_timesteps-2) * n_samples
    if options.get('class_out_factoring',0) == 0:
        pW = (tensor.dot(p[:-1,:,:] + rev_p[::-1,:,:][2:,:,:],tparams['Wd']) + tparams['bd']).reshape([n_out_samps,options['output_size']])
        pWSft = tensor.nnet.softmax(pW)
        totProb = pWSft[tensor.arange(n_out_samps), xW[1:-1,:].flatten()]
        out_list = [pWSft, totProb, p]
    else:
        ixtoclsinfo_t = tensor.as_tensor_variable(options['ixtoclsinfo'])
        xC = ixtoclsinfo_t[xW[1:,:].flatten(),0]
        pW = ((tparams['Wd'][:,xC,:].T*(p.reshape([1,n_out_samps,options['hidden_size']]))).sum(axis=-1).T
             + tparams['bd'][:,xC,:])
        pWSft   = tensor.nnet.softmax(pW[0,:,:])
        pC    = (tensor.dot(p,tparams['WdCls']) + tparams['bdCls']).reshape([n_out_samps,options['nClasses']])
        pCSft = tensor.nnet.softmax(pC)

        totProb = pWSft[tensor.arange(n_out_samps), ixtoclsinfo_t[xW[1:,:].flatten(),3]] * \
                  pCSft[tensor.arange(n_out_samps), xC]
        out_list = [pWSft, pCSft, totProb, p]

    # XXX : THIS IS VERY FISHY, CHECK THE MASK INDEXING AGAIN
    probs_valid = tensor.log(totProb + 1e-10) * mask[1:-1,:].flatten()
    tot_cost = -(probs_valid.sum())
    tot_pplx = -(tensor.log2(totProb + 1e-10) * mask[1:-1,:].flatten()).sum()
    cost = [tot_cost/options['batch_size'], tot_pplx]

    inp_list = [xW, mask, xI]

    if options.get('en_aux_inp',0):
        inp_list.append(xAux)

    if options.get('sched_sampling_mode',None) != None:
        inp_list.append(curr_epoch)

    per_sent_prob = probs_valid.reshape([n_timesteps-2,n_samples]).sum(axis=0)
    f_per_sentLogP = theano.function(inp_list, per_sent_prob, name='f_pred_logprob', updates=updatesLSTM)
    f_pred_prob = ['',f_per_sentLogP,'']


    return use_noise, inp_list, f_pred_prob, cost, out_list , updatesLSTM

# ========================================================================================
# Predictor Related Stuff!!

  def prepPredictor(self, model_npy=None, checkpoint_params=None, beam_size=5):
    if model_npy != None:
        zipp(model_npy, self.model_th)

    #theano.config.exception_verbosity = 'high'

	# Now we build a predictor model
    (inp_list, predLogProb, predIdx, predCand, wOut_emb, updates) = self.build_prediction_model(self.model_th, checkpoint_params, beam_size)
    self.f_pred_th = theano.function(inp_list, [predLogProb, predIdx, predCand], name='f_pred')

	# Now we build a training model which evaluates cost. This is for the evaluation part in the end
    (self.use_dropout, inp_list2,
     f_pred_prob, cost, predTh, updatesLSTM) = self.build_model(self.model_th, checkpoint_params)
    self.f_eval= theano.function(inp_list2, cost, name='f_eval')

# ========================================================================================
  def predict(self, batch, checkpoint_params, **kwparams):

    beam_size = kwparams.get('beam_size', 1)

    inp_list = [batch[0]['image']['feat'].reshape(1,checkpoint_params['image_feat_size']).astype(config.floatX)]

    if checkpoint_params.get('en_aux_inp',0):
        inp_list.append(batch[0]['image']['aux_inp'].reshape(1,checkpoint_params['aux_inp_size']).astype(config.floatX))

    Ax = self.f_pred_th(*inp_list)

    # Backtracking to decode the correct sequence of candidates
    Ys = []
    for i in xrange(beam_size):
        candI = []
        curr_cand = Ax[2][-1][i]
        for j in reversed(xrange(Ax[1].shape[0]-1)):
            candI.insert(0,Ax[1][j][curr_cand])
            curr_cand = Ax[2][j][curr_cand]

        Ys.append([Ax[0][i], candI])
    return [Ys]

  def build_prediction_model(self, tparams, options, beam_size):

    n_samples = 1

    xI = tensor.matrix('xI', dtype=config.floatX)
    xAux = tensor.matrix('xAux', dtype=config.floatX)

    if options.get('swap_aux',0):
       xAuxEmb = tensor.dot(xAux,tparams['WIemb_aux']) + tparams['b_Img_aux']
    else:
       xAuxEmb = xAux

    embImg = (tensor.dot(xI, tparams['WIemb']) + tparams['b_Img']).reshape([n_samples,options['image_encoding_size']]);

    if options.get('advers_gen',0) == 1:
        accLogProb, Idx, wOut_emb, updates = self.lstm_advers_gen_layer(tparams, embImg, xAuxEmb, options, beam_size, prefix=options['generator'])
        Cand = []
    else:
        accLogProb, Idx, Cand, wOut_emb, updates = self.lstm_predict_layer(tparams, embImg, xAuxEmb, options, beam_size, prefix=options['generator'])

    inp_list = [xI]
    if options.get('en_aux_inp',0):
        inp_list.append(xAux)

    return inp_list, accLogProb, Idx, Cand, wOut_emb, updates

# ========================================================================================
  # LSTM LAYER in Prediction mode. Here we don't provide the word sequences, just the image feature vector
  # The network starts first with forward propogatin the image feature vector. Then we pass the start word feature
  # i.e zeroth word vector. From then the network output word (i.e ML word) is fed as the input to the next time step.
  # In beam_size > 1 we could repeat a time step multiple times, once for each beam!!.

  def lstm_predict_layer(self, tparams, Xi, aux_input, options, beam_size, prefix='lstm'):

    nMaxsteps = options.get('maxlen',30)

    if nMaxsteps is None:
        nMaxsteps = 30
    n_samples = 1
    h_depth = options.get('hidden_depth',1)
    h_sz = options['hidden_size']

    # ----------------------  STEP FUNCTION  ---------------------- #
    def _stepP(x_, h_, c_, lP_, dV_, xAux):
        preact = tensor.dot(sliceT(h_, 0, h_sz), tparams[_p(prefix, 'W_hid')])
        preact += (tensor.dot(x_, tparams[_p(prefix, 'W_inp')]) +
                   tparams[_p(prefix, 'b')])
        if options.get('en_aux_inp',0):
            preact += tensor.dot(xAux,tparams[_p(prefix,'W_aux')])

        hL = [[]]*h_depth
        cL = [[]]*h_depth
        outp = [[]]*h_depth
        for di in xrange(h_depth):
            i = tensor.nnet.sigmoid(sliceT(preact, 0, h_sz))
            f = tensor.nnet.sigmoid(sliceT(preact, 1, h_sz))
            o = tensor.nnet.sigmoid(sliceT(preact, 2, h_sz))
            cL[di] = tensor.tanh(sliceT(preact, 3, h_sz))
            cL[di] = f * sliceT(c_, di, h_sz) + i * cL[di]
            hL[di] = o * tensor.tanh(cL[di])
            outp[di] = hL[di]
            if options.get('en_residual_conn',1):
                if (di > 0):
                  outp[di] += outp[di-1]
                  print "Connecting residual at %d"%(di)
            if di < (h_depth - 1):
                preact = tensor.dot(sliceT(h_, di+1, h_sz), tparams[_p(prefix, ('W_hid_' + str(di+1)))]) + \
                        tensor.dot(outp[di], tparams[_p(prefix, ('W_inp_' + str(di+1)))])

        c = tensor.concatenate(cL,axis=1)
        h = tensor.concatenate(hL,axis=1)

        if options.get('class_out_factoring',0) == 1:
            pC    = tensor.dot(outp[-1],tparams['WdCls']) + tparams['bdCls']
            pCSft = tensor.nnet.softmax(pC)
            xCIdx =  tensor.argmax(pCSft)
            pW = tensor.dot(outp[-1],tparams['Wd'][:,xCIdx,:]) + tparams['bd'][:,xCIdx,:]
            smooth_factor = tensor.as_tensor_variable(numpy_floatX(options.get('softmax_smooth_factor',1.0)), name='sm_f')
            pWSft = tensor.nnet.softmax(pW*smooth_factor)
            lProb = tensor.log(pWSft + 1e-20) + tensor.log(pCSft[0,xCIdx] + 1e-20)
        else:
            p = tensor.dot(outp[-1],tparams['Wd']) + tparams['bd']
            smooth_factor = tensor.as_tensor_variable(numpy_floatX(options.get('softmax_smooth_factor',1.0)), name='sm_f')
            p = tensor.nnet.softmax(p*smooth_factor)
            lProb = tensor.log(p + 1e-20)

        if beam_size > 1:
            def _FindB_best(lPLcl, lPprev, dVLcl):
                srtLcl = tensor.argsort(-lPLcl)
                srtLcl = srtLcl[:beam_size]
                deltaVec = tensor.fill( lPLcl[srtLcl], numpy_floatX(-10000.))
                deltaVec = tensor.set_subtensor(deltaVec[0], lPprev)
                lProbBest = ifelse(tensor.eq( dVLcl, tensor.zeros_like(dVLcl)), lPLcl[srtLcl] + lPprev, deltaVec)
                xWIdxBest = ifelse(tensor.eq( dVLcl, tensor.zeros_like(dVLcl)), srtLcl, tensor.zeros_like(srtLcl))
                return lProbBest, xWIdxBest

            rvalLcl, updatesLcl = theano.scan(_FindB_best, sequences = [lProb, lP_, dV_], name=_p(prefix, 'FindBest'), n_steps=x_.shape[0])
            xWIdxBest = rvalLcl[1]
            lProbBest = rvalLcl[0]

            xWIdxBest = xWIdxBest.flatten()
            lProb = lProbBest.flatten()
            # Now sort and find the best among these best extensions for the current beams
            srtIdx = tensor.argsort(-lProb)
            srtIdx = srtIdx[:beam_size]
            xCandIdx = srtIdx // beam_size # Floor division
            h = h.take(xCandIdx.flatten(),axis=0)
            c = c.take(xCandIdx.flatten(),axis=0)
            xWlogProb = lProb[srtIdx]
            xWIdx = xWIdxBest[srtIdx]
        else:
            xCandIdx = tensor.as_tensor_variable([0])
            lProb = lProb.flatten()
            xWIdx =  tensor.argmax(lProb,keepdims=True)
            xWlogProb = lProb[xWIdx] + lP_
            if options.get('class_out_factoring',0) == 1:
                clsoffset = tensor.as_tensor_variable(options['ixtoclsinfo'][:,0])
                xWIdx += clsoffset[xCIdx]
            h = h.take(xCandIdx.flatten(),axis=0)
            c = c.take(xCandIdx.flatten(),axis=0)

        if options.get('softmax_propogate',0) == 0:
            xW = tparams['Wemb'][xWIdx.flatten()]
        else:
            xW = p.dot(tparams['Wemb'])
        doneVec = tensor.eq(xWIdx,tensor.zeros_like(xWIdx))

        return [xW, h, c, xWlogProb, doneVec, xWIdx, xCandIdx], theano.scan_module.until(doneVec.all())
    # ------------------- END of STEP FUNCTION  -------------------- #

    if options.get('en_aux_inp',0) == 0:
       aux_input = []

    h = tensor.alloc(numpy_floatX(0.),beam_size,h_sz*h_depth)
    c = tensor.alloc(numpy_floatX(0.),beam_size,h_sz*h_depth)

    lP = tensor.alloc(numpy_floatX(0.), beam_size);
    dV = tensor.alloc(np.int8(0.), beam_size);

    # Propogate the image feature vector
    [xW, h, c, _, _, _, _], _ = _stepP(Xi, h[:1,:], c[:1,:], lP, dV,aux_input)

    xWStart = tparams['Wemb'][[0]]
    [xW, h, c, lP, dV, idx0, cand0], _ = _stepP(xWStart, h[:1,:], c[:1,:], lP, dV, aux_input)

    if options.get('en_aux_inp',0) == 1:
        aux_input = tensor.extra_ops.repeat(aux_input,beam_size,axis=0)

    # Now lets do the loop.
    rval, updates = theano.scan(_stepP, outputs_info=[xW, h, c, lP, dV, None, None], non_sequences = [aux_input], name=_p(prefix, 'predict_layers'), n_steps=nMaxsteps)

    return rval[3][-1], tensor.concatenate([idx0.reshape([1,beam_size]), rval[5]],axis=0), tensor.concatenate([cand0.reshape([1,beam_size]), rval[6]],axis=0), tensor.concatenate([tensor.shape_padleft(xW,n_ones=1),rval[0]],axis=0), updates

#================================================================================================================
  def lstm_advers_gen_layer(self, tparams, Xi, aux_input, options, beam_size, prefix='lstm'):
    nMaxsteps = options.get('maxlen',15)
    n_samples = 1
    h_depth = options.get('hidden_depth',1)
    h_sz = options['hidden_size']

    # ----------------------  STEP FUNCTION  ---------------------- #
    def _stepP(x_, h_, c_, lP_, dV_, xAux):
        preact = tensor.dot(sliceT(h_, 0, h_sz), tparams[_p(prefix, 'W_hid')])
        preact += (tensor.dot(x_, tparams[_p(prefix, 'W_inp')]) +
                   tparams[_p(prefix, 'b')])
        if options.get('en_aux_inp',0):
            preact += tensor.dot(xAux,tparams[_p(prefix,'W_aux')])

        hL = [[]]*h_depth
        cL = [[]]*h_depth
        for di in xrange(h_depth):
            i = tensor.nnet.sigmoid(sliceT(preact, 0, h_sz))
            f = tensor.nnet.sigmoid(sliceT(preact, 1, h_sz))
            o = tensor.nnet.sigmoid(sliceT(preact, 2, h_sz))
            cL[di] = tensor.tanh(sliceT(preact, 3, h_sz))
            cL[di] = f * sliceT(c_, di, h_sz) + i * cL[di]
            hL[di] = o * tensor.tanh(cL[di])
            if di < (h_depth - 1):
                preact = tensor.dot(sliceT(h_, di+1, h_sz), tparams[_p(prefix, ('W_hid_' + str(di+1)))]) + \
                        tensor.dot(hL[di], tparams[_p(prefix, ('W_inp_' + str(di+1)))])

        c = tensor.concatenate(cL,axis=1)
        h = tensor.concatenate(hL,axis=1)

        p = tensor.dot(hL[-1],tparams['Wd']) + tparams['bd']
        smooth_factor = tensor.as_tensor_variable(numpy_floatX(options.get('softmax_smooth_factor',1.0)), name='sm_f')
        p = tensor.nnet.softmax(p*smooth_factor)
        lProb = tensor.log(p + 1e-20)

        #xCandIdx = tensor.as_tensor_variable([0])
        lProb = lProb.flatten()
        xWIdx =  tensor.argmax(lProb,keepdims=True)
        xWlogProb = lProb[xWIdx] + lP_

        if options.get('softmax_propogate',0) == 0:
            xW = tparams['Wemb'][xWIdx.flatten()]
        else:
            xW = p.dot(tparams['Wemb'])
        doneVec = tensor.eq(xWIdx,tensor.zeros_like(xWIdx))

        return [xW, h, c, xWlogProb, doneVec, xWIdx, p], theano.scan_module.until(doneVec.all())
    # ------------------- END of STEP FUNCTION  -------------------- #

    if options.get('en_aux_inp',0) == 0:
       aux_input = []

    h = tensor.alloc(numpy_floatX(0.),n_samples,h_sz*h_depth)
    c = tensor.alloc(numpy_floatX(0.),n_samples,h_sz*h_depth)

    lP = tensor.alloc(numpy_floatX(0.), beam_size);
    dV = tensor.alloc(np.int8(0.), beam_size);

    # Propogate the image feature vector
    [xW, h, c, _, _, _, _], _ = _stepP(Xi, h, c, lP, dV,aux_input)

    xWStart = tparams['Wemb'][0,:]
    [xW, h, c, lP, dV, idx0, p0], _ = _stepP(xWStart, h, c, lP, dV, aux_input)

    #if options.get('en_aux_inp',0) == 1:
    #    aux_input = tensor.extra_ops.repeat(aux_input,beam_size,axis=0)

    # Now lets do the loop.
    rval, updates = theano.scan(_stepP, outputs_info=[xW, h, c, lP, dV, None, None], non_sequences = [aux_input], name=_p(prefix, 'predict_layers'), n_steps=nMaxsteps-1)

    return rval[3][-1], tensor.concatenate([idx0.reshape([1,beam_size]), rval[5]],axis=0), tensor.concatenate([tensor.shape_padleft(p0,n_ones=1),rval[6]],axis=0), updates


# ========================================================================================
  def build_eval_other_sent(self, tparams, options,model_npy):

    zipp(model_npy, self.model_th)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    xW = tensor.matrix('xW', dtype='int64')
    mask = tensor.matrix('mask', dtype=config.floatX)
    n_timesteps = xW.shape[0]
    n_samples = xW.shape[1]
    n_out_samps = (n_timesteps-1) * n_samples

    embW = tparams['Wemb'][xW.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['word_encoding_size']])
    xI = tensor.matrix('xI', dtype=config.floatX)
    xAux = tensor.matrix('xAux', dtype=config.floatX)

    if options.get('swap_aux',0):
       xAuxEmb = tensor.dot(xAux,tparams['WIemb_aux']) + tparams['b_Img_aux']
    else:
       xAuxEmb = xAux

    embImg = (tensor.dot(xI, tparams['WIemb']) + tparams['b_Img']).reshape([1,n_samples,options['image_encoding_size']]);
    emb = tensor.concatenate([embImg, embW], axis=0)


    rval, updatesLSTM = basic_lstm_layer(tparams, emb[:n_timesteps,:,:], xAuxEmb, use_noise, options, prefix=options['generator'])
    p = sliceT(rval[0][1:,:,:],options.get('hidden_depth',1),options['hidden_size'])

    pW = (tensor.dot(p,tparams['Wd']) + tparams['bd']).reshape([n_out_samps,options['output_size']])

    pWSft = tensor.nnet.softmax(pW)
    totProb = pWSft[tensor.arange(n_out_samps), xW[1:,:].flatten()]

#    #pred = tensor.nnet.softmax(p)
#
#    #pred = rval[2]
#
#    #pred = pred[1:,:,:]
#
#    def accumCost(pred,xW,m,c_sum,ppl_sum):
#        pred = tensor.nnet.softmax(pred)
#        c_sum += (tensor.log(pred[tensor.arange(n_samples), xW]+1e-20) * m)
#        ppl_sum += -(tensor.log2(pred[tensor.arange(n_samples), xW]+1e-10) * m)
#        return c_sum, ppl_sum
#
#    sums, upd = theano.scan(fn=accumCost,
#                                outputs_info=[tensor.alloc(numpy_floatX(0.), 1,n_samples),
#                                              tensor.alloc(numpy_floatX(0.), 1,n_samples)],
#                                sequences = [p, xW[1:,:], mask[1:,:]])
    # NOTE1: we are leaving out the first prediction, which was made for the image
    # and is meaningless. Here cost[0] contains log probability (log10) and cost[1] contains
    # perplexity (log2)
    tot_cost = -(tensor.log(totProb + 1e-10) * mask[1:,:].flatten()).sum()
    cost = tot_cost/options['batch_size']

    inp_list = [xW, mask, xI]

    if options.get('en_aux_inp',0):
        inp_list.append(xAux)

    self.f_pred_prob_other = theano.function(inp_list, p, name='f_pred_prob', updates=updatesLSTM)
    #f_pred = theano.function([xW, mask], pred.argmax(axis=1), name='f_pred')

    #cost = -tensor.log(pred[tensor.arange(n_timesteps),tensor.arange(n_samples), xW] + 1e-8).mean()

    self.f_eval_other = theano.function(inp_list, cost, name='f_eval')

    return use_noise, inp_list, self.f_pred_prob_other, cost, pW, updatesLSTM

# =================================== MULTI Model Ensemble Predictor related ==========================

  def prepMultiPredictor(self, tparams, checkpoint_params, beam_size,nmodels):
	# Now we build a predictor model
    (inp_list, predLogProb, predIdx, predCand,rval) = self.build_multi_prediction_model(tparams, checkpoint_params, beam_size,nmodels)
    self.f_multi_pred_th = theano.function(inp_list, [predLogProb, predIdx, predCand,rval], name='f_multi_pred')

	# Now we build a training model which evaluates cost. This is for the evaluation part in the end
    #(self.use_dropout, inp_list2,
    # f_pred_prob, cost, predTh, updatesLSTM) = self.build_model(self.model_th, checkpoint_params)
    #self.f_eval= theano.function(inp_list2, cost, name='f_multi_eval')


# ========================================================================================
  def predictMulti(self, batch, checkpoint_params, **kwparams):

    beam_size = kwparams.get('beam_size', 1)
    nmodels = kwparams.get('nmodels', 1)

    inp_list = []
    for i in xrange(nmodels):
        inp_list.append(batch[i]['image']['feat'].reshape(1,checkpoint_params[i]['image_feat_size']).astype(config.floatX))

    for i in xrange(nmodels):
        if checkpoint_params[i].get('en_aux_inp',0):
            inp_list.append(batch[i]['image']['aux_inp'].reshape(1,checkpoint_params[i]['aux_inp_size']).astype(config.floatX))

    Ax = self.f_multi_pred_th(*inp_list)

    Ys = []
    for i in xrange(beam_size):
        candI = []
        curr_cand = Ax[2][-1][i]
        for j in reversed(xrange(Ax[1].shape[0]-1)):
            candI.insert(0,Ax[1][j][curr_cand])
            curr_cand = Ax[2][j][curr_cand]

        Ys.append([Ax[0][i], candI])
    return [Ys]

  def build_multi_prediction_model(self, tparams, options, beam_size,nmodels):

    n_samples = 1
    xI = []
    xAux = []
    embImg = []
    for i in xrange(nmodels):
        xI.append(tensor.matrix('xI_' + str(i), dtype=config.floatX))
        xAux.append(tensor.matrix('xAux_'+str(i), dtype=config.floatX))
        embImg.append((tensor.dot(xI[i], tparams[i]['WIemb']) + tparams[i]['b_Img']).reshape([n_samples,options[i]['image_encoding_size']]));

    accLogProb, Idx, Cand,rval = self.lstm_multi_model_pred(tparams, embImg, xAux, options, beam_size, nmodels, prefix=options[0]['generator'])

    inp_list = []
    inp_list.extend(xI)
    for i in xrange(nmodels):
        if options[i].get('en_aux_inp',0):
            inp_list.append(xAux[i])

    return inp_list, accLogProb, Idx, Cand,rval


  def lstm_multi_model_pred(self,tparams, Xi, aux_input, options, beam_size, nmodels, prefix='lstm'):
    nMaxsteps = 30

    # ----------------------  STEP FUNCTION  ---------------------- #
    def _stepP(*in_list):
        x_inp = []
        h_inp = []
        c_inp = []
        for i in xrange(nmodels):
            x_inp.append(in_list[i])
            h_inp.append(in_list[nmodels+i])
            c_inp.append(in_list[2*nmodels+i])
        lP_ = in_list[3*nmodels]
        dV_ = in_list[3*nmodels+1]

        p_comb = tensor.alloc(numpy_floatX(0.), options[0]['output_size']);
        cf = []
        h = []
        xW = []
        for i in xrange(nmodels):
            preact = tensor.dot(h_inp[i], tparams[i][_p(prefix, 'W_hid')])
            preact += (tensor.dot(x_inp[i], tparams[i][_p(prefix, 'W_inp')]) +
                       tparams[i][_p(prefix, 'b')])
            if options[i].get('en_aux_inp',0):
                preact += tensor.dot(aux_input2[i],tparams[i][_p(prefix,'W_aux')])

            inp = tensor.nnet.sigmoid(sliceT(preact, 0, options[i]['hidden_size']))
            f = tensor.nnet.sigmoid(sliceT(preact, 1, options[i]['hidden_size']))
            o = tensor.nnet.sigmoid(sliceT(preact, 2, options[i]['hidden_size']))
            c = tensor.tanh(sliceT(preact, 3, options[i]['hidden_size']))

            cf.append(f * c_inp[i] + inp * c)

            h.append(o * tensor.tanh(cf[i]))
            p = tensor.dot(h[i],tparams[i]['Wd']) + tparams[i]['bd']
            if i == 0:
                p_comb = tparams[i]['comb_weight']*tensor.nnet.softmax(p)
            else:
                p_comb += tparams[i]['comb_weight']*tensor.nnet.softmax(p)

        lProb = tensor.log(p_comb + 1e-20)
        def _FindB_best(lPLcl, lPprev, dVLcl):
            srtLcl = tensor.argsort(-lPLcl)
            srtLcl = srtLcl[:beam_size]
            deltaVec = tensor.fill( lPLcl[srtLcl], numpy_floatX(-10000.))
            deltaVec = tensor.set_subtensor(deltaVec[0], lPprev)
            lProbBest = ifelse(tensor.eq( dVLcl, tensor.zeros_like(dVLcl)), lPLcl[srtLcl] + lPprev, deltaVec)
            xWIdxBest = ifelse(tensor.eq( dVLcl, tensor.zeros_like(dVLcl)), srtLcl, tensor.zeros_like(srtLcl))
            return lProbBest, xWIdxBest

        rvalLcl, updatesLcl = theano.scan(_FindB_best, sequences = [lProb, lP_, dV_], name=_p(prefix, 'FindBest'), n_steps=x_inp[0].shape[0])
        xWIdxBest = rvalLcl[1]
        lProbBest = rvalLcl[0]

        xWIdxBest = xWIdxBest.flatten()
        lProb = lProbBest.flatten()

        # Now sort and find the best among these best extensions for the current beams
        srtIdx = tensor.argsort(-lProb)
        srtIdx = srtIdx[:beam_size]
        xWlogProb = lProb[srtIdx]

        xWIdx = xWIdxBest[srtIdx]
        xCandIdx = srtIdx // beam_size # Floor division

        doneVec = tensor.eq(xWIdx,tensor.zeros_like(xWIdx))

        x_out = []
        h_out = []
        c_out = []
        for i in xrange(nmodels):
            x_out.append(tparams[i]['Wemb'][xWIdx.flatten()])
            h_out.append(h[i].take(xCandIdx.flatten(),axis=0))
            c_out.append(cf[i].take(xCandIdx.flatten(),axis=0))

        out_list = []
        out_list.extend(x_out)
        out_list.extend(h_out)
        out_list.extend(c_out)
        out_list.extend([xWlogProb, doneVec, xWIdx, xCandIdx])

        return out_list, theano.scan_module.until(doneVec.all())
    # ------------------- END of STEP FUNCTION  -------------------- #

    #Xi = tensor.extra_ops.repeat(Xi,beam_size,axis=0)

    lP = tensor.alloc(numpy_floatX(0.), beam_size);
    dV = tensor.alloc(np.int8(0.), beam_size);

    h_inp = []
    c_inp = []
    x_inp = []
    for i in xrange(nmodels):
      hidden_size = options[i]['hidden_size']
      h = theano.shared(np.zeros((1,hidden_size),dtype='float32'))
      c = theano.shared(np.zeros((1,hidden_size),dtype='float32'))
      h_inp.append(h)
      c_inp.append(c)
      x_inp.append(Xi[i])

    aux_input2 = aux_input

    in_list = []
    in_list.extend(x_inp); in_list.extend(h_inp); in_list.extend(c_inp)
    in_list.append(lP); in_list.append(dV)


    # Propogate the image feature vector
    out_list,_ = _stepP(*in_list)

    for i in xrange(nmodels):
        h_inp[i] = out_list[nmodels + i]
        c_inp[i] = out_list[2*nmodels + i]

    x_inp = []
    for i in xrange(nmodels):
      x_inp.append(tparams[i]['Wemb'][[0]])
      h_inp[i] = h_inp[i][:1,:]
      c_inp[i] = c_inp[i][:1,:]
      #if options[i].get('en_aux_inp',0):
      #  aux_input2.append(aux_input[i])

    in_list = []
    in_list.extend(x_inp); in_list.extend(h_inp); in_list.extend(c_inp)
    in_list.append(lP); in_list.append(dV)

    out_list, _ = _stepP(*in_list)
    aux_input2 = []
    for i in xrange(nmodels):
        x_inp[i] = out_list[i]
        h_inp[i] = out_list[nmodels + i]
        c_inp[i] = out_list[2*nmodels + i]
        aux_input2.append(tensor.extra_ops.repeat(aux_input[i],beam_size,axis=0))
    lP = out_list[3*nmodels]
    dV = out_list[3*nmodels+1]
    idx0 = out_list[3*nmodels+2]
    cand0 = out_list[3*nmodels+3]

    in_list = []
    in_list.extend(x_inp); in_list.extend(h_inp); in_list.extend(c_inp)
    in_list.append(lP); in_list.append(dV)
    in_list.append(None);in_list.append(None);

    # Now lets do the loop.
    rval, updates = theano.scan(_stepP, outputs_info=in_list, name=_p(prefix, 'predict_layers'), n_steps=nMaxsteps)

    return rval[3*nmodels][-1], tensor.concatenate([idx0.reshape([1,beam_size]), rval[3*nmodels+2]],axis=0), tensor.concatenate([cand0.reshape([1,beam_size]), rval[3*nmodels+3]],axis=0), rval[3*nmodels]
