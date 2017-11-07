import numpy as np
import code
import theano
from theano import config
import theano.tensor as tensor
from theano.ifelse import ifelse
from collections import OrderedDict
import time
from imagernn.utils import zipp, initwTh, numpy_floatX, _p, sliceT, basic_lstm_layer, dropout_layer, gumbel_softmax_sample, l2norm
#from theano.tensor.shared_randomstreams import RandomStreams as RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class LSTMGenerator:
  """
  A multimodal long short-term memory (LSTM) generator
  """
# ========================================================================================
  def __init__(self, params):

    image_encoding_size = params.get('image_encoding_size', 128)
    word_encoding_size = params.get('word_encoding_size', 128)

    hidden_size = params.get('hidden_size', 128)
    hidden_depth = params.get('hidden_depth', 1)
    generator = params.get('generator', 'lstm')
    vocabulary_size = params.get('vocabulary_size',-1)
    output_size = params.get('output_size',-1)
    image_feat_size = params.get('image_feat_size',-1)# size of CNN vectors hardcoded here

    aux_inp_size = params.get('aux_inp_size', -1)

    model = OrderedDict()
    # Recurrent weights: take x_t, h_{t-1}, and bias unit
    # and produce the 3 gates and the input to cell signal

    encoder = params.get('feat_encoder', None)
    use_feat_enc = params.get('use_encoder_for',0)

    if not (use_feat_enc & 1):
        model['WIemb'] = initwTh(image_feat_size, word_encoding_size) # image encoder
        model['b_Img'] = np.zeros((word_encoding_size)).astype(config.floatX)

    model['Wemb'] = initwTh(vocabulary_size, word_encoding_size) # word encoder
    model['lstm_W_hid'] = initwTh(hidden_size, 4 * hidden_size)
    model['lstm_W_inp'] = initwTh(word_encoding_size, 4 * hidden_size)

    for i in xrange(1,hidden_depth):
        model['lstm_W_hid_'+str(i)] = initwTh(hidden_size, 4 * hidden_size)
        model['lstm_W_inp_'+str(i)] = initwTh(hidden_size, 4 * hidden_size)

    model['lstm_b'] = np.zeros((4 * hidden_size,)).astype(config.floatX)
    # Decoder weights (e.g. mapping to vocabulary)

    if params.get('class_out_factoring',0) == 0:
        model['Wd'] = initwTh(hidden_size, output_size) # decoder
        model['bd'] = np.zeros((output_size,)).astype(config.floatX)
    else:
        clsinfo = params['ixtoclsinfo']
        self.clsinfo = clsinfo
        clsSizes = clsinfo[:,2] - clsinfo[:,1]
        self.clsSize = np.zeros(params['nClasses'])
        self.clsOffset = np.zeros(params['nClasses'],dtype=np.int32)
        self.clsSize[clsinfo[:,0]] = clsSizes
        self.clsOffset[clsinfo[:,0]] = np.int32(clsinfo[:,1])
        max_cls_size = np.max(clsSizes)
        self.max_cls_size = max_cls_size
        Wd = np.zeros((params['hidden_size'],params['nClasses'], max_cls_size),dtype=config.floatX)
        model['bd'] = np.zeros((1,params['nClasses'], max_cls_size),dtype=config.floatX)
        for cix in clsinfo[:,0]:
            Wd[:,cix,:clsSizes[cix]] = initwTh(params['hidden_size'],clsSizes[cix])
            model['bd'][0,cix,clsSizes[cix]:] = -100
        model['Wd'] = Wd

    update_list = ['lstm_W_hid', 'lstm_W_inp', 'lstm_b', 'Wd', 'bd','Wemb']
    self.regularize = ['lstm_W_hid', 'lstm_W_inp', 'Wd','Wemb']
    if not (use_feat_enc & 1):
        update_list.extend(['WIemb', 'b_Img'])
        self.regularize.extend(['WIemb'])

    if params.get('class_out_factoring',0) == 1:
        model['WdCls'] = initwTh(hidden_size, params['nClasses']) # decoder
        model['bdCls'] = np.zeros((params['nClasses'],)).astype(config.floatX)
        update_list.extend(['WdCls', 'bdCls'])
        self.regularize.extend(['WdCls'])

    for i in xrange(1,hidden_depth):
        update_list.append('lstm_W_hid_'+str(i))
        update_list.append('lstm_W_hid_'+str(i))
        self.regularize.append('lstm_W_inp_'+str(i))
        self.regularize.append('lstm_W_inp_'+str(i))

    if params.get('en_aux_inp',0):
        if params.get('swap_aux',1) == 1:
            if not (use_feat_enc & 2) or params.get('encode_gt_sentences',0):
                model['WIemb_aux'] = initwTh(aux_inp_size, image_encoding_size) # image encoder
                model['b_Img_aux'] = np.zeros((image_encoding_size)).astype(config.floatX)
                update_list.append('WIemb_aux')
                self.regularize.append('WIemb_aux')
                update_list.append('b_Img_aux')
            model['lstm_W_aux'] = initwTh(image_encoding_size, 4 * hidden_size, 0.00005)
        else:
            model['lstm_W_aux'] = initwTh(aux_inp_size, 4 * hidden_size, 0.001)
        update_list.append('lstm_W_aux')
        self.regularize.append('lstm_W_aux')

    if params.get('gen_input_noise',0):
        self.noise_dim = params.get('gen_inp_noise_dim',50)
        model['lstm_W_noise'] = initwTh(self.noise_dim, 4 * hidden_size, 0.001)

    self.model_th = self.init_tparams(model)
    del model
    if params.get('use_gumbel_mse',0):
        self.usegumbel = theano.shared(1)
        self.gumb_temp = theano.shared(numpy_floatX(params.get('gumbel_temp_init',0.5)))
        #self.model_th['gumb_temp'] = self.gumb_temp
        self.softmax_smooth_factor = theano.shared(numpy_floatX(params.get('softmax_smooth_factor',1.0)))
    else:
        self.usegumbel = theano.shared(0)
    self.update_list = update_list

# ========================================================================================
  def init_tparams(self,params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

# ========================================================================================
 # BUILD LSTM forward propogation model
  def build_model(self, tparams, options, xI=None, xAux = None, attn_nw = None):
    self.trng = RandomStreams(int(time.time()))

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    xW = tensor.matrix('xW', dtype='int64')

    mask = tensor.matrix('mask', dtype=config.floatX)
    n_timesteps = xW.shape[0]
    n_samples = xW.shape[1]

    embW = tparams['Wemb'][xW.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['word_encoding_size']])
    if xI == None:
        xI = tensor.matrix('xI', dtype=config.floatX)
        embImg = (tensor.dot(xI, tparams['WIemb']) + tparams['b_Img'])
        xI_is_inp = True
    else:
        embImg = xI
        xI_is_inp = False

    if xAux == None:
        xAux = tensor.matrix('xAux', dtype=config.floatX) if attn_nw == None else tensor.tensor3('xAux', dtype=config.floatX)
        if (options.get('swap_aux',1)) and (attn_nw == None):
           xAuxEmb = tensor.dot(xAux,tparams['WIemb_aux']) + tparams['b_Img_aux']
        else:
           xAuxEmb = xAux
        xA_is_inp = True
    else:
        xA_is_inp = False
        if options.get('encode_gt_sentences',0):
           xAuxEmb = tensor.dot(xAux,tparams['WIemb_aux']) + tparams['b_Img_aux']
        else:
            xAuxEmb = xAux


    embImg = embImg.reshape([1,n_samples,options['image_encoding_size']]);
    emb = tensor.concatenate([embImg, embW], axis=0)

    #This is implementation of input dropout !!
    if options['use_dropout']:
        emb = dropout_layer(emb, use_noise, self.trng, options['drop_prob_encoder'], shp = emb.shape)
        if (options.get('en_aux_inp',0)) and (attn_nw == None):
            xAuxEmb = dropout_layer(xAuxEmb, use_noise, self.trng, options['drop_prob_aux'], shp = xAuxEmb.shape)

    # Implement scehduled sampling!
    if options.get('sched_sampling_mode',None) != None:
        curr_epoch = tensor.scalar(name='curr_epoch',dtype=config.floatX)

        # Assign the probabilies according to the scheduling mode
        if options['sched_sampling_mode'] == 'linear':
            prob = tensor.maximum(options['sslin_min'],options['sched_sampling_const'] - options['sslin_slope'] * curr_epoch)
        elif options['sched_sampling_mode'] == 'exp':
            raise ValueError('ERROR: %s --> This solver type is not yet supported'%(options['sched_sampling_mode']))
        elif options['sched_sampling_mode'] == 'invsig':
            raise ValueError('ERROR: %s --> This solver type is not yet supported'%(options['sched_sampling_mode']))
        else:
            raise ValueError('ERROR: %s --> This scheduling type is unknown'%(options['sched_sampling_mode']))

        # Now to build the mask. We don't want to do this coin toss when
        # feeding in image feature and the start symbol
        sched_mask = self.trng.binomial((n_timesteps - 2, n_samples), p=prob, n=1, dtype='int64')
        sched_mask = tensor.concatenate([sched_mask, tensor.alloc(1, 2, n_samples)],axis=0)
    else:
        sched_mask = []


    #############################################################################################################################
    # This implements core lstm
    rval, updatesLSTM = basic_lstm_layer(tparams, emb[:n_timesteps,:,:], xAuxEmb, use_noise, options,
                                         prefix=options['generator'], sched_prob_mask = sched_mask, attn_nw = attn_nw)
    #############################################################################################################################


    # NOTE1: we are leaving out the first prediction, which was made for the image and is meaningless.
    if options['use_dropout']:
        # XXX : Size given to dropout is missing one dimension. This keeps the dropped units consistent across time!?.
        # ###   Is this a good bug ?
        p = dropout_layer(sliceT(rval[0][1:,:,:],options.get('hidden_depth',1),options['hidden_size']), use_noise, self.trng,
            options['drop_prob_decoder'], (n_samples,options['hidden_size']))
    else:
        p = sliceT(rval[0][1:,:,:],options.get('hidden_depth',1),options['hidden_size'])

    if options.get('class_out_factoring',0) == 1:
        if options.get('cls_diff_layer',0) == 1:
            pC_inp = dropout_layer(sliceT(rval[0][1:,:,:],options.get('hidden_depth',1)-2,options['hidden_size']), use_noise, self.trng,
            options['drop_prob_decoder'], (n_samples,options['hidden_size']))
        else:
            pC_inp = p

    n_out_samps = (n_timesteps-1) * n_samples
    if options.get('class_out_factoring',0) == 0:
        pW = (tensor.dot(p,tparams['Wd']) + tparams['bd']).reshape([n_out_samps,options['output_size']])
        if options.get('use_gumbel_mse',0) == 0:
            pWSft = tensor.nnet.softmax(pW)
        else:
            w_out = ifelse(self.usegumbel, gumbel_softmax_sample(self.trng, pW, self.gumb_temp,
                            hard=options.get('use_gumbel_hard',False)), tensor.nnet.softmax(pW))
            # This is not exactly right, but just testing
            pWSft = w_out

        totProb = pWSft[tensor.arange(n_out_samps), xW[1:,:].flatten()]
        out_list = [pWSft, totProb, pW]
    else:
        ixtoclsinfo_t = tensor.as_tensor_variable(self.clsinfo)
        xC = ixtoclsinfo_t[xW[1:,:].flatten(),0]
        if options.get('cls_zmean',1):
            pW = ((tparams['Wd'][:,xC,:].T*((p.reshape([1,n_out_samps,options['hidden_size']])-tparams['WdCls'][:,xC].T))).sum(axis=-1).T
                 + tparams['bd'][:,xC,:])
        else:
            pW = ((tparams['Wd'][:,xC,:].T*(p.reshape([1,n_out_samps,options['hidden_size']]))).sum(axis=-1).T
                 + tparams['bd'][:,xC,:])
        pWSft   = tensor.nnet.softmax(pW[0,:,:])

        pC    = (tensor.dot(pC_inp,tparams['WdCls']) + tparams['bdCls']).reshape([n_out_samps,options['nClasses']])
        pCSft = tensor.nnet.softmax(pC)

        totProb = pWSft[tensor.arange(n_out_samps), ixtoclsinfo_t[xW[1:,:].flatten(),3]] * \
                  pCSft[tensor.arange(n_out_samps), xC]
        out_list = [pWSft, pCSft, totProb, p]

    tot_cost = -(tensor.log(totProb + 1e-10) * mask[1:,:].flatten()).sum()
    tot_pplx = -(tensor.log2(totProb + 1e-10) * mask[1:,:].flatten()).sum()
    cost = [tot_cost/tensor.cast(n_samples,dtype = config.floatX), tot_pplx]

    inp_list = [xW, mask]
    if xI_is_inp:
        inp_list.append(xI)

    if options.get('en_aux_inp',0) and xA_is_inp:
        inp_list.append(xAux)

    if options.get('sched_sampling_mode',None) != None:
        inp_list.append(curr_epoch)

    f_pred_prob = theano.function([xW, xI, xAux], out_list, name='f_pred_prob', updates=updatesLSTM)


    return use_noise, inp_list, f_pred_prob, cost, out_list , updatesLSTM

# ========================================================================================
# Predictor Related Stuff!!

  def prepPredictor(self, model_npy=None, checkpoint_params=None, beam_size=5, xI=None, xAux = None, inp_list_prev=[], per_word_logweight = None):
    if model_npy != None:
        if type(model_npy[model_npy.keys()[0]]) == np.ndarray:
            zipp(model_npy, self.model_th)
        else:
            self.model_th = model_npy

    #theano.config.exception_verbosity = 'high'
    self.beam_size = beam_size

	# Now we build a predictor model
    if checkpoint_params.get('advers_gen',0) == 1:
        checkpoint_params['n_gen_samples'] = beam_size
    (inp_list_gen, predLogProb, predIdx, predCand, wOut_emb, updates, seq_lengths) = self.build_prediction_model(self.model_th, checkpoint_params, xI, xAux, per_word_logweight = per_word_logweight)
    self.f_pred_th = theano.function(inp_list_prev + inp_list_gen, [predLogProb, predIdx, predCand], name='f_pred')

	# Now we build a training model which evaluates cost. This is for the evaluation part in the end
    if checkpoint_params.get('advers_gen',0) == 0:
        (self.use_dropout, inp_list_gen2,
         f_pred_prob, cost, predTh, updatesLSTM) = self.build_model(self.model_th, checkpoint_params, xI, xAux)
        self.f_eval= theano.function(inp_list_prev+inp_list_gen2, cost, name='f_eval')

# ========================================================================================
  def predict(self, batch, checkpoint_params, ext_inp = []):

    inp_list = ext_inp
    if not checkpoint_params.get('use_encoder_for',0)&1:
        inp_list.extend([batch[0]['image']['feat'].reshape(1,checkpoint_params['image_feat_size']).astype(config.floatX)])

    if not checkpoint_params.get('use_encoder_for',0)&2:
        if checkpoint_params.get('en_aux_inp',0):
            inp_list.append(batch[0]['image']['aux_inp'].reshape(1,checkpoint_params['aux_inp_size']).astype(config.floatX))

    Ax = self.f_pred_th(*inp_list)

    # Backtracking to decode the correct sequence of candidates
    Ys = []
    for i in xrange(self.beam_size):
        candI = []
        curr_cand = Ax[2][-1][i]
        for j in reversed(xrange(Ax[1].shape[0]-1)):
            candI.insert(0,Ax[1][j][curr_cand])
            curr_cand = Ax[2][j][curr_cand]

        Ys.append([Ax[0][i], candI])
    return [Ys], Ax

  def build_prediction_model(self, tparams, options, xI=None, xAux = None, per_word_logweight = None):
    #Initialize random streams for other to use.
    self.trng = RandomStreams(int(time.time()))

    if xI == None:
        xI = tensor.matrix('xI', dtype=config.floatX)
        embImg = (tensor.dot(xI, tparams['WIemb']) + tparams['b_Img'])
        xI_is_inp = True
    else:
        xI_is_inp = False
        embImg = xI
    if xAux == None and options.get('en_aux_inp',0):
        xAux = tensor.matrix('xAux', dtype=config.floatX)
        xA_is_inp = True
        if options.get('swap_aux',1):
           xAuxEmb = tensor.dot(xAux,tparams['WIemb_aux']) + tparams['b_Img_aux']
        else:
           xAuxEmb = xAux
    else:
        xA_is_inp = False
        if options.get('encode_gt_sentences',0):
           xAuxEmb = tensor.dot(xAux,tparams['WIemb_aux']) + tparams['b_Img_aux']
        else:
           xAuxEmb = xAux

    if options.get('advers_gen',0) == 1:
        accLogProb, Idx, wOut_emb, updates, seq_lengths = self.lstm_advers_gen_layer(tparams, embImg, xAuxEmb, options, prefix=options['generator'])
        Cand = tensor.tile(tensor.arange(Idx.shape[1]),[Idx.shape[0],1])
    else:
        accLogProb, Idx, Cand, wOut_emb, updates = self.lstm_predict_layer(tparams, embImg, xAuxEmb, options, self.beam_size, prefix=options['generator'],
                per_word_logweight=per_word_logweight)
        seq_lengths = []

    inp_list = []
    if xI_is_inp:
        inp_list.append(xI)
    if options.get('en_aux_inp',0) and xA_is_inp:
        inp_list.append(xAux)

    return inp_list, accLogProb, Idx, Cand, wOut_emb, updates, seq_lengths

# ========================================================================================
  # LSTM LAYER in Prediction mode. Here we don't provide the word sequences, just the image feature vector
  # The network starts first with forward propogatin the image feature vector. Then we pass the start word feature
  # i.e zeroth word vector. From then the network output word (i.e ML word) is fed as the input to the next time step.
  # In beam_size > 1 we could repeat a time step multiple times, once for each beam!!.

  def lstm_predict_layer(self, tparams, Xi, aux_input, options, beam_size, prefix='lstm', per_word_logweight = None):

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
            if options.get('cls_diff_layer',0) == 1:
                pC    = tensor.dot(hL[-2],tparams['WdCls']) + tparams['bdCls']
            else:
                pC    = tensor.dot(outp[-1],tparams['WdCls']) + tparams['bdCls']

            pCSft = tensor.nnet.softmax(pC)
            xCIdx =  tensor.argmax(pCSft,axis=-1)
            #pW = tensor.dot(outp[-1],tparams['Wd'][:,xCIdx,:]) + tparams['bd'][:,xCIdx,:]
            #smooth_factor = tensor.as_tensor_variable(numpy_floatX(options.get('softmax_smooth_factor',1.0)), name='sm_f')
            #pWSft = tensor.nnet.softmax(pW*smooth_factor)
            #lProb = tensor.log(pWSft + 1e-20) + tensor.log(pCSft[0,xCIdx] + 1e-20)
            #########################################################
            # pW is now of size (beam_size, n_classes, class_size)
            if options.get('cls_zmean',0):
                pW = tensor.dot((outp[-1]-tparams['WdCls'][:,xCIdx].T),tparams['Wd'].swapaxes(0,1)) + tparams['bd'][0,:,:]
            else:
                pW = tensor.dot((outp[-1]),tparams['Wd'].swapaxes(0,1)) + tparams['bd'][0,:,:]
            #smooth_factor = tensor.as_tensor_variable(numpy_floatX(options.get('softmax_smooth_factor',1.0)), name='sm_f')
            pWSft = tensor.nnet.softmax(pW.reshape([pW.shape[0]*pW.shape[1],
                    pW.shape[2]])).reshape([pW.shape[0], pW.shape[1]*pW.shape[2]])
            ixtoclsinfo_t = tensor.as_tensor_variable(self.clsinfo)
            lProb = tensor.log(pWSft[:,ixtoclsinfo_t[:,0]*tparams['Wd'].shape[2]+ixtoclsinfo_t[:,3]] + 1e-20) + \
                    tensor.log(pCSft[0,ixtoclsinfo_t[:,0]] + 1e-20)
        else:
            p = tensor.dot(outp[-1],tparams['Wd']) + tparams['bd']
            smooth_factor = tensor.as_tensor_variable(numpy_floatX(options.get('softmax_smooth_factor',1.0)), name='sm_f')
            p = tensor.nnet.softmax(p*smooth_factor)
            lProb = tensor.log(p + 1e-20)
            if per_word_logweight is not None:
                log_w = theano.shared(per_word_logweight)#, dtype= theano.config.floatX)
                lProb = log_w+lProb

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
            if options.get('class_out_factoring',0) == 1:
                clsoffset = tensor.as_tensor_variable(self.clsOffset)
        else:
            xCandIdx = tensor.as_tensor_variable([0])
            lProb = lProb.flatten()
            xWIdx =  tensor.argmax(lProb,keepdims=True)
            xWlogProb = lProb[xWIdx] + lP_
            #if options.get('class_out_factoring',0) == 1:
            #    clsoffset = tensor.as_tensor_variable(self.clsOffset)
            #    xWIdx += clsoffset[xCIdx]
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
  def lstm_advers_gen_layer(self, tparams, xI, xAux, options, prefix='lstm'):
    nBatchSamps = xI.shape[0]
    nMaxsteps = options.get('maxlen',15)
    if nMaxsteps is None:
        nMaxsteps = 30
    n_samp = options.get('n_gen_samples',1)

    h_depth = options.get('hidden_depth',1)
    h_sz = options['hidden_size']

    # ----------------------  STEP FUNCTION  ---------------------- #
    def _stepP(U, xW_, h_, c_, lP_, dV_, xAux, xNoise):
        preact = tensor.dot(sliceT(h_, 0, h_sz), tparams[_p(prefix, 'W_hid')])
        preact += (tensor.dot(xW_, tparams[_p(prefix, 'W_inp')]) +
                   tparams[_p(prefix, 'b')])
        preact += xAux
        if options.get('gen_input_noise',0):
            preact += xNoise

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

        logits = tensor.dot(outp[-1],tparams['Wd']) + tparams['bd']
        #p = tensor.dot(outp[-1],l2norm(tparams['Wd'],axis=0))# + tparams['bd']
        if options.get('use_gumbel_mse',0) == 0 or options.get('greedy',0):
            p = tensor.nnet.softmax(logits)
        else:
            p = gumbel_softmax_sample(self.trng, logits*self.softmax_smooth_factor, self.gumb_temp, U, options.get('use_gumbel_hard',False))

        if options.get('computelogprob', 0):
            lProb = tensor.log(tensor.nnet.softmax(logits*self.softmax_smooth_factor)+1e-20)
        else:
            lProb = logits

        # Idx of the correct word should come from the
        xWIdx =  ~dV_ * tensor.argmax(p,axis = -1)

        xWlogProb = ~dV_ * lProb[tensor.arange(nBatchSamps*n_samp),xWIdx] + lP_
        #xW = tparams['Wemb'][xWIdx.flatten()]
        if options.get('use_gumbel_hard',0) and options.get('use_gumbel_mse', 0) and not options.get('greedy', 0):
            xW = p.dot(tparams['Wemb'])
        else:
            xW = theano.gradient.disconnected_grad(tparams['Wemb'][xWIdx.flatten()].reshape([xWIdx.shape[0],-1]))


        doneVec = tensor.eq(xWIdx,tensor.zeros_like(xWIdx))

        return [xW, h, c, xWlogProb, doneVec, xWIdx, p], theano.scan_module.until(doneVec.all())
    # ------------------- END of STEP FUNCTION  -------------------- #

    if options.get('use_gumbel_mse',0) == 0:
        U = self.trng.uniform((nMaxsteps, 1),low=0.,
                              high=1.,dtype=theano.config.floatX)
    else:
        U = self.trng.uniform((nMaxsteps+1, nBatchSamps * n_samp, options['vocabulary_size']),low=0.,
                              high=1.,dtype=theano.config.floatX)


    xI= tensor.extra_ops.repeat(xI, n_samp,axis=0)
    xAux = tensor.extra_ops.repeat(tensor.dot(xAux,
                tparams[_p(prefix,'W_aux')]), n_samp,axis=0)

    if options.get('gen_input_noise',0):
        xNoise = tensor.dot(self.trng.normal([nBatchSamps * n_samp, self.noise_dim]), tparams[_p(prefix,'W_noise')])
    else:
        xNoise = []

    if options.get('gen_use_rand_init',0) and not options.get('gen_input_noise',0):
        h = tensor.unbroadcast(self.trng.uniform([nBatchSamps * n_samp, h_sz*h_depth],low=-0.1, high= 0.1),0,1)
        c = tensor.unbroadcast(self.trng.uniform([nBatchSamps * n_samp,h_sz*h_depth],low=-0.1, high= 0.1),0,1)
    else:
        h = tensor.zeros([nBatchSamps * n_samp, h_sz*h_depth])
        c = tensor.zeros([nBatchSamps * n_samp,h_sz*h_depth])

    lP = tensor.alloc(numpy_floatX(0.), nBatchSamps * n_samp);
    dV = tensor.alloc(np.bool_(0.), nBatchSamps * n_samp);

    # Propogate the image feature vector
    [_, h, c, _, _, _, _], _ = _stepP(U[0,:], xI, h, c, lP, dV, xAux, xNoise)

    xWStart = tensor.unbroadcast(tensor.tile(tparams['Wemb'][[0]],[nBatchSamps * n_samp,1]) ,0,1)

    # Now lets do the loop.
    rval, updates = theano.scan(_stepP, sequences = [U[1:,:]], outputs_info=[xWStart, h, c, lP, dV, None, None],
                                        non_sequences = [xAux, xNoise], name=_p(prefix, 'adv_predict_layers'),
                                        n_steps=nMaxsteps)

    seq_lengths = theano.gradient.disconnected_grad(tensor.argmax(tensor.concatenate([rval[4][:-1,:],
                    tensor.ones((1,nBatchSamps * n_samp))], axis=0), axis=0)+1)

    return rval[3][-1], rval[5], rval[6], updates, seq_lengths


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


    if options.get('class_out_factoring',0) == 0:
        pW = (tensor.dot(p,tparams['Wd']) + tparams['bd']).reshape([n_out_samps,options['output_size']])
        pWSft = tensor.nnet.softmax(pW)
        totProb = pWSft[tensor.arange(n_out_samps), xW[1:,:].flatten()]
        out_list = [pWSft, totProb, p]
    else:
        ixtoclsinfo_t = tensor.as_tensor_variable(self.clsinfo)
        xC = ixtoclsinfo_t[xW[1:,:].flatten(),0]
        pW = ((tparams['Wd'][:,xC,:].T*((p.reshape([1,n_out_samps,options['hidden_size']])-tparams['WdCls'][:,xC].T))).sum(axis=-1).T
             + tparams['bd'][:,xC,:])
        pWSft   = tensor.nnet.softmax(pW[0,:,:])
        pC    = (tensor.dot(p,tparams['WdCls']) + tparams['bdCls']).reshape([n_out_samps,options['nClasses']])
        pCSft = tensor.nnet.softmax(pC)

        totProb = pWSft[tensor.arange(n_out_samps), ixtoclsinfo_t[xW[1:,:].flatten(),3]] * \
                  pCSft[tensor.arange(n_out_samps), xC]

    tot_cost = -(tensor.log(totProb + 1e-10) * mask[1:,:].flatten()).reshape([n_timesteps-1,n_samples])
    cost = tot_cost.sum(axis=0)

    inp_list = [xW, mask, xI]

    if options.get('en_aux_inp',0):
        inp_list.append(xAux)

    self.f_pred_prob_other = theano.function([xW,xI, xAux], pWSft, name='f_pred_prob', updates=updatesLSTM)
    #f_pred = theano.function([xW, mask], pred.argmax(axis=1), name='f_pred')

    #cost = -tensor.log(pred[tensor.arange(n_timesteps),tensor.arange(n_samples), xW] + 1e-8).mean()

    self.f_eval_other = theano.function(inp_list, cost, name='f_eval')

    return use_noise, inp_list, self.f_pred_prob_other, cost, pW, updatesLSTM

# =================================== MULTI Model Ensemble Predictor related ==========================
