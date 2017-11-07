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

class RecurrentFeatEncoder:
  """
  A long short-term memory (LSTM) network to embed and encode sequential features
  """
# ========================================================================================
  def __init__(self, feat_size, enc_size, params, mdl_prefix='', features=None):

    self.hidden_size = params.get('featenc_hidden_size', enc_size)
    hidden_size = self.hidden_size
    self.hidden_depth = params.get('featenc_hidden_depth', 1)
    self.en_residual_conn = params.get('featenc_en_residual_conn',0)
    encoder = params.get('feat_encoder', 'lstm')
    self.encoder = encoder
    self.image_feat_size = feat_size# size of CNN vectors hardcoded here

    self.mp = mdl_prefix
    mp = self.mp

    model = OrderedDict()
    # Recurrent weights: take x_t, h_{t-1}, and bias unit
    # and produce the 3 gates and the input to cell signal

    model[mp+'lstm_W_hid'] = initwTh(hidden_size, 4 * hidden_size)
    model[mp+'lstm_W_inp'] = initwTh(self.image_feat_size, 4 * hidden_size)

    if encoder == 'bilstm':
        model[mp+'rev_lstm_W_hid'] = initwTh(hidden_size, 4 * hidden_size)
        model[mp+'rev_lstm_W_inp'] = initwTh(word_encoding_size, 4 * hidden_size)
        model[mp+'rev_lstm_b'] = np.zeros((4 * hidden_size,)).astype(config.floatX)

    for i in xrange(1,self.hidden_depth):
        model[mp+'lstm_W_hid_'+str(i)] = initwTh(hidden_size, 4 * hidden_size)
        model[mp+'lstm_W_inp_'+str(i)] = initwTh(hidden_size, 4 * hidden_size)
        if encoder == 'bilstm':
            model[mp+'rev_lstm_W_hid_'+str(i)] = initwTh(hidden_size, 4 * hidden_size)
            model[mp+'rev_lstm_W_inp_'+str(i)] = initwTh(hidden_size, 4 * hidden_size)

    model[mp+'lstm_b'] = np.zeros((4 * hidden_size,)).astype(config.floatX)

    # This is to make sure that initial gradient flow can propogate a long time.
    model[mp+'lstm_b'][hidden_size:hidden_size*2] = 2*np.ones((hidden_size,)).astype(config.floatX)
    # Decoder weights (e.g. mapping to vocabulary)

    update_list = ['lstm_W_hid', 'lstm_W_inp', 'lstm_b']
    self.regularize = ['lstm_W_hid', 'lstm_W_inp']
    if encoder == 'bilstm':
        update_list.extend(['rev_lstm_W_hid', 'rev_lstm_W_inp', 'rev_lstm_b'])
        self.regularize.extend(['rev_lstm_W_hid', 'rev_lstm_W_inp'])

    for i in xrange(1,self.hidden_depth):
        update_list.append('lstm_W_hid_'+str(i))
        update_list.append('rev_lstm_W_hid_'+str(i))
        if encoder == 'bilstm':
            self.regularize.append('lstm_W_inp_'+str(i))
            self.regularize.append('rev_lstm_W_inp_'+str(i))

    self.model_th = self.init_tparams(model)
    ## Store the feature vectors on GPU already!!
    if params.get('use_shared_mem_enc',0):
        self.features = features
        self.use_shared_features = True


    self.regularize = [mp+rgl for rgl in self.regularize]
    self.update_list = [mp+upl for upl in update_list]

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
    self.use_noise = theano.shared(numpy_floatX(0.))

    if self.use_shared_features == False:
        xI = tensor.tensor3('xI', dtype=config.floatX)
        xIemb = xI
        n_timesteps = xI.shape[0]
        n_samples = xI.shape[1]
    else:
        xI = tensor.matrix('xI', dtype='int64')
        n_timesteps = xI.shape[0]
        n_samples = xI.shape[1]
        #feats = tensor.concatenate([self.features,tensor.alloc(numpy_floatX(0.),self.image_feat_size,1)],axis=1).T
        xIemb = self.features[xI.flatten(),:].reshape([n_timesteps,
                                                n_samples,
                                                self.image_feat_size])

    samp_lens = tensor.vector('sL', dtype='int64')


    #This is implementation of input dropout !!
    if options['use_dropout']:
        emb = dropout_layer(xIemb, self.use_noise, trng, options['drop_prob_encoder'], shp = xIemb.shape)

    #############################################################################################################################
    # This implements core lstm
    rval, updatesLSTM = self.lstm_enc_layer(tparams, emb, prefix=self.mp+'lstm')
    #############################################################################################################################
    # This implements core reverse lstm
    if self.encoder == 'bilstm':
        rev_rval, rev_updatesLSTM = basic_lstm_layer(tparams, emb[::-1,:,:], prefix=self.mp+'rev_lstm')
    #############################################################################################################################

    # NOTE1: we are leaving out the first prediction, which was made for the image and is meaningless.
    p = sliceT(rval[0][samp_lens,tensor.arange(n_samples),:],self.hidden_depth,self.hidden_size)

    if self.encoder == 'bilstm':
        rev_p = sliceT(rev_rval[0][-1,:,:],self.hidden_depth, self.hidden_size)

    feat_enc = p + rev_p if self.encoder == 'bilstm' else p

    if options.get('encoder_add_mean',0):
            feat_enc = feat_enc + (sliceT(rval[0],self.hidden_depth,self.hidden_size).sum(axis=0) / samp_lens[:,None])

    inp_list = [xI, samp_lens]

    return self.use_noise, inp_list, feat_enc, updatesLSTM

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
