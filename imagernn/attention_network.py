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

class AttentionNetwork:
  """
  Module which provides attention mechanism over a sequence of feature vectors
  """
# ========================================================================================
  def __init__(self, feat_size, state_size, params, mdl_prefix=''):

    self.hidden_config = params.get('attn_hidden_config', [])
    n_layers = len(self.hidden_config)
    assert(n_layers>0)
    self.en_residual_conn = params.get('attn_en_residual_conn',0)
    self.nw_type = params.get('attn_nw', 'mlp')
    self.image_feat_size = feat_size# size of CNN vectors hardcoded here
    self.non_linear_layer = params.get('attn_nonlin','relu')

    self.mp = mdl_prefix
    mp = self.mp

    model = OrderedDict()
    # Recurrent weights: take x_t, h_{t-1}, and bias unit
    # and produce the 3 gates and the input to cell signal

    model[mp+'W_att_img'] = initwTh(feat_size, self.hidden_config[0])
    model[mp+'W_att_sta'] = initwTh(state_size, self.hidden_config[0])
    model[mp+'b_att_img'] = np.zeros((self.hidden_config[0],)).astype(config.floatX)
    update_list = ['W_att_img', 'W_att_sta', 'b_att_img']
    self.regularize = ['W_att_img', 'W_att_sta']
    for i in xrange(1,n_layers):
        model[mp+'W_att_'+str(i)] = initwTh(self.hidden_config[i-1], self.hidden_config[i])
        model[mp+'b_att_'+str(i)] = np.zeros((self.hidden_config[i],)).astype(config.floatX)
        update_list.extend(['W_att_'+str(i),'b_att_'+str(i)])
        self.regularize.append('W_att_'+str(i))
    
    model[mp+'W_att_fin'] = initwTh(self.hidden_config[-1], 1)
    update_list.extend(['W_att_fin'])
    self.regularize.append('W_att_fin')
    self.model_th = self.init_tparams(model)

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
  def build_model(self, X_img, h_state, use_noise, img_emb_matrix, img_emb_bias):
    trng = RandomStreams(1234)

    # Used for dropout.
    self.use_noise = use_noise 
    n_time_steps = x_img.shape[0]
    mp = self.mp
    model = self.model_th
    n_layers = len(self.hidden_config)
    nLinType = self.non_linear_layer
    
    #X_2D = X_img.reshape([X_img.shape[0]*X_img.shape[1], X_img.shape[2]])

    def f_att(x_img, h_st, model):
        act = []
        img_emb = tensor.dot(x_img, model[mp+'W_att_img']) + model[mp+'b_att_img']
        sta_emb = tensor.dot(h_st, model[mp+'W_att_sta'])
        act.append(nonLinLayer(img_emb,nLinType) + nonLinLayer(sta_emb,nLinType))
        for i in xrange(1,n_layers):
            # Add as many layers as in the config
            act.append(nonLinLayer(tensor.dot(act[i-1], model[mp+'W_att_'+str(i)]) + model[mp+'b_att_'+str(i)], nLinType))
        
        fin_score = tensor.dot(act[-1], model[mp+'W_att_fin'])
        
        return fin_score
   
    attn_scrs = f_att(X_img, h_state, model)

    attn_probs = tensor.nnet.softmax(attn_scrs.reshape([attn_scrs.shape[0], attn_scrs.shape[1]]))

    curr_ctxt =  (attn_probs[:,:,None]*X_image).sum(axis=-1)

    emb_ctxt = tensor.dot(curr_ctxt, img_emb_matrix) + img_emb_bias 

    return emb_ctxt 

