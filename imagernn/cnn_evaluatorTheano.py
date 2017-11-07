import numpy as np
import code
import theano
from theano import config
import theano.tensor as T
from theano.ifelse import ifelse
from theano.tensor.extra_ops import fill_diagonal
from collections import OrderedDict
import time
from imagernn.utils import *
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor.nnet as tnnet

class CnnEvaluator:
  """
  A multimodal long short-term memory (LSTM) generator
  """
# ========================================================================================
  def __init__(self, params,Wemb = None):

    self.word_encoding_size = params.get('word_encoding_size', 512)
    image_feat_size = params.get('image_feat_size', 512)
    aux_inp_size = params.get('aux_inp_size', -1)

    self.n_fmaps_psz = params.get('n_fmaps_psz', 100)
    self.filter_hs = params.get('filter_hs', [])

    # Used for dropout.
    self.use_noise = theano.shared(numpy_floatX(0.))


    vocabulary_size = params.get('vocabulary_size',-1)
    self.sent_enc_size = params.get('sent_encoding_size',-1)# size of CNN vectors hardcoded here

    model = OrderedDict()
    # Recurrent weights: take x_t, h_{t-1}, and bias unit
    # and produce the 3 gates and the input to cell signal
    if Wemb == None:
        model['Wemb'] = initwTh(vocabulary_size-1, self.word_encoding_size) # word encoder
    model['WIemb'] = initwTh(image_feat_size, self.sent_enc_size,inittype='xavier') # image encoder
    #model['b_Img'] = np.zeros((self.sent_enc_size)).astype(config.floatX)


    model['Wfc_sent'] = initwTh(self.n_fmaps_psz * len(self.filter_hs), self.sent_enc_size,inittype='xavier') # word encoder
    #model['bfc_sent'] = np.zeros((self.sent_enc_size)).astype(config.floatX)

    #if params['advers_gen']:
        # Add a merging layer
        #model['Wm_sent'] = initwTh(self.sent_enc_size, params.get('merge_dim',50),inittype='xavier') # word encoder
        #model['Wm_img'] = initwTh(self.sent_enc_size, params.get('merge_dim',50),inittype='xavier') # word encoder
        #model['b_m'] = np.zeros((params.get('merge_dim',50))).astype(config.floatX)
        ## Final output weights
        #model['W_out'] = initwTh(params.get('merge_dim',50),1, 1.0) # word encoder

    # Decoder weights (e.g. mapping to vocabulary)

    update_list = ['Wemb','Wfc_sent','WIemb']
    self.regularize = ['Wemb','Wfc_sent','WIemb']

    if params.get('en_aux_inp',0) and not params['advers_gen']:
        model['WIemb_aux'] = initwTh(aux_inp_size, self.sent_enc_size) # image encoder
        model['b_Img_aux'] = np.zeros((self.sent_enc_size)).astype(config.floatX)

    self.model_th = self.init_tparams(model)

    # Share the Word embeddings with the generator model
    if Wemb != None:
        self.model_th['Wemb'] = Wemb
    self.updateP = OrderedDict()
    for vname in update_list:
        self.updateP[vname] = self.model_th[vname]

    # Instantiate a conv layer already so we don't end up creating new weights
    if params['advers_gen']:
        filter_w = self.word_encoding_size
        self.conv_layers = []
        max_sent_len = params.get('maxlen',0)
        for filter_h in self.filter_hs:
            filter_shape = (self.n_fmaps_psz, params['n_gen_samples'], filter_h, filter_w)
            pool_size = (max_sent_len-filter_h+1, self.word_encoding_size-filter_w+1)
            conv_layer = batch2DConvPoolLayer(filter_shape=filter_shape,
                                              poolsize=pool_size,
                                              non_linear=params['conv_non_linear'])
            # flatten all the filter outputs to a single vector
            self.conv_layers.append(conv_layer)
            self.updateP.update(conv_layer.params)
            self.regularize.extend(conv_layer.regularize)
            self.model_th.update(conv_layer.params)


# ========================================================================================
  def init_tparams(self,params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

# ========================================================================================
 # BUILD CNN evaluator forward propogation model
  def build_model(self, tparams, options, xI=None, prior_inp_list = []):
    trng = RandomStreams()
    rng = np.random.RandomState()

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    xWi = T.matrix('xW', dtype='int64')
    # Now input is transposed compared to the generator!!
    xW = xWi.T
    n_samples = xW.shape[0]
    n_words= xW.shape[1]

    Words = T.concatenate([tparams['Wemb'], T.alloc(numpy_floatX(0.),1,self.word_encoding_size)],axis=0)
    embW = Words[xW.flatten()].reshape([options['batch_size'], 1, n_words, self.word_encoding_size])

    if options.get('use_dropout',0):
        embW = dropout_layer(embW, use_noise, trng, options['drop_prob_encoder'], shp = embW.shape)

    sent_emb, cnn_out , tparams = self.sent_conv_layer(tparams, options, embW, options['batch_size'], use_noise, trng)

    if xI == None:
        xI = T.matrix('xI', dtype=config.floatX)
        xI_is_inp = True
    else:
        xI_is_inp = False


    if options.get('mode','batchtrain') != 'batchtrain':
        posSamp = T.ivector('posSamp')

    if xI_is_inp:
        embImg = T.dot(xI, tparams['WIemb']) + tparams['b_Img']
    else:
        embImg = xI + tparams['b_Img']

    if options.get('use_dropout',0):
        embImg = dropout_layer(embImg, use_noise, trng, options['drop_prob_encoder'], shp = embImg.shape)


    #-------------------------------------------------------------------------------------------------------------#
    # Curr prob is computed by applying softmax over (I0,c0), (I0,c1),... (I0,cn-1) pairs
    # It could also be computed with (I0,c0), (I1,c0),... (In,c0) pairs, but will lead to different discrimination
    # Maybe even sum of the two could be used
    #-------------------------------------------------------------------------------------------------------------#
    probMatchImg, sim_score = multimodal_cosine_sim_softmax(embImg, sent_emb, tparams, options.get('sim_smooth_factor',1.0))
    inp_list = [xWi]
    if xI_is_inp:
        inp_list.append(xI)

    if options.get('en_aux_inp',0):
        xAux = T.matrix('xAux', dtype=config.floatX)
        embAux = T.dot(xAux, tparams['WIemb_aux']) + tparams['b_Img_aux']
        xAuxEmb = dropout_layer(embAux, use_noise, trng, options['drop_prob_aux'], shp = embAux.shape)
        inp_list.append(xAux)
        probMatchAux, sim_scoreAux = multimodal_cosine_sim_softmax(embAux, sent_emb, tparams, options.get('sim_smooth_factor',1.0))
    else:
        probMatchAux = T.alloc(numpy_floatX(0.),1,1)

    probMatch = (probMatchImg + probMatchAux) / 2.

    sortedProb = T.argsort(probMatch,axis=1)

    batch_idces = T.arange(probMatch.shape[0])
    opponents = T.switch(T.eq(sortedProb[:,-1], batch_idces), sortedProb[:,-2], sortedProb[:,-1])

    violator_mask = (probMatch.diagonal() - probMatch[batch_idces,opponents]) < (options.get('cost_margin',0.02))

    n_violators = violator_mask.sum()

    if options.get('mode','batchtrain') == 'batchtrain':
        cost = [-((T.log(probMatch.diagonal())* (1+2.0*violator_mask)).sum())/probMatch.shape[0]]
    else:
        cost = [-(T.log(probMatch[0,posSamp]).sum())/posSamp.shape[0]]

    cost.append(n_violators)
    cost.append((probMatch.diagonal() - probMatch[batch_idces,opponents]))

    f_pred_sim_prob = theano.function(prior_inp_list + inp_list, [probMatchImg, probMatchAux, probMatch, opponents], name='f_pred_sim_prob')
    f_pred_sim_scr = theano.function(prior_inp_list + inp_list[:2], sim_score, name='f_pred_sim_scr')
    f_sent_emb = theano.function(inp_list[:1], cnn_out, name='f_sent_emb')

    if options.get('mode','batchtrain') != 'batchtrain':
        inp_list.append(posSamp)

    return use_noise, inp_list, [f_pred_sim_prob, f_pred_sim_scr, f_sent_emb], cost, sim_score, tparams

# ========================================================================================
 # BUILD CNN evaluator forward propogation model with taking direct inputs from lstm gen
  def build_advers_eval(self, tparams, options, gen_inp_list=None, gen_out=None, genUpdates = None, genLens = None):
    trng = RandomStreams()

    #n_words= xWRef.shape[1]

    zero_guy = T.alloc(numpy_floatX(0.),1,self.word_encoding_size)
    Word_Vecs = T.concatenate([zero_guy, tparams['Wemb']],axis=0)
    #Word_Vecs = tparams['Wemb']

    #Word_Vecs = tparams['Wemb']

    # These are of dimensions B x n_samp x time x Vocab
    if gen_out == None:
        discrim_inp = T.tensor4(name='disc_inp')
        inp_list = [discrim_inp]
        n_ref_samps = discrim_inp.shape[0]
    else:
        refData_inp = tensor.tensor4(name='disc_ref_inp')
        n_ref_samps = refData_inp.shape[0]
        n_words = refData_inp.shape[2]
        n_gen_words = gen_out.shape[2]
        z_shape = list(gen_out.shape)
        z_shape[2] = n_words - n_gen_words
        gen_pad = ifelse(tensor.gt(n_words, n_gen_words), tensor.concatenate([gen_out,
                                            tensor.zeros(z_shape)], axis=2), gen_out)
        discrim_inp = tensor.concatenate([refData_inp, gen_pad], axis=0)
        inp_list = [refData_inp]

    # Embed this input into size B x n_samp x time x word_vec_dim
    embW = T.dot(discrim_inp,Word_Vecs)

    #embGen = ifelse(tensor.gt(n_words, n_gen_words),tensor.concatenate([gen_out,theano.tensor.alloc(numpy_floatX(0.),n_words-n_gen_words,self.word_encoding_size)], axis=0),gen_out)
    #embGen = tensor.shape_padleft(embGen, n_ones=2)

    #embWRef = Words[xWRef.flatten()].reshape([options['eval_batch_size'], 1, n_words, self.word_encoding_size])
    #embW = tensor.concatenate([embWRef, embGen], axis=0)

    max_sent_len = options.get('maxlen',0)
    layer1_inputs = []
    for i,filter_h in enumerate(self.filter_hs):
        pool_size = (max_sent_len-filter_h+1,1)
        self.conv_layers[i].build(embW, poolsize = pool_size)
        # flatten all the filter outputs to a single vector
        cout = self.conv_layers[i].output.flatten(2)
        layer1_inputs.append(cout)

    layer1_input = T.concatenate(layer1_inputs,axis=1)

    # Now apply dropout on the cnn ouptut
    if options.get('use_dropout',0):
        cnn_out = dropout_layer(layer1_input, self.use_noise, trng, options['drop_prob_eval'],layer1_input.shape)
    else:
        cnn_out = layer1_input

    # Now transform this into a sent embedding
    sent_emb = T.dot(cnn_out, tparams['Wfc_sent'])# + tparams['bfc_sent']
    # Add a nonlinearity here
    #sent_emb = nonLinLayer(sent_emb, layer_type=options['conv_non_linear'])

    # Now to embed the image feature vector and calculate a similarity score
    if gen_out == None:
        xImg = T.matrix('xI', dtype=config.floatX)
    else:
        xImg = gen_inp_list[0]

    #Compute Image embedding:
    embImg = T.dot(xImg, tparams['WIemb'])# + tparams['b_Img']
    # Add a nonlinearity here
    #embImg = nonLinLayer(embImg, layer_type=options['conv_non_linear'])

    #if options.get('use_dropout',0):
    #    embImg = dropout_layer(embImg, self.use_noise, trng, options['drop_prob_eval'],embImg.shape)
    #else:
    #    embImg = embImg

    #m_img = l2norm(tensor.dot(embImg, tparams['Wm_img']))
    #m_sent = l2norm(tensor.dot(sent_emb, tparams['Wm_sent']))
    m_img = l2norm(embImg)
    m_sent = l2norm(sent_emb)

    #Now time to merge them
    #merge_out = m_img * m_sent + tparams['b_m']
    #merge_out = nonLinLayer(merge_out, layer_type=options['conv_non_linear'])

    scores = T.dot(m_img, m_sent.T)
    #merge_out = nonLinLayer(merge_out, layer_type='sigm')

    # Final output layer
    #p_out = nonLinLayer(tensor.dot(merge_out, tparams['W_out']), layer_type='sigm')
    p_out = (scores.diagonal())
    if gen_out !=None:
        p_out = T.concatenate([p_out, 0.5*(scores[:,n_ref_samps:].diagonal()+1.0)])
    #p_out = nonLinLayer(5.0*scores.diagonal(), layer_type='sigm').flatten()

    if gen_out !=None:
        for inp in gen_inp_list:
          if inp not in inp_list:
              inp_list.append(inp)
        print inp_list
    else:
        inp_list.append(xImg)

    xTarg = T.fvector('targ')
    inp_list.append(xTarg)
    #import pdb;pdb.set_trace()
    if options.get('eval_loss','contrastive')=='contrastive':
        #costEval, ic_s, ic_i = self.contrastive_loss(m_img, m_sent)
        probMatch = T.nnet.softmax(scores*2.0)
        costEval = -((T.log(probMatch[:,:n_ref_samps].diagonal())*xTarg).sum())
        if gen_out !=None:
            costGen = -((T.log(probMatch[:,n_ref_samps:].diagonal())).sum())
            # Also minimize the probability assigned to the generated fake samples
            #costEval += ((T.log(probMatch[:,n_ref_samps:].diagonal())).sum())
        else:
            costGen = []
        ic_s = probMatch
        ic_i = probMatch
    elif options.get('eval_loss','contrastive')=='wass':
        costEval = (scores[:,:n_ref_samps].diagonal()*xTarg).mean() - (scores[:,:n_ref_samps].diagonal()*(1.-xTarg)).mean()
        if gen_out !=None:
            costGen = -(scores[:,n_ref_samps:].diagonal()).mean()
            costEval += costGen
        costEval = -costEval
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


    f_pred_cost = theano.function(inp_list, costEval, name='f_pred_sim_scr', updates=genUpdates)

    f_pred_sim_prob = theano.function(inp_list[:-1], [p_out], name='f_pred_sim_prob', updates=genUpdates)
    #f_pred_sim_prob = theano.function(inp_list, [p_out, sent_emb, m_img, m_sent, embW, ic_s, ic_i, self.reg_cost], name='f_pred_sim_prob')
    f_sent_emb = theano.function(inp_list[:-1], [m_sent, m_img, scores], name='f_sent_emb', updates=genUpdates)


    return inp_list, [f_pred_sim_prob, f_pred_cost, f_sent_emb], [costEval, costGen], p_out, tparams

  def contrastive_loss(self, im, s, margin=0.1):
      """
      Compute contrastive loss
      """
      # compute image-sentence score matrix
      scores = T.dot(im, s.T)
      diagonal = scores.diagonal()

      # compare every diagonal score to scores in its column (i.e, all contrastive images for each sentence)
      cost_s = T.maximum(0, margin - diagonal + scores)
      # compare every diagonal score to scores in its row (i.e, all contrastive sentences for each image)
      cost_im = T.maximum(0, margin - diagonal.reshape((-1, 1)) + scores)

      # clear diagonals
      cost_s = fill_diagonal(cost_s, 0.)
      cost_im = fill_diagonal(cost_im, 0.)

      return cost_s.sum() + cost_im.sum(), cost_s, cost_im

# ========================================================================================
  ####################################################################################
  # Defines the convolution layer on sentences.
  # -- Input is word embeddings stacked as a n_word * enc_size "image"
  # -- Filters are all of width equal to enc_size, height varies (3,4,5 grams etc.)
  # -- Also pooling is taking max over entire filter output, i.e each filter output
  #    is converted to a single number!
  # -- Output is stacking all the filter outputs to a single vector,
  #    sz = (batch-size,  n_filters)
  ####################################################################################
  def sent_conv_layer(self, tparams, options, embW, batch_size, use_noise, trng, n_samp=1):
    # Used for dropout.
    rng = np.random.RandomState()
    max_sent_len = options.get('maxlen',0)
    filter_shapes = []
    self.conv_layers = []
    pool_sizes = []
    filter_w = self.word_encoding_size
    layer1_inputs = []
    for filter_h in self.filter_hs:
        filter_shapes.append((self.n_fmaps_psz, n_samp, filter_h, filter_w))
        if max_sent_len > 0:
            image_shape = [batch_size, n_samp, max_sent_len, self.word_encoding_size]
        else:
            image_shape = None
        pool_sizes.append((max_sent_len-filter_h+1, self.word_encoding_size-filter_w+1))
        conv_layer = LeNetConvPoolLayer(rng, input= embW, image_shape= image_shape, filter_shape=filter_shapes[-1],
                                poolsize=pool_sizes[-1], non_linear=options['conv_non_linear'])
        # flatten all the filter outputs to a single vector
        cout = conv_layer.output.flatten(2)
        self.conv_layers.append(conv_layer)
        layer1_inputs.append(cout)
        self.updateP.update(conv_layer.params)
        self.regularize.extend(conv_layer.regularize)
        tparams.update(conv_layer.params)

    layer1_input = T.concatenate(layer1_inputs,axis=1)

    # Now apply dropout on the cnn ouptut
    if options.get('use_dropout',0):
        cnn_out = dropout_layer(layer1_input, use_noise, trng, options['drop_prob_cnn'],layer1_input.shape)
    else:
        cnn_out = layer1_input

    # Now transform this into a sent embedding
    sent_emb = T.dot(cnn_out,tparams['Wfc_sent']) + tparams['bfc_sent']

    return sent_emb, cnn_out, tparams

# ========================================================================================
class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input_x, filter_shape, image_shape, poolsize=(2, 2), non_linear="tanh"):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

       # assert image_shape[1] == filter_shape[1]
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.non_linear = non_linear
        self.max_pool_method = 'downsamp'
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /np.prod(poolsize))
        # initialize weights with random weights
        if self.non_linear=="none" or self.non_linear=="relu":
            self.W = theano.shared(np.asarray(rng.uniform(low=-0.01,high=0.01,size=filter_shape),
                                                dtype=config.floatX),name="W_conv")
        else:
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=config.floatX),name="W_conv")
        b_values = np.zeros((filter_shape[0],), dtype=config.floatX)
        self.b = theano.shared(value=b_values, name="b_conv")

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input_x, filters=self.W,filter_shape=self.filter_shape, image_shape=self.image_shape)
        if self.non_linear=="tanh":
            conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output = myMaxPool(conv_out_tanh, ps=self.poolsize, method=self.max_pool_method)
        elif self.non_linear=="relu":
            conv_out_tanh = ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output = myMaxPool(conv_out_tanh, ps=self.poolsize, method=self.max_pool_method)
        else:
            pooled_out = myMaxPool(conv_out, ps=self.poolsize, method=self.max_pool_method)
            self.output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.params = {}
        self.params['CNN_W_h' + str(filter_shape[2]) + '_w' +str(filter_shape[3])] = self.W
        self.params['CNN_b_h' + str(filter_shape[2]) + '_w' +str(filter_shape[3])] = self.b
        self.regularize = ['CNN_W_h' + str(filter_shape[2]) + '_w' +str(filter_shape[3])]


    def predict(self, new_data, batch_size):
        """
        predict for new data
        """
        img_shape = (batch_size, 1, self.image_shape[2], self.image_shape[3])
        conv_out = conv.conv2d(input=new_data, filters=self.W, filter_shape=self.filter_shape, image_shape=img_shape)
        if self.non_linear=="tanh":
            conv_out_tanh = Tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            output = myMaxPool(conv_out_tanh, ps=self.poolsize, method=self.max_pool_method)
        if self.non_linear=="relu":
            conv_out_tanh = ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            output = myMaxPool(conv_out_tanh, ps=self.poolsize, method=self.max_pool_method)
        else:
            pooled_out = myMaxPool(conv_out, ps=self.poolsize, method=self.max_pool_method)
            output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        return output

# ========================================================================================
class batch2DConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, filter_shape, poolsize=(2, 2), non_linear="tanh"):
        """
        Allocate a 3D conv layer with shared variable internal parameters.

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

       # assert image_shape[1] == filter_shape[1]
        self.filter_shape = filter_shape
        self.poolsize = poolsize
        self.non_linear = non_linear
        self.max_pool_method = 'downsamp'
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /np.prod(poolsize))
        # initialize weights with random weights
        if self.non_linear=="none" or self.non_linear=="relu":
            self.W = theano.shared(np.asarray(np.random.uniform(low=-0.01,high=0.01,size=filter_shape),
                                                dtype=config.floatX),name="W_conv")
        else:
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(np.asarray(np.random.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=config.floatX),name="W_conv")
        b_values = np.zeros((filter_shape[0],), dtype=config.floatX)
        self.b = theano.shared(value=b_values, name="b_conv")
        self.params = {}
        self.params['CNN_W_h' + str(filter_shape[2]) + '_w' +str(filter_shape[3])] = self.W
        self.params['CNN_b_h' + str(filter_shape[2]) + '_w' +str(filter_shape[3])] = self.b
        self.regularize = ['CNN_W_h' + str(filter_shape[2]) + '_w' +str(filter_shape[3])]

    def build(self, input_x, poolsize=(2, 2)):

        # convolve input feature maps with filters
        conv_out = tnnet.conv2d(input=input_x, filters=self.W,filter_shape=self.filter_shape)
        if self.non_linear=="tanh":
            conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output = myMaxPool(conv_out_tanh, ps=self.poolsize, method=self.max_pool_method)
        elif self.non_linear=="relu":
            conv_out_tanh = ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output = myMaxPool(conv_out_tanh, ps=self.poolsize, method=self.max_pool_method)
        else:
            pooled_out = myMaxPool(conv_out, ps=self.poolsize, method=self.max_pool_method)
            self.output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')

