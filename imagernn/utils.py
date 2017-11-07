from random import uniform
import numpy as np
import theano
from theano import config
import theano.tensor as tensor
from theano.tensor.signal import pool
from collections import OrderedDict, defaultdict
from itertools import tee
import time

def randi(N):
  """ get random integer in range [0, N) """
  return int(uniform(0, N))

def merge_init_structs(s0, s1):
  """ merge struct s1 into s0 """
  for k in s1['model']:
    assert (not k in s0['model']), 'Error: looks like parameter %s is trying to be initialized twice!' % (k, )
    s0['model'][k] = s1['model'][k] # copy over the pointer
  s0['update'].extend(s1['update'])
  s0['regularize'].extend(s1['regularize'])

def initw(n,d): # initialize matrix of this size
  magic_number = 0.1
  return (np.random.rand(n,d) * 2 - 1) * magic_number # U[-0.1, 0.1]

def initwTh(n,d,magic_number=0.1, inittype = 'norm'): # initialize matrix of this size
  if inittype == 'xavier':
     magic_number = np.sqrt(6.) / np.sqrt(n + d)
  return ((np.random.rand(n,d) * 2 - 1) * magic_number).astype(config.floatX) # U[-0.1, 0.1]

def initwThNd(shape, magic_number=0.1, inittype = 'norm'): # initialize matrix of this size
  if inittype == 'xavier':
     magic_number = np.sqrt(6.) / np.sqrt(shape.sum())
  return (np.random.randn(*shape) * magic_number).astype(config.floatX) # U[-0.1, 0.1]

def init_tparams(params):
  tparams = OrderedDict()
  for kk, pp in params.iteritems():
      tparams[kk] = theano.shared(params[kk], name=kk)
  return tparams

def _p(pp, name):
    return '%s_%s' % (pp, name)

def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)

def accumNpDicts(d0, d1):
  """ forall k in d0, d0 += d1 . d's are dictionaries of key -> numpy array """
  for k in d1:
    if k in d0:
      d0[k] += d1[k]
    else:
      d0[k] = d1[k]

def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    if type(tparams) == list:
        for i in xrange(len(params)):
            tparams[i].set_value(params[i])
    else:
        for kk in tparams:
            if kk in params:
                tparams[kk].set_value(params[kk])
            else:
                print '%s not found in cp. Skipping...'%(kk)

def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    if type(zipped) == list:
        new_params = []
        for vv in zipped:
            new_params.append(vv.get_value())
    else:
        new_params = OrderedDict()
        for kk, vv in zipped.iteritems():
            new_params[kk] = vv.get_value()
    return new_params

def forwardSubRoutine(Hin,Hout, X, WLSTM,IFOG,IFOGf,C,n,d):

    for t in xrange(n):

      prev = np.zeros(d) if t == 0 else Hout[t-1]
      #tanhC_version = 1
      Hin[t,0] = 1
      Hin[t,1:1+d] = X[t]
      Hin[t,1+d:] = prev

      # compute all gate activations. dots:
      IFOG[t] = Hin[t].dot(WLSTM)

      IFOGf[t,:3*d] = 1.0/(1.0+np.exp(-IFOG[t,:3*d])) # sigmoids; these are the gates
      IFOGf[t,3*d:] = np.tanh(IFOG[t, 3*d:]) # tanh

      C[t] = IFOGf[t,:d] * IFOGf[t, 3*d:]
      if t > 0: C[t] += IFOGf[t,d:2*d] * C[t-1]

      Hout[t] = IFOGf[t,2*d:3*d] * np.tanh(C[t])

      #  Hout[t] = IFOGf[t,2*d:3*d] * C[t]
    return Hin, Hout, IFOG,IFOGf,C

def softmax(x,axis = -1):
    xs = x.shape
    ndim = len(xs)
    if axis == -1:
        axis = ndim -1

    z = np.max(x,axis=axis)
    y = x - z[...,np.newaxis] # for numerical stability shift into good numerical range
    e1 = np.exp(y)
    p1 = e1 / np.sum(e1,axis=axis)[...,np.newaxis]

    return p1

def cosineSim(x,y):
    n1 = np.sqrt(np.sum(x**2))
    n2 = np.sqrt(np.sum(y**2))
    sim = x.T.dot(y)/(n1*n2) if n1 !=0.0 and n2!= 0.0 else 0.0
    return sim

def sliceT(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n * dim:(n + 1) * dim]
    return _x[:, n * dim:(n + 1) * dim]

#Theano functions
def ReLU(x, alpha = 0.):
    y = tensor.nnet.relu(x, alpha=alpha)
    return(y)
def Sigmoid(x):
    y = tensor.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = tensor.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)

def l2norm(X,axis=1):
    """
    Compute L2 norm, row-wise
    """
    norm = tensor.sqrt(tensor.pow(X, 2).sum(axis=axis)+1e-20)
    if axis == 1:
        X /= (norm[:, None] + 1e-10)
    else:
        X /= (norm[None, :] + 1e-10)
    return X

def nonLinLayer(x,layer_type='ReLU', alpha=0.01):
    if layer_type == 'relu':
        return ReLU(x, alpha=0.01)
    elif layer_type == 'sigm':
        return Sigmoid(x)
    elif layer_type == 'tanh':
        return Tanh(x)
    elif layer_type == 'iden':
        return Iden(x)
    else:
        raise ValueError('Unknown nonlinear layer type %s'%(layer_type))

def myMaxPool(x, ps=[],method='downsamp'):
    if method == 'downsamp':
        y = pool.pool_2d(input= x, ds=ps, ignore_border=True, mode= 'max')
    elif method == 'max':
        y = tensor.max(x, axis=3).max(axis=2)
    return(y)

def preProBuildWordVocab(sentence_iterator, word_count_threshold, options = None):
  # count up all word counts so that we can threshold
  # this shouldnt be too expensive of an operation

  st_iter1, st_iter2 = tee(sentence_iterator)

  print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, )
  t0 = time.time()
  word_counts = {}
  nsents = 0
  for sent in st_iter1:
    nsents += 1
    for w in sent['tokens']:
      word_counts[w] = word_counts.get(w, 0) + 1
  vocab = {w:word_counts[w] for w in word_counts if word_counts[w] >= word_count_threshold}
  print 'filtered words from %d to %d in %.2fs' % (len(word_counts), len(vocab), time.time() - t0)

  if (options != None) and (options['class_out_factoring'] == 1):
    print 'Clustering words into %d calsses for output factorization ' % (options['nClasses'])
    t0 = time.time()
    if options['class_inp_file'] == None:
        import os
        fInName = 'input' + options['dataset'] + 'TrainTok'
        fIn = open(os.path.join(options['clust_tool_dir'], fInName + '.txt'),'w')
        for st in st_iter2:
            fIn.write(' '.join([w if w in vocab else '' for w in st['tokens']]))
            fIn.write('\n')
        fIn.close()
        owd = os.getcwd()
        os.chdir(options['clust_tool_dir'])
        clust_cmd = './wcluster --text '+ fInName + '.txt --c ' + str(options['nClasses'])
        print ' Invoking the clustering tool now...'
        os.system(clust_cmd)
        os.chdir(owd)
        options['class_inp_file'] = os.path.join(options['clust_tool_dir'], fInName + '-c' + str(options['nClasses']) + '-p1.out/paths')
        print 'Clustering is done in %.2fs ... Now onto processing the outputs' % (time.time() - t0)

    clustOut = open(options['class_inp_file'],'r').readlines()
    classes = defaultdict(list)
    treetocix = {}

    for cls in clustOut:
        lineS = cls.split()
        if lineS[0] not in treetocix:
            treetocix[lineS[0]] = len(treetocix)
        cix = treetocix[lineS[0]]
        if int(lineS[2]) >= word_count_threshold and (lineS[1] != '#UNK' or options.get('use_unk_token',0)):
            classes[cix].append({'w':lineS[1],'c':word_counts[lineS[1]]})

    # Re-arrange the vocabulary by grouping into classes
    vocab = []
    clstoix = {}
    wordtocls= {}
    class_counts = defaultdict(int)
    for cls in classes:
        # +1 is to compensate for insertion of '.' later
        clstoix[cls] = {'strt':len(vocab)+1,'len':len(classes[cls])}
        for wSt in classes[cls]:
            class_counts[cls] += wSt['c']
            wordtocls[wSt['w']] = cls
            vocab.append(wSt['w'])

    # Adding special STOP class containing only '.' to the class list
    # #START# is not needed because it is force fed to the model and
    # model doesn't ever have to produce the class output #START#
    treetocix['STOP'] = len(treetocix)
    cls = treetocix['STOP']
    classes[cls] = [{'w':'.', 'c':nsents}]
    wordtocls['.'] = cls
    clstoix[cls] = {'strt':0,'len':1}
    class_counts[cls] = nsents

    cixtotree = {}
    for treeHash in treetocix:
        cixtotree[treetocix[treeHash]] = treeHash

    print 'Class based factorization of output done %.2fs' % (time.time() - t0)
    max_cls_len = np.max([clstoix[cls]['len'] for cls in clstoix])
    print 'Maximum Class size is  %d' % (max_cls_len)



  # with K distinct words:
  # - there are K+1 possible inputs (START token and all the words)
  # - there are K+1 possible outputs (END token and all the words)
  # we use ixtoword to take predicted indices and map them to words for output visualization
  # we use wordtoix to take raw words and get their index in word vector matrix
  ixtoword = {}
  ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
  wordtoix = {}
  wordtoix['#START#'] = 0 # make first vector be the start token
  ix = 1
  for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

  word_counts['.'] = nsents

  if (options == None) or (options['class_out_factoring'] == 0):
    ##############################################################################################
    # compute bias vector, which is related to the log probability of the distribution
    # of the labels (words) and how often they occur. We will use this vector to initialize
    # the decoder weights, so that the loss function doesnt show a huge increase in performance
    # very quickly (which is just the network learning this anyway, for the most part). This makes
    # the visualizations of the cost function nicer because it doesn't look like a hockey stick.
    # for example on Flickr8K, doing this brings down initial perplexity from ~2500 to ~170.
    ##############################################################################################
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector
  else:
    ##############################################################################################
    # Ths matrices below is used withing the lstm generator module inorder to correctly
    # factorize the output and compute softmax locally within a class.
    # This maps word id to its class id and class location info within the composite decode matrix
    # idx 0 is for class id. idx 1 is for class start location. idx 2 is for class end location
    # idx 3 is for word id within its class, needed to map class softmax to overall vocab
    ##############################################################################################
    ixtoclsinfo = np.zeros((len(ixtoword),4),dtype=np.int32)
    for cls in classes:
        for i,wSt in enumerate(classes[cls]):
            cS = clstoix[cls]['strt']
            cE = clstoix[cls]['strt'] +  clstoix[cls]['len']
            ixtoclsinfo[wordtoix.get(wSt['w'],0), :] = [cls, cS, cE, i]
    ##############################################################################################
    # For class based output clustering, we need to intialize two sets of biases. One is inter-class
    # bias reflecting the frequencies of each aggr lass. Next is the n intra-class biases, which is
    # only for items words within a class
    ##############################################################################################
    bias_init_inter_class = np.array([1.0*class_counts[cls] for cls in clstoix])
    bias_init_inter_class /= np.sum(bias_init_inter_class) # normalize to frequencies
    bias_init_inter_class = np.log(bias_init_inter_class)
    bias_init_inter_class -= np.max(bias_init_inter_class) # shift to nice numeric range

    bias_init_intra_class = -100*np.ones((1,len(classes),max_cls_len))
    for cls in classes:
        idx = np.arange(0, clstoix[cls]['len'])
        bias_init_intra_class[0, cls, idx] = np.array([1.0*word_counts.get(wrd['w'],0) for wrd in classes[cls]])
        bias_init_intra_class[0, cls, idx] /= np.sum(bias_init_intra_class[0,cls,idx]) # normalize to frequencies
        bias_init_intra_class[0, cls, idx] = np.log(bias_init_intra_class[0,cls,idx])
        bias_init_intra_class[0, cls, idx] -= np.max(bias_init_intra_class[0,cls,idx]) # shift to nice numeric range

    return [wordtoix, classes] , [ixtoword, cixtotree, ixtoclsinfo], [bias_init_intra_class, bias_init_inter_class]

# ========================================================================================
# LSTM LAYER DEFINITION
# This is a simple forward propogating lstm layer with no bells and whistles,
# This can be used to encode an input sequence or for training mode in image captioning
# Supports arbitrarily deep lstm layer, but only forward propogation and is only
# ========================================================================================


def basic_lstm_layer(tparams, state_below, aux_input, use_noise, options, prefix='lstm', sched_prob_mask = [], attn_nw = None):
  nsteps = state_below.shape[0]
  h_depth = options.get('hidden_depth',1)
  h_sz = options['hidden_size']

  if state_below.ndim == 3:
      n_samples = state_below.shape[1]
  else:
      n_samples = 1

  def _step(x_in, xp_m,  h_, c_, xwout_, xAux):
      preact = tensor.dot(sliceT(h_, 0, h_sz), tparams[_p(prefix, 'W_hid')])
      if options.get('sched_sampling_mode',None) == None:
        preact += x_in
      else:
        xy_emb = tensor.dot(xwout_, tparams[_p(prefix, 'W_inp')] + tparams[_p(prefix, 'b')])
        temp_container = tensor.concatenate([xy_emb.dimshuffle('x',0,1), x_in.dimshuffle('x', 0, 1)],axis=0)
        preact += temp_container[ xp_m, tensor.arange(n_samples),:]

      if options.get('en_aux_inp',0):
          if attn_nw == None:
            preact += tensor.dot(xAux,tparams[_p(prefix,'W_aux')])
          else:
             # Which hidden state should be used here!? Now using the bottom most one
            emb_xAux = attn_nw(xAux, sliceT(h_, 0, h_sz), use_noise, tparams['WIemb_aux'], tparams['b_Img_aux'])
            preact += tensor.dot(emb_xAux,tparams[_p(prefix,'W_aux')])

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
          if options.get('en_residual_conn',1):
              if (di > 0):
                outp[di] += outp[di-1]
                print "Connecting residual at %d"%(di)
          if di < (h_depth - 1):
              preact = tensor.dot(sliceT(h_, di+1, h_sz), tparams[_p(prefix, ('W_hid_' + str(di+1)))]) + \
                      tensor.dot(outp[di], tparams[_p(prefix, ('W_inp_' + str(di+1)))])


      c_out = tensor.concatenate(c,axis=1)
      h_out = tensor.concatenate(h+[outp[-1]],axis=1)
      if options.get('sched_sampling_mode',None) == None:
        xw_out = xwout_
      else:
        y = tensor.dot(h[-1],tparams['Wd']) + tparams['bd']
        xWIdx =  tensor.argmax(y, axis=-1,keepdims=True)
        xw_out = tparams['Wemb'][xWIdx.flatten()].reshape([n_samples,options['word_encoding_size']])

      return h_out, c_out, xw_out

  state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W_inp')]) + tparams[_p(prefix, 'b')])

  if options.get('en_aux_inp',0) == 0:
     aux_input = []

  if options.get('sched_sampling_mode',None) == None:
    sched_prob_mask = tensor.alloc(1, nsteps, n_samples)
    xw_out = tensor.alloc(numpy_floatX(0.), 1, 1)
  else:
    xw_out = tensor.alloc(numpy_floatX(0.), n_samples, options['word_encoding_size'])


  rval, updates = theano.scan(_step,
                              sequences=[state_below, sched_prob_mask],
                              outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                         n_samples,
                                                         (h_depth+1)*h_sz),
                                            tensor.alloc(numpy_floatX(0.),
                                                         n_samples,
                                                         h_depth*h_sz),
                                            xw_out
                                            ],
                              non_sequences = [aux_input] ,
                              name=_p(prefix, '_layers'),
                              n_steps=nsteps)
  return rval, updates

# ======================== Dropout Layer =================================================
# Implements a simple dropout layer. When droput is on it drops units according to speeci-
# -fied prob and scales the rest. NOP otherwise
# ========================================================================================
def dropout_layer(inp, use_noise, trng, prob, shp):
  scale = 1.0/(1.0-prob)
  proj = tensor.switch(use_noise,
                       (inp *
                        trng.binomial(shp,
                                      p=prob, n=1,
                                      dtype=inp.dtype)*scale),
                       inp)
  return proj


# ======================== Multimodal cosine sim =========================================
# Embeds Image feature vector and computes the cosine sim and softmax with a given sent emb
# ========================================================================================
def multimodal_cosine_sim_softmax(embImg, sent_emb, tparams, sm_f):
  sim_score = tensor.dot(embImg,sent_emb.T)/tensor.dot(embImg.norm(2,axis=1)[:,None],sent_emb.norm(2,axis=1)[None,:])
  # Now to implement the cost function!
  # We can use two kinds of cost, ranking hinge loss or negative log likelihod
  # Below we implement negetive log_likelihood
  smooth_factor = tensor.as_tensor_variable(numpy_floatX(sm_f), name='sm_f')
  probMatch = tensor.nnet.softmax(sim_score*smooth_factor)

  return probMatch,sim_score

# ======================== Multimodal cosine sim =========================================
# Embeds Image feature vector and computes the cosine sim and softmax with a given sent emb
# ========================================================================================
def multimodal_euc_dist_softmax(embImg, sent_emb, tparams, sm_f):
  euc_dist = ((embImg[:,None,:] - sent_emb[None,:,:])**2).sum(axis=-1) / (xI.shape[0] * xI.shape[1])
  # Now to implement the cost function!
  # We can use two kinds of cost, ranking hinge loss or negative log likelihod
  # Below we implement negetive log_likelihood
  smooth_factor = tensor.as_tensor_variable(numpy_floatX(sm_f), name='sm_f')
  probMatch = tensor.nnet.softmax(-euc_dist*smooth_factor)

  return probMatch, euc_dist

def normalizeSal(sal, normType = 'L1'):
    if normType[0] == 'L':
        return np.power(sal,float(normType[-1]))/np.power(sal,float(normType[-1])).sum()
    elif normType[0] == 'S':
        return (1 + np.power(sal,float(normType[-1]))/np.power(sal,float(normType[-1])).sum())

def applyFeatPool(pooltype, featInp, proj_mat = None):
  if pooltype == 'max':
    return featInp.max(axis=1, keepdims=True)
  elif pooltype == 'mean':
    return featInp.mean(axis=1, keepdims=True)
  elif pooltype == 'min':
    return featInp.min(axis=1, keepdims=True)
  elif pooltype == 'sqrmean':
    return (featInp**2).mean(axis=1, keepdims=True)
  elif pooltype == 'none':
    return featInp
  elif pooltype == 'randprojconcat':
    return np.dot(proj_mat,featInp).T.flatten()[:,None]
  elif 'sal_' in pooltype:
    poolConfig = pooltype.split('_')[1:]
    dims_f = map(int,poolConfig[0].split('x'))
    dims_sc = map(int,poolConfig[1].split('x')) if poolConfig[1] != 'I' else poolConfig[1]
    sal_norm = poolConfig[2]
    dims_pool = map(int,poolConfig[3].split('x'))
    pool_type = poolConfig[4]
    fullFeat = featInp[:reduce(lambda x,y: x*y, dims_f),:].reshape(dims_f)
    if dims_sc != 'I':
        fullSc   = featInp[reduce(lambda x,y: x*y, dims_f):,:].reshape(dims_sc)
        fullSc = normalizeSal(fullSc, sal_norm)
    red_dims = [dims_f[-2] - dims_pool[0]+1, dims_f[-1] - dims_pool[ 1]+1]
    red_feat = np.zeros([fullFeat.shape[0]*dims_pool[0]*dims_pool[1],red_dims[0]*red_dims[1]], dtype=config.floatX)
    red_sc = np.zeros([1,red_dims[0]*red_dims[1]], dtype=np.float32)
    for i in xrange(red_dims[0]):
        for j in xrange(red_dims[1]):
            red_feat[:,i*red_dims[0]+j] = fullFeat[:,i:i+dims_pool[0],j:j+dims_pool[1]].T.flatten()
            red_sc[:,i*red_dims[0]+j] = fullSc[:,i:i+dims_pool[0],j:j+dims_pool[1]].sum() if dims_sc !='I' else 1.
    red_sc = normalizeSal(red_sc, sal_norm)
    return applyFeatPool(pool_type, red_feat*red_sc, proj_mat = proj_mat)
  else:
    raise ValueError('Unknown pooling type %s'%(pooltype))


# Utilitites related to gumbel softmax:
def sample_gumbel(trng, shape, eps=np.float32(1e-8), U=None):
  """Sample from Gumbel(0, 1)"""
  if U == None:
    U = trng.uniform(shape,low=0.,high=1.)
  return -tensor.log(-tensor.log(U + eps) + eps)

# Utilitites related to gumbel softmax:
def gumbel_softmax_sample(trng, logits, tau, U=None, hard=False):
  """Sample from Gumbel(0, 1)"""
  ylog = logits + sample_gumbel(trng, logits.shape, U=U)
  y = tensor.nnet.softmax(ylog/tau)

  if hard:
      print 'Using hard gumbel'
      one_hot = tensor.cast(tensor.eq(y, y.max(axis=-1,keepdims=1)),dtype=config.floatX)
      y = theano.gradient.disconnected_grad(one_hot -y) + y
  return y

# Utilitites related to gumbel softmax:
def sample_gumbel_np(shape, eps=1e-20):
  """Sample from Gumbel(0, 1)"""
  U = np.random.uniform(0.0,1.0,size=shape)
  return -np.log(-np.log(U + eps) + eps)

# Utilitites related to gumbel softmax:
def gumbel_softmax_sample_np(logits, tau):
  """Sample from Gumbel(0, 1)"""
  y = logits + sample_gumbel_np(logits.shape)
  return softmax(y/tau)

# -----------------------------------------------
def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

def compute_div_n(caps,n=1):
  aggr_div = []
  for k in caps:
      all_ngrams = set()
      lenT = 0.
      for c in caps[k]:
         tkns = c.split()
         lenT += len(tkns)
         ng = find_ngrams(tkns, n)
         all_ngrams.update(ng)
      aggr_div.append(float(len(all_ngrams))/ (1e-6 + float(lenT)))
  return np.array(aggr_div).mean(), np.array(aggr_div)

def compute_global_div_n(caps,n=1):
  aggr_div = []
  all_ngrams = set()
  lenT = 0.
  for k in caps:
      for c in caps[k]:
         tkns = c.split()
         lenT += len(tkns)
         ng = find_ngrams(tkns, n)
         all_ngrams.update(ng)
  if n == 1:
    aggr_div.append(float(len(all_ngrams)))
  else:
    aggr_div.append(float(len(all_ngrams))/ (1e-6 + float(lenT)))
  return aggr_div[0], np.repeat(np.array(aggr_div),len(caps))
