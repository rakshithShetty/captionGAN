import json
import os
import h5py
import scipy.io
import codecs
import numpy as np
import theano
import random
from collections import defaultdict, OrderedDict
from imagernn.dataproc.picsom_bin_data import picsom_bin_data
from imagernn.dataproc.numpy_list_data import numpy_list_data
from imagernn.utils import applyFeatPool
import re
import gc

class BasicDataProvider:
  def __init__(self, params):
    dataset = params.get('dataset', 'coco')
    feature_file = params.get('feature_file', 'vgg_feats.mat')
    data_file = params.get('data_file', 'dataset.json')
    mat_new_ver = params.get('mat_new_ver', -1)
    print 'Initializing data provider for dataset %s...' % (dataset, )
    self.hdf5Flag = 0 #Flag indicating whether the dataset is an HDF5 File.
                 #Large HDF5 files are stored (by Vik) as one image
                 #  per row, going against the conventions of the other
                 # storage formats

    # !assumptions on folder structure
    self.dataset_root = os.path.join('data', dataset)
    self.image_root = os.path.join('data', dataset, 'imgs')
    self.use_video_feat = params.get('use_video_feat',0)

    # which is the object id, depending on the dataset
    idstrList = {'coco':'cocoid','lsmdc2015':'id','msr-vtt':'id'}
    self.idstr = idstrList[dataset]

    # load the dataset into memory
    dataset_path = os.path.join(self.dataset_root, data_file)
    print 'BasicDataProvider: reading %s' % (dataset_path, )
    self.dataset = json.load(open(dataset_path, 'r'))
    allIdSet = set([img[self.idstr] for img in self.dataset['images']])

    if params.get('use_partial_sent',[]) != []:
        for i,img in enumerate(self.dataset['images']):
            self.dataset['images'][i]['sentences'] = [img['sentences'][si] for si in params['use_partial_sent']]
        self.dataset['capperimage'] = len(params['use_partial_sent'])
        print 'Using only the following sentence indices: ',params['use_partial_sent']


    # load the image features into memory
    features_path = os.path.join(self.dataset_root, feature_file)
    print 'BasicDataProvider: reading %s' % (features_path, )

    self.capperimage = self.dataset.get('capperimage',5)
    print self.capperimage


    if params['uselabel'] & 1:
        labelsF = os.path.join(self.dataset_root, params['labels'])
        self.featUseLbl = True
        if params.get('disk_feature',0) & 1:
            self.featIdx = parseLabels(labelsF, params['featfromlbl'].split()[0], idtype=int)
            self.feat_reader = prepPartialFileReader(features_path, params['poolmethod'].split()[0])
        else:
            self.features, self.featIdx = loadFromLbls(features_path, labelsF,
                params['featfromlbl'].split()[0], params['poolmethod'].split()[0], imgIdSet = allIdSet)
            self.feat_reader = lambda idxes: self.features[:,idxes].T # this is a 4096 x N numpy array of features
    else:
        raise ValueError('This mode is phased out now, use labels file based feature loading')

    self.img_feat_size = self.feat_reader([0]).shape[1]
    print self.img_feat_size

    #gc.collect()
    self.aux_pres = 0
    self.aux_inp_size = 0
    if params.get('aux_inp_file','None') != 'None':
        # Load Auxillary input file, one vec per image
        # NOTE: Assuming same order as feature file
        aux_inp_file = os.path.join(self.dataset_root,params['aux_inp_file'])
        self.aux_pres = 1
        if params['uselabel'] & 2:
            labelsF = os.path.join(self.dataset_root, params['labels'])
            self.auxUseLbl = True
            # If the feature file is too big to be loaded entirely into ram,
            # just read the labels and prepare the things needed for it.
            if params.get('disk_feature',0) & 2:
                self.auxIdx = parseLabels(labelsF, params['featfromlbl'].split()[-1], idtype=int)
                self.aux_reader = prepPartialFileReader(aux_inp_file, params['poolmethod'].split()[-1])
            else:
                self.aux_inputs, self.auxIdx = loadFromLbls(aux_inp_file, labelsF,
                                                    params['featfromlbl'].split()[-1],
                                                    params['poolmethod'].split()[-1],
                                                    imgIdSet = allIdSet)
                self.aux_reader = lambda idxes: self.aux_inputs[:,idxes].T # this is a 4096 x N numpy array of features
        else:
            raise ValueError('This mode is phased out now, use labels file based feature loading')

        self.aux_inp_size = self.aux_reader([0]).shape[1]
        print self.img_feat_size, self.aux_inp_size

    # group images by their train/val/test split into a dictionary -> list structure
    gc.collect()
    self.split = defaultdict(list)
    for img in self.dataset['images']:
      if params.get('use_train_subset',[]) == [] or  img['split'] != 'train' or img['sub_split'] in set(params.get('use_train_subset',[])):
        self.split[img['split']].append(img)
      if img['split'] != 'train':
        self.split['allval'].append(img)

    # load external data to augment the training captions
    self.use_extdata = 0
    if params.get('ext_data_file',None) != None:
        print 'BasicDataProvider: reading ext data file %s' % (params['ext_data_file'], )
        self.ext_data = json.load(open(params['ext_data_file'],'r'))
        self.use_extdata = 1
        self.edata_prob = params['ed_sample_prob']
        self.ed_features = None
        if params.get('ed_feature_file',None) != None:
            self.ed_features, _ = readFeaturesFromFile(params['ed_feature_file'],
                                        idxes = Ellipsis, mat_new_ver=mat_new_ver)
        self.ed_aux_inputs = None
        if params.get('ed_aux_inp_file',None) != None:
            self.ed_aux_inputs, _ = readFeaturesFromFile(params['ed_aux_inp_file'],
                                        idxes = Ellipsis, mat_new_ver=mat_new_ver)
        for img in self.ext_data['images']:
            if img['split'] == 'train' or img['split'] == 'val':
                self.split['ed_data'].append(img)

    print 'Final split sizes are:',[(spk,len(self.split[spk])) for spk in self.split]
    # Build tables for length based sampling
    lenHist = defaultdict(int)
    self.lenMap = defaultdict(list)
    if(dataset == 'coco'):
        self.min_len = 7
        self.max_len = 27
    elif(dataset == 'lsmdc2015'):
        self.min_len = 1
        self.max_len = 40
    elif(dataset == 'msr-vtt'):
        self.min_len = 2
        self.max_len = 30
    else:
        raise ValueError('ERROR: Dont know how to do len splitting for this dataset')


    # Build the length based histogram
    for iid, img in enumerate(self.split['train']):
      for sid, sent in enumerate(img['sentences']):
        ix = max(min(len(sent['tokens']),self.max_len),self.min_len)
        lenHist[ix] += 1
        self.lenMap[ix].append((iid,sid))

    self.lenCdist = np.cumsum(lenHist.values())


  # "PRIVATE" FUNCTIONS
  # in future we may want to create copies here so that we don't touch the
  # data provider class data, but for now lets do the simple thing and
  # just return raw internal img sent structs. This also has the advantage
  # that the driver could store various useful caching stuff in these structs
  # and they will be returned in the future with the cache present
  def _getImage(self, img, from_ed = 0):
    """ create an image structure for the driver """

    # lazily fill in some attributes
    if not 'local_file_path' in img: img['local_file_path'] = os.path.join(self.image_root, img['filename'])
    if not 'feat' in img: # also fill in the features
      # NOTE: imgid is an integer, and it indexes into features
      if from_ed ==0:
        feature_index = self.featIdx[img[self.idstr]]
        img['img_idx'] = feature_index
        img['feat'] = self.feat_reader(feature_index)
        if self.aux_pres:
          aux_feature_index = self.auxIdx[img[self.idstr]]
          img['aux_idx'] = aux_feature_index
          img['aux_inp'] = self.aux_reader(aux_feature_index)
      else:
          img['feat'] = self.ed_features[:,img['imgid']] if self.ed_features != None else np.zeros((self.img_feat_size,),dtype=theano.config.floatX)
          img['aux_inp'] = self.ed_aux_inputs[:,img['imgid']] if self.ed_aux_inputs != None else np.zeros((self.aux_inp_size,),dtype=theano.config.floatX)
    return img

  def _getSentence(self, sent):
    """ create a sentence structure for the driver """
    # NOOP for now
    return sent

  # PUBLIC FUNCTIONS

  def getSplitSize(self, split, ofwhat = 'sentences'):
    """ return size of a split, either number of sentences or number of images """
    if ofwhat == 'sentences':
      return sum(len(img['sentences']) for img in self.split[split])
    else: # assume images
      return len(self.split[split])

  def sampleImageSentences(self, split = 'train', pos=True, n_sent=5, shuf_sent=False):
    out = {}
    sentIds = np.random.choice(self.capperimage,size=n_sent, replace=False)
    if pos:
        # Put correct image and sentences together
        img_idx = np.random.choice(np.arange(len(self.split[split])),size=1, replace=False)
        out['image'] = self._getImage(self.split[split][img_idx[0]])
        out['sentences'] = [out['image']['sentences'][j] for j in sentIds]
        out['lbl'] = 1
    else:
        # Put incorrect image and sentences together
        img_idx = np.random.choice(np.arange(len(self.split[split])),size=2, replace=False)
        out['image'] = self._getImage(self.split[split][img_idx[0]])
        out['sentences'] = [self.split[split][img_idx[1]]['sentences'][j] for j in sentIds]
        out['lbl'] = 0
    if shuf_sent:
        if n_sent > 1:
            sent_samp = random.choice(out['sentences'])
        for i,st in enumerate(out['sentences']):
            if n_sent == 1:
                out['sentences'][i]['tokens'] = random.sample(st['tokens'],len(st['tokens']))
            else:
                out['sentences'][i] = sent_samp

        out['lbl'] = 0
    return out

  def sampleImageSentencePair(self, split = 'train'):
    """ sample image sentence pair from a split """

    if split != 'train' or self.use_extdata == 0 or random.random()>self.edata_prob:
        images = self.split[split]
        from_ed = 0
    else:
        from_ed = 1
        images = self.split['ed_data']

    img = random.choice(images)
    sent = random.choice(img['sentences'])

    out = {}
    out['image'] = self._getImage(img, from_ed = from_ed)
    out['sentence'] = self._getSentence(sent)
    return out

  def sampleImageSentencePairByLen(self, l):
    """ sample image sentence pair from a split """
    split='train'
    pair = random.choice(self.lenMap[l])

    img = self.split[split][pair[0]]
    #img = {self.idstr:imgLcl[self.idstr], 'img_id':imgLcl['img_id']}
    sent = img['sentences'][pair[1]]

    out = {}
    out['image'] = self._getImage(img)
    out['sentence'] = self._getSentence(sent)
    return out

  def getRandBatchByLen(self,batch_size):
    """ sample image sentence pair from a split """

    rn = np.random.randint(0,self.lenCdist[-1])
    for l in xrange(len(self.lenCdist)):
        if rn < self.lenCdist[l] and (len(self.lenMap[l + self.min_len]) > 0):
            break

    l += self.min_len
    batch = [self.sampleImageSentencePairByLen(l) for i in xrange(batch_size)]
    return batch,l

  # Used for CNN evaluator training
  def sampPosNegSentSamps(self, batch_size, mode = 'batchtrain', thresh = 1):
    """ sample image sentence pair from a split """
    batch = []
    if mode == 'batchtrain':
        img_idx = np.random.choice(np.arange(len(self.split['train'])),size=batch_size, replace=False)
        for i in img_idx:
            batch.append({'sentence':random.choice(self.split['train'][i]['sentences']),
                        'image':self._getImage(self.split['train'][i])})
        posSamp = np.arange(batch_size,dtype=np.int32)
    elif mode == 'multi_choice_mode':
        img_idx = np.random.choice(np.arange(len(self.split['train'])),size=1, replace=False)
        batch.extend([{'sentence':st} for st in self.split['train'][img_idx]['sentences']])
        batch[0]['image'] = self._getImage(self.split['train'][img_idx])
        posSamp = np.array([0],dtype=np.int32)
    elif mode == 'multimodal_lstm':
        img_idx = np.random.choice(np.arange(len(self.split['train'])), size=batch_size, replace=False)
        img = self._getImage(self.split['train'][img_idx[0]])
        for i in img_idx:
            batch.append({'sentence':random.choice(self.split['train'][i]['sentences'])})
        batch[0]['image'] = img
        posSamp = np.arange(1,dtype=np.int32)
    else:
        img_idx = np.random.choice(np.arange(len(self.split['train'])))
        img = self._getImage(self.split['train'][img_idx])
        for si in self.split['train'][img_idx]['prefOrder'][:batch_size]:
            batch.append({'sentence':self.split['train'][img_idx]['sentences'][si]})

        posSamp = np.arange(len(img['candScoresSorted']), dtype=np.int32)[np.array(img['candScoresSorted'])>thresh]
        if len(posSamp) == 0:
            posSamp = np.arange(1, dtype=np.int32)

        # To keep all the batches of same size, pad if necessary
        for i in xrange(batch_size - len(self.split['train'][img_idx]['prefOrder'])):
            batch.append({'sentence':self.split['train'][img_idx]['sentences'][-1]})
        # Finally store image feature
        batch[0]['image'] = img
    return batch, posSamp

  # Used for CNN evaluator training
  def sampAdversBatch(self, batch_size,split='train', probs = [0.5, 0.5, 0.0], n_sent=1):
    c_prob = np.cumsum(probs)

    batch = []
    for i in xrange(batch_size):
      rn = np.random.uniform()
      for l in xrange(len(c_prob)):
          if rn < c_prob[l]:
              break
      if l == 0:
          # Here we sample correct GT.
          batch.append(self.sampleImageSentences(split = split, pos=True, n_sent=n_sent))
      elif l == 1:
          # Here we sample incorrect GT.
          batch.append(self.sampleImageSentences(split = split, pos=False, n_sent=n_sent))
      elif l == 2:
          # Here we sample jumbled GT.
          batch.append(self.sampleImageSentences(split = split, pos=True, n_sent=n_sent, shuf_sent = True))
    return batch

  def iterImageSentencePair(self, split = 'train', max_images = -1):
    for i,img in enumerate(self.split[split]):
      if max_images >= 0 and i >= max_images: break
      for sent in img['sentences']:
        out = {}
        out['image'] = self._getImage(img)
        out['sentence'] = self._getSentence(sent)
        yield out

  def iterImageSentencePairBatch(self, split = 'train', max_images = -1, max_batch_size = 100,shuffle = False):
    batch = []
    imglist = self.split[split]
    ixd = np.array([np.repeat(np.arange(len(imglist)),self.capperimage),
            np.tile(np.arange(self.capperimage),len(imglist))]).T
    if shuffle:
      np.random.shuffle(ixd)
    for i,idx in enumerate(ixd):
      if max_images >= 0 and i >= (self.capperimage * max_images): break
      img = imglist[idx[0]]
      out = {}
      out['image'] = self._getImage(img)
      out['sentence'] = self._getSentence(img['sentences'][idx[1]])
      out['sentidx'] = idx[1]
      batch.append(out)
      if len(batch) >= max_batch_size:
        yield batch
        batch = []
    if batch:
      yield batch

  def iterImageBatch(self, split = 'train', max_images = -1, max_batch_size = 100, shuffle = False):
    batch = []
    imglist = self.split[split]
    ixd =  np.arange(len(imglist))
    if shuffle:
      np.random.shuffle(ixd)
    for i,idx in enumerate(ixd):
      if max_images >= 0 and i >= (self.capperimage * max_images): break
      batch.append(self._getImage(imglist[idx]))
      if len(batch) >= max_batch_size:
        yield batch
        batch = []
    if batch:
      yield batch

  def iterSentences(self, split = 'train'):
    for img in self.split[split]:
      for sent in img['sentences']:
        sent[self.idstr] = img[self.idstr]
        yield self._getSentence(sent)

  def iterImages(self, split = 'train', shuffle = False, max_images = -1):
    imglist = self.split[split]
    ix = range(len(imglist))
    if shuffle:
      random.shuffle(ix)
    if max_images > 0:
      ix = ix[:min(len(ix),max_images)] # crop the list
    for i in ix:
      yield self._getImage(imglist[i])


def process_seqtomatrix(seqs, maxlen):
    # seqs: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None and maxlen > 0:
      new_seqs = []
      new_lengths = []
      for l, s in zip(lengths, seqs):
          if l > maxlen:
              new_seqs.append(s[:maxlen-1]+[0])
              new_lengths.append(maxlen)
          else:
              new_seqs.append(s)
              new_lengths.append(l)
      lengths = new_lengths
      seqs = new_seqs

    if not (maxlen is not None and maxlen > 0):# or prep_for=='lstm_eval':
      maxlen = np.max(lengths)
    n_samples = len(seqs)

    xWId = np.zeros((n_samples,maxlen)).astype('int64')
    for idx, s in enumerate(seqs):
      xWId[idx,:lengths[idx]] = s

    return xWId, lengths

def prepare_adv_data(batch, wordtoix, maxlen=None, prep_for='cnn'):
    xI = np.row_stack([x['image']['feat'] for x in batch])
    if 'aux_inp' in batch[0]['image']:
        xAux = [np.row_stack(x['image']['aux_inp'] for x in batch)]
    else:
        xAux = []
    seqs = []
    prefix = []
    targ = []

    for ix,x in enumerate(batch):
      for j, st in enumerate(x['sentences']):
        alltks = st['tokens']
        seqs.append(prefix + [wordtoix[w] for w in alltks if w in wordtoix] + [0])
      targ.append(x['lbl'])

    xWId, lengths = process_seqtomatrix(seqs, maxlen)

    if prep_for == 'cnn':
        xW = np.eye(len(wordtoix),dtype=theano.config.floatX)[xWId.flatten()]
        xW = xW.reshape(len(batch), len(batch[0]['sentences']),-1,len(wordtoix))
        inp_list = [xW, xI] + xAux + [np.array(targ,dtype=theano.config.floatX)]
    else:
        lensXw = np.array(lengths, dtype=np.int64)
        xWId = xWId.T.reshape(-1,len(batch), len(batch[0]['sentences']))
        inp_list = [xWId, lensXw, xI] + xAux + [np.array(targ,dtype=theano.config.floatX)]

    return inp_list

def prepare_seq_features(batch, use_enc_for=0, maxlen=None, use_shared_mem = 0, pos_samp=None,
                         enc_gt_sent = 0, n_enc_sent = 5, wordtoix = None):
    """Create the matrices from the datasets.

    If maxlen is set, we will cut all sequence to this maximum
    length. In case of CNN we will also extend all the sequences
    to reach this maxlen.

    Prepare masks if needed and in approp form.

    Allowed values for prep_for are: ['lstm_gen', 'lstm_eval', 'cnn']
    """
    #prep_for_cls = prep_for.split('_')[0]
    #if prep_for_cls == 'bilstm':
    #    prep_for_cls = 'lstm'
    #prep_for_subcls = prep_for.split('_')[1] if len(prep_for.split('_')) > 1 else ''

    inp_list = []
    if pos_samp != None:
        batch = [batch[i] for i in pos_samp]

    if (use_enc_for & 1) and not enc_gt_sent:
        lensXI = np.array([x['image']['feat'].shape[0] for x in batch], dtype=np.int64)
        if use_shared_mem:
            xI = np.zeros((lensXI.max(),len(batch)),dtype=np.int64) - 1
        else:
            xI = np.zeros((lensXI.max(),len(batch),x['image']['feat'].shape[1]),dtype=theano.config.floatX)
        for i,bat in enumerate(batch):
            xI[:lensXI[i],i] = bat['image']['img_idx'] if use_shared_mem else bat['image']['feat']
        inp_list.extend([xI, lensXI-1])


    if (use_enc_for & 2):
        if enc_gt_sent:
            seqs = []
            prefix = []
            for ix,x in enumerate(batch):
                for j, st in enumerate(x['image']['sentences'][:n_enc_sent]):
                    alltks = st['tokens']
                    seqs.append(prefix + [wordtoix[w] for w in alltks if w in wordtoix] + [0])

            xWId, lens = process_seqtomatrix(seqs, maxlen)
            inp_list.extend([xWId.T, np.array(lens, dtype=np.int64)-1])
        else:
            lensAuxI = np.array([x['image']['aux_inp'].shape[0] for x in batch], dtype=np.int64)
            if use_shared_mem:
                xAux = np.zeros((lensAuxI.max(),len(batch)),dtype=np.int64) -1
            else:
                xAux = np.zeros((lensAuxI.max(),len(batch),x['image']['aux_inp'].shape[1]),dtype=theano.config.floatX)
            for i,bat in enumerate(batch):
                xAux[:lensAuxI[i],i] = bat['image']['aux_idx'] if use_shared_mem else bat['image']['aux_inp']
            inp_list.extend([xAux, lensAuxI-1])

    return inp_list

def prepare_data(batch, wordtoix, maxlen=None, sentTagMap=None, ixw = None, pos_samp = [],
                prep_for = 'lstm_gen', rand_negs = 0, rev_sents = 0, use_enc_for=0, use_unk_token=0):
    """Create the matrices from the datasets.

    If maxlen is set, we will cut all sequence to this maximum
    length. In case of CNN we will also extend all the sequences
    to reach this maxlen.

    Prepare masks if needed and in approp form.

    Allowed values for prep_for are: ['lstm_gen', 'lstm_eval', 'cnn']
    """
    prep_for_cls = prep_for.split('_')[0]
    if prep_for_cls == 'bilstm':
        prep_for_cls = 'lstm'
    prep_for_subcls = prep_for.split('_')[1] if len(prep_for.split('_')) > 1 else ''

    seqs = []
    if pos_samp == []:
      xI = np.row_stack(x['image']['feat'] for x in batch)
    else:
      xI = np.row_stack(batch[i]['image']['feat'] for i in pos_samp)

    prefix = [0] if prep_for_cls == 'lstm' else []
    for ix,x in enumerate(batch):
      alltks = x['sentence']['tokens'] if rev_sents == 0 else reversed(x['sentence']['tokens'])
      if use_unk_token == 0:
        seqs.append(prefix + [wordtoix[w] for w in alltks if w in wordtoix] + [0])
      else:
        seqs.append(prefix + [wordtoix[w] if w in wordtoix else 1 for w in alltks] + [0])

    if rand_negs > 0:
        for i in xrange(rand_negs):
            seqs.append(np.random.choice(seqs[0],np.maximum(maxlen,len(seqs[0]))).tolist())
    # x: a list of sentences
    lengths = [len(s) for s in seqs]
    orig_lengths = lengths

    if maxlen is not None and maxlen > 0:
      new_seqs = []
      new_lengths = []
      for l, s in zip(orig_lengths, seqs):
          if l > maxlen:
              new_seqs.append(s[:maxlen-1]+[0])
              new_lengths.append(maxlen)
          else:
              new_seqs.append(s)
              new_lengths.append(l)
      lengths = new_lengths
      seqs = new_seqs

    if not (maxlen is not None and maxlen > 0) or (prep_for_cls == 'lstm'): # and prep_for_subcls != 'eval'):
      maxlen = np.max(lengths)

    n_samples = len(seqs)

    xW = np.zeros((maxlen, n_samples)).astype('int64')
    # Masks are only for lstms
    if prep_for_cls == 'lstm':
      if prep_for_subcls == 'gen':
        x_mask = np.zeros((maxlen, n_samples)).astype(theano.config.floatX)
      else:
        x_mask = np.array(lengths,dtype=np.int64) -1

    for idx, s in enumerate(seqs):
      xW[:lengths[idx], idx] = s
      if prep_for_cls == 'cnn':
        xW[lengths[idx]:, idx] = -1
      # Masks are only for lstms
      elif prep_for_cls == 'lstm' and prep_for_subcls == 'gen':
        x_mask[:lengths[idx], idx] = 1.
        if sentTagMap != None:
          for i,sw in enumerate(s):
            if sentTagMap[batch[idx]['sentence']['sentid']].get(ixw[sw],'') == 'JJ':
              x_mask[i,idx] = 2

    inp_list = [xW]
    if prep_for_cls == 'lstm':
      inp_list.append(x_mask)

    if not (use_enc_for & 1):
        inp_list.append(xI)

    if 'aux_inp' in batch[0]['image']:
      if pos_samp == []:
        xAux = np.row_stack(x['image']['aux_inp'] for x in batch)
      else:
        xAux = np.row_stack(batch[i]['image']['aux_inp'] for i in pos_samp)
      if not (use_enc_for & 2):
        inp_list.append(xAux)

    return inp_list, (np.sum(lengths) - n_samples*len(prefix))

def getDataProvider(params):
  """ we could intercept a special dataset and return different data providers """
  assert params['dataset'] in ['flickr8k', 'flickr30k', 'coco', 'lsmdc2015', 'msr-vtt'], 'dataset %s unknown' % (dataset, )
  return BasicDataProvider(params)

def parseLabels(labelsFile, featfromlbl, idtype=str):
  lbls = open(labelsFile).read().splitlines()
  imgIdtoidxPrePool = defaultdict(list)
  feat_load_list = []
  for lb in lbls:
    lbParts = lb.split()
    lbParts[1] = lbParts[1][1:-1]
    if (len(lbParts[1].split(':')) == 1):
        if featfromlbl == 'ALL':
            feat_load_list.append(int(lbParts[0][1:]))
            imgIdtoidxPrePool[idtype(lbParts[1].split(':')[0])].append(len(feat_load_list)-1)
    elif re.match(featfromlbl,lbParts[1].split(':')[1]):
        feat_load_list.append(int(lbParts[0][1:]))
        imgIdtoidxPrePool[idtype(lbParts[1].split(':')[0])].append(len(feat_load_list)-1)
  return imgIdtoidxPrePool

def loadFromLbls(features_path, labelsFile, featfromlbl, poolmethod, features = None, imgIdtoidxPrePool = None, imgIdSet = None):
  print 'parsing labels with %s and pooling with %s'%(featfromlbl, poolmethod)
  if features == None:
    imgIdtoidxPrePool = parseLabels(labelsFile, featfromlbl, idtype=int)
    assert(len(imgIdtoidxPrePool.keys())>0)
    if imgIdSet != None:
        imgIdtoidxPrePool = {imgId: imgIdtoidxPrePool[imgId] for imgId in imgIdSet}
    imgIdtoidxPrePool_v2 = OrderedDict()
    feat_load_list = []
    for i,imgId in enumerate(imgIdtoidxPrePool):
        imgIdtoidxPrePool_v2[imgId] = []
        for j,ridx in enumerate(imgIdtoidxPrePool[imgId]):
            feat_load_list.append(ridx)
            imgIdtoidxPrePool_v2[imgId].append(len(feat_load_list)-1)
    features, hdf5Flag = readFeaturesFromFile(features_path, idxes = feat_load_list)
  elif imgIdtoidxPrePool == None:
    assert('illegal inputs, both features and prepoolIds need to be given')

  featOut, idxPostPool = poolFeatById(features, imgIdtoidxPrePool_v2, poolmethod)

  return featOut, idxPostPool

def poolFeatById(features, imgIdtoidxPrePool, poolmethod):
  imgIdtoidxPostPool = {}
  featOut = []
  curr_idx = 0
  for i,imgid in enumerate(imgIdtoidxPrePool):
    featOut.append(applyFeatPool(poolmethod,features[:,imgIdtoidxPrePool[imgid]]))
    imgIdtoidxPostPool[imgid] = np.arange(curr_idx, curr_idx+featOut[-1].shape[1])
    curr_idx = curr_idx+featOut[-1].shape[1]

  featOut = np.concatenate(featOut,axis=1)
  print 'Done with pooling, final sizes are ', featOut.shape, ' ', imgIdtoidxPostPool[imgid]
  return featOut, imgIdtoidxPostPool

def preprocess_feat_Flist(root, flist):
  ret = []
  for f in flist:
    if f.lstrip()== '' or f[0] == '#':
      continue
    df = f
    if f[0] != '/':
      df = root[0:root.rfind('/')]+'/'+f
    ret = ret + [ df ]
  return ret
def getReadStructByFileType(filename):
    if filename.rsplit('.',1)[1] == 'bin':
        print "Working on bin file now"
        return picsom_bin_data(filename)
    elif filename.rsplit('.',1)[1] == 'npzl':
        print "Working on list of numpy files now"
        return numpy_list_data(filename)
    else:
        raise ValueError('Unknown feature file type')

def prepPartialFileReader(filename, poolmethod):
    if filename.rsplit('.',1)[1] == 'txt':
        print "Working on txt file now"
        #This is for feature concatenation.
        # NOTE: Assuming same order of features in all the files listed in the txt file
        feat_Flist = open(filename, 'r').read().splitlines()
        feat_Flist = preprocess_feat_Flist(filename, feat_Flist)
        f_struct = [getReadStructByFileType(f) for f in feat_Flist]
    else:
        f_struct = [getReadStructByFileType(filename)]

    if 'randprojconcat' in poolmethod:
        dimred = poolmethod.split('_')[-1]
        st_bkup = np.random.get_state()
        np.random.seed(int(dimred.split('+')[-1]))
        dimMat = np.random.standard_normal(map(int,dimred.split('+')[0].split('x'))).astype(np.float32)
        np.random.set_state(st_bkup)
    else:
        dimMat = None

    # The mega one line lambda function!!
    fread_idx = lambda idxes, f_struct=f_struct, poolmethod=poolmethod, dimMat = dimMat: applyFeatPool(poolmethod,np.concatenate([np.array(fst.get_float_list(idxes)).T.astype(theano.config.floatX) for fst in f_struct], axis=0), proj_mat = dimMat).T
    return fread_idx

def loadArbitraryFeatures(params, idxes = Ellipsis,auxidxes = []):

  feat_all = []
  aux_all = []
  feat_idx = []
  aux_idx = []
  if params.get('multi_model',0) == 0:
    params['nmodels'] = 1

  for im in xrange(params['nmodels']):
      #----------------------------- Loop -------------------------------------#
      features_path = params['feat_file'][im] if params.get('multi_model',0) else params['feat_file']

      if (params.get('uselabel',0) &1):
        features, idPostPool = loadFromLbls(features_path, params['labels'],
                params['featfromlbl'].split()[0], params['poolmethod'].split()[0],
                imgIdSet =  set(idxes))
        feat_idx = [idPostPool[i] for i in idxes]
        #features = features[:,[idPostPool[i] for i in idxes]]
      elif idxes != Ellipsis and type(idxes) != int and type(idxes[0]) == list:
        features,_ = readFeaturesFromFile(features_path, idxes=Ellipsis)
        imgIdtoidxPrePool = {i:idxes[i] for i in xrange(len(idxes))}
        features, idPostPool = loadFromLbls(None, None, None, params['poolmethod'], features,
                imgIdtoidxPrePool = imgIdtoidxPrePool)
        feat_idx = [idPostPool[i] for i in xrange(len(idxes))]
        #features = features[:,[idPostPool[i] for i in xrange(len(idxes))]]
      else:
        features,_ = readFeaturesFromFile(features_path, idxes=idxes)
        feat_idx = np.arange(features.shape[1])

      aux_reader = None
      aux_inp_file = params['aux_inp_file'][im] if params.get('multi_model',0) else params.get('aux_inp_file',None)
      if aux_inp_file != None:
        # Load Auxillary input file, one vec per image
        # NOTE: Assuming same order as feature file
        auxidxes = idxes if auxidxes == [] else auxidxes
        if (params.get('uselabel',0) &2):
          if (params.get('disk_feature',0)&2):
            aux_idx = parseLabels(params['labels'], params['featfromlbl'].split()[-1])
            aux_idx = [aux_idx[i] for i in auxidxes]
            aux_reader = prepPartialFileReader(aux_inp_file, params['poolmethod'].split()[-1])
          else:
            aux_inp, idPostPool = loadFromLbls(aux_inp_file, params['labels'],
                    params['featfromlbl'].split()[-1],
                    params['poolmethod'].split()[-1], imgIdSet =  set(auxidxes))
            aux_idx = [idPostPool[i] for i in auxidxes]
          #aux_inp = aux_inp[:,[idPostPool[i] for i in auxidxes]]
        elif auxidxes != Ellipsis and type(auxidxes[0]) == list:
          aux_inp,_ = readFeaturesFromFile(aux_inp_file, idxes=Ellipsis)
          aux_inp, idPostPool = loadFromLbls(None, None, None, params['poolmethod'], aux_inp,
                  imgIdtoidxPrePool = {i:auxidxes[i] for i in xrange(len(auxidxes))})
          aux_idx = [idPostPool[i] for i in xrange(len(auxidxes))]
          aux_inp= aux_inp[:,[idPostPool[i] for i in xrange(len(auxidxes))]]
        else:
          aux_inp,_ = readFeaturesFromFile(aux_inp_file, idxes=auxidxes,mat_new_ver = params.get('mat_new_ver',1))
          aux_idx = np.arange(aux_inp.shape[1])

        if aux_reader == None:
          aux_reader = lambda idxes,aux_inp=aux_inp: aux_inp[:,idxes].T # this is a 4096 x N numpy array of features


      feat_all.append(features)
      aux_all.append(aux_reader)

  if params.get('multi_model',0) == 0:
    return features, aux_reader, feat_idx, aux_idx
  else:
    return feat_all, aux_all

def readFeaturesFromFile(filename, idxes = Ellipsis, mat_new_ver = 1):

  def loadSingleFeat(filename, idxes, mat_new_ver):
    hdf5_flag = 0
    if filename.rsplit('.',1)[1] == 'mat':
      print "Working on mat file now"
      if mat_new_ver == 1:
          features_struct = h5py.File(filename)
          features = np.array(features_struct[features_struct.keys()[0]]).astype(theano.config.floatX)[:,idxes]
      else:
          features_struct = scipy.io.loadmat(filename)
          features = features_struct['feats'][idxes,:].astype(theano.config.floatX)
    elif filename.rsplit('.',1)[1] == 'hdf5':
      #If the file is one of Vik's HDF5 Files
      print "Working on hdf5 file now"
      features_struct = h5py.File(filename,'r')
      features = features_struct['features'][idxes,:].astype(theano.config.floatX) # this is a N x 2032128 array of features
      hdf5_flag = 1
    elif filename.rsplit('.',1)[1] == 'bin':
      print "Working on bin file now"
      features_struct = picsom_bin_data(filename)
      if idxes == Ellipsis:
        idxes_inp = -1
      else:
        idxes = np.array(idxes)
        asort = np.argsort(idxes)
        idxes_inp = list(idxes[asort])
        idxes_op = np.argsort(asort)
      features = np.array(features_struct.get_float_list(idxes_inp)).T.astype(theano.config.floatX) # this is a 4096 x N numpy array of features
      if (idxes != Ellipsis).any():
        features = features[:,idxes_op]
    elif filename.rsplit('.',1)[1] == 'npy':
      print "Working on npy file now"
      features = np.load(open(filename,'rb'))[idxes,:].T.astype(theano.config.floatX)
    return features, hdf5_flag

  if filename.rsplit('.',1)[1] == 'txt':
    print "Working on txt file now"
    #This is for feature concatenation.
    # NOTE: Assuming same order of features in all the files listed in the txt file
    feat_Flist = open(filename, 'r').read().splitlines()
    feat_Flist = preprocess_feat_Flist(filename, feat_Flist)
    feat_list = []
    for f in feat_Flist:
        fOut, hdf5_flag = loadSingleFeat(f, idxes, mat_new_ver)
        feat_list.append(fOut)
        print feat_list[-1].shape
  	# this is a 4096 x N numpy array of features
    features = np.concatenate(feat_list, axis=0)
    print "Combined all the features. Final size is %d %d"%(features.shape[0],features.shape[1])
  else:
    features, hdf5_flag = loadSingleFeat(filename, idxes, mat_new_ver)


  return features, hdf5_flag
