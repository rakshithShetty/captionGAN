from matplotlib.backends.backend_pdf import PdfPages
import json
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from picsom_bin_data import picsom_bin_data
import matplotlib.patches as mpatches

def visMaps(inpMap, grdsize, brd=1):
    shp = inpMap.shape
    strX = (shp[0]+brd)
    strY = (shp[1]+brd)
    mX = shp[0]
    mY = shp[1]
    resArry = np.zeros((grdsize[0]*(strX),grdsize[1]*(strY)))
    #first to draw the gridMap
    for i in xrange(1,grdsize[0]):
        resArry[i*strX-brd:i*strX,:] = np.ones((brd,resArry.shape[1]))

    for j in xrange(1,grdsize[1]):
        resArry[:,j*strY-brd:j*strY] = np.ones((resArry.shape[0],brd))

    for i in xrange(shp[2]):
        x = i//grdsize[1]
        y = i%grdsize[1]
        # First put the maps in
        resArry[x*strX: x*strX + mX, y*strY:y*strY + mY] = inpMap[:,:,i]
        #print i

    return resArry


ncat = 80;
gridSz = (8,8)
cmap = cm.copper_r;
pGridSz = (8,10)
n_samples = 10

#spatMapFeat = np.load(open('data/coco/spatMapFeat_IOU_8x8_all80.npy','rb')).T
spatMapFeat = np.load(open('data/coco/spatMapFeat_IOU_8x8_all80_corrected.npy','rb')).T

features_struct = picsom_bin_data('data/coco/linear::c_in14_o6_fc6_d_c::hkm-int2::8x8x80.bin')
detMapFeat = np.array(features_struct.get_float_list(-1)).T.astype('float16')
randIdces = np.random.randint(0,spatMapFeat.shape[1],n_samples)

pp = PdfPages('plots/spatMapFeatVsdetFeat_grey.pdf')
fig = plt.figure()
resZeros = visMaps(np.zeros((gridSz[0],gridSz[1],ncat)),pGridSz)
plt.imshow(resZeros,cmap=cmap)
for c in xrange(ncat):
    i = (c//pGridSz[1])*(gridSz[0]+1) + gridSz[0]/3
    j = (c%pGridSz[1])*(gridSz[1]+1) + gridSz[1]/3
    plt.text(j,i,str(c))
    #print i,j
pp.savefig(fig)

fig = plt.figure()
red_patch = mpatches.Patch(color='red', label='The red data')
catList = ['%d - %s'%(i, ann['name']) for i,ann in enumerate(data['categories'])]
plt.legend([red_patch]*80, catList,ncol=4, fontsize='x-small',markerscale=0.5)
pp.savefig(fig)

for i in randIdces:
    fig = plt.figure()
    resSpat = visMaps(spatMapFeat[:,i].reshape(ncat,gridSz[0],gridSz[1]).T/np.max(spatMapFeat[:,i]),(pGridSz[0],pGridSz[1]))
    resdet = visMaps(detMapFeat[:,i].reshape(ncat,gridSz[0],gridSz[1]).T,(pGridSz[0],pGridSz[1]))
    ax = plt.subplot(221);ax.imshow(resSpat,cmap=cmap);ax.set_title('GrndTruth')
    ax = plt.subplot(222);ax.imshow(resdet,cmap=cmap);ax.set_title('detMap')
    ax = plt.subplot(223);ax.imshow(plt.imread(os.path.join(basepath,data['images'][i]['filepath'], data['images'][i]['filename'])))
    ax.set_title('%d, %d'%(i, data['images'][i]['cocoid']))
    pp.savefig(fig)

plt.close("all")
pp.close()


##### Visualize saptial maps

from matplotlib.backends.backend_pdf import PdfPages
import json
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt

def visMaps(inpMap, grdsize, brd=1):
    shp = inpMap.shape
    strX = (shp[0]+brd)
    strY = (shp[1]+brd)
    mX = shp[0]
    mY = shp[1]
    resArry = np.zeros((grdsize[0]*(strX),grdsize[1]*(strY)))
    #first to draw the gridMap
    for i in xrange(1,grdsize[0]):
        resArry[i*strX-brd:i*strX,:] = np.ones((brd,resArry.shape[1]))

    for j in xrange(1,grdsize[1]):
        resArry[:,j*strY-brd:j*strY] = np.ones((resArry.shape[0],brd))

    for i in xrange(shp[2]):
        x = i//grdsize[1]
        y = i%grdsize[1]
        # First put the maps in
        resArry[x*strX: x*strX + mX, y*strY:y*strY + mY] = inpMap[:,:,i]
        #print i

    return resArry


mdlSpat = pickle.load(open('trainedModels/model_checkpoint_coco_gpu008_spatMap_SwapFeat_Goog_10.38.p','r'))
wS = mdlSpat['model']['WIemb_aux']
mapS = wS.reshape((80,8,8,512))
mapS = (mapS - np.min(wS))/ (np.max(wS)-np.min(wS))

pp = PdfPages('plots/spatMap8x8AllClassVis_color.pdf')
for i in xrange(mapS.shape[0]):
    fig = plt.figure()
    res = visMaps(mapS[i,:,:,:],(16,32))
    #plt.imshow(res,cmap=cm.Greys_r)
    plt.imshow(res)
    plt.title('W_emb for the category %s'%(data['categories'][i]['name']))
    print 'Now in %s'%(data['categories'][i]['name'])
    pp.savefig(fig)

pp.close()

################################################
for img in resMulti['imgblobs']:
    imgid = int(img['img_path'].split('_')[-1].split('.')[0])
    resByIdS[imgid] = img['candidate']
for img in resGS['imgblobs']:
    imgid = int(img['img_path'].split('_')[-1].split('.')[0])
    resByIdG[imgid] = img['candidate']


for i,k in enumerate(resByIdS.keys()):
    scrsBoth[i,:] = [k,resByIdS[k]['meteor_sc'],resByIdG[k]['meteor_sc']]


srtidxS = np.argsort(scrsBoth[:,1])
srtidxG = np.argsort(scrsBoth[:,2])


lines = plt.plot(np.arange(scrsBoth.shape[0]),scrsBoth[srtidxS,1],np.arange(scrsBoth.shape[0]),scrsBoth[srtidxS,2],'*')
plt.setp(lines[0], linewidth=3)
plt.setp(lines[1], markersize=1.0)
plt.xlabel('instance no')
plt.ylabel('Meteor score')
plt.legend(['Score of the SpatMap model','score of GSwap'])
plt.show()


lines = plt.plot(np.arange(scrsBoth.shape[0]),scrsBoth[srtidxG,1],'*',np.arange(scrsBoth.shape[0]),scrsBoth[srtidxG,2])
plt.setp(lines[1], linewidth=3)
plt.setp(lines[0], markersize=1.0)
plt.xlabel('instance no')
plt.ylabel('Meteor score')
plt.legend(['Score of the SpatMap model','score of GSwap'])
plt.show()





############### Plotting class trees #######################################
import pydot
import cPickle as pickle
cv = pickle.load(open('trainedModels/model_checkpoint_coco_g68_c_in14_GSWAP_cls200_10.86.p','r'))
from graphviz import Digraph
from graphviz import Graph
misc = cv['misc']
from collections import defaultdict
import json
genDict = defaultdict(int)
resMulti = json.load(open('example_images/result_struct_cls200SWAP.json','r'))
for img in resMulti['imgblobs']:
    for w in img['candidate']['text'].split():
        genDict[w] += 1

clstotree = misc['clstotree']
treedepth = (max([len(cw) for cw in clstotree.values()]))
classes = misc['classes']

nodes = {}
nodeNameLR = ['L','R']
for c in clstotree:
    if clstotree[c] == 'STOP':
        continue
    pathToRoot = ['Root']
    curr_name = ''
    for i,cp in enumerate(clstotree[c]):
        curr_name += cp
        pathToRoot.append(curr_name)
    pathToRoot[-1] = 'cls_' + str(c)
    for i in xrange(len(pathToRoot)-1):
        pN = pathToRoot[i]
        if pN not in nodes:
            nodes[pN] = {'children':{}}
        if pathToRoot[i+1] not in nodes[pN]['children']:
            nodes[pN]['children'][pathToRoot[i+1]] = 1
        else:
            nodes[pN]['children'][pathToRoot[i+1]] += 1
    nodes[pathToRoot[-1]] = {'children':{},'words':classes[c]}

G = Digraph('Word_cluster_200', filename='word_cluster_200cls.gv', engine='dot')
for n in nodes:
    if n == 'Root':
        G.attr('node', shape='ellipse')
        G.node(n, label='ROOT')
    elif len(nodes[n]['children']) > 0:
        G.attr('node', shape='point')
        G.node(n, label='')
    else:
        G.attr('node', shape='box')
        G.node(n, label=r"\n".join([w['w'] + ' ' + str(w['c']) for w in nodes[n]['words'][-10:]]))
        #G.node(n, label=nodes[n]['words'][-1]['w'] + ' ' + str(nodes[n]['words'][-1]['c']))

for n in nodes:
    if len(nodes[n]['children']) > 0:
        for ch in nodes[n]['children']:
            G.edge(n, ch)
G.render()

##############################################################################
resFileList = [ 'example_images/result_struct_Train_Goog_feat_aux_swap.json',
     'example_images/result_struct_Train_gr_pool5_d_aA3_ca3_80Aux.json',
     'example_images/result_struct_Train_gr_pool5_d_aA3_ca3_o9fc8Aux.json',
     'example_images/result_struct_Train_posJJ2_10p62.json',
     'example_images/result_struct_Train_Vggfc7_80Aux.json',
     'example_images/result_struct_Train_VGGfc7_fc8Aux.json' ]
src_mod_names = ['gSw','g80','g1k','pJJ','v80','v1k']

resJs = []
for f in resFileList:
    resJs.append(json.load(open(f,'r')))
imgblobs = []
for i in xrange(len(resJs[0]['imgblobs'])):
    curr_cands_txt = []
    curr_cand_final = []
    for m in xrange(len(resFileList)):
        img = resJs[m]['imgblobs'][i]
        if img['candidate']['text'] not in curr_cands_txt:
            curr_cands_txt.append(img['candidate']['text'])
            curr_cand_final.append(img['candidate'])
            curr_cand_final[-1]['src'] = [src_mod_names[m]]
        else:
            idx = curr_cands_txt.index(img['candidate']['text'])
            curr_cand_final[idx]['src'].append(src_mod_names[m])
        for c in img['candidatelist']:
            if c['text'] not in curr_cands_txt:
                curr_cands_txt.append(c['text'])
                curr_cand_final.append(c)
                curr_cand_final[-1]['src'] = [src_mod_names[m]]
            else:
                idx = curr_cands_txt.index(c['text'])
                curr_cand_final[idx]['src'].append(src_mod_names[m])
    imgblobs.append({'candidate':curr_cand_final[0],'candidatelist':curr_cand_final,'img_path':img['img_path']})


resCandsImgid = defaultdict(list)
icnt = 0
for img in resMulti['imgblobs']:
    imgid = int(img['img_path'].split('_')[-1].split('.')[0])
    for s in img['candidatelist']:
        resCandsImgid[imgid].append({'imgid':imgid,'raw':s['text'],'sentid':icnt,'tokens':s['text'].split(' ')})
        icnt+=1

for img in resNew['imgblobs']:
    imgid = int(img['img_path'].split('_')[-1].split('.')[0])
    resNewDict[imgid] = img['candidate']

for i,img in enumerate(resMulti['imgblobs']):
    imgid = int(img['img_path'].split('_')[-1].split('.')[0])
    resMulti['imgblobs'][i]['candidatelist'].append(resNewDict[imgid])

###################################### Compute Mert for each candidate against all 5 references##############################
import json
import time
from collections import defaultdict
import numpy as np
import cPickle as pickle
dataset = json.load(open('/triton/ics/project/imagedb/picsom/databases/COCO/download/annotations/captions_val2014.json','r'))
resMulti = json.load(open('example_images/result_struct_4AuxCmmePgoogSwapPposJJ_fullVal.json','r'))
resAllImgid = defaultdict(list)
for img in dataset['annotations']:
    resAllImgid[img['image_id']].append(img)
resCandsImgid = defaultdict(list)
icnt = 0
for img in resMulti['imgblobs']:
    imgid = int(img['img_path'].split('_')[-1].split('.')[0])
    for s in img['candidatelist']:
        resCandsImgid[imgid].append({'image_id':imgid,'caption':s['text'],'id':icnt})
        icnt+=1
from eval.mseval.pycocoevalcap.meteor.meteor import Meteor
from eval.mseval.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
tokenizer = PTBTokenizer()
resCandsImgid = tokenizer.tokenize(resCandsImgid)
resAllImgid = tokenizer.tokenize(resAllImgid)

scorer = Meteor()

lenDict = defaultdict(list)
for k in resCandsImgid:
   lenDict[len(resCandsImgid[k])].append(k)

maxlen = max(lenDict.keys())
print maxlen
candScoresImgid = defaultdict(list)
for i in xrange(maxlen):
    res ={}
    gts = {}
    for k in resAllImgid.keys():
        if i < len(resCandsImgid[k]):
            res[k] = [resCandsImgid[k][i]]
            gts[k] = resAllImgid[k]
    print 'Now in %d, Lengths %d'%(i, len(gts))
    t0 = time.time()
    score, scores = scorer.compute_score(gts, res)
    dt = time.time() - t0
    print 'Done %d in %.3fs, score = %.3f' %(i, dt, score)
    icnt = 0
    for si,k in enumerate(gts.keys()):
        candScoresImgid[k].append(scores[si])

    assert(len(scores) == si+1)

pickle.dump(candScoresImgid,open('candScrMeteor_4AuxCmmePgoogSwapPposJJ_fullVal.json','w'))
resDump = []
for k in resCandsImgid:
    b_idx = np.argmax(candScoresImgid[k])
    resDump.append({'image_id': k, 'caption':resCandsImgid[k][b_idx]})


#################################################################################################################
for img in candDb['imgblob']:
    candidatelist = []
    for i,c in enumerate(img['cands']):
        if set(img['src_mods'][i]) & allowed_mod_list:
            candidatelist.append({'text':c.rstrip(' ').lstrip(' '),'logprob': 0})
    resMertInp['imgblobs'].append({'candidatelist':candidatelist,'imgid':cocoIdtodbId[int(img['imgid'])]['imgid'],
                                'img_path':cocoIdtodbId[int(img['imgid'])]['path']})

for i,img in enumerate(resMertInp['imgblobs']):
  if len(img['candidatelist']) < maxlen:
      c_len_diff = maxlen - len(img['candidatelist'])
      for z in xrange(c_len_diff):
        resMertInp['imgblobs'][i]['candidatelist'].append(resMertInp['imgblobs'][i]['candidatelist'][-1])

for i,img in enumerate(resMulti['imgblobs']):
    resMulti['imgblobs'][i]['candidatelist'].append(resCand['imgblobs'][i]['candidate'])
    resMulti['imgblobs'][i]['candidatelist'].extend(resCand['imgblobs'][i]['candidatelist'])



#################################

for i,img in enumerate(resMultiFinal['imgblobs']):
    img['candidatelist'][0] = resMulti[0]['imgblobs'][i]['candidate']
    img['candidatelist'][1] = resMulti[1]['imgblobs'][i]['candidate']
    img['candidatelist'][2] = resMulti[2]['imgblobs'][i]['candidate']
    img['candidatelist'][3] = resMulti[3]['imgblobs'][i]['candidate']
    img['candidatelist'].append(resMulti[4]['imgblobs'][i]['candidate'])


########### Eval using coco toolkit ##########

resDump = []
for img in resMulti['imgblobs']:
    imgid = int(img['img_path'].split('_')[-1].split('.')[0])
    resDump.append({'image_id': imgid, 'caption':img['candidate']['text'].lstrip(' ').rstrip(' ')})

 json.dump(resDump,open('eval/mseval/results/captions_val2014_cNNEvalPicked_results.json','w'))



#################################
f = open('CandCommitteCocoMert.txt','w')
scrs = evalSc['logProb_feat']
cnt = 0
icnt = 0
for img in resOrig['images']:
    for s in img['sentences']:
        f.writelines(('%d ||| %s ||| %d '%(icnt,s['raw'],len(s['tokens']))) + ' '.join(map(str,scrs[:,cnt])) +'\n')
        cnt += 1
    icnt += 1


mod_names = {}
rootdir = '/ssdscratch/jormal/picsom/databases/COCO/objects/'
for r in os.walk(rootdir):
    if len(r[1]) == 0:
        for fl in r[2]:
            if 'eval' in fl:
               Cands =


all_references = {}
for img in dataset['images']:
    references = [' '.join(x['tokens']) for x in img['sentences']] # as list of lists of tokens
    all_references[img['cocoid']] = references

trn_refernces = [[] for q in xrange(5)]

for img in trnData['imgblob']:
    for q in xrange(5):
        trn_refernces[q].append(all_references[int(img['imgid'])][q])

for q in xrange(5):
   open('./nlpUtils/zmert_v1.50/zmert_ex_coco/referencefull.'+`q`, 'w').write('\n'.join(trn_refernces[q]))


nnStatsF = open('NNStats/Fast_SearchResult_MertAllModel.txt','r').read().splitlines()
nnStats = []
for ln in nnStatsF:
    if 'NN for tweet' == ln[:12]:
        nnStats.append(float(ln.split('= ')[1]))


import theano
from theano import config
import theano.tensor as tensor
import cPickle as pickle
from imagernn.data_provider import getDataProvider, prepare_data
params = {}
params['dataset'] = 'coco'
params['data_file'] = 'dataset.json'
params['feature_file'] = 'ConvAE_Test_e50_f32.hdf5'



Proj = np.random.rand(4096,out['image']['feat'].shape[0])
row_sum = Proj.sum(axis=1)
Proj = Proj/row_sum[:,np.newaxis]


i=0
for i in xrange(totImgs):
    cFeats[:,i] = Proj.dot(dp.features[i,:])
    if i %10 == 1:
        print('%d'%i)


############### Dump result Struct #############################


caps  =[]
for img in resOrig['imgblobs']:
    imgid = int(img['img_path'].rsplit('_')[-1].split('.')[0])
    caps.append({'image_id':imgid, 'caption':img['candidate']['text']})



############### Computing mutual Information #################

import numpy as np
import cPickle as pickle
import json
from operator import itemgetter

checkpoint = pickle.load(open('trainedModels/model_checkpoint_coco_gpu001_c_in14_o9_fc7_d_a_Auxo9_fc8_11.96.p','r'))
wix = checkpoint['wordtoix']
dataset = json.load(open('/triton/ics/project/imagedb/picsom/databases/COCO/download/annotations/instances_train2014.json','r'))
ixw = checkpoint['ixtoword']

from collections import defaultdict
from imagernn.data_provider import getDataProvider, prepare_data
params = {}
params['dataset'] = 'coco'
params['data_file'] = 'dataset.json'
dp = getDataProvider(params)

# Map category id to list of images it appears in
catIdImgs = defaultdict(set)
for ann in dataset['annotations']:
    catIdImgs[ann['category_id']].add(ann['image_id'])

catIdtoIx = {}
for i,cat in enumerate(catIdImgs.keys()):
    catIdtoIx[cat] = i

nTrnSamp = len(dataset['images'])
wordsIdList = defaultdict(set)

# Map word to list of images it appears in
for img in dp.split['train']:
    for sent in img['sentences']:
        for tk in sent['tokens']:
            if tk in wix:
                wordsIdList[tk].add(img['cocoid'])

# Prob of finding this word in a randomly sampled image
# Note that a sample is an image and not a sentence
wordProbs = np.zeros(len(wix))
for w in wordsIdList:
    wordProbs[wix[w]] = float(len(wordsIdList[w]))/nTrnSamp

# probablility is 1 for a fullstop
wordProbs[0] = 1

# Prob of finding a object category in a randomly sampled image
catProbs = np.zeros(len(catIdtoIx))
for i in catIdImgs:
    catProbs[catIdtoIx[i]] = float(len(catIdImgs[i]))/nTrnSamp

mi = np.zeros((len(catIdImgs),len(wix)))
jp = np.zeros((len(catIdImgs),len(wix)))
delt = np.zeros((len(catIdImgs),len(wix)))

totImgs = float(len(dp.split['train']))
eps = 1e-10

# unnormalized joint probability of finding a category and a word together in an image
for cid in catIdImgs:
    for tk in wordsIdList:
        jp[catIdtoIx[cid],wix[tk]] = float(len(wordsIdList[tk] & catIdImgs[cid])+eps)

sumd = nTrnSamp #np.sum(jp)

for cid in catIdImgs:
    for tk in wordsIdList:
        mind = min(wordProbs[wix[tk]], catProbs[catIdtoIx[cid]])
        delt[catIdtoIx[cid],wix[tk]] = (jp[catIdtoIx[cid],wix[tk]] /(jp[catIdtoIx[cid],wix[tk]] + 1)) * (mind*nTrnSamp/(mind*nTrnSamp+1))

jp = jp/sumd
jp[:,0] = 1.0 + eps


for cid in catIdImgs:
    for tk in wordsIdList:
        mi[catIdtoIx[cid],wix[tk]] = np.log((jp[catIdtoIx[cid],wix[tk]]) / (wordProbs[wix[tk]] * catProbs[catIdtoIx[cid]]))

mi = mi*delt/ -np.log(jp)
ixtocat = {}
for nm in dataset['categories']:
    ixtocat[catIdtoIx[nm['id']]] = nm['name']

#wVecMat = mi
#wVecMat = checkpoint['model']['Wemb'].T
wVecMat = np.concatenate((checkpoint['model']['Wemb'].T,checkpoint['model']['Wd']),axis=0)

normWords = np.sqrt(np.sum(wVecMat**2,axis = 0)[:,np.newaxis]) + 1e-20
wordsMutualSim = wVecMat.T.dot(wVecMat) / normWords.dot(normWords.T)

# For each word find the category it best matches to!!. Hence each word has only one matching category
catToWords = defaultdict(list)
for i in xrange(mi.shape[1]):
    cid = mi[:,i].argmax()
    catToWords[cid].append((i,mi[cid,i]))

k = 10
fid = open('./nlpExpts/results/wordToCatMap/cat2WordFinal.txt','w')
for cid in catToWords:
    fid.write('\n%s : '%(ixtocat[cid]))
    srtedK = sorted(catToWords[cid],key=itemgetter(1),reverse=True)[:k]
    widx = [Wc[0] for Wc in  srtedK]
    scrs = mi[cid,widx]
    newScrs = (scrs/scrs[0] + wordsMutualSim[widx,widx[0]])/ 2
    widx2 = np.argsort( newScrs)[::-1]
    for w in widx2:
        fid.write('%s (%.2f), '%(ixw[widx[w]],newScrs[w]))
fid.close()

# For each category find top 10 matching words, hence each word can map to many categories
fid = open('./nlpExpts/results/wordToCatMap/word2catFinal.txt','w')
for cid in ixtocat:
    fid.write('\n%s : '%(ixtocat[cid]))
    for i in mi[cid,:].argsort()[::-1][:k]:
        fid.write('%s (%.2f), '%(ixw[i],mi[cid,i]))
fid.close()

visWordsDict = defaultdict(set)

fid = open('./nlpExpts/results/wordToCatMap/word2cat_SimCombFinal.txt','w')
for cid in ixtocat:
    fid.write('\n%s : '%(ixtocat[cid]))
    widx = mi[cid,:].argsort()[::-1][:k]
    newScrs = (mi[cid,widx]/mi[cid,widx[0]] + wordsMutualSim[widx,widx[0]])/ 2
    widx2 = np.argsort(newScrs)[::-1]
    for i in widx2:
        if newScrs[i] > 0.70:
            visWordsDict[ixw[widx[i]]].add(ixtocat[cid])
        fid.write('%s (%.2f), '%(ixw[widx[i]],newScrs[i]))
fid.close()
############################################################################################################################################
fid = open('./nlpExpts/data/dbSentencesRaw.txt','w')
fid.write(' .\n'.join([re.sub('\\n+','. ',ann['caption'].lstrip(' .\n').rstrip(' .\n')) for ann in dbTrn['annotations']]) + ' .')
fid.close()

########################################### Visualizing Generated vocabulary ################################################################
from collections import defaultdict
import numpy as np
import json
import bokeh
from bokeh.io import output_file
from bokeh.plotting import figure, output_file, save, ColumnDataSource, VBox, HBox
from bokeh.models import HoverTool


#dataset = json.load(open('data/coco/dataset.json','r'))
subResultsT = json.load(open('eval/mseval/results/captions_val2014_CMME_results.json','r'))
subResultsV = json.load(open('eval/mseval/results/captions_val2014_posJJ210p74_results.json','r'))

nSampT = len(subResultsT)
nSampV = len(subResultsV)

genDictT = defaultdict(int)
for res in subResultsT:
    for w in res['caption'].split(' '):
        genDictT[w] += 1

genCntsT = np.zeros(len(genDictT))
ixwGenT = {}
for i,w in enumerate(genDictT):
    genCntsT[i] = genDictT[w]
    ixwGenT[i] = w


genDictV = defaultdict(int)
for res in subResultsV:
    for w in res['caption'].split(' '):
        genDictV[w] += 1

genCntsV = np.zeros(len(genDictV))
ixwGenV = {}
for i,w in enumerate(genDictV):
    genCntsV[i] = genDictV[w]
    ixwGenV[i] = w



nSamples = 0
trnDict = defaultdict(int)
for img in dataset['images']:
    if img['split'] == 'train':
        for s in img['sentences']:
            for w in s['tokens']:
                trnDict[w] += 1.0
            nSamples += 1

#nSampVRef = 0
#valDict = defaultdict(int)
#for img in dataset['images']:
#    if img['split'] != 'train':
#        for s in img['sentences']:
#            for w in s['tokens']:
#                valDict[w] += 1.0
#            nSampVRef += 1

#XXX HACK!!! DELETE IMMEDIATELY
nSampVRef = 0
valDict = defaultdict(int)
for res in subResultsT:
    for w in res['caption'].split(' '):
        valDict[w] += 1.0
    nSampVRef += 1

srtidx = np.argsort(genCntsT)[::-1]
genwordsSrtedT = [ixwGenT[i] for i in srtidx]
genCntsT = genCntsT[srtidx]/nSampT
trnCntsT = np.array([trnDict[w] for w in genwordsSrtedT]) / nSamples

srtidx = np.argsort(genCntsV)[::-1]
genwordsSrtedV = [ixwGenV[i] for i in srtidx]
genCntsV = genCntsV[srtidx]/nSampV
trnCntsV = np.array([trnDict[w] for w in genwordsSrtedV]) / nSamples
valCntsV = np.array([valDict[w] for w in genwordsSrtedV]) / nSampVRef

trnCntsAll = np.zeros(len(trnDict))
ixwTrn = {}
for i,w in enumerate(trnDict):
    trnCntsAll[i] = trnDict[w]
    ixwTrn[i] = w
srtidx = np.argsort(trnCntsAll)[::-1][:8900]
trnwordsSrted = [ixwTrn[i] for i in srtidx]
trnCntsAll = trnCntsAll[srtidx]/ nSamples
genCntsTall = np.zeros(trnCntsAll.shape[0]) + 1e-6
genCntsVall = np.zeros(trnCntsAll.shape[0]) + 1e-6
valCntsall = np.zeros(trnCntsAll.shape[0]) + 1e-6

for i,w in enumerate(trnwordsSrted):
    if w in genDictT:
        genCntsTall[i] = float(genDictT[w]) / nSampT
    if w in genDictV:
        genCntsVall[i] = float(genDictV[w]) / nSampV
    if w in valDict:
        valCntsall[i] = float(valDict[w]) / nSampVRef

filepath = 'generateWordsVisTest.html'
output_file(filepath)
TOOLS="pan,wheel_zoom,box_zoom,reset,hover"
source1 = ColumnDataSource(data=dict(x=range(genCntsT.shape[0]), y=np.log10(genCntsT),cnt = genCntsT*nSampT,lab = genwordsSrtedT))
p1 = figure(title="WC/sent(log10) in generted (TEST) vs Train", tools=TOOLS)
p1.circle(range(genCntsT.shape[0]),np.log10(genCntsT),source = source1,color="blue",legend = "Cnt in Generated")
source2 = ColumnDataSource(data=dict(x=range(genCntsT.shape[0]), y=np.log10(trnCntsT),cnt=trnCntsT*nSampT,lab = genwordsSrtedT))
p1.circle(range(genCntsT.shape[0]),np.log10(trnCntsT),source = source2,legend = "Cnt in Train",color="red")
hover1 = p1.select(dict(type=HoverTool))
hover1.tooltips = [("(x,cnt)","(@x,@cnt)"),("text","@lab")]

source3 = ColumnDataSource(data=dict(x=range(genCntsT.shape[0]), y=np.log10(genCntsT/trnCntsT),cnt = genCntsT/trnCntsT,lab = genwordsSrtedT))
p2 = figure(title="Ratio of WC/sent in log TEST vs Train", tools=TOOLS)
p2.square(range(genCntsT.shape[0]),np.log10(genCntsT/trnCntsT),source = source3,color="blue",legend = "Gen Test / Train")
p2.line(range(genCntsT.shape[0]),np.log10(genCntsT/trnCntsT),source = source3,color="blue",legend = "Gen Test / Train")
hover2 = p2.select(dict(type=HoverTool))
hover2.tooltips = [("(x,ratio)","(@x,@cnt)"),("text","@lab")]


source4 = ColumnDataSource(data=dict(x=range(trnCntsAll.shape[0]), y=np.log10(genCntsTall),cnt = genCntsTall*nSampT,lab = trnwordsSrted))
p3 = figure(title="WC/sent(log10) in generted (TEST) vs Train", tools=TOOLS,plot_width=1200)
p3.circle(range(genCntsTall.shape[0]),np.log10(genCntsTall),source = source4,color="blue",legend = "Cnt in Generated")
p3.line(range(genCntsTall.shape[0]),np.log10(genCntsTall),line_dash=[4, 4],source = source4,color="blue",legend = "Cnt in Generated")
source5 = ColumnDataSource(data=dict(x=range(genCntsTall.shape[0]), y=np.log10(trnCntsAll),cnt=trnCntsAll*nSamples,lab = trnwordsSrted))
p3.circle(range(genCntsTall.shape[0]),np.log10(trnCntsAll),source = source5,legend = "Cnt in Train",color="red")
hover3 = p3.select(dict(type=HoverTool))
hover3.tooltips = [("(x,cnt)","(@x,@cnt)"),("text","@lab")]

save(VBox(HBox(p1,p2),p3))


filepath = 'generateWordsVisValPosJJ.html'
output_file(filepath)
TOOLS="pan,wheel_zoom,box_zoom,reset,hover"
p1 = figure(title="WC/sent(log10) in generted (Val) vs Train", tools=TOOLS)
##
source1 = ColumnDataSource(data=dict(x=range(genCntsV.shape[0]), y=np.log10(genCntsV),cnt = genCntsV*nSampV,lab = genwordsSrtedV))
p1.circle(range(genCntsV.shape[0]),np.log10(genCntsV),source = source1,color="blue",legend = "Cnt in Generated")
##
source2 = ColumnDataSource(data=dict(x=range(genCntsV.shape[0]), y=np.log10(trnCntsV),cnt=trnCntsV*nSampV,lab = genwordsSrtedV))
p1.circle(range(genCntsV.shape[0]),np.log10(trnCntsV),source = source2,fill_alpha = 0.5,legend = "Cnt in Train",color="red")
##
source1V = ColumnDataSource(data=dict(x=range(valCntsV.shape[0]), y=np.log10(valCntsV),cnt=valCntsV*nSampV,lab = genwordsSrtedV))
p1.circle(range(valCntsV.shape[0]),np.log10(valCntsV),source = source1V,fill_alpha = 0.5,legend = "Cnt in Val Ref",color="black")
##
hover1 = p1.select(dict(type=HoverTool))
hover1.tooltips = [("(x,cnt)","(@x,@cnt)"),("text","@lab")]

trnCntsV = valCntsV
source3 = ColumnDataSource(data=dict(x=range(genCntsV.shape[0]), y=np.log10(genCntsV/trnCntsV),cnt = genCntsV/trnCntsV,lab = genwordsSrtedV))
p2 = figure(title="Ratio of WC/sent in log Val vs Train", tools=TOOLS)
p2.square(range(genCntsV.shape[0]),np.log10(genCntsV/trnCntsV),fill_alpha = 0.5,source = source3,color="blue",legend = "Gen Val / Train")
p2.line(range(genCntsV.shape[0]),np.log10(genCntsV/trnCntsV),line_dash=[4, 4],source = source3,color="blue",legend = "Gen Val / Train")

source3V = ColumnDataSource(data=dict(x=range(genCntsV.shape[0]), y=np.log10(valCntsV/trnCntsV),cnt = valCntsV/trnCntsV,lab = genwordsSrtedV))
#p2.square(range(genCntsV.shape[0]),np.log10(valCntsV/trnCntsV),fill_alpha = 0.5,source = source3V,color="red",legend = "Val Ref / Train")
#p2.line(range(genCntsV.shape[0]),np.log10(valCntsV/trnCntsV),line_dash=[4, 4],fill_alpha = 0.3,source = source3V,color="red",legend = "Val Ref / Train")
hover2 = p2.select(dict(type=HoverTool))
hover2.tooltips = [("(x,ratio)","(@x,@cnt)"),("text","@lab")]


p3 = figure(title="WC/sent(log10) in generted (Val) vs Train", tools=TOOLS,plot_width=1200)
##
source4 = ColumnDataSource(data=dict(x=range(trnCntsAll.shape[0]), y=np.log10(genCntsVall),cnt = genCntsVall*nSampV,lab = trnwordsSrted))
p3.circle(range(genCntsVall.shape[0]),np.log10(genCntsVall),source = source4,color="blue",legend = "Cnt in Generated")
p3.line(range(genCntsVall.shape[0]),np.log10(genCntsVall),line_dash=[4, 4],source = source4,color="blue",legend = "Cnt in Generated")
##
source5 = ColumnDataSource(data=dict(x=range(genCntsVall.shape[0]), y=np.log10(trnCntsAll),cnt=trnCntsAll*nSamples,lab = trnwordsSrted))
p3.circle(range(genCntsVall.shape[0]),np.log10(trnCntsAll),source = source5,fill_alpha = 0.5,legend = "Cnt in Train",color="red")
##
source5V = ColumnDataSource(data=dict(x=range(valCntsall.shape[0]), y=np.log10(valCntsall),cnt=valCntsall*nSampVRef,lab = trnwordsSrted))
p3.circle(range(genCntsVall.shape[0]),np.log10(valCntsall),source = source5V,fill_alpha = 0.3,legend = "Cnt in Val Reference",color="black")
##
hover3 = p3.select(dict(type=HoverTool))
hover3.tooltips = [("(x,cnt)","(@x,@cnt)"),("text","@lab")]

save(VBox(HBox(p1,p2),p3))




########################

srcWrds = [ixw[i] for i in ixw]
len(srcWrds)
genCntsTall = np.zeros(len(srcWrds))
for i,w in enumerate(srcWrds):
    if w in genDictT:
        genCntsTall[i] = float(genDictT[w])

trnCntsAll = np.zeros(len(srcWrds))
for i,w in enumerate(srcWrds):
    if w in trnDict:
        trnCntsAll[i] = float(trnDict[w])
trnCntsAll[0] = nSamples

params['fname_append'] = 'c_in14_o9_fc7_d_a_Aux3gr_2o_an012_10.51_valSet'

colormap = (np.log10(np.log10(trnCntsAll)) - np.min(np.log10(np.log10(trnCntsAll))))*126 + 1
colorsL = ["#%02x%02x%02x" % (2*t,128-t,250*g ) for  t,g in zip(np.floor(colormap),np.floor(genCntsTall>0))]
colors = ["blue","red"]
colorsL2 = [colors[tf] for tf in genCntsTall>0]
radiiGen = (np.log10(genCntsTall/nSampT+1e-6) + 7 )*2 + 3

filepath = 'scatter_Wemb_Callback%s.html' % (params['fname_append'] )
output_file(filepath)
TOOLS="pan,wheel_zoom,box_zoom,reset,hover"

radiiGenLcl = radiiGen.copy()

p1 = figure(title="Word embedding Matrix rows", tools=TOOLS,plot_width=1200,plot_height = 900)
source1 = ColumnDataSource(data=dict(x=z1[:,0], y=z1[:,1],cntOrig = trnCntsAll, cntGen = genCntsTall, r = radiiGenLcl, lab = srcWrds))
p1.circle('x', 'y',size = 'r',fill_alpha = (0.8 - 0.5*(genCntsTall > 0)),color = colorsL2, source=source1,line_color=None)
hover1 = p1.select(dict(type=HoverTool))
hover1.tooltips = [("(cT,cG)","(@cntOrig,@cntGen)"),("text","@lab")]

p2 = figure(title="Word decoding Matrix rows", tools=TOOLS,plot_width=1200,plot_height = 900)
source2 = ColumnDataSource(data=dict(x=z2[:,0], y=z2[:,1],cntOrig = trnCntsAll, cntGen = genCntsTall, r = radiiGenLcl,lab = srcWrds))
p2.circle('x', 'y',size = 'r',fill_alpha = (0.8 - 0.5*(genCntsTall > 0)),fill_color = colorsL,line_color=line_colors, source=source2)
hover2 = p2.select(dict(type=HoverTool))
hover2.tooltips = [("(cT,cG)","(@cntOrig,@cntGen)"),("text","@lab")]

callback = Callback(args=dict(source=source2), code="""
    var data = source.get('data');
    var f = cb_obj.get('value')
    r = data['r']
    x = data['x']
    y = data['y']
    wrds = data['lab']
    for (i = 0; i < wrds.length; i++) {
        if (f == wrds[i]){
            r[i] = 30
            break
        }
    }
    source.trigger('change');
""")
text = TextInput(title="Word", name='search', value='one',callback=callback)
save(VBox(text,p2,p1))

###########################################################################################
import json
from imagernn.data_provider import readFeaturesFromFile
from collections import defaultdict
import numpy as np
import cPickle as pickle


split_str = 'val'
detsCOCOVal = json.load(open('../py-faster-rcnn/output/featExtract/coco_2014_val/coco80Cls_vgg16_faster_rcnn_iter_290000/detections_val2014_results.json','r'))
cocoSalMap,_ = readFeaturesFromFile('data/coco/salMapFeat_50x50.mat')
catsCoco = json.load(open('/projects/databases/coco/download/annotations/instances_val2014.json','r'))

data = json.load(open('data/coco/dataset.json','r'))
cocoId2FeatId = {}
for img in data['images']:
    cocoId2FeatId[img['cocoid']] = img['imgid']
del data

catIdtoIdx = {}
for i,cat in enumerate(catsCoco['categories']):
    catIdtoIdx[cat['id']] = i

salDim = (50.0, 50.0)

allImgDets = defaultdict(list)
for det in detsCOCOVal:
    allImgDets[det['image_id']].append(det)

objSalScores = {}
CONF_THRESH = 0.3
for img in catsCoco['images']:
    objSalScores[int(img['id'])] = np.zeros(len(catIdtoIdx), np.float32)
    sal = cocoSalMap[:,cocoId2FeatId[img['id']]].reshape(*salDim).T
    for det in allImgDets[img['id']]:
        if det['score'] > CONF_THRESH:
            x = det['bbox'][0]/img['width']
            w = det['bbox'][2]/img['width']
            y = det['bbox'][1]/img['height']
            h = det['bbox'][3]/img['height']
            salScore = sal[np.floor(x*salDim[0]):np.ceil((x+w)*salDim[0]), np.floor(y*salDim[1]):np.ceil((y+h)*salDim[1])].sum()
            objSalScores[int(img['id'])][catIdtoIdx[det['category_id']]] += float(salScore*det['score'])

pickle.dump({'salScores':objSalScores,'categories':catsCoco['categories']}, open('data/coco/saliencyPerObjValSet.p','wb'))


###########################################################################################

import cPickle as pickle
from collections import defaultdict

salScoresData = pickle.load(open('data/coco/saliencyPerObjValSet.p','rb'))
objSalScores = salScoresData['salScores']
cat2MapData = open('nlpExpts/results/wordToCatMap/word2cat_SimCombFinal.txt','r').read().splitlines()

catNameTOId = {}
for i,cat in enumerate(salScoresData['categories']):
    catNameTOId[cat['name']] = i

word2Cat = defaultdict(list)
wordMaxSc = defaultdict(float)
for catL in cat2MapData:
    catN = catL.split(':')[0].rstrip()
    words = catL.split(':')[1].split(',')
    for w in words[:-1]:
        wStr = w.split('(')[0].lstrip().rstrip()
        if float(w.split('(')[1][:-1]) > wordMaxSc[wStr]:
            word2Cat[wStr] = [catNameTOId[catN]]
            wordMaxSc[wStr] = float(w.split('(')[1][:-1])

resD = json.load(open('example_images/merge_result_top6.json','r'))
for i,res in enumerate(resD['imgblobs']):
    resId = int(res['img_path'].split('_')[-1].split('.')[0])
    maxSc = 0.0
    for j,cand in enumerate(res['candidatelist']):
        scIdx = [wCi for w in cand['text'].split() if w in word2Cat for wCi in word2Cat[w]]
        scPen = [0.0001*wi for wi,w in enumerate(cand['text'].split()) if w in word2Cat for wCi in word2Cat[w]]
        resD['imgblobs'][i]['candidatelist'][j]['logprob'] = float(objSalScores[resId][scIdx].sum() + sum(scPen))
        if resD['imgblobs'][i]['candidatelist'][j]['logprob'] >= maxSc:
            maxSc = resD['imgblobs'][i]['candidatelist'][j]['logprob']
            resD['imgblobs'][i]['candidate'] = resD['imgblobs'][i]['candidatelist'][j]

json.dump(resD, open('example_images/salScore_reranked_top6Models.json','w'))

############################  Tracker plotting ###############################################################
import cPickle as pkl
import matplotlib.pyplot as plt
import numpy as np
cp = pkl.load(open('trainedModels/cocoModels/model_checkpoint_coco_rs-titanx_r-dep3WSc-frcnn80detP3+3SpatGaussScaleP6grRBFsun397-gaA3cA3_per9.74.p','r'))

vP = np.array([tp[1] for tp in cp['trackers']['val_perplex']])
vPit = np.array([tp[0] for tp in cp['trackers']['val_perplex']])
vScrs = np.array([[tp[1]['bleu_4'], tp[1]['cider'], tp[1]['meteor'], tp[1]['rouge']] for tp in cp['trackers']['trackScores']])
vScrsit = np.array([tp[0] for tp in cp['trackers']['trackScores']])

tP = np.array([tp[1] for tp in cp['trackers']['train_perplex']])
tPit = np.array([tp[0] for tp in cp['trackers']['train_perplex']])

fig, ax1 = plt.subplots()
ax1.plot(tPit, tP, 'b.-',vPit, vP, 'b--*',linewidth=2)
ax1.set_xlabel('iterations')
ax1.set_ylabel('perplex', color='b')
for tl in ax1.get_yticklabels():
  tl.set_color('b')
ax1.set_xscale('log')
ax2 = ax1.twinx()
ax1.grid()
ax1.set_yscale('log')
ax2.plot(vPit, vScrs[:,0],'r:o',vPit, vScrs[:,1],'r:s',vPit, vScrs[:,2],'r:+',vPit, vScrs[:,3],'r:x')
ax2.set_ylabel('Automatic Evaluation Metrics', color='r')
for tl in ax2.get_yticklabels():
  tl.set_color('r')

ax1.legend(["train perplexity", "val perplexity"], loc=locP1)
ax2.legend(["bleu4", "cider","meteor","rouge"], loc=locP2)

plt.title('Training and validation perplexity Vs evaluation metrics')
plt.show()

############################  Tracker plotting 2 ###############################################################
import cPickle as pkl
import matplotlib.pyplot as plt
import numpy as np
cp = pkl.load(open('trainedModels/cocoModels/model_checkpoint_coco_rs-titanx_r-dep4-frcnn80detP6grRBFsun397-gaA3cA3_per9.75.p','r'))
vPRes = np.array([tp[1] for tp in cp['trackers']['val_perplex']])
vPResit = np.array([tp[0] for tp in cp['trackers']['val_perplex']])
tPRes = np.array([tp[1] for tp in cp['trackers']['train_perplex']])
tPResit = np.array([tp[0] for tp in cp['trackers']['train_perplex']])

cp = pkl.load(open('trainedModels/cocoModels/model_checkpoint_coco_rs-titanx_r-dep4reg-frcnn80detP6grRBFsun397-gaA3cA3_per10.69.p','r'))
vPReg = np.array([tp[1] for tp in cp['trackers']['val_perplex']])
vPRegit = np.array([tp[0] for tp in cp['trackers']['val_perplex']])
tPReg = np.array([tp[1] for tp in cp['trackers']['train_perplex']])
tPRegit = np.array([tp[0] for tp in cp['trackers']['train_perplex']])

fig, ax1 = plt.subplots()
ax1.plot(tPResit, tPRes, 'b-',vPResit, vPRes, 'b--',markerfacecolor='none')
ax1.plot(tPRegit, tPReg, 'r-',vPRegit, vPReg, 'r--',markerfacecolor='none')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Perplexity')
ax1.set_xscale('log')
ax1.grid()
ax1.set_yscale('log')
ax1.set_xlim([588,10**5])
ax1.set_ylim([4,200])

ax1.legend(["Residual train perplexity", "Residual val perplexity","Regular train perplexity", "Regular val perplexity"], loc=0)

plt.title('Effect of residual connection on  perplexity evolution for 4-layer network')
plt.show()

############################ Error Analysis vid ###############################################################
import json
import os
import os.path as osp
import cv2
import numpy as np
import matplotlib.pyplot as plt

vidpath = '/projects/databases/msr-vtt/download/TrainValVideo/'
res = json.load(open('example_images/cnn_reranked_val2016_msr-vtt_r-ensemb-cnnTanh0p15-3MixModelsCorr.json','r'))
resById = {}

data =  json.load(open('data/msr-vtt/dataset.json','r'))
resEv = json.load(open('eval/mseval/msr-vtt/results/captions_val2016_r-ensemb-cnnTanh0p15-3MixModelsCorr_evalImgs.json','r'))
for img in res['imgblobs']:
    resById[img['img_path'].split('.')[0]] = img

for img in data['images']:
    if img['video_id'] in resById:
        resById[img['video_id']]['length'] = img['end time'] - img['start time']
        resById[img['video_id']]['category'] = img['category']

cats = np.array([resById[img]['category'] for img in resById])
for ev in resEv:
    resById[ev['video_id']]['scrs'] = ev

lens = np.zeros(len(resById))
scrsC = np.zeros(len(resById))
for i,img in enumerate(resById):
    lens[i] = resById[img]['length']
    scrsC[i] = resById[img]['scrs']['CIDEr']

cidBins = np.zeros(21)
cidErrs = np.zeros(21)

for j in np.arange(10,31):
    cidBins[j-10] = scrsC[(np.floor(lens+0.49)).astype('int8') == j].sum()/ ((np.floor(lens+0.49)) == j).sum()
    cidErrs[j-10] = np.std(scrsC[(np.floor(lens+0.49)).astype('int8') == j])/ np.sqrt(((np.floor(lens+0.49)) == j).sum())

for img in data['images']:
    if img['video_id'] in resById: resById[img['video_id']]['category'] = img['category']

catBins = np.zeros(20)
catErrs = np.zeros(20)
catNms = [data['categories'][str(i)].split('/')[0] for i in xrange(20)]
for i in np.arange(20):
   catBins[i] = scrsC[cats == i].sum()/ (cats == i).sum()
   catErrs[i] = np.std(scrsC[cats == i])/ np.sqrt((cats == i).sum())

fig, ax1 = plt.subplots()
ax1.bar(np.arange(20), catBins, yerr=catErrs, align='center',color='MediumSlateBlue', error_kw={'ecolor':'Tomato','linewidth':2})
ax1.set_xticks(np.arange(20))
ax1.set_xticklabels(catNms,rotation=45,horizontalalignment='right')
ax1.set_xlim([-2,22])
ax1.set_ylabel('Mean CIDEr Score',weight='semibold')
#ax1.set_xlabel('Video Categories',weight='semibold')
plt.title('Mean CIDEr score across categories of MSR-VTT validation set')
#plt.show()

fig2, ax2 = plt.subplots()
ax2.bar(np.arange(10,31), cidBins, yerr=cidErrs, align='center',color='MediumSlateBlue', error_kw={'ecolor':'Tomato','linewidth':2})
ax2.set_xticks(np.arange(10,31))
ax2.set_xlim([8,33])
ax2.set_ylabel('Mean CIDEr Score',weight='semibold')
plt.title('Mean CIDEr score Vs video lengths in MSR-VTT validation set')
ax2.set_xlabel('Video length in seconds',weight='semibold')
plt.show()

#-------------------- Visualize dense trajectories --------------------------
import scipy
import scipy.cluster.vq as vq
import numpy as np
import cv2

vid_fname = '/projects/databases/msr-vtt/download/TrainValVideo/video245.mp4'
cap = cv2.VideoCapture(vid_fname)
ret, img = cap.read()
#print img.shape

vid_frames = [img]
while(cap.isOpened() and ret==True):
  ret, img= cap.read()
  if ret:
    vid_frames.append(img)

filepath = 'video245.npz'
index = [[11,41,137,245,341],[40,136,244,340,436]]
idt = np.load(filepath)
idt = idt['X']
idt = np.reshape(idt, (-1,index[1][4]))

font = cv2.FONT_HERSHEY_SIMPLEX
fscale = 1.0
fsFeat = 0.7
fTC = 2
fTF = 1

sz = vid_frames[0].shape

for i in xrange(len(vid_frames)):
  cdt = idt[((idt[:,0] >= i) * ((idt[:,0]-idt[:,5]) <= i)),:]
  frm = vid_frames[i]


#-------------------- Visualize cost function and metrics --------------------------
import matplotlib.pyplot as plt
import numpy as np

fname = 'logs/advmodel_checkpoint_coco_wks-12-46_r-reg-vgg-1samp-crossent-lstmeval-cosine-pretrainBOTH-zerohid_log.npz'
ld = 5e-4
lg = 1e-6
lenRefMean = 10.464429841414992
costs = np.load(fname)

fig, ax1 = plt.subplots()
ax1.plot(np.arange(len(costs['eval_cost'])),costs['eval_cost'],'b--', np.arange(len(costs['gen_cost'])),costs['gen_cost'], 'k--')

#metric_types = ['mean', 'max','min']
metric_types = ['mean'] #,'mean'] #, 'max','min']
mets  = np.array([[tsc[1]['meteor'+'_'+mt]   for mt in metric_types] for tsc in costs['tracksc']])
cid   = np.array([[tsc[1]['cider'+'_'+mt]    for mt in metric_types] for tsc in costs['tracksc']])
lens  = np.array([[tsc[1]['len'+'_'+mt]      for mt in metric_types] for tsc in costs['tracksc']])
div_1 = np.array([[tsc[1]['lcldiv_1'+'_'+mt] for mt in metric_types] for tsc in costs['tracksc']])
div_2 = np.array([[tsc[1]['lcldiv_2'+'_'+mt] for mt in metric_types] for tsc in costs['tracksc']])
#spice = np.array([[tsc[1]['spice'+'_'+mt]   for mt in metric_types] for tsc in costs['tracksc']])
its = np.array([tsc[0]  for tsc in costs['tracksc']])

ax1.set_xlabel('iterations')
ax1.set_ylabel('costs', color='b')
for tl in ax1.get_yticklabels():
  tl.set_color('b')
#ax1.set_xscale('log')
ax2 = ax1.twinx()
ax1.grid()
#ax1.set_yscale('log')
ax2.plot(its, mets,'r:o', its, cid,'r:+', its, div_1,'r:^', its, div_2,'r:*',linewidth=2) #its, lens/lenRefMean,'r:s',
ax2.set_ylabel('Evaluation Metrics', color='r')
for tl in ax2.get_yticklabels():
  tl.set_color('r')

ax1.legend(('eval_cost', 'gen_cost'),loc=2)
ax2.legend(["rev_meteor"]*len(metric_types)+["rev_cider"]*len(metric_types) +["img_div1", "imgdiv_2"],loc=1)# ["lens"]*len(metric_types)+

plt.title('Evaluator vs generator cost - Gumbel, ld: %.1e lg: %.1e'%(ld,lg))
plt.show()



fig, ax1 = plt.subplots()
ax1.set_xlabel('softmax smooth factor')
for tl in ax1.get_yticklabels():
  tl.set_color('b')
#ax1.set_xscale('log')
#ax2 = ax1.twinx()
ax1.grid()
#ax1.set_yscale('log')
ax1.plot(its, mets,'r:o',its, cid,'r:+', its, div_1,'k:^', its, div_2,'k:*',linewidth=1)
#ax1.plot(its, mets,'r:o', its, cid,'g:+', its, spice,'b:+',  its, lens/lenRefMean,'b:+', its, div_1,'k:^', its, div_2,'k:*',linewidth=1)
#ax1.legend(["met","cider","spice","len", "div1", "div2"],loc=1)
ax1.set_ylabel('Evaluation Metrics')
#for tl in ax2.get_yticklabels():
#  tl.set_color('r')

ax1.plot(its, mets,'r-o', its, cid,'r-+',its, div_1,'k-^', its, div_2,'k-*',linewidth=1)
ax1.plot(its, mets,'r-.o',its, div_1,'k-.^', its, div_2,'k-.*',linewidth=1)
ax1.legend(["ref_met","ref_cider", "ref_div1", "ref_div2","met","cider", "div1", "div2"],loc=1)


plt.title('Mean Metrics with varying softmax scaling')

plt.title('Mean Metrics with varying gumbel temperature')
plt.show()



maxM  = np.array([tsc[1]['max']   for tsc in costs['tracksc']])
meanM = np.array([tsc[1]['mean']  for tsc in costs['tracksc']])
minM  = np.array([tsc[1]['min']   for tsc in costs['tracksc']])
varM  = np.array([tsc[1]['var']   for tsc in costs['tracksc']])
mVec  = np.array([tsc[1]['m_vec'] for tsc in costs['tracksc']])
its = np.array([tsc[0]  for tsc in costs['tracksc']])

maxM_sft  = np.array([tsc[1]['max']   for tsc in costs['tracksc']])
meanM_sft = np.array([tsc[1]['mean']  for tsc in costs['tracksc']])
minM_sft  = np.array([tsc[1]['min']   for tsc in costs['tracksc']])
varM_sft  = np.array([tsc[1]['var']   for tsc in costs['tracksc']])
mVec_sft  = np.array([tsc[1]['m_vec'] for tsc in costs['tracksc']])
its_sft = np.array([tsc[0]  for tsc in costs['tracksc']])



fig, ax1 = plt.subplots()
ax1.grid()
#ax1.set_yscale('log')
its = np.arange(prog_max_meteor_base_beamsearch.shape[0])
ax1.plot(its, prog_max_meteor_base_beamsearch,its, prog_max_meteor_base_sf1, its, prog_max_meteor_base_sf2, its, prog_max_meteor, its, prog_max_meteor_sf3)
ax1.set_ylabel('Max Meteor score')
ax1.set_xlabel('Number of sentences')
ax1.legend(["baseline-beamsearch","baseline-sf1","baseline-sf2", "advers-sf2", "advers-sf3"],loc=2)

plt.show()



# -----------------------------------------------------------------------------------------------------
idx = 0
n_b = 1
n_images = allMetsBase.shape[-1]

prog_max_oracmeteor_base = np.zeros((allMetsBase.shape[2]))
meteorF = allMetsBase[0,:,:,:].max(axis=0)
for i in xrange(meteorF.shape[0]):
    prog_max_oracmeteor_base[i] = meteorF[np.argsort(meteorF[:i+1,:],axis=0)[-n_b:,:],np.arange(n_images)].mean(axis=0).mean(axis=-1)

prog_max_oracmeteor_adv = np.zeros((allMetsAdv.shape[2]))
meteorF = allMetsAdv[0,:,:,:].max(axis=0)
for i in xrange(meteorF.shape[0]):
    prog_max_oracmeteor_adv[i] = meteorF[np.argsort(meteorF[:i+1,:],axis=0)[-n_b:,:],np.arange(n_images)].mean(axis=0).mean(axis=-1)

prog_max_oracmeteor_advbeam = np.zeros((allMetsAdvBeam['met'].shape[2]))
meteorF = allMetsAdvBeam['met'][0,:,:,:].max(axis=0)
for i in xrange(meteorF.shape[0]):
    prog_max_oracmeteor_advbeam[i] = meteorF[np.argsort(meteorF[:i+1,:],axis=0)[-n_b:,:],np.arange(n_images)].mean(axis=0).mean(axis=-1)


prog_max_oracmeteor_beam = np.zeros((allMetsBeam.shape[2]))
meteorF = allMetsBeam[0,:,:,:].max(axis=0)
for i in xrange(meteorF.shape[0]):
    prog_max_oracmeteor_beam[i] = meteorF[np.argsort(meteorF[:i+1,:],axis=0)[-n_b:,:],np.arange(n_images)].mean(axis=0).mean(axis=-1)

plt.plot(np.arange(prog_max_oracmeteor_beam.shape[0]), prog_max_oracmeteor_beam, np.arange(prog_max_oracmeteor_base.shape[0]), prog_max_oracmeteor_base, np.arange(prog_max_oracmeteor_advbeam.shape[0]), prog_max_oracmeteor_advbeam, np.arange(prog_max_oracmeteor_adv.shape[0]), prog_max_oracmeteor_adv)
plt.xlabel('number of samples');plt.ylabel('meteor');plt.title('Oracle-Meteor (max %d of n)'%(n_b));plt.legend(['base-beam','base-samp','adv-beam', 'adv'],loc=2); plt.show()



# -----------------------------------------------------------------------------------------------------
idx = 0
n_b = 5
indivSpice_base = spice_all_base
indivSpice_beam = spice_all_basebeam
indivSpice_adv = spice_all_adv
indivSpice_advbeam= spice_all_advbeam

n_images = indivSpice_base['prec'].shape[1]
names = ['Overall', 'Color', 'Attribute', 'Object', 'Relation','Cardinality', 'Size']
prog_max_oracspice_base = np.zeros((indivSpice_base['prec'].shape[0]))
spiceF = np.nan_to_num(2*indivSpice_base['prec'][:,:,idx]*indivSpice_base['rec'][:,:,idx]/(indivSpice_base['prec'][:,:,idx]+indivSpice_base['rec'][:,:,idx]))
for i in xrange(indivSpice_base['rec'].shape[0]):
    prog_max_oracspice_base[i] = spiceF[np.argsort(spiceF[:i+1,:],axis=0)[-n_b:,:],np.arange(n_images)].mean(axis=0).mean(axis=-1)


prog_max_oracspice_adv = np.zeros((indivSpice_adv['prec'].shape[0]))
spiceF = np.nan_to_num(2*indivSpice_adv['prec'][:,:,idx]*indivSpice_adv['rec'][:,:,idx]/(indivSpice_adv['prec'][:,:,idx]+indivSpice_adv['rec'][:,:,idx]))
for i in xrange(indivSpice_adv['prec'].shape[0]):
    prog_max_oracspice_adv[i] = spiceF[np.argsort(spiceF[:i+1,:],axis=0)[-n_b:,:],np.arange(n_images)].mean(axis=0).mean(axis=-1)

prog_max_oracspice_advbeam= np.zeros((indivSpice_advbeam['prec'].shape[0]))
spiceF = np.nan_to_num(2*indivSpice_advbeam['prec'][:,:,idx]*indivSpice_advbeam['rec'][:,:,idx]/(indivSpice_advbeam['prec'][:,:,idx]+indivSpice_advbeam['rec'][:,:,idx]))
for i in xrange(indivSpice_advbeam['prec'].shape[0]):
    prog_max_oracspice_advbeam[i] = spiceF[np.argsort(spiceF[:i+1,:],axis=0)[-n_b:,:],np.arange(n_images)].mean(axis=0).mean(axis=-1)

prog_max_oracspice_beam = np.zeros((indivSpice_beam['prec'].shape[0]))
spiceF = np.nan_to_num(2*indivSpice_beam['prec'][:,:,idx]*indivSpice_beam['rec'][:,:,idx]/(indivSpice_beam['prec'][:,:,idx]+indivSpice_beam['rec'][:,:,idx]))
for i in xrange(indivSpice_beam['prec'].shape[0]):
    prog_max_oracspice_beam[i] = spiceF[np.argsort(spiceF[:i+1,:],axis=0)[-n_b:,:],np.arange(n_images)].mean(axis=0).mean(axis=-1)

plt.plot(np.arange(prog_max_oracspice_beam.shape[0]), prog_max_oracspice_beam, np.arange(prog_max_oracspice_base.shape[0]), prog_max_oracspice_base, np.arange(prog_max_oracspice_advbeam.shape[0]), prog_max_oracspice_advbeam, np.arange(prog_max_oracspice_adv.shape[0]), prog_max_oracspice_adv)
plt.xlabel('number of samples');plt.ylabel('Spice F-score');plt.title('Oracle-Spice %s F-score (max %d of n)'%(names[idx], n_b));plt.legend(['base-beam','base-samp', 'adv-beam', 'adv-samp'],loc=2); plt.show()



prog_max_oracmeteor_base = np.zeros((allMetsBase.shape[2]))
for i in xrange(allMetsBase.shape[2]):
    prog_max_oracmeteor_base[i] = allMetsBase[0,:,:i+1,:].max(axis=0).max(axis=0).mean(axis=-1)

prog_max_oracmeteor_beam= np.zeros((allMetsBeam.shape[2]))
for i in xrange(allMetsBase.shape[2]):
    prog_max_oracmeteor_beam[i] = allMetsBeam[0,:,:i+1,:].max(axis=0).max(axis=0).mean(axis=-1)

prog_max_oracmeteor_adv= np.zeros((allMetsAdv.shape[2]))
for i in xrange(allMetsBase.shape[2]):
    prog_max_oracmeteor_adv[i] = allMetsAdv[0,:,:i+1,:].max(axis=0).max(axis=0).mean(axis=-1)


plt.plot(np.arange(prog_max_oracmeteor_beam.shape[0]), prog_max_oracmeteor_beam, np.arange(prog_max_oracmeteor_base.shape[0]), prog_max_oracmeteor_base, np.arange(prog_max_oracmeteor_adv.shape[0]), prog_max_oracmeteor_adv)
plt.xlabel('number of samples');plt.ylabel('Meteor score');plt.title('Oracle-Meteor (max 1 of n)');plt.legend(['base-beam','base-samp', 'adv'],loc=2); plt.show()




# Do n-unique assignments

metScores = allMetsBase[0,:,:,:]; name = 'test-base-sampling-5'
n_cands = 5

n_refs = metScores.shape[0]
best_p = np.zeros((metScores.shape[-1], n_refs))
max_sc = np.zeros((n_cands,metScores.shape[-1],n_refs))
max_idces = np.zeros((n_cands,metScores.shape[-1],n_refs),dtype=np.int32)
n_imgs = metScores.shape[-1]
n_combs = n_refs ** n_refs

for j in np.arange(4,n_cands):
    idces = np.argsort(metScores[:,:j+1,:],axis=1)
    all_idces = np.array([[np.array(np.meshgrid(*idces[:,-n_refs:,i])).reshape(n_refs,-1).T] for i in xrange(n_imgs)])
    bool_mask = np.array([len(set(x)) == n_refs for x in all_idces.reshape([-1,n_refs])]).reshape((metScores.shape[-1], n_combs))
    max_idces[j,:,:] = [all_idces[i,0,(metScores[np.tile(np.arange(n_refs),n_combs),all_idces[i,0,:,:].flatten(),i].reshape(-1,n_refs)*bool_mask[i,:][:,None]).sum(axis=1).argmax(),:] for i in xrange(n_imgs)]
    max_sc[j,:,:] = [metScores[np.arange(n_refs),max_idces[j,i,:],i] for i in xrange(n_imgs)]
    print j
    np.savez('scorelogs/meteor_non_overalpping_%s.npz'%(name), max_sc = max_sc, max_idces = max_idces)


##--------------------------------------------------------------------------------------------------------------


metSc = allMetsBeamSmall['met'][0,:,:,:]
for i,imgid in enumerate(capsById):
    metSccurr = metSc[:,:,metById[imgid]]
    spiceCurr = spiceF[:,spiceById[imgid]]
    res['imgblobs'][resById[imgid]]['max-met'] = metSccurr.max(axis=0).max(axis=0)
    res['imgblobs'][resById[imgid]]['mean-met'] = metSccurr.max(axis=0).mean(axis=0)
    res['imgblobs'][resById[imgid]]['max-spice'] = spiceCurr.max(axis=0)
    res['imgblobs'][resById[imgid]]['mean-spice'] = spiceCurr.mean(axis=0)
    res['imgblobs'][resById[imgid]]['candidate']['spice'] = spiceCurr[0]
    res['imgblobs'][resById[imgid]]['candidate']['meteor'] = metSccurr.max(axis=0)[0]
    for j in xrange(len(res['imgblobs'][resById[imgid]]['candidatelist'])):
        res['imgblobs'][resById[imgid]]['candidatelist'][j]['spice'] = spiceCurr[j+1]
        res['imgblobs'][resById[imgid]]['candidatelist'][j]['meteor'] = metSccurr.max(axis=0)[j+1]


##--------------------------------------------------------------------------------------------------------------

win = 20;
fig, ax1 = plt.subplots();
ax1.grid();
x_data = np.median(rolling_window(tricountWithId[tricSrt,1],win),axis=-1)
y_data_base = np.median(rolling_window(metBase[[metByIdBase[int(imgid)] for imgid in tricountWithId[tricSrt,0]]], win),axis=-1)
y_data_adv = np.median(rolling_window(metAdv[[metByIdAdv[int(imgid)] for imgid in tricountWithId[tricSrt,0]]],win), axis=-1)
ax1.plot(x_data, y_data_base,'b*', x_data, y_data_adv,'r.')

win = 1;
filter_func = np.median
x_srt_idx = tricSrt_ps
x_scores = tricountWithId_ps_scores
x_ids = tricountWithId_ps_ids

base_data = logProbBase['pplx'] #metBase
base_idtoidx = lpBasKtoId #metByIdbaseB_ps

adv_data = logProbAdv['pplx'] #metAdv
adv_idtoidx = lpAdvKtoId #metByIdAdvB_ps

fig, ax1 = plt.subplots();
ax1.grid();
x_data = np.median(rolling_window(x_scores[x_srt_idx],win),axis=-1)
y_data_base = filter_func(rolling_window(base_data[[base_idtoidx[imgid] for imgid in x_ids[x_srt_idx]]], win),axis=-1)
y_data_adv = filter_func(rolling_window(adv_data[[adv_idtoidx[imgid] for imgid in x_ids[x_srt_idx]]],win), axis=-1)
ax1.plot(x_data, y_data_base,'b*', x_data, y_data_adv,'r.')
plt.xlabel(' in individual reference sentences')
plt.xlabel('Corresponding counts in individual reference sentences')
plt.show()

win =5
filter_func = np.mean
x_srt_idx = tricSrt_ps
x_scores = tricountWithId_ps_scores
x_ids = tricountWithId_ps_ids

base_data = logProbBase['pplx'] #metBase
base_idtoidx = lpBasKtoId #metByIdbaseB_ps

adv_data = logProbAdv['pplx'] #metAdv
adv_idtoidx = lpAdvKtoId #metByIdAdvB_ps

fig, ax1 = plt.subplots();
ax1.grid();
x_data = filter_func(x_scores[x_srt_idx].reshape((-1,win)),axis=-1)
y_data_base = filter_func(base_data[[base_idtoidx[imgid] for imgid in x_ids[x_srt_idx]]].reshape((-1,win)),axis=-1)
y_data_adv = filter_func(adv_data[[adv_idtoidx[imgid] for imgid in x_ids[x_srt_idx]]].reshape((-1,win)), axis=-1)
ax1.plot(x_data, y_data_base,'b*', x_data, y_data_adv,'r.'); ax1.set_xscale('log'); ax1.set_yscale('log');
plt.ylabel('Perplexity of the model')
plt.xlabel('Corresponding counts in individual reference sentences')
plt.title('Perplexity vs trigram count in individual reference captions'); plt.legend(['baseline', 'adversarial']);
plt.show()


win =5
x_srt_idx = tricSrt_ps
x_scores = tricountWithId_ps_scores
x_ids = tricountWithId_ps_ids

base_data = logProbBase['pplx'] #metBase
base_idtoidx = lpBasKtoId #metByIdbaseB_ps

adv_data = logProbAdv['pplx'] #metAdv
adv_idtoidx = lpAdvKtoId #metByIdAdvB_ps

fig, ax1 = plt.subplots();
ax1.grid();
x_data = filter_func(x_scores[x_srt_idx].reshape((-1,win)),axis=-1)
y_data_base = filter_func(base_data[[base_idtoidx[imgid] for imgid in x_ids[x_srt_idx]]].reshape((-1,win)),axis=-1)
y_data_adv = filter_func(adv_data[[adv_idtoidx[imgid] for imgid in x_ids[x_srt_idx]]].reshape((-1,win)), axis=-1)
ax1.plot(x_data, y_data_base,'b*', x_data, y_data_adv,'r.'); ax1.set_xscale('log'); ax1.set_yscale('log');
plt.ylabel('Perplexity of the model')
plt.xlabel('Corresponding counts in individual reference sentences')
plt.title('Perplexity vs trigram count in individual reference captions'); plt.legend(['baseline', 'adversarial']);
plt.show()

win = 1000

a = stats.binned_statistic(x_scores, base_data[[base_idtoidx[imgid] for imgid in x_ids]],statistic='mean',bins=win)
b = stats.binned_statistic(x_scores, adv_data[[adv_idtoidx[imgid] for imgid in x_ids]],statistic='mean',bins=win)
plt.semilogx((a[1][:-1] + a[1][1:]) / 2.,a[0], (b[1][:-1] + b[1][1:]) / 2.,b[0]); plt.xlabel('Trigram count of ref captions in training set');plt.ylabel('Perplexity assigned by the models');plt.title('Perplexity vs trigram count of references'); plt.legend(['baseline', 'adversarial']);plt.show()



win =5
x_srt_idx = tricSrt_ps
x_scores = tricountWithId_ps_scores
x_ids = tricountWithId_ps_ids

base_data =  metBase #nonovmet_base['max_sc'][-1,:,:].flatten() #metBase #logProbBase['pplx']
base_idtoidx = metByIdbaseB_ps #lpBasKtoId

adv_data =  metAdv #nonovmet_adv['max_sc'][-1,:,:].flatten() #metAdv #logProbAdv['pplx']
adv_idtoidx = metByIdAdvB_ps #lpAdvKtoId

win = 1000 #x_scores.shape[0]#1000
fig, ax1 = plt.subplots();
ax1.grid();
a = stats.binned_statistic(x_scores, base_data[[base_idtoidx[imgid] for imgid in x_ids]],statistic='sum',bins=win)
b = stats.binned_statistic(x_scores, adv_data[[adv_idtoidx[imgid] for imgid in x_ids]],statistic='sum',bins=win)
plt.semilogx((a[1][:-1] + a[1][1:]) / 2.,a[0], (b[1][:-1] + b[1][1:]) / 2.,b[0]);

#x_data = x_scores[x_srt_idx]
#y_data_base = base_data[[base_idtoidx[imgid] for imgid in x_ids[x_srt_idx]]]
#y_data_adv = adv_data[[adv_idtoidx[imgid] for imgid in x_ids[x_srt_idx]]]
#fig, ax1 = plt.subplots();
#ax1.grid();
#ax1.plot(x_data, y_data_base,'b*', x_data, y_data_adv,'r.'); ax1.set_xscale('log');# ax1.set_yscale('log');

plt.xlabel('Trigram count of ref captions in training set');
plt.ylabel('Reverse meteor score');
plt.title('Reverse Meteor vs trigram count of references');
plt.legend(['baseline', 'adversarial']);plt.show()


#------------------------------------------------------------------------------------------------------------------------------
from imagernn.utils import find_ngrams

tV = testBiGramCounts
bV = baseBiGramCounts #baseVocab
aV = advBiGramCounts #advVocab

tV = testTriGramCounts
#bV2 = baseVocabDiv #baseVocab
bV = baseTriGramCounts #baseVocab
aV = advTriGramCounts #advVocab

tV = testvocab
bV2 = baseVocabDiv #baseVocab
bV = baseVocab
aV = advVocab


clip=0.;
alpha = 0.2;
n_words = None;
x_data = np.array([w[1] for w in tV.most_common(n_words)],dtype=np.float32)
win = np.unique(x_data) #x_scores.shape[0]#1000
y_base = np.array([bV.get(w[0],clip) for w in tV.most_common(n_words)],dtype=np.float32)
y_adv = np.array([aV.get(w[0],clip) for w in tV.most_common(n_words)],dtype=np.float32)
a = stats.binned_statistic(x_data, y_base/x_data, statistic='median',bins=win)
b = stats.binned_statistic(x_data, y_adv/x_data, statistic='median',bins=win)
plt.semilogx((a[1][:-1] + a[1][1:]) / 2.,np.nan_to_num(a[0]), 'bo', (b[1][:-1] + b[1][1:]) / 2.,np.nan_to_num(b[0]),'ro',(a[1][:-1] + a[1][1:]) / 2., np.ones(a[1].shape[0]-1), 'g-',alpha=0.3, basex=10);
plt.xlabel('Word count on test set'); plt.ylabel('Word count ratio of generated and reference captions on the test set');
plt.title('Median word count ratio on the test set plotted against word fequency');
plt.legend(['baseline', 'adversarial'],loc=2);
plt.show()



tV = testvocab
bV2 = baseVocabDiv #baseVocab
bV = baseVocabDiv
aV = advVocab

clip=0.;
alpha = 0.2;
n_words = None;
x_data = np.array([w[1] for w in tV.most_common(n_words)],dtype=np.float32)
y_base = np.array([bV.get(w[0],clip) for w in tV.most_common(n_words)],dtype=np.float32)
y_base2 = np.array([bV2.get(w[0],clip) for w in tV.most_common(n_words)],dtype=np.float32)
y_adv = np.array([aV.get(w[0],clip) for w in tV.most_common(n_words)],dtype=np.float32)
#plt.semilogx(x_data, y_base/x_data,'bo',alpha=alpha);
plt.semilogx(x_data, y_base2/x_data, 'go', alpha=alpha*2);
plt.semilogx(x_data, y_adv/x_data,'r.',alpha=alpha*2);
plt.semilogx(x_data, np.ones(x_data.shape[0]),'g-');
plt.xlabel('word index'); plt.ylabel('Word count ratio of generated and reference captions on the test set');
plt.legend(['baseline', 'baseline_diverse', 'adversarial'],loc=2);
plt.title('Word count ratio on the test set for the 1000 most common words');
plt.show()

### Scatter plots ####
alpha = 0.2;
base = 10;
clip = 0.1;
x_data = np.array([w[1] for w in tV.most_common(n_words)],dtype=np.float32)
y_base = np.array([bV.get(w[0],clip) for w in tV.most_common(n_words)],dtype=np.float32)
y_adv = np.array([aV.get(w[0],clip) for w in tV.most_common(n_words)],dtype=np.float32)

plt.loglog(x_data, y_base, 'bo', alpha=alpha, nonposy='clip', basex=base,basey=base);
plt.loglog(x_data, y_adv , 'r.', alpha=2*alpha, nonposy='clip', basex=base,basey=base);
plt.loglog(x_data, x_data,'g-');
plt.xlabel('word count on test set'); plt.ylabel('Word count generated by the models');
plt.legend(['baseline', 'adversarial'],loc=2);
plt.title('Scatter plot of word counts in reference captiosn vs the generated captions'); plt.show()


skip = 1
fig, ax1 = plt.subplots()
ax1.set_xlabel('temperature')
ax1.set_ylabel('diversity metrics', color='b')
ax1.plot(temps[skip:], ldiv1_base[skip:],'b:o', temps[skip:], ldiv2_base[skip:],'b:*', temps[skip:], ldiv1_adv[skip:],'r:o' , temps[skip:], ldiv2_adv[skip:],'r:*',alpha=0.6);
ax1.legend(('baseline_div_1', 'baseline_div_2', 'adversarial_div_1', 'adversarial_div_2'),loc=2)
ax1.grid()
ax2 = ax1.twinx();
ax2.plot(temps[skip:], vocab_base[skip:],'b:^', temps[skip:], vocab_adv[skip:],'r:^',alpha=0.6);
ax2.set_ylabel('Vocabulary size', color='r');
ax2.legend(('baseline_vocab', 'adversarial_vocab'),loc=1)
plt.title('Diversity statistcs with varying temperature')
plt.show()



#Vocabulary count ratios

tV = trainvocab
bV2 = baseVocab #baseVocab
bV = testvocab
aV = advVocab
clip=0.;
alpha = 0.2;
n_words = None;
x_data = np.array([w[1] for w in tV.most_common(n_words)],dtype=np.float32)
x_data = x_data/x_data.sum()
win = 1000 #x_scores.shape[0]#1000
y_test = np.array([bV.get(w[0],clip) for w in tV.most_common(n_words)],dtype=np.float32)
y_test = y_test/y_test.sum()
y_adv = np.array([aV.get(w[0],clip) for w in tV.most_common(n_words)],dtype=np.float32)
y_adv = y_adv/y_adv.sum()
y_base = np.array([bV2.get(w[0],clip) for w in tV.most_common(n_words)],dtype=np.float32)
y_base = y_base/y_base.sum()


n_train = len([img for img in data['images'] if img['split']=='train'])*5.0
n_test = len([img for img in data['images'] if img['split']=='test'])*5.0

tV = trainvocab
bV = baseVocab #baseVocab
teV = testvocab
aV = advVocab
n_words = None

alpha =0.3
x_data = np.array([w[1] for w in tV.most_common(n_words)],dtype=np.float32)*n_test/n_train
y_test = np.array([teV.get(w[0],clip) for w in tV.most_common(n_words)],dtype=np.float32)
y_base = np.array([bV.get(w[0],clip) for w in tV.most_common(n_words)],dtype=np.float32)
y_adv = np.array([aV.get(w[0],clip) for w in tV.most_common(n_words)],dtype=np.float32)

#Scale train data to match test data


plt.loglog(x_data, y_test, 'g.', alpha=2*alpha, nonposy='clip', basex=base,basey=base);
plt.loglog(x_data, y_base, 'bo', alpha=alpha, nonposy='clip', basex=base,basey=base);
plt.loglog(x_data, y_adv , 'r.', alpha=2*alpha, nonposy='clip', basex=base,basey=base);
plt.loglog(x_data, x_data,'k-',alpha=alpha);
plt.xlabel('Word count on the training set reference captions'); plt.ylabel('Word count generated by the models');
plt.legend(['test-references','baseline', 'adversarial'],loc=2);
plt.title('Scatter plot of word counts in training captions vs the generated captions'); plt.show()


base_counts = np.array(bV.values())
y_base = np.array([(base_counts>=bc).sum() for bc in np.unique(base_counts)])
adv_counts = np.array(aV.values())
y_adv = np.array([(adv_counts>=bc).sum() for bc in np.unique(adv_counts)])
test_counts = np.array(teV.values())
y_test = np.array([(test_counts>=bc).sum() for bc in np.unique(test_counts)])


plt.loglog(np.unique(test_counts), y_test, 'g.', alpha=2*alpha)#, nonposy='clip', basex=base,basey=base);
plt.loglog(np.unique(base_counts), y_base, 'bo', alpha=alpha  )#, nonposy='clip', basex=base,basey=base);
plt.loglog(np.unique(adv_counts), y_adv , 'r.', alpha=2*alpha)#, nonposy='clip', basex=base,basey=base);


################ Feature extract from resnet- Pytorch ########################
import torch
import torchvision.models
import numpy as np
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from __future__ import print_function
from resnet import resnet152

def extractFeat(inpResnet, x):
    x = inpResnet.conv1(x)
    x = inpResnet.bn1(x)
    x = inpResnet.relu(x)
    x = inpResnet.maxpool(x)

    x = inpResnet.layer1(x)
    x = inpResnet.layer2(x)
    x = inpResnet.layer3(x)
    x = inpResnet.layer4(x)
    x = torch.nn.AvgPool2d(14)(x)
    x = x.view(x.size()[0], -1)

    return x

resMod = torch.load('/BS/rshetty-wrk/work/auxData/pytorch-models/resnet152.pth')
resnet152 = resnet152()#pretrained=True)
resnet152.load_state_dict(resMod)
resnet152.cuda()
traindir = 'data/social-new/rawdata/folders/'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
dataset = datasets.ImageFolder(traindir, transforms.Compose([transforms.Scale([448,448]), transforms.ToTensor(), normalize]))
batch_size = 128
outfile = 'data/social-new/resnet152-meanpool-altresnet.npy'
train_loader = torch.utils.data.DataLoader(dataset, batch_size = 128, shuffle = False, pin_memory=True, num_workers = 16)

n_images = len(dataset)
n_iters = ((n_images -1)// batch_size) + 1
labels = []
features = np.zeros((len(dataset),2048),dtype=np.float32)
for i,(input_x, target) in enumerate(train_loader):#xrange(n_iters):
    b_sz = input_x.size()[0]
    input_var = torch.autograd.Variable(input_x, volatile=True)
    output = extractFeat(resnet152,input_var.cuda())
    features[i*batch_size:i*batch_size +b_sz,:] = output.cpu().data.numpy()
    labels.extend([dataset.imgs[i*batch_size+idx][0] for idx in xrange(b_sz)])
    if i%100 == 1:
        np.save(outfile,features)
    print(i)
np.save(outfile,features)
outf = open('data/social-new/featorder.txt','w')
outf.write('\n'.join(labels))
outf.close()





