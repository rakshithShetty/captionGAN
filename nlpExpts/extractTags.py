import json
tSents =  open('tagAllSentMerged', 'r').read().splitlines()
tag_hist = {}

for s in tSents:
    words = s.split()
    for wt in words:
        w = wt.rsplit('_')[0]
        t = wt.rsplit('_')[1]
        if w in tag_hist.keys() and t in tag_hist[w].keys():
            tag_hist[w][t] += 1
        elif w in tag_hist.keys():
            tag_hist[w][t] = 1
        else:
            tag_hist[w] = {}
            tag_hist[w][t] = 1

word_analysis_data = {}
word_analysis_data['all_tags'] = tag_hist
json.dump(word_analysis_data, open('word_analysis_data_coco.json','w'))

setD = lambda : defaultdict(set)
wordsIdList = defaultdict(setD)
tSents =  open('./nlpExpts/results/posTagging/tagAllSentsPuncts', 'r').read().splitlines()
for i,sent in enumerate(dbTrn['annotations']):
    words = tSents[i].split()
    for wt in words:
        w = wt.rsplit('_')[0]
        t = wt.rsplit('_')[1]
        if w in wix:
            wordsIdList[t][w].add(sent['image_id'])

i = 0
sentTagMap = defaultdict(dict)
for i,sent in enumerate(dbTrn['annotations']):
    words = tSents[i].split()
    i += 1
    for wt in words:
        w = wt.rsplit('_')[0].lower()
        t = wt.rsplit('_')[1]
        if w in wix:
           sentTagMap[sent['id']][w] = t




############### Collate dependency parse tree ############################
import json
txt = open('nlpExpts/results/depParsing/dbSentencesRaw.txt.stp','r').read().splitlines()
dbTrn = json.load(open('/triton/ics/project/imagedb/picsom/databases/COCO/download/annotations/captions_train2014.json','r'))

parseDataPerSent = []
c_strt = 0
for i,t in enumerate(txt):
    if t == '':
        parseDataPerSent.append(txt[c_strt:i])
        c_strt = i+1

parseDataByImgId = defaultdict(list)
for i,ann in enumerate(dbTrn['annotations']):
    parseDataByImgId[ann['image_id']].append({'cap':ann['caption'],'parse':parseDataPerSent[i]})

dataset = json.load(open('data/coco/dataset.json'))
imgidToFeat = {}
for img in dataset['images']:
    imgidToFeat[img['cocoid']] = img['imgid']


imgIds = [ann['image_id'] for ann in dbTrn['annotations']]
idxes = [imgidToFeat[i] for i in imgIds[:20]]


fid = open('words2CatReplaced.txt','w')
for sent in dbTrn['annotations']:
    sw = sent['caption'].split()
    for i,w in enumerate(sw):
        if w in word2CMap:
            sw[i] = '<' + word2CMap[w]+'>'
    fid.writelines(sent['caption'] + ' ||| ' + ' '.join(sw) +' \n')

fid.close()

caseWords = defaultdict(set)
for id in parseDataByImgId:
    for sent in parseDataByImgId[id]:
        for pLine in sent['parse']:
            if pLine.startswith('case'):
                caseWords[pLine.split(',')[1].split('-')[0].lower().lstrip(' ')].add(id)
