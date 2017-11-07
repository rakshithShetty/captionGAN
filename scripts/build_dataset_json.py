import os

dL = {}
dL['dataset'] = 'lsmdc2015'
dL['images'] = {} 

basepath = '/triton/ics/project/imagedb/picsom/databases/lsmdc2015'

lbls = open(os.path.join(basepath,'labels.txt'),'r').read().splitlines()
splitidtostr = ['train', 'val','test','blindtest']


for L in lbls:
    if 'ks' in L:
        img = {}
        dbid = L.split('<')[1].split(':')[0]
        seqid = int(L.split('<')[0][1:])
        split = dbid[0]

        if int(split) < 3: 
            movieid = dbid[1:5]
            clipid = dbid[5:]

            img['dbid'] = int(dbid)
            img['filename'] = dbid + ':kf1.jpg'
            img['filepath'] = os.path.join(basepath,'objects',split,movieid)
            img['imgid'] = seqid
            img['split'] = splitidtostr[int(split)]
            img['sentences'] = []


            #if int(split) < 3:
            #    caption = open(os.path.join(img['filepath'],dbid+'.d', dbid+'-gt.txt'),'r').read()
            #    sents = {'imgid':seqid, 'raw':caption, 'sentid':seqid, 'tokens':caption.split()}

            #    img['sentences'].append(sents)
            
            dL['images'][int(dbid)] = img

        if (seqid %100) == 0:
            print('Now in %d\r'%(seqid)),

#json.dump(dL, open('data/lsmdc2015/datasetBlind.json','w'))
        
