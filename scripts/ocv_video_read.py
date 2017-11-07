import os
import numpy as np
import cv2
import argparse
import json


def tileImages(frames,idcs,cent,grid=None,final_size=[150.0,800.0],brd=10,const_bright=0):
    if grid == None:
        grid = [1,len(idcs)]
    widthO  = frames[0].shape[0]+2*brd
    heightO = frames[0].shape[1]+2*brd
    width  = widthO * grid[0];
    height = heightO * grid[1];
    accum_frame = np.zeros((width,height,frames[0].shape[2]),dtype=frames[0].dtype)
    k = 0;
    BLUE = [255,0,0]
    SPL = [0,0,255]
    for j in xrange(grid[1]):
        for i in xrange(grid[0]):
            col = SPL if idcs[k] == cent else BLUE
            accum_frame[i*widthO:(i+1)*widthO,j*heightO:(j+1)*heightO,:] = cv2.copyMakeBorder(frames[idcs[k]]+const_bright,brd,brd,brd,brd,cv2.BORDER_CONSTANT,value=col)            
            k = k+1
    return cv2.resize(accum_frame,None,fx=final_size[1]/height,fy=final_size[0]/width, interpolation = cv2.INTER_CUBIC)

vidId = '210540520'


if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  parser.add_argument(dest='vidList', nargs='+',type=str, default=[], help='List of video ids')
  parser.add_argument('--cb', dest='cb', type=int, default=0, help='constant brightness increase')
  parser.add_argument('--ref', dest='ref', type=int, default=0, help='constant brightness increase')
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  
  
  basepath = '/triton/ics/project/imagedb/picsom/databases/lsmdc2015/objects/'
  genCaptionSrcL = ['example_images/result_struct_lsmdcTrajSwapTestb1.json', 'example_images/result_struct_lsmdcTrajSwapValb1.json' ] 
  refData = 'data/lsmdc2015/dataset.json'
  saveDir = 'video_visualizes'
  #cap = cv2.VideoCapture('/triton/ics/project/imagedb/picsom/databases/lsmdc2015/objects/2/3035/230350439.avi')
  #2/3010/230100536.avi
  #1/1051/110510033.avi
  #1/4076/140760125.avi
  #1/1027/110270280.avi
  #1/1026/110260059.avi 
  #1/1051/110510496.avi 
  candbId = {}
  
  if params['ref']==0:
    for genCaptionSrc in genCaptionSrcL:
      res = json.load(open(genCaptionSrc,'r'))
      for img in res['imgblobs']:
        candbId[img['img_path'].split('/')[-1].split(':')[0]] = img['candidate']['text']
  else:
    data = json.load(open(refData,'r'))
    for img in data['images']:
        candbId[img['filename'].split(':')[0]] = img['sentences'][0]['raw']
    
  
  for vidId in params['vidList']:
    vidName = basepath + vidId[0] + '/' + vidId[1:5] + '/' + vidId + '.avi'
    saveFile = vidId + '_genCap.png' if params['ref'] == 0 else  vidId + '_refCap.png'
    cap = cv2.VideoCapture(vidName)
    captext = candbId[vidId] 
    
    ret, frame = cap.read()
    accum_frame = np.zeros(frame.shape[:2],dtype=np.float32)
    i = 0
    all_frames = []
    while(cap.isOpened() and ret==True):
        all_frames.append(frame)
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #accum_frame = accum_frame * float(i) / (i+1) + 1.0/(i+1) * gray
        #if i == 0:
        #    accum_frame = gray
        #else:
        #    accum_frame = accum_frame + 1.0* np.abs(gray.astype(float) - prev_gray.astype(float))   
        #cv2.imshow('frame',frame)
        #prev_gray = gray.copy()
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        ret, frame = cap.read()
        i = i+1
    
    cap.release()
    aclip = accum_frame.clip(0.0,512.0)
    cent = int(len(all_frames)/2)
    step = int((len(all_frames)-cent)/3)
    idces = [ cent-2*step,cent-step,cent,cent+step, cent+2*step]
    
    megaFrame = tileImages(all_frames,idces,cent,const_bright=params['cb'])
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    megaFrame = cv2.putText(megaFrame,captext,(2,140), font, 1,(255,255,255),2,cv2.LINE_AA)
    
    cv2.imwrite(os.path.join(saveDir,saveFile),megaFrame)
    #cv2.imshow('frame',megaFrame)
    #
    #if cv2.waitKey(0) & 0xFF == ord('q'):
    #    cv2.destroyAllWindows()
