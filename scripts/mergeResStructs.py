import argparse
import json

def mergeRes(params):
  
  blb = {}
  model_list_f = open(params['res_list'], 'r').read().splitlines()
  mdlNames = []
  mdlLabels = []
  for fnms in model_list_f:
    model_name = fnms.split(',')[0]
    model_lbl = fnms.split(',')[1] 
    mdlNames.append(model_name)
    mdlLabels.append(model_lbl)
    res = json.load(open(model_name,'r'))

    for r in res['imgblobs']:
        imgid = r['img_path'].split('_')[-1].split('.')[0]
        if blb.get(imgid,[]) == []:
            blb[imgid] = {'candidatelist':[r['candidate']]}
            blb[imgid]['img_path'] = r['img_path']
        else:
            blb[imgid]['candidatelist'].append(r['candidate'])

  resM = {}
  resM['params'] = res['params']
  resM['checkpoint_params'] = res['checkpoint_params']
  resM['imgblobs'] = blb.values()
  resM['mdlNames'] = mdlNames
  resM['lbls'] = mdlLabels
  
  json.dump(resM,open(params['result_struct_filename'],'w'))

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('res_list', type=str, help='the input list of result structures to form committee from')
  parser.add_argument('--result_struct_filename', type=str, default='merge_result.json', help='filename of the result struct to save')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print 'parsed parameters:'
  print json.dumps(params, indent = 2)

  mergeRes(params)
