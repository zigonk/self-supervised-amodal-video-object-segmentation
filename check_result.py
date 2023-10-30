import json
from pycocotools import _mask as coco_mask

vm_result = 'VideoAmodal/FishBowl/fishbowl_test_annotations/a2vis_fishbowl/inference/visible/results.json'
test_infos = '/home/nqthuc/Documents/MOT/AMOT/savos/VideoAmodal/FishBowl/fishbowl_test_annotations/fishbowl_test50.json'

with open(vm_result, 'r') as f:
    f = json.load(f)
    print(len(f[0]['segmentations']))
        
with open(test_infos, 'r') as f:
    f = json.load(f)
    print(f['videos'][0].keys())
