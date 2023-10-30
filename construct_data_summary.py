import json

import numpy as np
from pycocotools import _mask as coco_mask
from tqdm import tqdm

"""Load predicted data from json file and construct required data fields
    - predicted_vm is a list of dict with keys: video_id, category_id, segmentations, score
    - mapped_data_info is a dict with keys: videos, categories, annotations
"""
predicted_vm_path = "VideoAmodal/FishBowl/test_data/a2vis/visible/results.json"
mapped_data_info_path = "VideoAmodal/FishBowl/test_data/a2vis/fishbowl_test50.json"

predicted_vm = json.load(open(predicted_vm_path, "r"))
mapped_data_info = json.load(open(mapped_data_info_path, "r"))

obj_lists = []
data_summary = {}

cur_obj_id = 0
# Sort the predicted_vm by video_id to optimize the loading time
predicted_vm.sort(key=lambda x: x["video_id"])
for vm in tqdm(predicted_vm):
    video_info = list(filter(lambda x: x["id"] == vm["video_id"], mapped_data_info["videos"]))[0]
    video_name = video_info["file_names"][0].split("/")[0]
    obj_lists.append(str(video_name)+"_"+str(cur_obj_id))
    cur_obj_id += 1
    
    # Construct data_summary
    if str(video_name)+"_"+str(cur_obj_id-1) not in data_summary:
        data_summary[str(video_name)+"_"+str(cur_obj_id-1)] = {}

    current_data_key = obj_lists[-1]
    masks = vm['segmentations']
    bboxes = coco_mask.toBbox(vm["segmentations"])
    for index, (seg, bbox) in enumerate(zip(masks, bboxes)):
        data_summary[current_data_key][index] = {}
        data_summary[current_data_key][index]["VM"] = [seg]
        data_summary[current_data_key][index]["VM_bx"] = bbox
        # data_summary[current_data_key][index]["loss_mask_weight"] = np.ones((seg['size'][1], seg['size'][0]))

# Save data_summary to pkl file
import pickle

with open("VideoAmodal/FishBowl/test_data/custom_test_data.pkl", "wb") as f:
    pickle.dump(data_summary, f)