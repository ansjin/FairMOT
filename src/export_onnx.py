from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import json
import torch
import torch.onnx
import torch.utils.data
from torchvision.transforms import transforms as T
from opts import opts
from models.model import create_model, load_model, save_model

import cv2
import numpy as np
import torch
import torch.onnx.utils as onnx
import models.networks.pose_dla_dcn as net
from collections import OrderedDict
import torchvision.models as models
import torch
from ptflops import get_model_complexity_info

opt = opts().parse()
    

model = net.get_pose_net(num_layers=34, heads={'hm': 1, 'wh': 4, 'reg': 2, 'id':128})

checkpoint = torch.load(r"../models/fairmot_dla34.pth", map_location="cpu")
checkpoint = checkpoint["state_dict"]
change = OrderedDict()
for key, op in checkpoint.items():
    change[key.replace("module.", "", 1)] = op

model.load_state_dict(change)
model.eval()
if opt.gpus[0] >= 0: 
    model.cuda()

    input = torch.zeros((1, 3, 608, 1088)).cuda()
    [hm, wh, reg, hm_pool, id_feature] = model(input)
    onnx.export(model, (input), "../models/fairmot_dla34_1088x608.onnx", output_names=["hm", "wh", "reg", "hm_pool", "id"], verbose=True)

    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (3, 608, 1088), as_strings=True,
                                                print_per_layer_stat=True, verbose=True)

        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
else:
    input = torch.zeros((1, 3, 608, 1088)).cuda()
    [hm, wh, reg, hm_pool, id_feature] = model(input)
    onnx.export(model, (input), "../models/fairmot_dla34_1088x608.onnx", output_names=["hm", "wh", "reg", "hm_pool", "id"], verbose=True)

    macs, params = get_model_complexity_info(model, (3, 608, 1088), as_strings=True,
                                            print_per_layer_stat=True, verbose=True)

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

