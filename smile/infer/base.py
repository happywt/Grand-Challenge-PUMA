import argparse
import glob
import json
import math
import multiprocessing
import os
import re
import sys
from importlib import import_module
from multiprocessing import Lock, Pool

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import tqdm

from run_utils.utils import convert_pytorch_checkpoint


####
class InferManager(object):
    def __init__(self, **kwargs):
        self.run_step = None
        for variable, value in kwargs.items():
            self.__setattr__(variable, value)
        self.__load_model()
        self.nr_types = self.method["model_args"]["nr_types"]
        # create type info name and colour

        # default
        self.type_info_dict = {
            None: ["no label", [0, 0, 0]],
        }

        if self.nr_types is not None and self.type_info_path is not None:
            self.type_info_dict = json.load(open(self.type_info_path, "r"))
            self.type_info_dict = {
                int(k): (v[0], tuple(v[1])) for k, v in self.type_info_dict.items()
            }
            # availability check
            for k in range(self.nr_types):
                if k not in self.type_info_dict:
                    assert False, "Not detect type_id=%d defined in json." % k

        if self.nr_types is not None and self.type_info_path is None:
            cmap = plt.get_cmap("hot")
            colour_list = np.arange(self.nr_types, dtype=np.int32)
            colour_list = (cmap(colour_list)[..., :3] * 255).astype(np.uint8)
            # should be compatible out of the box wrt qupath
            self.type_info_dict = {
                k: (str(k), tuple(v)) for k, v in enumerate(colour_list)
            }
        return

    def __load_model(self):
        """Create the model, load the checkpoint and define
        associated run steps to process each data batch.
        
        """
        model_desc = import_module("models.smile.net_desc_smile")
        model_creator = getattr(model_desc, "create_model")

        net = model_creator(**self.method["model_args"])
        saved_state_dict = torch.load(self.method["model_path"])["desc"]
        saved_state_dict = convert_pytorch_checkpoint(saved_state_dict)

        net.load_state_dict(saved_state_dict, strict=True)
        net = torch.nn.DataParallel(net)
        net = net.to("cuda")

        module_lib = import_module("models.smile.run_desc")
        run_step = getattr(module_lib, "infer_step")
        self.run_step = lambda input_batch: run_step(input_batch, net)

        module_lib = import_module("models.smile.post_proc")
        self.post_proc_func = getattr(module_lib, "process")
        return


    def __save_json(self, path, old_dict, mag=None):
        
        # type_map = {
        #     1: "nuclei_lymphocyte",
        #     2: "nuclei_tumor",
        #     3: "nuclei_other"
        # }
        
        type_map = {
            1: "nuclei_endothelium",
            2: "nuclei_plasma_cell",
            3: "nuclei_stroma",
            4: "nuclei_tumor",
            5: "nuclei_histiocyte",
            6: "nuclei_apoptosis",
            7: "nuclei_epithelium",
            8: "nuclei_melanophage",
            9: "nuclei_neutrophil",
            10: "nuclei_lymphocyte"
        } 
        
        output = {
            "type": "Multiple polygons",
            "polygons": [],
            "version": {  # 添加版本信息
                "major": 1,
                "minor": 0
            }
        }
        
        for inst_id, inst_info in old_dict.items():
            # 获取轮廓的第一个点作为种子点
            first_contour_point = inst_info["contour"][0]
            polygon_name = type_map.get(inst_info["type"], "unknown")
            
            if polygon_name != "unknown": # 如果模型存在未预测到类别，就忽略这个实例的预测
                polygon = {
                    "name": polygon_name,
                    "seed_point": [
                        float(first_contour_point[0]),  # 转换为浮点数保持格式统一
                        float(first_contour_point[1]),
                        0.5
                    ],
                    "path_points": [
                        [float(p[0]), float(p[1]), 0.5]  # 全部转为浮点数
                        for p in inst_info["contour"]
                    ],
                    "sub_type": "",
                    "groups": [],
                    "probability": 1
                }
                output["polygons"].append(polygon)
        
        with open(path, "w") as handle:
            json.dump(output, handle, indent=2)
        
        # # old_dict : inst_info_dict 
        # new_dict = {}
        
        # print("*********")
        # print("save_path:", path)
        # print("*********")
        # for inst_id, inst_info in old_dict.items():
        #     new_inst_info = {}
        #     for info_name, info_value in inst_info.items():
        #         # convert to jsonable
        #         if isinstance(info_value, np.ndarray):
        #             info_value = info_value.tolist()
        #         new_inst_info[info_name] = info_value
        #     new_dict[int(inst_id)] = new_inst_info

        # json_dict = {"mag": mag, "nuc": new_dict}  # to sync the format protocol
        
        # with open(path, "w") as handle:
        #     json.dump(json_dict, handle)
        return old_dict


    # def __save_json(self, path, old_dict, mag=None):
    #     new_dict = {}
    #     for inst_id, inst_info in old_dict.items():
    #         new_inst_info = {}
    #         for info_name, info_value in inst_info.items():
    #             # convert to jsonable
    #             if isinstance(info_value, np.ndarray):
    #                 info_value = info_value.tolist()
    #             new_inst_info[info_name] = info_value
    #         new_dict[int(inst_id)] = new_inst_info

    #     json_dict = {"mag": mag, "nuc": new_dict}  # to sync the format protocol
    #     with open(path, "w") as handle:
    #         json.dump(json_dict, handle)
    #     return new_dict
