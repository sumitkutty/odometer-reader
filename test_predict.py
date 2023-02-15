
from opts import set_cfg, base_opts, make_integer
import detectron2, cv2, random
import os, json, itertools
import numpy as np
import torch, torchvision
import matplotlib.pyplot as plt
from PIL import Image
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from matplotlib import pyplot as plt
import argparse
from ocr import odometer_reader
import pandas as pd
import shutil
from warnings import filterwarnings as w
w('ignore')

#OCR
from scripts.utils import AttnLabelConverter
from scripts.model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def detect_roi(path, opt):
    
    print("Processing Images.......")
    predictor = DefaultPredictor(cfg)
    all_classes = ["LCD", "odometer", "M","not_touching", "screen"]
    c = 0
    for img_name in os.listdir(path):
        if img_name.split(".")[-1].lower() not in ["jpg", "jpeg", "png"]:
            continue
        img_path = os.path.join(path, img_name)
        img = np.array(Image.open(img_path))
        
        outputs = predictor(img)

        instance = outputs['instances']
        classes = np.array(instance.pred_classes)
        class_inds = np.where(classes == 1)[0]
        
        if len(class_inds) >= 1:
            class_ind = int(class_inds[0])
        else:
            print(img_name)
            continue
        
        x1, y1, x2, y2 =  make_integer(instance.pred_boxes[class_ind].tensor[0])
        crop = img[y1-1:y2+2, x1-1:x2+2]
        crop = Image.fromarray(crop)
        
        temp_path = f"temp/{img_name}"
        crop.save(temp_path, quality = 95)
        c += 1
    print(f"{c} ROIs Saved")
    
if __name__ == '__main__':
    
    
    cfg = get_cfg()
    cfg = set_cfg(cfg)
    
    parser = argparse.ArgumentParser()
    base_opts(parser)
    opt = parser.parse_args()
    
    path = opt.images_path
    shutil.rmtree(opt.image_folder, ignore_errors = True)
    os.makedirs(opt.image_folder, exist_ok = True)
    detect_roi(path, opt)
    
    
    #Initialize OCR Model
    converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)

    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    
    result  = odometer_reader(opt, model, converter)
    
    shutil.rmtree(opt.image_folder, ignore_errors = True)
    
    if opt.generate_csv:
        result.to_csv("Odometer_Results.csv", index = False)
    print("=====RESULTS======")
    print(result)
    print("===================")
    
    
    