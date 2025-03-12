import glob
import json
import os
import random
import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.engine import DefaultPredictor, DefaultTrainer, default_setup
import numpy as np
from detectron2.modeling import build_model
from detectron2.engine import default_argument_parser
from data.datasets import builtin
from modeling import add_stn_config


def setup(args):
    cfg = get_cfg()
    add_stn_config(cfg)
    # hack to add base yaml
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_file(model_zoo.get_config_file(cfg.BASE_YAML))
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):

    cfg = setup(args)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.65
    cfg.OUTPUT_DIR = "./output/results_Gap"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    predictor = DefaultPredictor(cfg)

    files_path = glob.glob("/home/zzh/domaingen/dw"
                           "/*.jpg")
    print("total imgs " + str(len(files_path)))

    for filepath in files_path:
        im = cv2.imread(filepath)
        outputs = predictor(
            im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                       metadata=metadata,
                       scale=0.5,
                       # remove the colors of unsegmented pixels. This option is only available for segmentation models
                       )

        new_basename = filepath.rsplit('/', 1)[-1]
        domain_name = filepath.split('/')[-4]
        stored_dir = os.path.join(cfg.OUTPUT_DIR, domain_name)
        if not os.path.exists(stored_dir):
            os.makedirs(stored_dir)
        stored_path = os.path.join(stored_dir, new_basename)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(stored_path, out.get_image()[:, :, ::-1])


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    print("Command Line Args:", args)

    main(args)




