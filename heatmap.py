import argparse
import cv2
import numpy as np
import os
import torch
import tqdm
from detectron2.data.detection_utils import read_image
import time
from detectron2.utils.logger import setup_logger


def setup_cfg(args):  # 获取cfg，并合并，不用细看，和demo.py中的一样
    # load config from file and command-line arguments
    from detectron2.config import get_cfg
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="",  # 此处是配置文件，在config下选择你的yaml文件
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", default='', nargs="+",
                        help="A list of space separated input images")  # 图片文件夹路径，目前只支持图片输入，
    # 要输入视频或者调用摄像头，可以自行修改代码
    parser.add_argument(
        "--output",
        default='',  # 输出文件夹路径
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,  # 置信度阈值
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)

    # 1*256*200*256 # feat的维度要求，四维
    feature_map = feature_map.detach()

    # 1*256*200*256->1*200*256
    heatmap = feature_map[:, 0, :, :] * 0
    for c in range(feature_map.shape[1]):
        heatmap += feature_map[:, c, :, :]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap


def draw_feature_map(img_path, save_dir):
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    from predictor import VisualizationDemo
    demo = VisualizationDemo(cfg)
    for imgs in tqdm.tqdm(os.listdir(img_path)):
        img = read_image(os.path.join(img_path, imgs), format="BGR")
        start_time = time.time()
        predictions = demo.run_on_image(img)  # 后面需对网络输出做一定修改，
        # 会得到一个字典P3-P7的输出
        logger.info(
            "{}: detected in {:.2f}s".format(
                imgs, time.time() - start_time))
        i = 0
        for featuremap in list(predictions.values()):
            heatmap = featuremap_2_heatmap(featuremap)
            # 200*256->512*640
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的
            # 大小调整为与原始图像相同         
            heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
            # 512*640*3
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原
            # 始图像       
            superimposed_img = heatmap * 0.7 + 0.3 * img  # 热力图强度因子，修改参数，
            cv2.imwrite(os.path.join(save_dir, imgs + str(i) + '.jpg'),
                        superimposed_img)  # 将图像保存
            i = i + 1


from argparse import ArgumentParser


def main():
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    draw_feature_map(args.input, args.output)


if __name__ == '__main__':
    main()