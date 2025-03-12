import glob
import json
import os
import random
import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.modeling import build_backbone
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.engine import DefaultPredictor, DefaultTrainer, default_setup
import numpy as np
from detectron2.modeling import build_model
from detectron2.engine import default_argument_parser
from data.datasets import builtin
from modeling import add_stn_config
from PIL import Image
import torch
import cv2
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from torchvision import transforms
import torch.nn.functional as F
import cv2
import torchvision
import torch
from matplotlib import pyplot as plt
import numpy as np

def setup(args):
    cfg = get_cfg()
    add_stn_config(cfg)
    #hack to add base yaml 
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_file(model_zoo.get_config_file(cfg.BASE_YAML))
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    #cfg.freeze()
    default_setup(cfg, args)
    return cfg

transformss = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((224, 224)),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def random_num(size,end):
    range_ls=[i for i in range(end)]
    num_ls=[]
    for i in range(size):
        num=random.choice(range_ls)
        range_ls.remove(num)
        num_ls.append(num)
    return num_ls


def main(args):
    cfg = setup(args)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = "all_outs/diverse_weather_f1_dis_p8_newp_night_rainy_newd/model_best.pth"
    cfg.OUTPUT_DIR = "./output/result_7"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    model = build_model(cfg)
    model.eval()
    model = model.to("cuda")
    path = "/home/zzh/domaingen/foggy-103.jpg"
    transformss = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((224, 224)),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 注意如果有中文路径需要先解码，最好不要用中文
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 转换维度
    img = transformss(img).unsqueeze(0)
    img = img.to("cuda")
    for name in model.di.named_children():
        print(name[0])

    # selected_layer = model.backbone.enc.conv1
    # selected_layer2 = model.backbone.enc.bn1
    # selected_layer3 = model.backbone.enc.relu1
    selected_layers = [model.backbone.enc.conv1, model.backbone.enc.bn1, model.backbone.enc.relu1,
                       model.backbone.enc.conv2, model.backbone.enc.bn2, model.backbone.enc.relu2,
                       model.backbone.enc.conv3, model.backbone.enc.bn3, model.backbone.enc.relu3,
                        model.backbone.enc.avgpool, model.backbone.enc.layer1,model.backbone.enc.layer2
                        , model.backbone.enc.layer3, model.backbone.enc.layer4,
                       ]
    for layer in selected_layers:
        if layer == model.backbone.enc.conv1 :
            k = 1
            intermediate_output = layer(img)
            # for i in range(8):
            #     map_glo = torch.unsqueeze(intermediate_output[:, 2*i, :, :], 1)
            #     map_glo = (map_glo - map_glo.min()) / (map_glo.max() - map_glo.min() + 1e-8)
            #     map_glo = F.interpolate(map_glo, size=(192, 192), mode='bilinear', align_corners=True)
            #     map_glo = np.squeeze(torch.Tensor(map_glo.cpu().data.numpy()).cpu().data.numpy())
            #     heatmap = cv2.applyColorMap(np.uint8(255 * map_glo), cv2.COLORMAP_JET)
            #     cv2.imwrite(os.path.join('./feature_map/', 'fus-e-3-' + str(2*i) + '.png'), heatmap)
        else:
            k = k+1
            intermediate_output = layer(intermediate_output)
            if k <= 10:
                for i in range(16):
                    map_glo = torch.unsqueeze(intermediate_output[:, 2 * i, :, :], 1)
                    map_glo = (map_glo - map_glo.min()) / (map_glo.max() - map_glo.min() + 1e-8)
                    map_glo = F.interpolate(map_glo, size=(640, 640), mode='bilinear', align_corners=True)
                    map_glo = np.squeeze(torch.Tensor(map_glo.cpu().data.numpy()).cpu().data.numpy())
                    heatmap = cv2.applyColorMap(np.uint8(255 * map_glo), cv2.COLORMAP_JET)
                    cv2.imwrite(os.path.join('./feature_map2/', str(k) + '——fus-e-3-' + str(2 * i) + '.png'), heatmap)
            else:
                for i in range(64):
                    map_glo = torch.unsqueeze(intermediate_output[:, 2 * i, :, :], 1)
                    map_glo = (map_glo - map_glo.min()) / (map_glo.max() - map_glo.min() + 1e-8)
                    map_glo = F.interpolate(map_glo, size=(640, 640), mode='bilinear', align_corners=True)
                    map_glo = np.squeeze(torch.Tensor(map_glo.cpu().data.numpy()).cpu().data.numpy())
                    heatmap = cv2.applyColorMap(np.uint8(255 * map_glo), cv2.COLORMAP_JET)
                    cv2.imwrite(os.path.join('./feature_map2/', str(k) + '——fus-e-3-' + str(2 * i) + '.png'), heatmap)




    # fig, axs = plt.subplots(4, 4, figsize=(12, 12))  # 根据需要调整网格大小
    # original_image = img
    # for i in range(16):  # 调整要可视化的特征图数量
    #     map_glo = torch.unsqueeze(intermediate_output[:, 2 * i, :, :], 1)
    #     map_glo = (map_glo - map_glo.min()) / (map_glo.max() - map_glo.min() + 1e-8)
    #     map_glo = F.interpolate(map_glo, size=(original_image.shape[2], original_image.shape[3]), mode='bilinear',
    #                             align_corners=True)
    #     map_glo = np.squeeze(torch.Tensor(map_glo.cpu().data.numpy()).cpu().data.numpy())
    #     heatmap = cv2.applyColorMap(np.uint8(255 * map_glo), cv2.COLORMAP_JET)

        # 将 original_image 转移到CPU并转换为numpy数组
        # original_image_np = np.array(original_image.cpu())
        #
        # # 将 heatmap 转换为 numpy 数组并归一化到 0-255
        # heatmap_np = np.uint8(255 * heatmap)
        #
        # # 调整 heatmap_resized 的通道数以匹配 original_image_np
        # if original_image_np.shape[-1] == 3 and heatmap_np.shape[-1] == 1:
        #     heatmap_np = cv2.cvtColor(heatmap_np, cv2.COLOR_GRAY2RGB)
        #
        # # 将图像混合在一起
        # heatmap_with_overlay = cv2.addWeighted(original_image_np.astype(np.uint8), 0.5, heatmap_np.astype(np.uint8),
        #                                        0.5, 0)

        # 显示结果
    #     axs[i // 4, i % 4].imshow(heatmap)
    #     axs[i // 4, i % 4].set_title(f'层{k} - 特征图{2 * i}')
    #
    #
    # plt.show()

    # v = intermediate_output.squeeze(0)
    # print("Intermediate Output Mean:", v.mean())
    # print("Intermediate Output Std:", v.std())
    # # 取消Tensor的梯度并转成三维tensor，否则无法绘图
    # v = v.data.cpu()
    # channel_num = random_num(25, v.shape[0])
    #
    # plt.figure(figsize=(10, 10))
    # for index, channel in enumerate(channel_num):
    #     ax = plt.subplot(5, 5, index + 1, )
    #     plt.imshow(v[channel, :, :])
    #
    # plt.show()
    # plt.savefig("feature5_layer4.jpg", dpi=300)
    # 保存图像



    # metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    # predictor = DefaultPredictor(cfg)
    # # self.backbone.set_backbone_model(self.roi_heads.box_predictor.cls_score.visual_enc)
    # files_path = glob.glob("/home/zzh/domaingen/datasets/diverseWeather/night_rainy/VOC2007/pic/*.jpg")
    # print("total imgs " + str(len(files_path)))

    # def print_model_modules(model, prefix="model", depth=0):
    #     for name, module in model.named_children():
    #         print(f"{'  ' * depth}{prefix}.{name}")
    #         if len(list(module.children())) > 0:
    #             print_model_modules(module, f"{prefix}.{name}", depth + 1)

    # 打印模型结构
    # print_model_modules(model)
    #
    # for filepath in files_path:
    #     im = cv2.imread(filepath)
    #
    #     outputs_dict = {}
    #
    #     def hook_fn(module, input, output):
    #         outputs_dict["intermediate_output"] = output
    #
    #     # 找到模型中某个层，设置钩子函数
    #     target_layer = model.backbone.backbone_unchanged[15].relu3  # 替换为你的目标中间层
    #     hook_handle = target_layer.register_forward_hook(hook_fn)
    #
    #     outputs = predictor(im)
    #
    #     hook_handle.remove()
    #
    #     # 获取中间层的输出
    #     intermediate_output = outputs_dict["intermediate_output"]
    #
    #     # 在这里，你可以对中间输出进行可视化或其他处理
    #     print(f"Intermediate Output Shape: {intermediate_output.shape}")
    #
    #     v = Visualizer(im[:, :, ::-1],
    #                    metadata=metadata,
    #                    scale=0.5,
    #                    # remove the colors of unsegmented pixels. This option is only available for segmentation models
    #                    )
    #
    #     new_basename = filepath.rsplit('/', 1)[-1]
    #     domain_name = filepath.split('/')[-4]
    #     stored_dir = os.path.join(cfg.OUTPUT_DIR, domain_name)
    #     if not os.path.exists(stored_dir):
    #         os.makedirs(stored_dir)
    #     stored_path = os.path.join(stored_dir, new_basename)
    #     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #     cv2.imwrite(stored_path, out.get_image()[:, :, ::-1])


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    print("Command Line Args:", args)
    
    main(args)
    

    
    
