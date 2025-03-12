from ast import mod
import math
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from typing import Dict, List, Optional

from detectron2.modeling import META_ARCH_REGISTRY, GeneralizedRCNN
from detectron2.structures import ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.layers import batched_nms
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.utils.visualizer import Visualizer
from .singe_prototype import Singe_prototype
# import ipdb
import os

def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),  # 卷积块 卷积加批量标准化 激活
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )

class invariant(nn.Module):
    def __init__(self):
        super(invariant, self).__init__()
        self.di = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(1024),
            )
    def forward(self, x):
        x = self.di(x)
        return x

class specific(nn.Module):
    def __init__(self):
        super(specific, self).__init__()
        self.di = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Conv2d(512, 512, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Conv2d(512, 1024, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(1024),
            )
    def forward(self, x):
        x = self.di(x)
        return x

@META_ARCH_REGISTRY.register()
class ClipRCNNWithClipBackbone(GeneralizedRCNN):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.colors = self.generate_colors(7)
        self.backbone.set_backbone_model(self.roi_heads.box_predictor.cls_score.visual_enc)
        self.pro = Singe_prototype(1024, 40)
        self.di = invariant()
        self.ds = specific()
        self.conv_out = convblock(2 * 1024, 1024, 3, 1, 1)

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        clip_images = [x["image"].to(self.pixel_mean.device) for x in batched_inputs]
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]

        clip_images = [T.functional.normalize(ci.flip(0) / 255, mean, std) for ci in clip_images]
        clip_images = ImageList.from_tensors(
            [i for i in clip_images])
        return clip_images

    def forward(self, batched_inputs):

        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        b = images.tensor.shape[0]  # batchsize

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            if self.training:
                logits, proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            else:
                logits, proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        try:
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, None, self.backbone)
        except Exception as e:
            print(e)
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, None)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
                with torch.no_grad():
                    ogimage = batched_inputs[0]['image']
                    ogimage = convert_image_to_rgb(ogimage.permute(1, 2, 0), self.input_format)
                    o_pred = Visualizer(ogimage, None).overlay_instances().get_image()

                    vis_img = o_pred.transpose(2, 0, 1)
                    storage.put_image('og-tfimage', vis_img)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def generate_colors(self, N):
        import colorsys
        '''
            Generate random colors.
            To get visually distinct colors, generate them in HSV space then
            convert to RGB.
        '''
        brightness = 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: tuple(round(i * 255) for i in colorsys.hsv_to_rgb(*c)), hsv))
        perm = np.arange(7)
        colors = [colors[idx] for idx in perm]
        return colors

    def inference(
            self,
            batched_inputs: List[Dict[str, torch.Tensor]],
            detected_instances: Optional[List[Instances]] = None,
            do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        base_di = self.di(features['res4'])
        # domain specific
        base_ds = self.ds(features['res4'])

        # for i in range(20):
        #     map_glo = torch.unsqueeze(base_di[:1, 2*i, :, :], 1)
        #     map_glo = (map_glo - map_glo.min()) / (map_glo.max() - map_glo.min() + 1e-8)
        #     map_glo = F.interpolate(map_glo, size=(192, 192), mode='bilinear', align_corners=True)
        #     map_glo = np.squeeze(torch.Tensor(map_glo.cpu().data.numpy()).cpu().data.numpy())
        #     heatmap = cv2.applyColorMap(np.uint8(255 * map_glo), cv2.COLORMAP_JET)
        #     cv2.imwrite(os.path.join('/home/zzh/domaingen/feature map/', 'base_di_' + str(2*i) + '.png'), heatmap)
        #     print("save success")

        # for i in range(20):
        #     map_glo = torch.unsqueeze(base_di[:1, 2*i, :, :], 1)
        #     map_glo = (map_glo - map_glo.min()) / (map_glo.max() - map_glo.min() + 1e-8)
        #     map_glo = F.interpolate(map_glo, size=(192, 192), mode='bilinear', align_corners=True)
        #     map_glo = np.squeeze(torch.Tensor(map_glo.cpu().data.numpy()).cpu().data.numpy())
        #     heatmap = cv2.applyColorMap(np.uint8(255 * map_glo), cv2.COLORMAP_JET)
        #     cv2.imwrite(os.path.join('/home/zzh/domaingen/feature map/', 'base_ds_' + str(2*i) + '.png'), heatmap)

        # # Sequential Once
        # base_ii = self.di(base_feat_di)
        # base_is = self.ds(base_feat_di)
        #
        # base_si = self.di(base_feat_ds)
        # base_ss = self.ds(base_feat_ds)

        base_di = self.pro(base_di)
        basef = torch.cat((base_di, base_ds), dim=1)
        features['res4'] = self.conv_out(basef) + features['res4']
        features['res4'] = self.pro(features['res4'])

        # for i in range(20):
        #     map_glo = torch.unsqueeze(features['res4'][:, 2 * i, :, :], 1)
        #     map_glo = (map_glo - map_glo.min()) / (map_glo.max() - map_glo.min() + 1e-8)
        #     map_glo = F.interpolate(map_glo, size=(192, 192), mode='bilinear', align_corners=True)
        #     map_glo = np.squeeze(torch.Tensor(map_glo.cpu().data.numpy()).cpu().data.numpy())
        #     heatmap = cv2.applyColorMap(np.uint8(255 * map_glo), cv2.COLORMAP_JET)
        #     cv2.imwrite(os.path.join('/home/zzh/domaingen/feature map/', 'res4_' + str(2 * i) + '.png'), heatmap)

        if detected_instances is None:
            if self.proposal_generator is not None:
                logits, proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            # boxes = batched_inputs[0]['instances'].gt_boxes.to(images.tensor.device)
            # logits = 10*torch.ones(len(boxes)).to(images.tensor.device)
            # dictp = {'proposal_boxes':boxes,'objectness_logits':logits}
            # new_p = Instances(batched_inputs[0]['instances'].image_size,**dictp)
            # proposals = [new_p]

            try:
                results, _ = self.roi_heads(images, features, proposals, None, None, self.backbone)
            except:
                results, _ = self.roi_heads(images, features, proposals, None, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."

            allresults = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)

            return allresults
        else:
            return results


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.reshape(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.reshape(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


# class PIN(nn.Module):
#     def __init__(self,shape,content_feat):
#         super(PIN,self).__init__()
#         self.shape = shape
#         self.content_feat = content_feat.clone().detach()
#         self.content_mean, self.content_std = calc_mean_std(self.content_feat)
#         self.size = self.content_feat.size()
#         self.content_feat_norm = (self.content_feat - self.content_mean.expand(
#         self.size)) / self.content_std.expand(self.size)

#         self.style_mean =   self.content_mean.clone().detach()
#         self.style_std =   self.content_std.clone().detach()

#         self.style_mean = nn.Parameter(self.style_mean, requires_grad = True)
#         self.style_std = nn.Parameter(self.style_std, requires_grad = True)
#         self.relu  = nn.ReLU(inplace=True)

#     def forward(self):

#         self.style_std.data.clamp_(min=0)
#         target_feat =  self.content_feat_norm * self.style_std.expand(self.size) + self.style_mean.expand(self.size)
#         target_feat = self.relu(target_feat)
#         return target_feat

@META_ARCH_REGISTRY.register()
class ClipRCNNWithClipBackboneWithOffsetGenTrainable(ClipRCNNWithClipBackbone):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        domain_text = {'day': 'an image taken during the day'}
        with open('prunedprompts2.txt', 'r') as f:
            for ind, l in enumerate(f):
                domain_text.update({str(ind): l.strip()})
        # self.offsets = nn.Parameter(offsets)
        #         self.offsets = nn.Parameter(torch.zeros(len(domain_text)-1,1024,14,14)) #skip day

        # self.pin = PIN()
        self.stylemean = nn.Parameter(torch.zeros(len(domain_text) - 1, 1024, 1, 1))
        self.stylestd = nn.Parameter(torch.ones(len(domain_text) - 1, 1024, 1, 1))

        import clip
        #         ipdb.set_trace()
        self.domain_tk = dict([(k, clip.tokenize(t)) for k, t in domain_text.items()])
        self.apply_aug = 0
        self.pro = Singe_prototype(1024, 40)
        self.di = invariant()
        self.ds = specific()
        self.conv_out = convblock(2 * 1024, 1024, 3, 1, 1)

    def forward(self, batched_inputs):
        #         idpb.set_trace()

        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        b = images.tensor.shape[0]  # batchsize

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        features = self.backbone(images.tensor)

        base_di = self.di(features['res4'])
        # domain specific
        base_ds = self.ds(features['res4'])

        # Sequential Once

        zero_loss_di_p = F.normalize(base_di) * F.normalize(features['res4'])
        zero_loss_di_n = F.normalize(base_ds) * F.normalize(features['res4'])

        zero_loss_di_p = torch.exp(torch.sum(zero_loss_di_p, dim=1))
        zero_loss_di_n = torch.exp(torch.sum(zero_loss_di_n, dim=1))

        log_result_di = torch.log(zero_loss_di_p / (zero_loss_di_p + zero_loss_di_n)) * -1.0
        zero_loss_di = torch.mean(log_result_di)

        if np.random.rand(1) > self.apply_aug:
            #             ipdb.set_trace()
            #             oids = np.random.choice(np.arange(len(self.offsets)),b)
            #             change = torch.cat([self.offsets[oid:oid+1].cuda().mean(dim=(2,3),keepdims=True) for oid in oids ],0)
            #             features['res4']=features['res4']+ change

            oids = np.random.choice(np.arange(len(self.stylemean)), b)
            mean = torch.cat([self.stylemean[oid:oid + 1].cuda().mean(dim=(2, 3), keepdims=True) for oid in oids], 0)
            std = torch.cat([self.stylestd[oid:oid + 1].cuda().mean(dim=(2, 3), keepdims=True) for oid in oids], 0)
            base_ds = base_ds * std.expand(base_ds.size()) + mean.expand(
                base_ds.size())
        #             features['res4']= 0.5*features['res4']+ 0.5*(features['res4']*std.expand(features['res4'].size()) + mean.expand(features['res4'].size()))
        #         ipdb.set_trace()

        base_di = self.pro(base_di)
        basef = torch.cat((base_di, base_ds), dim=1)
        features['res4'] = self.conv_out(basef) + features['res4']
        features['res4'] = self.pro(features['res4'])

        if self.proposal_generator is not None:
            if self.training:
                logits, proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            else:
                logits, proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        try:
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, None, self.backbone)
        except Exception as e:
            print(e)
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, None)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
                with torch.no_grad():
                    ogimage = batched_inputs[0]['image']
                    ogimage = convert_image_to_rgb(ogimage.permute(1, 2, 0), self.input_format)
                    o_pred = Visualizer(ogimage, None).overlay_instances().get_image()

                    vis_img = o_pred.transpose(2, 0, 1)
                    storage.put_image('og-tfimage', vis_img)

        loss_di = {'zero_loss_di': zero_loss_di}
        losses = {}
        losses.update(loss_di)
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def opt_offsets(self, batched_inputs):
        #         ipdb.set_trace()

        crops_clip = None
        if 'randomcrops' in batched_inputs[0]:
            rcrops = [x['randomcrops'] for x in batched_inputs]
            rcrops = torch.cat(rcrops, 0)
            crops_clip = rcrops.flip(1) / 255
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
            crops_clip = T.functional.normalize(crops_clip, mean, std)
            crops_clip = crops_clip.cuda()

        with torch.no_grad():
            features = self.backbone(crops_clip)

        losses = {}
        total_dist = 0
        total_reg = 0
        total_chgn = 0
        for i, val in enumerate(self.domain_tk.items()):
            name, dtk = val
            if name == 'day':
                continue
            with torch.no_grad():
                #                 ipdb.set_trace()
                # print(self.backbone.forward_res5(features['res4']))
                #                 wo_aug_im_embed = self.backbone.attention_global_pool(self.backbone.forward_res5(features['res4']))
                #                 wo_aug_im_embed  = wo_aug_im_embed/wo_aug_im_embed.norm(dim=-1,keepdim=True)

                #                 day_text_embed = self.roi_heads.box_predictor.cls_score.model.encode_text(self.domain_tk['day'].cuda()) #day
                #                 day_text_embed = day_text_embed/day_text_embed.norm(dim=-1,keepdim=True)
                new_text_embed = self.roi_heads.box_predictor.cls_score.model.encode_text(dtk.cuda())  # new_d
                new_text_embed = new_text_embed / new_text_embed.norm(dim=-1, keepdim=True)
            #                 text_off = (new_text_embed - day_text_embed)
            #                 text_off = text_off/text_off.norm(dim=-1,keepdim=True)

            #                 wo_aug_im_tsl = wo_aug_im_embed + text_off
            #                 wo_aug_im_tsl = wo_aug_im_tsl/wo_aug_im_tsl.norm(dim=-1,keepdim=True)
            #                 wo_aug_im_tsl = wo_aug_im_tsl.unsqueeze(1).permute(0,2,1)

            #             aug_feat = features['res4'].detach()+self.offsets[i-1:i]
            #             aug_feat = 0.5*features['res4']+0.5*(features['res4']*self.stylestd[i-1].expand(features['res4'].size()) + self.stylemean[i-1].expand(features['res4'].size()))

            aug_feat = features['res4'] * self.stylestd[i - 1].expand(features['res4'].size()) + self.stylemean[
                i - 1].expand(features['res4'].size())
            x = self.backbone.forward_res5(aug_feat)
            im_embed = self.backbone.attention_global_pool(x)

            im_embed = im_embed / im_embed.norm(dim=-1, keepdim=True)

            #             cos_dist = 1 - im_embed.unsqueeze(1).bmm(wo_aug_im_tsl)
            cos_dist = 1 - torch.mm(im_embed, new_text_embed.T)

            dist_loss = cos_dist.mean()

            #             l1loss = torch.nn.functional.l1_loss(im_embed,wo_aug_im_embed)

            total_dist += dist_loss
        #             total_reg += l1loss

        losses.update(
            {
                f'cos_dist_loss_{name}': total_dist / len(self.domain_tk),
                #             f'reg_loss_{name}': total_reg/len(self.domain_tk)
            }
        )

        return losses


@META_ARCH_REGISTRY.register()
class ClipRCNNWithClipBackboneWithOffsetGenTrainableVOC(ClipRCNNWithClipBackbone):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        domain_text = {'real': 'a realistic image'}

        domain_text.update({str(0): 'an image in the comics style'})
        domain_text.update({str(1): 'an image in the painting style'})
        domain_text.update({str(2): 'an image in the cartoon style'})
        domain_text.update({str(3): 'an image in the digital-art style'})
        domain_text.update({str(4): 'an image in the sketch style'})
        domain_text.update({str(5): 'an image in the watercolor painting style'})
        domain_text.update({str(6): 'an image in the oil painting style'})
        # self.offsets = nn.Parameter(offsets)
        self.offsets = nn.Parameter(torch.zeros(len(domain_text) - 1, 1024, 14, 14))  # skip day

        import clip
        self.domain_tk = dict([(k, clip.tokenize(t)) for k, t in domain_text.items()])
        self.apply_aug = cfg.AUG_PROB

    def forward(self, batched_inputs):

        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        b = images.tensor.shape[0]  # batchsize

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        features = self.backbone(images.tensor)

        if np.random.rand(1) > self.apply_aug:
            oids = np.random.choice(np.arange(len(self.offsets)), b)
            change = torch.cat([self.offsets[oid:oid + 1].cuda().mean(dim=(2, 3), keepdims=True) for oid in oids], 0)
            features['res4'] = features['res4'] + change

        if self.proposal_generator is not None:
            if self.training:
                logits, proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            else:
                logits, proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        try:
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, None, self.backbone)
        except Exception as e:
            print(e)
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, None)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
                with torch.no_grad():
                    ogimage = batched_inputs[0]['image']
                    ogimage = convert_image_to_rgb(ogimage.permute(1, 2, 0), self.input_format)
                    o_pred = Visualizer(ogimage, None).overlay_instances().get_image()

                    vis_img = o_pred.transpose(2, 0, 1)
                    storage.put_image('og-tfimage', vis_img)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def opt_offsets(self, batched_inputs):

        crops_clip = None
        if 'randomcrops' in batched_inputs[0]:
            rcrops = [x['randomcrops'] for x in batched_inputs]
            rcrops = torch.cat(rcrops, 0)
            crops_clip = rcrops.flip(1) / 255
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
            crops_clip = T.functional.normalize(crops_clip, mean, std)
            crops_clip = crops_clip.cuda()

        with torch.no_grad():
            features = self.backbone(crops_clip)

        losses = {}
        total_dist = 0
        total_reg = 0
        total_chgn = 0
        for i, val in enumerate(self.domain_tk.items()):
            name, dtk = val
            if name == 'real':
                continue
            with torch.no_grad():

                # print(self.backbone.forward_res5(features['res4']))
                wo_aug_im_embed = self.backbone.attention_global_pool(self.backbone.forward_res5(features['res4']))
                wo_aug_im_embed = wo_aug_im_embed / wo_aug_im_embed.norm(dim=-1, keepdim=True)

                day_text_embed = self.roi_heads.box_predictor.cls_score.model.encode_text(
                    self.domain_tk['real'].cuda())  # day
                day_text_embed = day_text_embed / day_text_embed.norm(dim=-1, keepdim=True)
                new_text_embed = self.roi_heads.box_predictor.cls_score.model.encode_text(dtk.cuda())  # new_d
                new_text_embed = new_text_embed / new_text_embed.norm(dim=-1, keepdim=True)
                text_off = (new_text_embed - day_text_embed)
                text_off = text_off / text_off.norm(dim=-1, keepdim=True)

                wo_aug_im_tsl = wo_aug_im_embed + text_off
                wo_aug_im_tsl = wo_aug_im_tsl / wo_aug_im_tsl.norm(dim=-1, keepdim=True)
                wo_aug_im_tsl = wo_aug_im_tsl.unsqueeze(1).permute(0, 2, 1)

            aug_feat = features['res4'].detach() + self.offsets[i - 1:i]

            x = self.backbone.forward_res5(aug_feat)
            im_embed = self.backbone.attention_global_pool(x)

            im_embed = im_embed / im_embed.norm(dim=-1, keepdim=True)

            cos_dist = 1 - im_embed.unsqueeze(1).bmm(wo_aug_im_tsl)

            dist_loss = cos_dist.mean()

            l1loss = torch.nn.functional.l1_loss(im_embed, wo_aug_im_embed)

            total_dist += dist_loss
            total_reg += l1loss

        losses.update({f'cos_dist_loss_{name}': total_dist / len(self.domain_tk),
                       f'reg_loss_{name}': total_reg / len(self.domain_tk)})
        import pdb;
        pdb.set_trace()
        return losses







