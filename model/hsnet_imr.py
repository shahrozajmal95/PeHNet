r""" Hypercorrelation Squeeze Network """
import pdb
from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg

from .base.feature import extract_feat_vgg, extract_feat_res
from .base.correlation import Correlation
from .learner import HPNLearner


class Prototype_Adaptive_Module(nn.Module):
    """ 
    Prototype adaptive module for query-support feature enhancement.
    produce enhanced features for Few-shot segmentation task.
    Args:
          dim (int): Number of input channels.
          input_resolution (tuple): input_resolution for high and weight
          hidden_ratio(float): the compressed ratio of feature dim
          class_num(int): Number of adaptive class
          momentum(float): the momentum updating ratio for prototype
    """
    def __init__(self, 
                 dim, 
                 input_resolution, 
                 hidden_ratio=16,
                 drop=0.,
                 class_num=1,
                 momentum=0.9):
          super().__init__()
          self.dim = dim
          self.dim_low = dim // hidden_ratio
          self.input_resolution = input_resolution
          self.act = nn.ReLU()
          self.linears_down = nn.Linear(dim, self.dim_low)
          self.linears_up = nn.Linear(self.dim_low, dim)

          self.register_buffer("prototype", torch.randn(dim, class_num).requires_grad_(False))
          self.prototype = nn.functional.normalize(self.prototype, dim=0)

          self.drop = drop
          self.proj_drop = nn.Dropout(drop)
          self.momentum = momentum
          self.act_enhance = nn.ReLU6()

    def forward(self, x, s_f, s_y, class_idx=None):
          """
          Args:
          x: torch.Tensor
              [B , h * w, dim], query
          s_f: torch.Tensor
              [B * S , h * w, dim], support
          s_y: torch.Tensor
              [B, S, H, W],    support mask one-shot
          class_id: list
              len(class_id) = B
          Outputs:
          registered_feas: torch.Tensor
              [B * (S + 1), h * w, dim], injected features
          """
          B, N, D = x.shape
          s_f = s_f.reshape(B, -1, N, D)
          S = s_f.size(1)

          s_y = F.interpolate(s_y, (self.input_resolution[0], self.input_resolution[1]), mode='nearest')
          sup_mask_fg = (s_y == 1).float().reshape(B, S, -1).reshape(B, -1).unsqueeze(-1) #[B, S * N, 1]
          semantic_prototype, sign_fore_per_batch = self.extract_semantic_prototype(s_f, sup_mask_fg)

          if class_idx is not None and self.training:
              new_semantic_prototype = self.updata_prototype_bank(semantic_prototype, class_idx, sign_fore_per_batch)
          else:
              new_semantic_prototype = self.select_prototype_bank(semantic_prototype, self.prototype)

          enhanced_feat_q = self.enhanced_feature(x.unsqueeze(1), new_semantic_prototype, sign_fore_per_batch)
          enhanced_feat_sup = self.enhanced_feature(s_f, new_semantic_prototype, sign_fore_per_batch)

          registered_feas = torch.cat([enhanced_feat_sup.reshape(-1, N, D), enhanced_feat_q.squeeze(1)], dim=0)
          registered_feas = self.proj_drop(self.linears_up(self.act(self.linears_down(registered_feas))))
          return registered_feas


    def extract_semantic_prototype(self, s_f, s_y):
        """
        extract temporary class prototype according to support features and masks
        input:
          s_f: torch.Tensor
              [B, S, N, D], support features
          s_y: torch.Tensor
              [B, S * N, 1], support masks
        output:
          semantic_prototype: torch.Tensor
              [B, D], temporary prototypes
          sign_fore_per_batch: torch.Tensor
              [B], the signal of whether including foreground region in this image
        """
        B, S, N, D = s_f.shape
        num_fore_per_batch = torch.count_nonzero(s_y.reshape(B, -1), dim=1)
        s_y = s_y.repeat(1, 1, D)
        semantic_prototype = s_y * s_f.reshape(B, -1, D)
        semantic_prototype = semantic_prototype.mean(1) * (N * S) / (num_fore_per_batch.unsqueeze(1)+1e-4)
        one = torch.ones_like(num_fore_per_batch).cuda()
        sign_fore_per_batch =  torch.where(num_fore_per_batch > 0.5, one, num_fore_per_batch)
        return semantic_prototype, sign_fore_per_batch

    def updata_prototype_bank(self, semantic_prototype, class_idx, sign_fore_per_batch):
        """
        updata prototype in class prototype bank during traning
        input:
          semantic_prototype: torch.Tensor
              [B, D]
          class_id: list
              len(class_id) = B
          sign_fore_per_batch: torch.Tensor
              [B], the signal of whether including foreground region in this image
        output:
          new_semantic_prototype: torch.Tensor
              [B, D], the updated prototypes for feature enhancement
        """
        B, D = semantic_prototype.shape
        self.prototype = nn.functional.normalize(self.prototype, dim=0)
        semantic_prototype = nn.functional.normalize(semantic_prototype, dim=1)
        new_semantic_prototype_list = []
        for i in range(B):
             semantic_prototype_per = semantic_prototype[i,: ]
             class_idx_per = class_idx[i]
             if sign_fore_per_batch[i] == 1:
                  new_semantic_prototype_per = self.prototype[:, class_idx_per] * self.momentum + (1 - self.momentum) * semantic_prototype_per
                  self.prototype[:, class_idx_per] = new_semantic_prototype_per
             else:
                  new_semantic_prototype_per = self.prototype[:, class_idx_per]
             new_semantic_prototype_list.append(new_semantic_prototype_per)
        new_semantic_prototype = torch.stack(new_semantic_prototype_list, dim=0)
        return new_semantic_prototype

    def select_prototype_bank(self, semantic_prototype, prototype_bank):
        """
        select prototypes in class prototype bank during testing
        input:
          semantic_prototype: torch.Tensor
              shape = [B, D]
          prototype_bank: torch.Tensor
              shape = [D, class_num]
        output:
          new_semantic_prototype: torch.Tensor
              [B, D], the prototypes for feature enhancement
        """
        B, D = semantic_prototype.shape
        prototype_bank = nn.functional.normalize(prototype_bank, dim=0)
        semantic_prototype = nn.functional.normalize(semantic_prototype, dim=1)
        similar_matrix = semantic_prototype @ prototype_bank  # [B, class_num]
        idx = similar_matrix.argmax(1)

        new_semantic_prototype_list = []
        for i in range(B):
            new_semantic_prototype_per = prototype_bank[:, idx[i]]
            new_semantic_prototype_list.append(new_semantic_prototype_per)
        new_semantic_prototype = torch.stack(new_semantic_prototype_list, dim=0)
        return new_semantic_prototype

    def enhanced_feature(self, feature, new_semantic_prototype, sign_fore_per_batch):
        """
          Input:
              feature: torch.Tensor
                  [B, S, N, D]
              new_semantic_prototype: torch.Tensor
                  [B, D]
          Outputs:
              enhanced_feature: torch.Tensor
                  [B, S, N, D]
          """
        B, D = new_semantic_prototype.shape
        feature_sim = nn.functional.normalize(feature, p=2, dim=-1)
        new_semantic_prototype = nn.functional.normalize(new_semantic_prototype, p=2, dim=1)
        similarity_matrix_list = []
        for i in range(B):
            feature_sim_per = feature_sim[i,:, :, :]
            new_semantic_prototype_er = new_semantic_prototype[i, :]
            similarity_matrix_per = feature_sim_per @ new_semantic_prototype_er
            similarity_matrix_list.append(similarity_matrix_per)
        similarity_matrix = torch.stack(similarity_matrix_list, dim=0)

        similarity_matrix = (similarity_matrix * self.dim ** 0.5) *  sign_fore_per_batch.unsqueeze(-1).unsqueeze(-1)

        enhanced_feature = self.act_enhance(similarity_matrix).unsqueeze(-1).repeat(1, 1, 1, D) * feature + feature

        return enhanced_feature


class HypercorrSqueezeNetwork_imr(nn.Module):
    def __init__(self, backbone, use_original_imgsize):
        super(HypercorrSqueezeNetwork_imr, self).__init__()

        # 1. Backbone network initialization
        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize
        if backbone == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=False)
            ckpt = torch.load('Datasets_HSN/Pretrain/vgg16-397923af.pth')
            self.backbone.load_state_dict(ckpt)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]
        elif backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=False)
            ckpt = torch.load('Datasets_HSN/resnet50-19c8e357.pth')
            self.backbone.load_state_dict(ckpt)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
            self.conv1024_512 = nn.Conv2d(1024, 512, kernel_size=1)

        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.backbone.eval()
        self.hpn_learner = HPNLearner(list(reversed(nbottlenecks[-3:])))
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # IMR
        self.state = nn.Parameter(torch.zeros([1, 128, 50, 50]))
        self.convz0 = nn.Conv2d(769, 512, kernel_size=1, padding=0)
        self.convz1 = nn.Conv2d(256, 64, kernel_size=3, padding=1, groups=8, dilation=1)
        self.convz2 = nn.Conv2d(256, 64, kernel_size=3, padding=1, groups=8, dilation=1)

        self.convr0 = nn.Conv2d(769, 512, kernel_size=1, padding=0)
        self.convr1 = nn.Conv2d(256, 64, kernel_size=3, padding=1, groups=8, dilation=1)
        self.convr2 = nn.Conv2d(256, 64, kernel_size=3, padding=1, groups=8, dilation=1)

        self.convh0 = nn.Conv2d(769, 512, kernel_size=1, padding=0)
        self.convh1 = nn.Conv2d(256, 64, kernel_size=3, padding=1, groups=8, dilation=1)
        self.convh2 = nn.Conv2d(256, 64, kernel_size=3, padding=1, groups=8, dilation=1)

        # copied from hsnet-learner
        outch1, outch2, outch3 = 16, 64, 128
        self.decoder1 = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

        self.decoder2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch2, 2, (3, 3), padding=(1, 1), bias=True))

        self.res = nn.Sequential(nn.Conv2d(3, 10, kernel_size=1),
                                 nn.GELU(),
                                 nn.Conv2d(10, 2, kernel_size=1))

        # PAM Integration - added to existing IMR_Hsnet
        # Use appropriate dimension based on backbone
        pam_dim = 512 if backbone == 'vgg16' else 1024
        self.pam = Prototype_Adaptive_Module(
            dim=pam_dim,
            input_resolution=(50, 50),
            hidden_ratio=8,
            class_num=20,
            momentum=0.95,
            drop=0.1
        )

    def forward(self, query_img, support_img, support_cam, query_cam,
                query_mask=None, support_mask=None, stage=2, w='same', class_idx=None):
        with torch.no_grad():
            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = self.extract_feats(
                support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)

            # extracting feature
            if len(query_feats) == 7:
                isvgg = True  # VGG
                q_mid_feat = F.interpolate(query_feats[3] + query_feats[4] + query_feats[5],
                                           (50, 50), mode='bilinear', align_corners=True)
                s_mid_feat = F.interpolate(support_feats[3] + support_feats[4] + support_feats[5],
                                           (50, 50), mode='bilinear', align_corners=True)
            else:
                isvgg = False  # R50
                q_mid_feat = F.interpolate(
                    query_feats[4] + query_feats[5] + query_feats[6] + query_feats[7] + query_feats[8] + query_feats[9],
                    (50, 50), mode='bilinear', align_corners=True)

                s_mid_feat = F.interpolate(
                    support_feats[4] + support_feats[5] + support_feats[6] + support_feats[7] + support_feats[8] +
                    support_feats[9],
                    (50, 50), mode='bilinear', align_corners=True)

            # PAM Feature Enhancement - added to existing flow
            # Use original features for PAM (before dimension reduction)
            if isvgg:
                # VGG features are already 512 channels
                q_mid_feat_pam = q_mid_feat
                s_mid_feat_pam = s_mid_feat
                pam_dim = 512
            else:
                # ResNet features are 1024 channels before conv1024_512
                q_mid_feat_pam = q_mid_feat  # This is still 1024 channels
                s_mid_feat_pam = s_mid_feat  # This is still 1024 channels
                pam_dim = 1024

            B, C, H, W = q_mid_feat_pam.shape
            q_mid_flat = q_mid_feat_pam.view(B, C, -1).transpose(1, 2)  # [B, 2500, C]
            s_mid_flat = s_mid_feat_pam.view(B, C, -1).transpose(1, 2)  # [B, 2500, C]

            support_mask_pam = F.interpolate(support_cam.unsqueeze(1), (H, W), mode='nearest')

            enhanced_feats = self.pam(q_mid_flat, s_mid_flat, support_mask_pam, class_idx)

            enhanced_sup, enhanced_qry = torch.split(enhanced_feats, [B, B], dim=0)

            # Convert back to spatial format
            if isvgg:
                q_mid_feat_enhanced = enhanced_qry.transpose(1, 2).view(B, C, H, W)
                s_mid_feat_enhanced = enhanced_sup.transpose(1, 2).view(B, C, H, W)
            else:
                # For ResNet, we need to reduce dimensions after PAM
                q_mid_feat_enhanced = enhanced_qry.transpose(1, 2).view(B, C, H, W)
                s_mid_feat_enhanced = enhanced_sup.transpose(1, 2).view(B, C, H, W)
                # Apply conv1024_512 to get 512 channels
                q_mid_feat_enhanced = self.conv1024_512(q_mid_feat_enhanced)
                s_mid_feat_enhanced = self.conv1024_512(s_mid_feat_enhanced)

            # Use enhanced features for the rest of the pipeline
            q_mid_feat = q_mid_feat_enhanced
            s_mid_feat = s_mid_feat_enhanced

            query_feats_masked = self.mask_feature(query_feats, support_cam.clone())
            support_feats_masked = self.mask_feature(support_feats, query_cam.clone())

            corr_query = Correlation.multilayer_correlation(query_feats, support_feats_masked, self.stack_ids)
            corr_support = Correlation.multilayer_correlation(support_feats, query_feats_masked, self.stack_ids)

            query_cam = query_cam.unsqueeze(1)
            support_cam = support_cam.unsqueeze(1)

        # Remove the duplicate conv1024_512 calls since we already applied them after PAM
        # if not isvgg:
        #     # make feat dim in R50 same as VGG
        #     q_mid_feat = self.conv1024_512(q_mid_feat)
        #     s_mid_feat = self.conv1024_512(s_mid_feat)

        bsz = query_img.shape[0]
        state_query = self.state.expand(bsz, -1, -1, -1)
        state_support = self.state.expand(bsz, -1, -1, -1)

        losses = 0
        for ss in range(stage):
            # query
            after4d_query = self.hpn_learner.forward_conv4d(corr_query)
            imr_x_query = torch.cat([query_cam, after4d_query, q_mid_feat, state_query], dim=1)

            imr_x_query_z = self.convz0(imr_x_query)
            imr_z_query1 = self.convz1(imr_x_query_z[:, :256])
            imr_z_query2 = self.convz2(imr_x_query_z[:, 256:])
            imr_z_query = torch.sigmoid(torch.cat([imr_z_query1, imr_z_query2], dim=1))

            imr_x_query_r = self.convr0(imr_x_query)
            imr_r_query1 = self.convr1(imr_x_query_r[:, :256])
            imr_r_query2 = self.convr2(imr_x_query_r[:, 256:])
            imr_r_query = torch.sigmoid(torch.cat([imr_r_query1, imr_r_query2], dim=1))

            imr_x_query_h = self.convh0(
                torch.cat([query_cam, after4d_query, q_mid_feat, imr_r_query * state_query], dim=1))
            imr_h_query1 = self.convh1(imr_x_query_h[:, :256])
            imr_h_query2 = self.convh2(imr_x_query_h[:, 256:])
            imr_h_query = torch.cat([imr_h_query1, imr_h_query2], dim=1)

            state_new_query = torch.tanh(imr_h_query)
            state_query = (1 - imr_z_query) * state_query + imr_z_query * state_new_query

            # support
            after4d_support = self.hpn_learner.forward_conv4d(corr_support)
            imr_x_support = torch.cat([support_cam, after4d_support, s_mid_feat, state_support], dim=1)

            imr_x_support_z = self.convz0(imr_x_support)
            imr_z_support1 = self.convz1(imr_x_support_z[:, :256])
            imr_z_support2 = self.convz2(imr_x_support_z[:, 256:])
            imr_z_support = torch.sigmoid(torch.cat([imr_z_support1, imr_z_support2], dim=1))

            imr_x_support_r = self.convr0(imr_x_support)
            imr_r_support1 = self.convr1(imr_x_support_r[:, :256])
            imr_r_support2 = self.convr2(imr_x_support_r[:, 256:])
            imr_r_support = torch.sigmoid(torch.cat([imr_r_support1, imr_r_support2], dim=1))

            imr_x_support_h = self.convh0(
                torch.cat([support_cam, after4d_support, s_mid_feat, imr_r_support * state_support], dim=1))
            imr_h_support1 = self.convh1(imr_x_support_h[:, :256])
            imr_h_support2 = self.convh2(imr_x_support_h[:, 256:])
            imr_h_support = torch.cat([imr_h_support1, imr_h_support2], dim=1)

            state_new_support = torch.tanh(imr_h_support)
            state_support = (1 - imr_z_support) * state_support + imr_z_support * state_new_support

            # decoder
            hypercorr_decoded_s = self.decoder1(state_support + after4d_support)
            upsample_size = (hypercorr_decoded_s.size(-1) * 2,) * 2
            hypercorr_decoded_s = F.interpolate(hypercorr_decoded_s, upsample_size, mode='bilinear', align_corners=True)
            logit_mask_support = self.decoder2(hypercorr_decoded_s)

            hypercorr_decoded_q = self.decoder1(state_query + after4d_query)
            upsample_size = (hypercorr_decoded_q.size(-1) * 2,) * 2
            hypercorr_decoded_q = F.interpolate(hypercorr_decoded_q, upsample_size, mode='bilinear', align_corners=True)
            logit_mask_query = self.decoder2(hypercorr_decoded_q)

            logit_mask_support = self.res(
                torch.cat(
                    [logit_mask_support, F.interpolate(support_cam, (100, 100), mode='bilinear', align_corners=True)],
                    dim=1))
            logit_mask_query = self.res(
                torch.cat([logit_mask_query, F.interpolate(query_cam, (100, 100), mode='bilinear', align_corners=True)],
                          dim=1))

            # loss
            if query_mask is not None:  # for training
                if not self.use_original_imgsize:
                    logit_mask_query_temp = F.interpolate(logit_mask_query, support_img.size()[2:], mode='bilinear',
                                                          align_corners=True)
                    logit_mask_support_temp = F.interpolate(logit_mask_support, support_img.size()[2:], mode='bilinear',
                                                            align_corners=True)
                loss_q_stage = self.compute_objective(logit_mask_query_temp, query_mask)
                loss_s_stage = self.compute_objective(logit_mask_support_temp, support_mask)
                losses = losses + loss_q_stage + loss_s_stage

            if ss != stage - 1:
                support_cam = logit_mask_support.softmax(dim=1)[:, 1]
                query_cam = logit_mask_query.softmax(dim=1)[:, 1]
                query_feats_masked = self.mask_feature(query_feats, query_cam)
                support_feats_masked = self.mask_feature(support_feats, support_cam)
                corr_query = Correlation.multilayer_correlation(query_feats, support_feats_masked, self.stack_ids)
                corr_support = Correlation.multilayer_correlation(support_feats, query_feats_masked, self.stack_ids)

                query_cam = F.interpolate(query_cam.unsqueeze(1), (50, 50), mode='bilinear', align_corners=True)
                support_cam = F.interpolate(support_cam.unsqueeze(1), (50, 50), mode='bilinear', align_corners=True)

        if query_mask is not None:
            return logit_mask_query_temp, logit_mask_support_temp, losses
        else:
            # test
            if not self.use_original_imgsize:
                logit_mask_query = F.interpolate(
                    logit_mask_query, support_img.size()[2:], mode='bilinear', align_corners=True)
                logit_mask_support = F.interpolate(
                    logit_mask_support, support_img.size()[2:], mode='bilinear', align_corners=True)
            return logit_mask_query, logit_mask_support

    def mask_feature(self, features, support_mask):
        for idx, feature in enumerate(features):
            mask = F.interpolate(
                support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear', align_corners=True)
            features[idx] = features[idx] * mask
        return features

    def predict_mask_nshot(self, batch, nshot, stage):
        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        for s_idx in range(nshot):
            logit_mask, logit_mask_s = self(query_img=batch['query_img'],
                                            support_img=batch['support_imgs'][:, s_idx],
                                            support_cam=batch['support_cams'][:, s_idx],
                                            query_cam=batch['query_cam'], stage=stage)
            if self.use_original_imgsize:
                org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
                logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)

            logit_mask_agg += logit_mask.argmax(dim=1).clone()
            if nshot == 1:
                return logit_mask_agg

        # Average & quantize predictions given threshold (=0.5)
        bsz = logit_mask_agg.size(0)
        max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_agg.float() / max_vote
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1

        return pred_mask

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()
        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging