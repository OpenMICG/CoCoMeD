# Project:
#   VQA
# Description:
#   Model classes definition
# Author: 
#   Sergio Tascon-Morales

import torch
import torch.nn as nn
from torchvision import transforms
from .components.attention import apply_attention
from .components import image, text, attention, fusion, classification


class VQA_Base(nn.Module):
    # base class for simple VQA model
    def __init__(self, config, vocab_words, vocab_answers):
        super().__init__()
        self.visual_feature_size = config['visual_feature_size']#图形维度2048
        self.question_feature_size = config['question_feature_size']#问题维度1024
        self.pre_visual = config['pre_extracted_visual_feat']
        self.use_attention = config['attention']
        self.number_of_glimpses = config['number_of_glimpses']
        self.MFB_hidden_size=config["MFB_hidden_size"]
        self.MFB_out_size=config["MFB_out_size"]
        self.dropout_attention = config['attention_dropout']  # 0.25
        self.visual_size_before_fusion = self.visual_feature_size # 2048 by default, changes if attention
        self.drop = nn.Dropout(self.dropout_attention)#设置 MFB的dropout
        # Create modules for the model

        # if necesary, create module for offline visual feature extraction
        if not self.pre_visual:
            self.image = image.get_visual_feature_extractor(config)

        # create module for text feature extraction
        self.text = text.get_text_feature_extractor(config, vocab_words)

        # if necessary, create attention module
        if self.use_attention:
            if config["attention_special"]=="AttDMN":
                self.visual_size_before_fusion = 512 #self.number_of_glimpses*self.visual_feature_size
                self.attention_mechanism = attention.get_attention_mechanism(config, config["attention_special"])
            else:
                self.visual_size_before_fusion = self.number_of_glimpses*self.visual_feature_size
                self.attention_mechanism = attention.get_attention_mechanism(config, config["attention_special"])
        else:
            self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))

        # create  MFB multimodal fusion module
        self.MFB_lin_visual = nn.Linear(self.visual_size_before_fusion, self.MFB_hidden_size)
        self.MFB_lin_question = nn.Linear(self.question_feature_size, self.MFB_hidden_size)
        self.MFB_lin_output = nn.Linear(self.MFB_hidden_size, self.MFB_out_size)
        # self.MFB_lin_down = nn.Linear(self.MFB_out_size, self.question_feature_size)

        # create multimodal fusion module
        self.fuser, fused_size = fusion.get_fuser(config['fusion'], self.visual_size_before_fusion, self.question_feature_size)

        # create classifier
        #self.classifer = classification.get_classfier(self.MFB_out_size, config)
        #nomfb
        self.classifer = classification.get_classfier(self.MFB_out_size, config)

    def forward(self, v, q):
        # if required, extract visual features from visual input 
        if not self.pre_visual:
            v = self.image(v) # [B, 2048, 14, 14]

        # l2 norm
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
        
        # extract text features
        q = self.text(q)

        # if required, apply attention
        if self.use_attention:
            v = self.attention_mechanism(v, q) # should apply attention too
        else:
            v = self.avgpool(v).squeeze_() # [B, 2048]

        # apply multimodal fusion
        fused = self.fuser(v, q)
        #[B,514,14,14]
        # apply MLP
        x = self.classifer(fused)

        return x
        

class VQARS_1(VQA_Base):
    # First model for region-based VQA, with single mask. Input image is multiplied with the mask to produced a masked version which is sent to the model as normal
    def __init__(self, config, vocab_words, vocab_answers):
        # call mom
        super().__init__(config, vocab_words, vocab_answers)

    # override forward method to accept mask
    def forward(self, v, q):
        # if required, extract visual features from visual input
        # print(v.shape)
        # print("m="+str(m.shape))
        # if self.pre_visual:
        #     raise ValueError("This model does not allow pre-extracted features")
        # else:
        #     v = self.image(torch.mul(v, q)) # [B, 2048, 14, 14]   MASK IS INCLUDED HERE
        v = self.image(v)
        # print(v.shape)
        # extract text features
        q = self.text(q)
        # if required, apply attention
        if self.use_attention:
            v = self.attention_mechanism(v, q) # should apply attention too
        else:
            v = self.avgpool(v).squeeze_(dim=-1).squeeze_(dim=-1) # [B, 2048]
        # apply multimodal fusion
        #fused = self.fuser(v, q)
        # print(v.shape)

        #11/13use dropout in MFB
        v = self.MFB_lin_visual(v)#[batch,4096]
        q = self.MFB_lin_question(q)#[batch,4096]
        fused = self.drop(self.fuser(v, q))#[batch,4096]
        fused = fused / (fused.norm(p=2, dim=1, keepdim=True).expand_as(fused) + 1e-8)
        fused = self.MFB_lin_output(fused)
        # fused = self.MFB_lin_down(fused)#[1024]
        # print(fused.shape)
        # apply MLP
        x = self.classifer(fused)

        return x

class VQANOMFB(VQA_Base):
    # First model for region-based VQA, with single mask. Input image is multiplied with the mask to produced a masked version which is sent to the model as normal
    def __init__(self, config, vocab_words, vocab_answers):
        # call mom
        super().__init__(config, vocab_words, vocab_answers)

    # override forward method to accept mask
    def forward(self, v, q, m):
        # if required, extract visual features from visual input
        if self.pre_visual:
            raise ValueError("This model does not allow pre-extracted features")
        else:
            v = self.image(torch.mul(v, m)) # [B, 2048, 14, 14]   MASK IS INCLUDED HERE

        # extract text features
        q = self.text(q)

        # if required, apply attention
        if self.use_attention:
            v = self.attention_mechanism(v, q) # should apply attention too
        else:
            v = self.avgpool(v).squeeze_(dim=-1).squeeze_(dim=-1) # [B, 2048]

        # apply multimodal fusion
        fused = self.fuser(v, q)

        # apply MLP
        x = self.classifer(fused)

        return x

class SQuINT(VQARS_1):
    # SQuINTed version of model 1. See Selvaraju et al. 2020 (CVPR). Re-implemented for comparison purposes, since their code is not open-source.
    def __init__(self, config, vocab_words, vocab_answers):
        super().__init__(config, vocab_words, vocab_answers)

    # override forward so that attention maps are returned too
    def forward(self, v, q, m):
        # if required, extract visual features from visual input 
        if self.pre_visual:
            raise ValueError("This model does not allow pre-extracted features")
        else:
            v = self.image(torch.mul(v, m)) # [B, 2048, 14, 14]   MASK IS INCLUDED HERE

        # extract text features
        q = self.text(q)

        # if required, apply attention
        if self.use_attention:
            v, maps = self.attention_mechanism(v, q, return_maps=True) # should apply attention too
        else:
            raise ValueError("Attention is necessary for SQuINT")

        # apply multimodal fusion
        fused = self.fuser(v, q)

        # apply MLP
        x = self.classifer(fused)

        return x, maps

