# Project:
#   VQA
# Description:
#   Attention related functions and classes
# Author: 
#   Sergio Tascon-Morales
from torch.autograd import Variable

from . import fusion
from . import utils
from torch import nn
import torch.nn.functional as F
import torch

def get_attention_mechanism(config, special=None):
    # get attention parameters
    visual_features_size = config['visual_feature_size']#2048
    question_feature_size = config['question_feature_size']#1024
    attention_middle_size = config['attention_middle_size']#512
    number_of_glimpses = config['number_of_glimpses']#2
    attention_fusion = config['attention_fusion']#mul
    dropout_attention = config['attention_dropout']#0.25
    if special is None: # Normal attention mechanism
        attention = AttentionMechanismBase(visual_features_size, question_feature_size, attention_middle_size, number_of_glimpses, attention_fusion, drop=dropout_attention)
    elif special == 'Att1': # special attention mechanism 1
        attention = AttentionMechanism_1(visual_features_size, question_feature_size, attention_middle_size, number_of_glimpses, attention_fusion, drop=dropout_attention)
    elif special == 'Att2':
        attention = AttentionMechanism_2(visual_features_size, question_feature_size, attention_middle_size, number_of_glimpses, attention_fusion, drop=dropout_attention)
    elif special == 'Att3':
        attention = AttentionMechanism_3(visual_features_size, question_feature_size, attention_middle_size, number_of_glimpses, attention_fusion, drop=dropout_attention)
    elif special == 'AttDMN':
        attention_num_layers_DMN = config['attention_num_layers_DMN']  # DMN层数
        attention = AttentionDMNtied(visual_features_size, question_feature_size, attention_middle_size, attention_num_layers_DMN, attention_fusion, config, drop=dropout_attention)
    return attention


def apply_attention(visual_features, attention):
    # visual features has size [b, m, k, k]
    # attention has size [b, glimpses, k, k]
    b, m = visual_features.size()[:2] # batch size, number of feature maps[B,512],
    glimpses = attention.size(1)#[2]
    visual_features = visual_features.view(b, 1, m, -1) # vectorize feature maps [b, 1, m, k*k]
    attention = attention.view(b, glimpses, -1) # vectorize attention maps [b, glimpses, k*k]
    attention = F.softmax(attention, dim = -1).unsqueeze(2) # [b, glimpses, 1, k*k]
    attended = attention*visual_features # use broadcasting to weight the feature maps
    attended = attended.sum(dim=-1) # sum in the spatial dimension [b, glimpses, m]
    return attended.view(b, -1) # return vectorized version with size [b, glimpses*m]

#########################################################
class AttentionDMNtied(nn.Module):
    """Attention mechanism in Dynamic Memory Networks"""
    def __init__(self, visual_features_size, question_feature_size, attention_middle_size, attention_num_layers_DMN, attention_fusion, config, drop=0.0):
        super().__init__()
        self.config = config
        self.attention_num_layers_DMN = attention_num_layers_DMN
        self.conv1 = nn.Conv2d(visual_features_size, attention_middle_size, 1, bias=False)  # 图像降维度下降到512
        self.lin1 = nn.Linear(question_feature_size, attention_middle_size)#问题降维度到512
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)
        self.input_module = nn.GRU(input_size=attention_middle_size, hidden_size=attention_middle_size, bidirectional=True)
        self.DMN = EpisodicMemory(attention_middle_size)
        self.DMN1 = EpisodicMemory(attention_middle_size)
        self.DMN2 = EpisodicMemory(attention_middle_size)
        self.DMN3 = EpisodicMemory(attention_middle_size)
        self.DMN4 = EpisodicMemory(attention_middle_size)
        self.DMN5 = EpisodicMemory(attention_middle_size)
    #override forward
    def forward(self, visual_features, question_features, return_maps=False):
        # visual features has size [b, m, k, k]
        # question_features has size [b,1024]
        v = self.conv1(self.drop(visual_features))#visual features has size [b,512,14,14]
        v = self.relu(v)
        q = self.lin1(self.drop(question_features))#question features has size [b,512]
        q = self.relu(q)
        b, m = v.size()[:2]
        v = v.view(b, m, -1)#[b,512,14*14]
        v = v.permute(2, 0, 1)#[14*14,b,512]将图像的长度放在第一个维度
        # print(v.shape)
        output, hn = self.input_module(v)#output维度为[14*14,b,512],output为输出全部的f_collate,hn为最后一层gru的记忆
        fwd_output, bwd_output = output.split(m, dim=2)
        # print(fwd_output.shape)
        # print(bwd_output.shape)
        output = fwd_output+bwd_output
        # print(output.shape)
        output = output.permute(1, 0, 2)#output size is [b,14*14,512]
        v = output#使用f作为attended
        #v = v.permute(1, 0, 2)#transpose限制为2个维度交换，permute不限制维度交换
        q = q.unsqueeze(1)
        prevm = q
        # print(q.shape)
        # print(prevm.shape)
        # print(output.shape)
        if (self.attention_num_layers_DMN == 5):
            output = self.DMN1(v, q, prevm)
            prevm = output
            output = self.DMN2(v, q, prevm)
            prevm = output
            output = self.DMN3(v, q, prevm)
            prevm = output
            output = self.DMN4(v, q, prevm)
            prevm = output
            output = self.DMN5(v, q, prevm)
        if (self.attention_num_layers_DMN == 4):
            output = self.DMN1(v, q, prevm)
            prevm = output
            output = self.DMN2(v, q, prevm)
            prevm = output
            output = self.DMN3(v, q, prevm)
            prevm = output
            output = self.DMN4(v, q, prevm)
        if(self.attention_num_layers_DMN == 3):
            output = self.DMN1(v, q, prevm)
            prevm = output
            output = self.DMN2(v, q, prevm)
            prevm = output
            output = self.DMN3(v, q, prevm)
        if(self.attention_num_layers_DMN == 2):
            output = self.DMN1(v, q, prevm)
            prevm = output
            output = self.DMN2(v, q, prevm)
        if(self.attention_num_layers_DMN == 1):
            output = self.DMN1(v, q, prevm)
        #if config["untied"]==True:
        # for i in range(self.attention_num_layers_DMN):
        #     output = self.DMN(v, q, prevm)
        #     prevm = output
        # print(output.shape)
        # print(v.shape)
        #attention = output*q
        #attention = attention.sum(dim=1)
        # print(output.shape)
        # print(v.shape)
        # attention = output*v
        #print(attention.shape)
        #print(visual_features.shape)
        attention = output*v
        attention = attention.sum(dim=1)
        #print(attention.shape)
        return attention

#########################################################
class AttentionGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.Wr = nn.Linear(input_size, hidden_size)
        nn.init.xavier_normal_(self.Wr.state_dict()['weight'])
        self.Ur = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_normal_(self.Ur.state_dict()['weight'])
        self.W = nn.Linear(input_size, hidden_size)
        nn.init.xavier_normal_(self.W.state_dict()['weight'])
        self.U = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_normal_(self.U.state_dict()['weight'])


    def forward(self, fact, C, g):
        '''
        fact.size() -> (#batch, #hidden = #embedding)
        c.size() -> (#hidden, ) -> (#batch, #hidden = #embedding)
        r.size() -> (#batch, #hidden = #embedding)
        h_tilda.size() -> (#batch, #hidden = #embedding)
        g.size() -> (#batch, )
        '''

        r = torch.sigmoid(self.Wr(fact) + self.Ur(C))
        h_tilda = torch.tanh(self.W(fact) + r * self.U(C))
        g = g.unsqueeze(1).expand_as(h_tilda)
        h = g * h_tilda + (1 - g) * C
        return h

class AttentionGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionGRU, self).__init__()
        self.hidden_size = hidden_size
        self.AGRUCell = AttentionGRUCell(input_size, hidden_size)

    def forward(self, facts, G):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        fact.size() -> (#batch, #hidden = #embedding)
        G.size() -> (#batch, #sentence)
        g.size() -> (#batch, )
        C.size() -> (#batch, #hidden)
        '''
        batch_num, sen_num, embedding_size = facts.size()
        C = Variable(torch.zeros(self.hidden_size)).cuda()
        for sid in range(sen_num):
            fact = facts[:, sid, :]
            g = G[:, sid]
            if sid == 0:
                C = C.unsqueeze(0).expand_as(fact)
            C = self.AGRUCell(fact, C, g)
        return C

class EpisodicMemory(nn.Module):
    def __init__(self, hidden_size):
        super(EpisodicMemory, self).__init__()
        self.AGRU = AttentionGRU(hidden_size, hidden_size)
        self.z1 = nn.Linear(4 * hidden_size, hidden_size)#融合后的，即cat后的
        self.z2 = nn.Linear(hidden_size, 1)
        self.next_mem = nn.Linear(3 * hidden_size, hidden_size)
        nn.init.xavier_normal_(self.z1.state_dict()['weight'])
        nn.init.xavier_normal_(self.z2.state_dict()['weight'])
        (self.next_mem.state_dict()['weight'])

    def make_interaction(self, facts, questions, prevM):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, 1, #hidden)
        prevM.size() -> (#batch, #sentence = 1, #hidden = #embedding)
        z.size() -> (#batch, #sentence, 4 x #embedding)
        G.size() -> (#batch, #sentence)
        '''
        batch_num, sen_num, embedding_size = facts.size()
        questions = questions.expand_as(facts)#questions size is [batch,sentence,hidden]
        # print(questions.shape)
        prevM = prevM.expand_as(facts)
        # print(prevM.shape)
        z = torch.cat([
            facts * questions,
            facts * prevM,
            torch.abs(facts - questions),
            torch.abs(facts - prevM)
        ], dim=2)

        z = z.view(-1, 4 * embedding_size)

        G = F.tanh(self.z1(z))
        G = self.z2(G)
        G = G.view(batch_num, -1)
        G = F.softmax(G)

        return G

    def forward(self, facts, questions, prevM):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, #sentence = 1, #hidden)
        prevM.size() -> (#batch, #sentence = 1, #hidden = #embedding)
        G.size() -> (#batch, #sentence)
        C.size() -> (#batch, #hidden)
        concat.size() -> (#batch, 3 x #embedding)
        '''
        G = self.make_interaction(facts, questions, prevM)
        C = self.AGRU(facts, G)
        concat = torch.cat([prevM.squeeze(1), C, questions.squeeze(1)], dim=1)
        next_mem = F.relu(self.next_mem(concat))
        next_mem = next_mem.unsqueeze(1)
        return next_mem

################################################################

class AttentionMechanismBase(nn.Module):
    """Normal attention mechanism"""
    def __init__(self, visual_features_size, question_feature_size, attention_middle_size, glimpses, fusion_method, drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(visual_features_size, attention_middle_size, 1, bias=False)#降维度下降到512
        self.lin1 = nn.Linear(question_feature_size, attention_middle_size)#降维度到512
        self.fuser, self.size_after_fusion = fusion.get_fuser(fusion_method, attention_middle_size, attention_middle_size)#输出哈德蒙德积和维度512
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)
        self.conv2 = nn.Conv2d(self.size_after_fusion, glimpses, 1)
        #输入为512，输出为2

    def forward(self, visual_features, question_features, return_maps=False):
        # first, compute attention vectors
        v = self.conv1(self.drop(visual_features))
        q = self.lin1(self.drop(question_features))

        q = utils.expand_like_2D(q, v)#把q的维度扩充成像v一样的维度
        x = self.relu(self.fuser(v, q))
        x = self.conv2(self.drop(x))

        if return_maps: # if maps have to be returned, save them in a variable
            maps = x.clone()

        # then, apply attention vectors to input visual features
        x = apply_attention(visual_features, x)
        if return_maps:
            return x, maps
        else:
            return x


class AttentionMechanism_1(AttentionMechanismBase):
    """Attention mechanism for model VQARS_4 to include mask before softmax and then softmax only part that mask keeps"""
    def __init__(self, visual_features_size, question_feature_size, attention_middle_size, glimpses, fusion_method, drop=0.0):
        super().__init__(visual_features_size, question_feature_size, attention_middle_size, glimpses, fusion_method, drop=drop)

    # same as general function above but receiving mask and applying it before the softmax
    def apply_attention(self, visual_features, attention, mask):
        # visual features has size [b, m, k, k]
        # attention has size [b, glimpses, k, k]
        # mask has size [b, 1, k*k]
        b, m = visual_features.size()[:2] # batch size, number of feature maps
        glimpses = attention.size(1)
        visual_features = visual_features.view(b, 1, m, -1) # vectorize feature maps [b, 1, m, k*k]
        attention = attention.view(b, glimpses, -1) # vectorize attention maps [b, glimpses, k*k]
        attention = attention*mask #! Apply mask
        for i in range(glimpses):
            attention[:,i,:][mask.squeeze().to(torch.bool)] = F.softmax(attention[:,i,:][mask.squeeze().to(torch.bool)], dim=-1)
        attention.unsqueeze_(2)
        attended = attention*visual_features # use broadcasting to weight the feature maps
        attended = attended.sum(dim=-1) # sum in the spatial dimension [b, glimpses, m]
        return attended.view(b, -1) # return vectorized version with size [b, glimpses*m] 

    # override forward method
    def forward(self, visual_features, mask, question_features):
        # first, compute attention vectors
        v = self.conv1(self.drop(visual_features))
        q = self.lin1(self.drop(question_features))
        q = utils.expand_like_2D(q, v)
        x = self.relu(self.fuser(v, q))
        x = self.conv2(self.drop(x))

        # then, apply attention vectors to input visual features
        x = self.apply_attention(visual_features, x, mask)
        return x

class AttentionMechanism_2(AttentionMechanismBase):
    """Attention mechanism for model VQARS_6 = VQARS_4 to include mask before softmax and then softmax only part that mask keeps"""
    def __init__(self, visual_features_size, question_feature_size, attention_middle_size, glimpses, fusion_method, drop=0.0):
        super().__init__(visual_features_size, question_feature_size, attention_middle_size, glimpses, fusion_method, drop=drop)

    # same as general function above but receiving mask and applying it before the softmax
    def apply_attention(self, visual_features, attention, mask):
        # visual features has size [b, m, k, k]
        # attention has size [b, glimpses, k, k]
        # mask has size [b, 1, k*k]
        b, m = visual_features.size()[:2] # batch size, number of feature maps
        glimpses = attention.size(1)
        visual_features = visual_features.view(b, 1, m, -1) # vectorize feature maps [b, 1, m, k*k]
        attention = attention.view(b, glimpses, -1) # vectorize attention maps [b, glimpses, k*k]
        attention[:,0,:] = attention[:,0,:]*mask.squeeze() # apply to first glimpse only
        attention[:,0,:][mask.squeeze().to(torch.bool)] = F.softmax(attention[:,0,:][mask.squeeze().to(torch.bool)], dim=-1) # again, only first glimpse
        attention.unsqueeze_(2)
        attended = attention*visual_features # use broadcasting to weight the feature maps
        attended = attended.sum(dim=-1) # sum in the spatial dimension [b, glimpses, m]
        return attended.view(b, -1) # return vectorized version with size [b, glimpses*m] 

    # override forward method
    def forward(self, visual_features, mask, question_features):
        # first, compute attention vectors
        v = self.conv1(self.drop(visual_features))
        q = self.lin1(self.drop(question_features))
        q = utils.expand_like_2D(q, v)
        x = self.relu(self.fuser(v, q))
        x = self.conv2(self.drop(x))

        # then, apply attention vectors to input visual features
        x = self.apply_attention(visual_features, x, mask)
        return x


class AttentionMechanism_3(AttentionMechanismBase):
    """Attention mechanism for model VQARS_7 to include mask after softmax and then softmax only part that mask keeps"""
    def __init__(self, visual_features_size, question_feature_size, attention_middle_size, glimpses, fusion_method, drop=0.0):
        super().__init__(visual_features_size, question_feature_size, attention_middle_size, glimpses, fusion_method, drop=drop)

    # same as general function above but receiving mask and applying it before the softmax
    def apply_attention(self, visual_features, attention, mask):
        # visual features has size [b, m, k, k]
        # attention has size [b, glimpses, k, k]
        # mask has size [b, 1, k*k]
        b, m = visual_features.size()[:2] # batch size, number of feature maps
        glimpses = attention.size(1)
        visual_features = visual_features.view(b, 1, m, -1) # vectorize feature maps [b, 1, m, k*k]
        attention = attention.view(b, glimpses, -1) # vectorize attention maps [b, glimpses, k*k]
        attention = F.softmax(attention, dim = -1) # [b, glimpses, k*k]
        attention = attention*mask #! Apply mask
        attention.unsqueeze_(2)
        attended = attention*visual_features # use broadcasting to weight the feature maps
        attended = attended.sum(dim=-1) # sum in the spatial dimension [b, glimpses, m]
        return attended.view(b, -1) # return vectorized version with size [b, glimpses*m] 

    # override forward method
    def forward(self, visual_features, mask, question_features, return_maps=False):
        # first, compute attention vectors
        v = self.conv1(self.drop(visual_features))
        q = self.lin1(self.drop(question_features))
        q = utils.expand_like_2D(q, v)
        x = self.relu(self.fuser(v, q))
        x = self.conv2(self.drop(x))

        if return_maps: # if maps have to be returned, save them in a variable
            maps = x.clone()

        # then, apply attention vectors to input visual features
        x = self.apply_attention(visual_features, x, mask)

        if return_maps:
            return x, maps
        else:
            return x