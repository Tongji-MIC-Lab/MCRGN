import torch
from torchvision import models
from torch import nn as nn
import torch.nn.functional as F
import numpy as np
from ResNet50 import resnet50



class Referring_relationships_Model(nn.Module):
    def __init__(self, config=None):
        super(Referring_relationships_Model, self).__init__()
        self.config = config
        self.input_dim = self.config["input_dim"]
        self.feat_map_dim = self.config["feat_map_dim"]
        #self.hidden_dim = self.config["hidden_dim"]
        self.hidden_dim = 512
        self.num_objects = self.config["num_objects"]
        self.num_predicates = self.config["num_predicates"]
        self.output_dim = self.config["output_dim"]
        self.embedding_dim = self.config["embedding_dim"]


        # load cnn model to extract image features
        self.cnn_model = resnet50(pretrained=True)

        self.prj_5 = nn.Conv2d(2048, 512, kernel_size=1)
        self.prj_4 = nn.Conv2d(1024, 512, kernel_size=1)
        self.prj_3 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv_5 =nn.Conv2d(512, self.hidden_dim, kernel_size=3, padding=1)
        self.conv_4 =nn.Conv2d(512, self.hidden_dim, kernel_size=3, padding=1)
        self.conv_3 =nn.Conv2d(512, self.hidden_dim, kernel_size=3, padding=1)
        
        self.subject_embedding = nn.Embedding(self.num_objects, self.embedding_dim)
        self.object_embedding = nn.Embedding(self.num_objects, self.embedding_dim)
        self.rel_embedding = nn.Embedding(self.num_predicates, self.embedding_dim)
        self.subject_fc = nn.Linear(self.embedding_dim, self.hidden_dim, bias=True)
        self.object_fc = nn.Linear(self.embedding_dim, self.hidden_dim, bias=True)
        self.rel_fc = nn.Linear(self.embedding_dim, self.hidden_dim, bias=True)

        self.triple_subj_fc = nn.Linear(512*3, 512, bias=True)
        self.triple_obj_fc = nn.Linear(512*3, 512, bias=True)
        
        self.metric_so_conv1 = nn.Conv2d(512, 256, 5, stride=1, padding=2, bias=False)
        self.metric_so_conv2 = nn.Conv2d(256, 128, 5, stride=1, padding=2, bias=False)
        self.metric_so_conv3 = nn.Conv2d(128, 1, 5, stride=1, padding=2, bias=False)
        self.so_bn1 = nn.BatchNorm2d(256)
        self.so_bn2 = nn.BatchNorm2d(128)
        
        self.metric_os_conv1 = nn.Conv2d(512, 256, 5, stride=1, padding=2, bias=False)
        self.metric_os_conv2 = nn.Conv2d(256, 128, 5, stride=1, padding=2, bias=False)
        self.metric_os_conv3 = nn.Conv2d(128, 1, 5, stride=1, padding=2, bias=False)
        self.os_bn1 = nn.BatchNorm2d(256)
        self.os_bn2 = nn.BatchNorm2d(128)
        
        self.att_subj_conv1 = nn.Conv2d(512, 256, 5, stride=1, padding=2, bias=False)
        self.att_subj_conv2 = nn.Conv2d(256, 128, 5, stride=1, padding=2, bias=False)
        self.att_subj_conv3 = nn.Conv2d(128, 1, 5, stride=1, padding=2, bias=False)
        
        self.att_obj_conv1 = nn.Conv2d(512, 256, 5, stride=1, padding=2, bias=False)
        self.att_obj_conv2 = nn.Conv2d(256, 128, 5, stride=1, padding=2, bias=False)
        self.att_obj_conv3 = nn.Conv2d(128, 1, 5, stride=1, padding=2, bias=False)


    def attention(self, cnn_features, subj_obj_features, flag=True):
        if flag:
            subj_obj_features = subj_obj_features.unsqueeze(dim=-1)
            subj_obj_features = subj_obj_features.unsqueeze(dim=-1)
        attention_weight = F.relu(torch.sum(torch.mul(cnn_features, subj_obj_features), dim=1).unsqueeze(dim=1))
        #attention_weight = F.relu(torch.mul(cnn_features, subj_obj_features))
        return attention_weight    

    def upsample(self, input):
        return F.interpolate(input, size=(input.shape[2]*2, input.shape[3]*2)) 

    def forward(self, input_img, input_subj, input_rel, input_obj):
        C3, C4, C5 = self.cnn_model(input_img.cuda())
        P5 = self.prj_5(C5)
        P4 = self.prj_4(C4)
        P3 = self.prj_3(C3)
        P4 = P4 + self.upsample(P5)
        P3 = P3 + self.upsample(P4)
        # [b,self.hidden_dim,28,28,]
        P3 = self.conv_3(P3)
        # [b,self.hidden_dim,14,14]
        P4 = self.conv_4(P4)
        # [b,self.hidden_dim,7,7]
        P5 = self.conv_5(P5)
 
        subj_emb_features = self.subject_embedding(input_subj.long().cuda())
        obj_emb_features = self.object_embedding(input_obj.long().cuda())
        rel_emb_features = self.rel_embedding(input_rel.long().cuda())

        subj_features = F.relu(self.subject_fc(subj_emb_features))
        obj_features = F.relu(self.object_fc(obj_emb_features))
        rel_features = F.relu(self.rel_fc(rel_emb_features))

        
        P3_subj = self.attention(P3, subj_features).reshape(-1, 1, 28, 28).expand(-1, 512, 28, 28)
        P3_obj = self.attention(P3, obj_features).reshape(-1, 1, 28, 28).expand(-1, 512, 28, 28)
        
        P4_subj = self.attention(P4, subj_features).reshape(-1, 1, 14, 14).expand(-1, 512, 14, 14)
        P4_obj = self.attention(P4, obj_features).reshape(-1, 1, 14, 14).expand(-1, 512, 14, 14)
        
        P5_subj = self.attention(P5, subj_features).reshape(-1, 1, 7, 7).expand(-1, 512, 7, 7)
        P5_obj = self.attention(P5, obj_features).reshape(-1, 1, 7, 7).expand(-1, 512, 7, 7)

        # [b,512] 
        triple_subj = F.relu(self.triple_subj_fc(torch.cat([subj_features, rel_features, obj_features], dim=-1))).reshape(-1, 512, 1, 1)#.expand(-1, 512, 14, 14)
        triple_obj = F.relu(self.triple_obj_fc(torch.cat([subj_features, rel_features, obj_features], dim=-1))).reshape(-1, 512, 1, 1)#.expand(-1, 512, 14, 14)
        
        # P3_S
        
        P33_S = torch.mul(F.interpolate(P3_subj, size=[14, 14], mode="bilinear", align_corners=True)+F.interpolate(P3_obj, size=[14, 14], mode="bilinear", align_corners=True), triple_subj)
        P34_S = torch.mul(F.interpolate(P3_subj, size=[14, 14], mode="bilinear", align_corners=True)+P4_obj, triple_subj)
        P35_S = torch.mul(F.interpolate(P3_subj, size=[14, 14], mode="bilinear", align_corners=True)+F.interpolate(P5_obj, size=[14, 14], mode="bilinear", align_corners=True), triple_subj)
        
        P33_S = F.leaky_relu(self.so_bn1(self.metric_so_conv1(P33_S)))
        P33_S = F.leaky_relu(self.so_bn2(self.metric_so_conv2(P33_S)))
        P33_S = F.leaky_relu(self.metric_so_conv3(P33_S))
        
        P34_S = F.leaky_relu(self.so_bn1(self.metric_so_conv1(P34_S)))
        P34_S = F.leaky_relu(self.so_bn2(self.metric_so_conv2(P34_S)))
        P34_S = F.leaky_relu(self.metric_so_conv3(P34_S))
               
        P35_S = F.leaky_relu(self.so_bn1(self.metric_so_conv1(P35_S)))
        P35_S = F.leaky_relu(self.so_bn2(self.metric_so_conv2(P35_S)))
        P35_S = F.leaky_relu(self.metric_so_conv3(P35_S))
        
        P3_S = P33_S + P34_S + P35_S
        
        # P3_O
        
        P33_O = torch.mul(F.interpolate(P3_obj, size=[14, 14], mode="bilinear", align_corners=True)+F.interpolate(P3_subj, size=[14, 14], mode="bilinear", align_corners=True), triple_obj)
        P34_O = torch.mul(F.interpolate(P3_obj, size=[14, 14], mode="bilinear", align_corners=True)+P4_subj, triple_obj)
        P35_O = torch.mul(F.interpolate(P3_obj, size=[14, 14], mode="bilinear", align_corners=True)+F.interpolate(P5_subj, size=[14, 14], mode="bilinear", align_corners=True), triple_obj)
        
        P33_O = F.leaky_relu(self.os_bn1(self.metric_os_conv1(P33_O)))
        P33_O = F.leaky_relu(self.os_bn2(self.metric_os_conv2(P33_O)))
        P33_O = F.leaky_relu(self.metric_os_conv3(P33_O))
        
        P34_O = F.leaky_relu(self.os_bn1(self.metric_os_conv1(P34_O)))
        P34_O = F.leaky_relu(self.os_bn2(self.metric_os_conv2(P34_O)))
        P34_O = F.leaky_relu(self.metric_os_conv3(P34_O))
               
        P35_O = F.leaky_relu(self.os_bn1(self.metric_os_conv1(P35_O)))
        P35_O = F.leaky_relu(self.os_bn2(self.metric_os_conv2(P35_O)))
        P35_O = F.leaky_relu(self.metric_os_conv3(P35_O))
        
        P3_O = P33_O + P34_O + P35_O
        
        
        # P4_S
        
        P43_S = torch.mul(P4_subj+F.interpolate(P3_obj, size=[14, 14], mode="bilinear", align_corners=True), triple_subj)
        P44_S = torch.mul(P4_subj+P4_obj, triple_subj)
        P45_S = torch.mul(P4_subj+F.interpolate(P5_obj, size=[14, 14], mode="bilinear", align_corners=True), triple_subj)
        
        P43_S = F.leaky_relu(self.so_bn1(self.metric_so_conv1(P43_S)))
        P43_S = F.leaky_relu(self.so_bn2(self.metric_so_conv2(P43_S)))
        P43_S = F.leaky_relu(self.metric_so_conv3(P43_S))
        
        P44_S = F.leaky_relu(self.so_bn1(self.metric_so_conv1(P44_S)))
        P44_S = F.leaky_relu(self.so_bn2(self.metric_so_conv2(P44_S)))
        P44_S = F.leaky_relu(self.metric_so_conv3(P44_S))
               
        P45_S = F.leaky_relu(self.so_bn1(self.metric_so_conv1(P45_S)))
        P45_S = F.leaky_relu(self.so_bn2(self.metric_so_conv2(P45_S)))
        P45_S = F.leaky_relu(self.metric_so_conv3(P45_S))
        
        P4_S = P43_S + P44_S + P45_S
        
        # P4_O
        
        P43_O = torch.mul(P4_obj+F.interpolate(P3_subj, size=[14, 14], mode="bilinear", align_corners=True), triple_obj)
        P44_O = torch.mul(P4_obj+P4_subj, triple_obj)
        P45_O = torch.mul(P4_obj+F.interpolate(P5_subj, size=[14, 14], mode="bilinear", align_corners=True), triple_obj)
        
        P43_O = F.leaky_relu(self.os_bn1(self.metric_os_conv1(P43_O)))
        P43_O = F.leaky_relu(self.os_bn2(self.metric_os_conv2(P43_O)))
        P43_O = F.leaky_relu(self.metric_os_conv3(P43_O))
        
        P44_O = F.leaky_relu(self.os_bn1(self.metric_os_conv1(P44_O)))
        P44_O = F.leaky_relu(self.os_bn2(self.metric_os_conv2(P44_O)))
        P44_O = F.leaky_relu(self.metric_os_conv3(P44_O))
               
        P45_O = F.leaky_relu(self.os_bn1(self.metric_os_conv1(P45_O)))
        P45_O = F.leaky_relu(self.os_bn2(self.metric_os_conv2(P45_O)))
        P45_O = F.leaky_relu(self.metric_os_conv3(P45_O))
        
        P4_O = P43_O + P44_O + P45_O

        # P5_S
        
        P53_S = torch.mul(F.interpolate(P5_subj, size=[14, 14], mode="bilinear", align_corners=True)+F.interpolate(P3_obj, size=[14, 14], mode="bilinear", align_corners=True), triple_subj)
        P54_S = torch.mul(F.interpolate(P5_subj, size=[14, 14], mode="bilinear", align_corners=True)+P4_obj, triple_subj)
        P55_S = torch.mul(F.interpolate(P5_subj, size=[14, 14], mode="bilinear", align_corners=True)+F.interpolate(P5_obj, size=[14, 14], mode="bilinear", align_corners=True), triple_subj)
        
        P53_S = F.leaky_relu(self.so_bn1(self.metric_so_conv1(P53_S)))
        P53_S = F.leaky_relu(self.so_bn2(self.metric_so_conv2(P53_S)))
        P53_S = F.leaky_relu(self.metric_so_conv3(P53_S))
        
        P54_S = F.leaky_relu(self.so_bn1(self.metric_so_conv1(P54_S)))
        P54_S = F.leaky_relu(self.so_bn2(self.metric_so_conv2(P54_S)))
        P54_S = F.leaky_relu(self.metric_so_conv3(P54_S))
               
        P55_S = F.leaky_relu(self.so_bn1(self.metric_so_conv1(P55_S)))
        P55_S = F.leaky_relu(self.so_bn2(self.metric_so_conv2(P55_S)))
        P55_S = F.leaky_relu(self.metric_so_conv3(P55_S))
        
        P5_S = P53_S + P54_S + P55_S
        
        # P5_O
        
        P53_O = torch.mul(F.interpolate(P5_obj, size=[14, 14], mode="bilinear", align_corners=True)+F.interpolate(P3_subj, size=[14, 14], mode="bilinear", align_corners=True), triple_obj)
        P54_O = torch.mul(F.interpolate(P5_obj, size=[14, 14], mode="bilinear", align_corners=True)+P4_subj, triple_obj)
        P55_O = torch.mul(F.interpolate(P5_obj, size=[14, 14], mode="bilinear", align_corners=True)+F.interpolate(P5_subj, size=[14, 14], mode="bilinear", align_corners=True), triple_obj)
        
        
        P53_O = F.leaky_relu(self.os_bn1(self.metric_os_conv1(P53_O)))
        P53_O = F.leaky_relu(self.os_bn2(self.metric_os_conv2(P53_O)))
        P53_O = F.leaky_relu(self.metric_os_conv3(P53_O))
        
        P54_O = F.leaky_relu(self.os_bn1(self.metric_os_conv1(P54_O)))
        P54_O = F.leaky_relu(self.os_bn2(self.metric_os_conv2(P54_O)))
        P54_O = F.leaky_relu(self.metric_os_conv3(P54_O))
               
        P55_O = F.leaky_relu(self.os_bn1(self.metric_os_conv1(P55_O)))
        P55_O = F.leaky_relu(self.os_bn2(self.metric_os_conv2(P55_O)))
        P55_O = F.leaky_relu(self.metric_os_conv3(P55_O))
        
        P5_O = P53_O + P54_O + P55_O
        

        
        subject_regions = torch.mul(P4, (P3_S+P4_S+P5_S))
        subj_att_w = torch.mul(subject_regions, subj_features.reshape(-1,512,1,1))      
        subj_att_w = F.leaky_relu(self.att_subj_conv1(subj_att_w))
        subj_att_w = F.leaky_relu(self.att_subj_conv2(subj_att_w))
        subject_regions = torch.tanh(F.relu(self.att_subj_conv3(subj_att_w))).reshape(-1, 14*14)
        
        
        
        object_regions = torch.mul(P4, (P3_O+P4_O+P5_O))
        obj_att_w = torch.mul(object_regions, obj_features.reshape(-1,512,1,1))        
        obj_att_w = F.leaky_relu(self.att_obj_conv1(obj_att_w))
        obj_att_w = F.leaky_relu(self.att_obj_conv2(obj_att_w))
        object_regions = torch.tanh(F.relu(self.att_obj_conv3(obj_att_w))).reshape(-1, 14*14)
        

        return subject_regions, object_regions
