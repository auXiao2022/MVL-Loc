import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from network.att import AttentionBlock

class FourDirectionalLSTM(nn.Module):
    def __init__(self, seq_size, origin_feat_size, hidden_size):
        super(FourDirectionalLSTM, self).__init__()
        self.feat_size = origin_feat_size // seq_size
        self.seq_size = seq_size
        self.hidden_size = hidden_size
        self.lstm_rightleft = nn.LSTM(self.feat_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.lstm_downup = nn.LSTM(self.seq_size, self.hidden_size, batch_first=True, bidirectional=True)

    def init_hidden_(self, batch_size, device):
        return (torch.randn(2, batch_size, self.hidden_size).to(device),
                torch.randn(2, batch_size, self.hidden_size).to(device))

    def forward(self, x):
        batch_size = x.size(0)
        x_rightleft = x.view(batch_size, self.seq_size, self.feat_size)
        x_downup = x_rightleft.transpose(1, 2)
        hidden_rightleft = self.init_hidden_(batch_size, x.device)
        hidden_downup = self.init_hidden_(batch_size, x.device)
        _, (hidden_state_lr, _) = self.lstm_rightleft(x_rightleft, hidden_rightleft)
        _, (hidden_state_ud, _) = self.lstm_downup(x_downup, hidden_downup)
        hlr_fw = hidden_state_lr[0, :, :]
        hlr_bw = hidden_state_lr[1, :, :]
        hud_fw = hidden_state_ud[0, :, :]
        hud_bw = hidden_state_ud[1, :, :]
        return torch.cat([hlr_fw, hlr_bw, hud_fw, hud_bw], dim=1)

class AtLoc(nn.Module):
    def __init__(self, feature_extractor, droprate=0.5, pretrained=True, feat_dim=512, lstm=False, scene_feature=None,text_feature=None):  #feat_dim = 2048
        super(AtLoc, self).__init__()
        self.droprate = droprate
        self.lstm = lstm
        #self.global_text_feature = nn.Parameter(text_feature,requires_grad=False)
        self.scene_feature = nn.Parameter(scene_feature,requires_grad=False)
        # replace the last FC layer in feature extractor
        self.feature_extractor = feature_extractor # efficient VIT
        #self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        #fe_out_planes = 512 #384 FOR M5 #192for m0     ## 512 atloT  effvit
        fe_out_planes = 384  #768 #384
        #fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.head = nn.Linear(fe_out_planes, feat_dim)
        #self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim) #linear 512,2048

        if self.lstm:
            self.lstm4dir = FourDirectionalLSTM(seq_size=32, origin_feat_size=feat_dim, hidden_size=256)
            self.fc_xyz = nn.Linear(feat_dim // 2, 3)
            self.fc_wpqr = nn.Linear(feat_dim // 2, 3)
        else:
            #self.att = AttentionBlock(feat_dim) #2048
            #self.fc_add1 = nn.Linear(feat_dim, 1024)
            #self.relu = nn.ReLU()
            #self.fc_xyz = nn.Linear(1024, 3)
            #self.fc_wpqr = nn.Linear(1024, 3)
            decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
            self.att = nn.TransformerDecoder(decoder_layer, num_layers=4) #AttentionBlock(feat_dim)
            self.relu = nn.ReLU()
            self.fc_xyz = nn.Linear(feat_dim, 3)
            self.fc_wpqr = nn.Linear(feat_dim, 3)
        # initialize
        if pretrained:
            init_modules = [self.feature_extractor.head, self.fc_xyz, self.fc_wpqr]
            #init_modules = [self.feature_extractor.fc, self.fc_xyz, self.fc_wpqr]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x,scene):#(1,3,256,256)
        # clip encoder
        with torch.no_grad():
            x = self.feature_extractor.encode_image(x)#[1,50,512]
            #x = x + self.global_text_feature
        tgt = x # [1,50,512]
        if x.size(0) == 1:
            mem = self.scene_feature[scene] # [1,77,512] [:,:tgt.size(1),:] 
        else:
            mem = self.scene_feature[scene]#self.global_text_feature.repeat(x.size(0),1,1) # [1,77,512] [:,:tgt.size(1),:]
        
        #mem = 0.3*mem + x
        
        #x = F.relu(x)

        if self.lstm:
            x = self.lstm4dir(x)
        else:
            # x = self.att(x.view(x.size(0), -1))# [1,2048]
            x = self.att(tgt.transpose(0,1), mem.transpose(0,1)).transpose(0,1)# [1,50,512]
            x = torch.mean(x, 1) #[1,512]
            x = F.dropout(x, p=self.droprate) #[1,2048]

        #fc_add1 = self.fc_add1(x)
        #x = F.relu(fc_add1)#1,1024
        # x = self.relu(x)
        xyz = self.fc_xyz(x)#[1,3]
        wpqr = self.fc_wpqr(x) #[1,3]
        return torch.cat((xyz, wpqr), 1) #[1,6]

class AtLocPlus(nn.Module):
    def __init__(self, atlocplus):
        super(AtLocPlus, self).__init__()
        self.atlocplus = atlocplus

    def forward(self, x):
        s = x.size()
        x = x.view(-1, *s[2:])
        poses = self.atlocplus(x)
        poses = poses.view(s[0], s[1], -1)
        return poses
