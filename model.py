import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ==========================================
# ARSITEKTUR 1: KONVENSIONAL (SHALLOW)
# ==========================================

class SoftAttention_Conv(nn.Module):
    def __init__(self, in_channels, k_maps=16):
        super(SoftAttention_Conv, self).__init__()
        self.in_channels = in_channels
        self.k = k_maps
        self.attn_conv = nn.Conv2d(in_channels, k_maps, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        attn_maps = self.attn_conv(x)
        b, k, h, w = attn_maps.size()
        attn_maps = attn_maps.view(b, k, -1)
        attn_maps = F.softmax(attn_maps, dim=-1)
        attn_maps = attn_maps.view(b, k, h, w)
        scaled_attn = attn_maps * self.gamma
        
        out = torch.cat([x, scaled_attn], dim=1)
        # RETURN DUA VALUE: Output & Attention Maps (rata-rata antar channel k)
        return out, torch.mean(attn_maps, dim=1, keepdim=True)

class SoftAttentionModule_Conv(nn.Module):
    def __init__(self, in_channels, k_maps=16):
        super(SoftAttentionModule_Conv, self).__init__()
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.soft_attention = SoftAttention_Conv(in_channels, k_maps)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu1(x)
        path1 = self.maxpool1(x)
        
        # Terima output dan map
        path2_out, attn_map = self.soft_attention(x)
        path2 = self.maxpool2(path2_out)
        
        out = torch.cat([path1, path2], dim=1)
        out = self.relu2(out)
        out = self.dropout(out)
        return out, attn_map

class ResNet50_Conventional(nn.Module):
    def __init__(self, num_classes=7, k_attention_maps=16):
        super(ResNet50_Conventional, self).__init__()
        original_resnet = models.resnet50(weights=None) 
        self.features = nn.Sequential(
            original_resnet.conv1,
            original_resnet.bn1,
            original_resnet.relu,
            original_resnet.maxpool,
            original_resnet.layer1,
            original_resnet.layer2,
            original_resnet.layer3 
        )
        feature_channels = 1024 
        self.sa_module = SoftAttentionModule_Conv(in_channels=feature_channels, k_maps=k_attention_maps)
        final_channels = feature_channels + (feature_channels + k_attention_maps)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(final_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        # Dapatkan attention map dari modul
        x, attn_map = self.sa_module(x)
        
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        # Return Logits dan Attention Map
        return x, attn_map


# ==========================================
# ARSITEKTUR 2: GAN-BASED (DEEP)
# ==========================================

class SoftAttention_Deep(nn.Module):
    def __init__(self, in_channels, k_maps=16):
        super(SoftAttention_Deep, self).__init__()
        self.attn_conv = nn.Conv2d(in_channels, k_maps, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        attn_maps = self.attn_conv(x)
        b, k, h, w = attn_maps.size()
        attn_maps = attn_maps.view(b, k, -1)
        attn_maps = F.softmax(attn_maps, dim=-1)
        attn_maps = attn_maps.view(b, k, h, w)
        scaled_attn = attn_maps * self.gamma
        
        attn_avg = torch.mean(scaled_attn, dim=1, keepdim=True)
        out = x * (1 + attn_avg)
        
        # Return Output dan Rata-rata Attention Map (sebelum scaling gamma agar visualisasi murni)
        return out, torch.mean(attn_maps, dim=1, keepdim=True)

class SoftAttentionModule_Deep(nn.Module):
    def __init__(self, in_channels, k_maps=16):
        super(SoftAttentionModule_Deep, self).__init__()
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.soft_attention = SoftAttention_Deep(in_channels, k_maps)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu1(x)
        path1 = self.maxpool1(x)
        
        path2_out, attn_map = self.soft_attention(x)
        path2 = self.maxpool2(path2_out)
        
        out = torch.cat([path1, path2], dim=1)
        out = self.relu2(out)
        out = self.dropout(out)
        return out, attn_map

class ResNet50_Deep(nn.Module):
    def __init__(self, num_classes=7, k_attention_maps=16):
        super(ResNet50_Deep, self).__init__()
        original_resnet = models.resnet50(weights=None) 
        self.features = nn.Sequential(
            original_resnet.conv1,
            original_resnet.bn1,
            original_resnet.relu,
            original_resnet.maxpool,
            original_resnet.layer1,
            original_resnet.layer2,
            original_resnet.layer3,
            original_resnet.layer4 
        )
        feature_channels = 2048 
        self.sa_module = SoftAttentionModule_Deep(in_channels=feature_channels, k_maps=k_attention_maps)
        final_channels = 4096
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(final_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        # Dapatkan map
        x, attn_map = self.sa_module(x)
        
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, attn_map