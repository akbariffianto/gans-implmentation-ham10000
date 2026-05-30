import torch
import torch.nn as nn

class GeneratorACGAN(nn.Module):
    def __init__(self, num_classes, latent_dim=100, img_size=128, channels=3, feature_size=64):
        super(GeneratorACGAN, self).__init__()
        
        self.init_size = 4
        self.latent_dim = latent_dim
        # feature_size=64 adalah kunci agar channel awal menjadi 2048 (64*32)
        self.feature_size = feature_size 
        
        # Embedding label
        self.label_emb = nn.Embedding(num_classes, latent_dim)

        # Perhitungan channel awal: 64 * 32 = 2048
        # Checkpoint Anda mengharapkan input channel 2048 di block pertama
        self.initial_channels = self.feature_size * 32 

        # Linear projection
        # Input: Noise (100) + Label (100) = 200
        # Output: 2048 * 4 * 4
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + latent_dim, self.initial_channels * self.init_size ** 2)
        )

        self.model = nn.Sequential(
            # Block 1: 4x4 -> 8x8
            # Checkpoint: ConvTranspose2d (2048 -> 1024)
            nn.ConvTranspose2d(self.initial_channels, self.feature_size * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.feature_size * 16),
            nn.ReLU(inplace=True),
            
            # Block 2: 8x8 -> 16x16
            # Checkpoint: ConvTranspose2d (1024 -> 512)
            nn.ConvTranspose2d(self.feature_size * 16, self.feature_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.feature_size * 8),
            nn.ReLU(inplace=True),
            
            # Block 3: 16x16 -> 32x32
            # Checkpoint: ConvTranspose2d (512 -> 256)
            # (Ini yang tadi error mismatch di model.6.weight)
            nn.ConvTranspose2d(self.feature_size * 8, self.feature_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.feature_size * 4),
            nn.ReLU(inplace=True),

            # Block 4: 32x32 -> 64x64
            # Checkpoint: ConvTranspose2d (256 -> 128)
            nn.ConvTranspose2d(self.feature_size * 4, self.feature_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.feature_size * 2),
            nn.ReLU(inplace=True),

            # Block 5: 64x64 -> 128x128
            # Checkpoint: ConvTranspose2d (128 -> 64)
            nn.ConvTranspose2d(self.feature_size * 2, self.feature_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.feature_size),
            nn.ReLU(inplace=True),
            
            # Output Layer: 128x128
            # Output ke RGB (3 Channel)
            nn.Conv2d(self.feature_size, channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        
        out = self.l1(gen_input)
        # Reshape ke (Batch, 2048, 4, 4)
        out = out.view(out.shape[0], self.initial_channels, self.init_size, self.init_size)
        
        img = self.model(out)
        return img

class GeneratorWGANGP(nn.Module):
    def __init__(self, latent_dim=100, img_size=128, channels=3, feature_size=32):
        super(GeneratorWGANGP, self).__init__()
        
        self.init_size = img_size // 32
        self.g_features = feature_size 
        
        # 64 * 32 = 2048 channels awal
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, self.g_features * 32 * self.init_size ** 2)
        )

        self.model = nn.Sequential(
            # Block 1: 4x4 -> 8x8
            nn.ConvTranspose2d(self.g_features * 32, self.g_features * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.g_features * 16),
            nn.ReLU(inplace=True),
            
            # Block 2: 8x8 -> 16x16
            nn.ConvTranspose2d(self.g_features * 16, self.g_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.g_features * 8),
            nn.ReLU(inplace=True),
            
            # Block 3: 16x16 -> 32x32
            nn.ConvTranspose2d(self.g_features * 8, self.g_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.g_features * 4),
            nn.ReLU(inplace=True),
            
            # Block 4: 32x32 -> 64x64
            nn.ConvTranspose2d(self.g_features * 4, self.g_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.g_features * 2),
            nn.ReLU(inplace=True),
            
            # Block 5: 64x64 -> 128x128
            nn.ConvTranspose2d(self.g_features * 2, self.g_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.g_features),
            nn.ReLU(inplace=True),
            
            # Output Layer: 128x128
            nn.Conv2d(self.g_features, channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.model(out)
        return img

## --- ACWGAN-GP Generator (Fixed: ConvTranspose2d, ReLU, Conditional) ---
class GeneratorACWGANGP(nn.Module):
    def __init__(self, num_classes, latent_dim=100, img_size=128, channels=3, feature_size=32, embed_size=100):
        super(GeneratorACWGANGP, self).__init__()
        
        self.init_size = img_size // 32  
        self.g_features = feature_size   
        self.embed_size = embed_size     
        
        # 1. Label Embedding (ACGAN Part)
        self.label_emb = nn.Embedding(num_classes, self.embed_size)

        # Input: Latent Dim (Z) + Embedding Dim (Label)
        # Output: 64 * 32 * 4 * 4 = 32768
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + self.embed_size, self.g_features * 32 * self.init_size ** 2)
        )

        self.model = nn.Sequential(
            # Block 1: 4x4 -> 8x8
            # Input: 2048 -> Output: 1024
            nn.ConvTranspose2d(self.g_features * 32, self.g_features * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.g_features * 16),
            nn.ReLU(inplace=True),
            
            # Block 2: 8x8 -> 16x16
            # Input: 1024 -> Output: 512
            nn.ConvTranspose2d(self.g_features * 16, self.g_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.g_features * 8),
            nn.ReLU(inplace=True),
            
            # Block 3: 16x16 -> 32x32
            # Input: 512 -> Output: 256
            nn.ConvTranspose2d(self.g_features * 8, self.g_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.g_features * 4),
            nn.ReLU(inplace=True),
            
            # Block 4: 32x32 -> 64x64
            # Input: 256 -> Output: 128
            nn.ConvTranspose2d(self.g_features * 4, self.g_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.g_features * 2),
            nn.ReLU(inplace=True),
            
            # Block 5: 64x64 -> 128x128
            # Input: 128 -> Output: 64
            nn.ConvTranspose2d(self.g_features * 2, self.g_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.g_features),
            nn.ReLU(inplace=True),
            
            # Output Layer: 128x128
            # Input: 64 -> Output: RGB
            nn.Conv2d(self.g_features, channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # 1. Concatenate Noise + Label Embedding
        # noise: (batch, latent_dim)
        # label_emb: (batch, embed_size)
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.model(out)
        return img