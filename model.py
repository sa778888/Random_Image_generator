import torch
import torch.nn as nn

# Define the SelfAttention class
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        proj_query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, H * W)
        attention = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(attention, dim=-1)

        proj_value = self.value(x).view(batch_size, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)

        out = self.gamma * out + x
        return out

# Define the Generator class
class Generator(nn.Module):
    def __init__(self, latent_dim=100, channels=3, features_g=64):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, features_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(True)
        )

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(features_g * 8, features_g * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features_g * 4),
                nn.ReLU(True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features_g * 2),
                nn.ReLU(True)
            ),
            SelfAttention(features_g * 2),
            nn.Sequential(
                nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features_g),
                nn.ReLU(True)
            )
        ])

        self.final = nn.Sequential(
            nn.ConvTranspose2d(features_g, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = x.view(x.size(0), self.latent_dim, 1, 1)
        x = self.initial(x)
        for layer in self.layers:
            x = layer(x)
        return self.final(x)
