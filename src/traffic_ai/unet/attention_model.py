import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels):
        super(AttentionGate, self).__init__()
        inter_channels = in_channels // 2

        self.W_g = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class Attention_UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, base_channels=64):
        super(Attention_UNet, self).__init__()

        features = [base_channels * (2 ** i) for i in range(4)]  # [64, 128, 256, 512]
        self.encoders = nn.ModuleList([
            DoubleConv(in_channels, features[0]),
            DoubleConv(features[0], features[1]),
            DoubleConv(features[1], features[2]),
            DoubleConv(features[2], features[3])
        ])
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(features[3], features[3] * 2)

        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(features[3]*2, features[3], kernel_size=2, stride=2),
            nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2),
            nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2),
            nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2),
        ])

        self.att_blocks = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()

        # decoder_channels must match output channels of upconv at each stage
        decoder_channels = [features[3], features[2], features[1], features[0]]
        for i in range(4):
            self.att_blocks.append(
                AttentionGate(in_channels=features[3 - i], gating_channels=decoder_channels[i])
            )
            self.decoder_convs.append(
                DoubleConv(features[3 - i] + decoder_channels[i], features[3 - i])
            )

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        enc_feats = []
        for encoder in self.encoders:
            x = encoder(x)
            enc_feats.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for idx in range(4):
            x = self.upconvs[idx](x)
            skip = enc_feats[3 - idx]

            att = self.att_blocks[idx](g=x, x=skip)
            x = torch.cat([x, att], dim=1)
            x = self.decoder_convs[idx](x)

        return self.final_conv(x)
