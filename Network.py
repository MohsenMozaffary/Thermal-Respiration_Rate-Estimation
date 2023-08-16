import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
tr = torch

# -------------------------------------------------------------------------------------------------------------------
# PhysNet network
# -------------------------------------------------------------------------------------------------------------------
class PhysNetED(nn.Module):
    def __init__(self):
        super().__init__()

        self.start = self.build_start_layers()

        # 1x
        self.loop1 = self.build_loop1_layers()

        # encoder
        self.encoder = self.build_encoder_layers()

        #
        self.loop4 = self.build_loop4_layers()

        # decoder to reach back initial temporal length
        self.decoder = self.build_decoder_layers()

        self.end = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=5, dropout=0.3, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[2]
        x = self.start(x)
        x = self.loop1(x)
        x = self.encoder(x)
        x = self.loop4(x)
        x = self.decoder(x)
        x = self.end(x)
        x = torch.squeeze(x, (3, 4))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = x.view(batch_size, seq_len)
        return x

    def build_start_layers(self):
        return nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ELU()
        )

    def build_loop1_layers(self):
        return nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

    def build_encoder_layers(self):
        return nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),

            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

    def build_loop4_layers(self):
        return nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

    def build_decoder_layers(self):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=(4, 1, 1), stride=(2, 1, 1),
                               padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),

            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=(4, 1, 1), stride=(2, 1, 1),
                               padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )