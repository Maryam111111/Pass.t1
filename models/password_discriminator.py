import torch.nn as nn
from config import MAX_LEN
from datasets.password_dataset import CHAR2IDX

VOCAB_SIZE = len(CHAR2IDX) + 1

class PasswordDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, 32)
        self.conv = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * MAX_LEN, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.embed(x).transpose(1, 2)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
