import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.autograd as autograd
import os

# --- Configuration ---
class Config:
    LATENT_DIM = 128
    VOCAB_SIZE = 95  # Printable ASCII characters
    MAX_LEN = 20
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    DATA_FILE = "data/rockyou.txt"
    N_CRITIC = 5  # Critic updates per generator update
    LAMBDA_GP = 10  # Gradient penalty

config = Config()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Character Tokenizer ---
class CharTokenizer:
    def __init__(self):
        self.chars = ''.join([chr(i) for i in range(32, 127)])
        self.char_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s):
        return [self.char_to_int.get(c, 0) for c in s]

    def decode(self, l):
        return ''.join([self.int_to_char.get(i, '?') for i in l])

tokenizer = CharTokenizer()

# --- Dataset ---
class PasswordDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len):
        self.passwords = []
        with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
            for line in f:
                pw = line.strip()
                if 0 < len(pw) <= max_len:
                    self.passwords.append(pw)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.passwords)

    def __getitem__(self, idx):
        encoded = self.tokenizer.encode(self.passwords[idx])
        padded = encoded + [0] * (self.max_len - len(encoded))
        return torch.tensor(padded, dtype=torch.long)

dataset = PasswordDataset(config.DATA_FILE, tokenizer, config.MAX_LEN)
dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)

# --- Generator ---
class Generator(nn.Module):
    def __init__(self, latent_dim, vocab_size, max_len):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, max_len * vocab_size)
        )

    def forward(self, z):
        out = self.model(z)
        return out.view(-1, self.max_len, self.vocab_size)  # (batch, seq_len, vocab)

# --- Critic / Discriminator ---
class Critic(nn.Module):
    def __init__(self, vocab_size, max_len):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(max_len * vocab_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

# Instantiate models
generator = Generator(config.LATENT_DIM, config.VOCAB_SIZE, config.MAX_LEN).to(DEVICE)
critic = Critic(config.VOCAB_SIZE, config.MAX_LEN).to(DEVICE)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.9))
optimizer_C = torch.optim.Adam(critic.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.9))

# --- Gradient Penalty (WGAN-GP) ---
def compute_gradient_penalty(critic, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1).to(DEVICE)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates)
    fake = torch.ones(d_interpolates.size()).to(DEVICE)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

# --- Training Loop ---
print("🚀 Starting PassGAN training...")
for epoch in range(config.EPOCHS):
    for i, real_passwords in enumerate(dataloader):
        real_passwords = real_passwords.to(DEVICE)

        # Convert real passwords to one-hot encoding
        real_onehot = torch.nn.functional.one_hot(real_passwords, num_classes=config.VOCAB_SIZE).float()

        # Train Critic multiple times
        for _ in range(config.N_CRITIC):
            z = torch.randn(config.BATCH_SIZE, config.LATENT_DIM).to(DEVICE)
            fake_passwords = generator(z)
            c_real = critic(real_onehot)
            c_fake = critic(fake_passwords.detach())

            gp = compute_gradient_penalty(critic, real_onehot, fake_passwords)
            loss_C = -(torch.mean(c_real) - torch.mean(c_fake)) + config.LAMBDA_GP * gp

            optimizer_C.zero_grad()
            loss_C.backward()
            optimizer_C.step()

        # Train Generator
        z = torch.randn(config.BATCH_SIZE, config.LATENT_DIM).to(DEVICE)
        fake_passwords = generator(z)
        loss_G = -torch.mean(critic(fake_passwords))

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{config.EPOCHS}] | Loss_C: {loss_C.item():.4f} | Loss_G: {loss_G.item():.4f}")

print("✅ Training finished.")
