import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os


# --- Configuration ---
# It's better to have configs in the main file or a dedicated class
# unless the project is very large.
class Config:
    LATENT_DIM = 128
    VOCAB_SIZE = 95  # Common printable ASCII characters
    MAX_LEN = 20
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    DATA_FILE = "data/rockyou.txt"


config = Config()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# --- Character Tokenizer ---
# A simple tokenizer for printable ASCII characters.
# You can replace this with a more sophisticated one.
class CharTokenizer:
    def __init__(self):
        # All printable ASCII characters from space to ~
        self.chars = ''.join([chr(i) for i in range(32, 127)])
        self.char_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(self.chars)}
        # Ensure vocab size in config matches tokenizer
        if len(self.chars) != config.VOCAB_SIZE:
            print(
                f"Warning: Tokenizer vocab size ({len(self.chars)}) does not match config.VOCAB_SIZE ({config.VOCAB_SIZE}).")

    def encode(self, s):
        return [self.char_to_int.get(c, 0) for c in s]  # Default to token 0 if char not found

    def decode(self, l):
        return ''.join([self.int_to_char.get(i, '?') for i in l])  # Default to '?' if int not found


tokenizer = CharTokenizer()


# --- Dataset ---
class PasswordDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.passwords = []
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Data file not found at {file_path}. Please download rockyou.txt and place it in a 'data' folder.")
        with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
            for line in f:
                stripped_line = line.strip()
                if 0 < len(stripped_line) <= self.max_len:
                    self.passwords.append(stripped_line)

    def __len__(self):
        return len(self.passwords)

    def __getitem__(self, idx):
        password = self.passwords[idx]
        encoded = self.tokenizer.encode(password)
        # Pad the sequence to max_len
        padded = encoded + [0] * (self.max_len - len(encoded))
        return torch.tensor(padded, dtype=torch.long)


# --- Generator Model ---
class PasswordGenerator(nn.Module):
    def __init__(self, latent_dim, vocab_size, max_len):
        super(PasswordGenerator, self).__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, max_len * vocab_size)
        )

    def forward(self, z):
        # Input 'z' is the latent vector, shape: (batch_size, latent_dim)
        output = self.model(z)
        # We need to reshape the output to be compatible with CrossEntropyLoss
        # Expected shape: (batch_size, vocab_size, sequence_length)
        return output.view(-1, self.vocab_size, self.max_len)


# --- Main Training Script ---
# Load dataset
dataset = PasswordDataset(config.DATA_FILE, tokenizer, config.MAX_LEN)
dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)

# Instantiate generator, optimizer, and loss function
generator = PasswordGenerator(
    latent_dim=config.LATENT_DIM,
    vocab_size=config.VOCAB_SIZE,
    max_len=config.MAX_LEN
).to(DEVICE)

optimizer = torch.optim.Adam(generator.parameters(), lr=config.LEARNING_RATE)
criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification

# Training loop
print("🚀 Starting training...")
for epoch in range(config.EPOCHS):
    total_loss = 0
    for i, real_passwords in enumerate(dataloader):
        real_passwords = real_passwords.to(DEVICE)
        batch_size = real_passwords.size(0)

        # Generate a batch of latent vectors
        z = torch.randn(batch_size, config.LATENT_DIM).to(DEVICE)

        # Generate fake passwords (logits)
        fake_passwords_logits = generator(z)

        # Calculate loss
        # The generator outputs logits of shape (N, C, L) where C=vocab_size
        # The real_passwords are the target indices of shape (N, L)
        loss = criterion(fake_passwords_logits, real_passwords)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{config.EPOCHS}] | Loss: {avg_loss:.4f}")

print("✅ Training finished.")