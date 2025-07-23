import torch
from torch.utils.data import Dataset
import string
from config import MAX_LEN  # Make sure config.py defines MAX_LEN

# Define allowed characters and mappings
CHARS = string.printable[:95]  # Common printable chars
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 reserved for padding
IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}

def encode(pw):
    """
    Encode a password string into a list of indices.
    Pads with 0 to length MAX_LEN.
    """
    pw = pw[:MAX_LEN]
    encoded = [CHAR2IDX.get(c, 0) for c in pw]  # unknown chars -> 0
    encoded += [0] * (MAX_LEN - len(encoded))  # padding
    return encoded

def decode(seq):
    """
    Decode a list of indices into a password string.
    Ignores padding zeros.
    """
    return ''.join(IDX2CHAR.get(i, '') for i in seq if i != 0)

class PasswordDataset(Dataset):
    def __init__(self, path, max_lines=100000):
        """
        Load passwords from file path, limit to max_lines for speed.
        Only passwords between length 4 and MAX_LEN included.
        """
        lines = []
        with open(path, 'r', encoding='latin-1') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                line = line.strip()
                if 4 <= len(line) <= MAX_LEN:
                    lines.append(line)

        # Encode all passwords as tensors
        self.data = [torch.tensor(encode(line), dtype=torch.long) for line in lines]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
