"""Mini-BERT Transformer for command sequence classification."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter


@dataclass
class TransformerConfig:
    epochs: int = 10
    batch_size: int = 32
    lr: float = 2e-4
    max_seq_len: int = 128
    max_cmd_len: int = 64
    weight_decay: float = 0.01
    warmup_steps: int = 500


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


import math


class CommandTransformer(nn.Module):
    """
    Mini-BERT architecture for command sequence classification.
    - 4 layers, 4 heads, d_model=128
    - Dual heads: CLS classification (benign/scripted/agentic) + LM next-token
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 128,
        max_cmd_len: int = 64,
        num_classes: int = 3,  # benign, scripted, agentic
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.max_cmd_len = max_cmd_len
        self.num_classes = num_classes

        # Embeddings
        self.cmd_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head (CLS token)
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        # Language modeling head (next token prediction)
        self.lm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, vocab_size),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len, cmd_len) tokenized commands
            attention_mask: (batch, seq_len) padding mask
        Returns:
            cls_logits: (batch, num_classes)
            lm_logits: (batch, seq_len, cmd_len, vocab_size)
        """
        batch, seq_len, cmd_len = input_ids.shape

        # Flatten commands for embedding
        flat_ids = input_ids.view(batch * seq_len, cmd_len)
        cmd_embeds = self.cmd_embedding(flat_ids)  # (batch*seq_len, cmd_len, d_model)

        # Average pool command tokens to get command-level representation
        cmd_repr = cmd_embeds.mean(dim=1)  # (batch*seq_len, d_model)
        cmd_repr = cmd_repr.view(batch, seq_len, self.d_model)

        # Positional encoding
        x = self.pos_encoding(cmd_repr)
        x = self.dropout(x)

        # Transformer
        if attention_mask is not None:
            # Convert to src_key_padding_mask (True = ignore)
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (batch, seq_len, d_model)

        # CLS token (first position) for classification
        cls_token = x[:, 0]  # (batch, d_model)
        cls_logits = self.cls_head(cls_token)

        # LM head on all positions
        lm_logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        # Expand to match command length for per-token loss
        lm_logits = lm_logits.unsqueeze(2).expand(-1, -1, cmd_len, -1)

        return cls_logits, lm_logits

    def save_pretrained(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "model.pt")
        config = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": 4,
            "n_layers": 4,
            "d_ff": 512,
            "max_seq_len": self.max_seq_len,
            "max_cmd_len": self.max_cmd_len,
            "num_classes": self.num_classes,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f)

    @classmethod
    def from_pretrained(cls, path: Path) -> CommandTransformer:
        path = Path(path)
        with open(path / "config.json") as f:
            config = json.load(f)
        model = cls(**config)
        model.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
        return model


class CommandTokenizer:
    """
    Maps commands to vocabulary with special tokens for LOLBins, args, paths, IPs, etc.
    """

    SPECIAL_TOKENS = {
        "[PAD]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "[UNK]": 3,
        "[ARG]": 4,
        "[PATH]": 5,
        "[IP]": 6,
        "[URL]": 7,
        "[BASE64]": 8,
        "[HASH]": 9,
        "[CRED]": 10,
        "[ENV]": 11,
    }

    LOLBINS = {
        "powershell", "pwsh", "cmd", "wscript", "cscript", "mshta", "regsvr32",
        "rundll32", "certutil", "bitsadmin", "wmic", "schtasks", "at", "sc",
        "net", "netsh", "bash", "sh", "zsh", "python", "python3", "perl",
        "ruby", "php", "curl", "wget", "nc", "ncat", "socat", "openssl",
        "base64", "xxd", "awk", "sed", "grep", "find", "xargs", "tar",
        "gzip", "ssh", "scp", "ldapsearch", "dig", "nslookup", "host",
    }

    def __init__(self, max_vocab: int = 8000):
        self.max_vocab = max_vocab
        self.vocab = dict(self.SPECIAL_TOKENS)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self._fitted = False

    def _normalize_arg(self, arg: str) -> str:
        """Normalize argument to special token or keep as-is."""
        # IP addresses
        if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', arg):
            return "[IP]"
        # URLs
        if re.match(r'^https?://', arg):
            return "[URL]"
        # Base64
        if re.match(r'^[A-Za-z0-9+/]{20,}={0,2}$', arg):
            return "[BASE64]"
        # Hashes
        if re.match(r'^[a-fA-F0-9]{32,}$', arg):
            return "[HASH]"
        # File paths
        if re.match(r'^(/|[A-Za-z]:\\\\|\.\\\\|\.\./)', arg) or "/" in arg or "\\\\" in arg:
            return "[PATH]"
        # Environment variables
        if re.match(r'^\$\w+|%\w+%|\$\{.*\}$', arg):
            return "[ENV]"
        # Credentials (API keys, tokens)
        if re.match(r'^(sk-|ghp_|gho_|glpat-|xoxb-|ya29\.)', arg):
            return "[CRED]"
        return arg

    def fit(self, sessions: list):
        """Build vocabulary from sessions."""
        counter = Counter()

        for session in sessions:
            for cmd in session.commands:
                exe = cmd.exe.split("\\")[-1].split("/")[-1].lower()
                counter[exe] += 1
                for arg in cmd.argv:
                    norm = self._normalize_arg(arg)
                    if norm != arg or norm in self.SPECIAL_TOKENS:
                        counter[norm] += 1
                    else:
                        counter[arg] += 1

        # Add most common tokens
        for token, _ in counter.most_common(self.max_vocab - len(self.vocab)):
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.id_to_token[len(self.vocab) - 1] = token

        self._fitted = True

    def encode_command(self, exe: str, argv: list[str], max_cmd_len: int) -> list[int]:
        """Encode a single command to token IDs."""
        exe_base = exe.split("\\")[-1].split("/")[-1].lower()
        tokens = [self.vocab.get(exe_base, self.vocab["[UNK]"])]

        for arg in argv:
            norm = self._normalize_arg(arg)
            tokens.append(self.vocab.get(norm, self.vocab["[UNK]"]))
            if len(tokens) >= max_cmd_len - 1:
                break

        # Pad
        tokens += [self.vocab["[PAD]"]] * (max_cmd_len - len(tokens))
        return tokens[:max_cmd_len]

    def encode_session(self, session, max_seq_len: int, max_cmd_len: int) -> list[list[int]]:
        """Encode full session to 2D token array."""
        encoded = []
        for cmd in session.commands[:max_seq_len]:
            encoded.append(self.encode_command(cmd.exe, cmd.argv, max_cmd_len))

        # Pad sequence
        while len(encoded) < max_seq_len:
            encoded.append([self.vocab["[PAD]"]] * max_cmd_len)

        return encoded

    def save(self, path: Path):
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.vocab, f)

    @classmethod
    def load(cls, path: Path) -> CommandTokenizer:
        with open(path) as f:
            vocab = json.load(f)
        tokenizer = cls()
        tokenizer.vocab = vocab
        tokenizer.id_to_token = {v: k for k, v in vocab.items()}
        tokenizer._fitted = True
        return tokenizer


class CommandSequenceDataset(Dataset):
    def __init__(
        self,
        sessions: list,
        tokenizer: CommandTokenizer,
        max_seq_len: int = 128,
        max_cmd_len: int = 64,
        labels: list[int] | None = None,
    ):
        self.sessions = sessions
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_cmd_len = max_cmd_len
        self.labels = labels

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        session = self.sessions[idx]
        encoded = self.tokenizer.encode_session(session, self.max_seq_len, self.max_cmd_len)
        input_ids = torch.tensor(encoded, dtype=torch.long)  # (seq_len, cmd_len)

        # Attention mask (1 for real commands, 0 for padding)
        attention_mask = torch.tensor(
            [1 if any(t != self.tokenizer.vocab["[PAD]"] for t in cmd) else 0 for cmd in encoded],
            dtype=torch.long,
        )

        item = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_transformer(
    model: CommandTransformer,
    train_ds: CommandSequenceDataset,
    val_ds: CommandSequenceDataset,
    config: TransformerConfig,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """Train the transformer model."""
    model = model.to(device)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Cosine annealing with warmup
    total_steps = len(train_loader) * config.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    cls_criterion = nn.CrossEntropyLoss()
    lm_criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD

    history = {"train_loss": [], "val_loss": [], "train_cls_acc": [], "val_cls_acc": []}

    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_cls_correct = 0
        train_total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            cls_logits, lm_logits = model(input_ids, attention_mask)

            # Classification loss
            cls_loss = cls_criterion(cls_logits, labels)

            # LM loss (next token prediction)
            # Shift for next-token prediction
            lm_logits_flat = lm_logits[:, :-1].reshape(-1, model.vocab_size)
            lm_targets = input_ids[:, 1:].reshape(-1)
            lm_loss = lm_criterion(lm_logits_flat, lm_targets)

            # Combined loss
            loss = cls_loss + 0.3 * lm_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_cls_correct += (cls_logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        # Validation
        model.eval()
        val_loss = 0.0
        val_cls_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                cls_logits, lm_logits = model(input_ids, attention_mask)

                cls_loss = cls_criterion(cls_logits, labels)
                lm_logits_flat = lm_logits[:, :-1].reshape(-1, model.vocab_size)
                lm_targets = input_ids[:, 1:].reshape(-1)
                lm_loss = lm_criterion(lm_logits_flat, lm_targets)

                loss = cls_loss + 0.3 * lm_loss

                val_loss += loss.item()
                val_cls_correct += (cls_logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        history["train_loss"].append(train_loss / len(train_loader))
        history["val_loss"].append(val_loss / len(val_loader))
        history["train_cls_acc"].append(train_cls_correct / train_total)
        history["val_cls_acc"].append(val_cls_correct / val_total)

        print(f"Epoch {epoch+1}/{config.epochs}: "
              f"Train Loss: {history['train_loss'][-1]:.4f}, "
              f"Val Loss: {history['val_loss'][-1]:.4f}, "
              f"Train Acc: {history['train_cls_acc'][-1]:.4f}, "
              f"Val Acc: {history['val_cls_acc'][-1]:.4f}")

    return history