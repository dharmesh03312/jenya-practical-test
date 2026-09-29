"""
Generative AI Engineer Task
LSTM Text Generation with PyTorch

Dataset:
    Project Gutenberg - The Complete Works of William Shakespeare
    https://www.gutenberg.org/ebooks/100

What this script does:
1. Downloads/loads a public-domain text dataset.
2. Converts text to lowercase and removes punctuation.
3. Tokenizes the text at word level.
4. Creates input -> next-word training pairs.
5. Splits data into training/validation sets.
6. Trains an Embedding + LSTM + Dense(softmax-equivalent) model.
7. Uses validation loss and early stopping/checkpointing.
8. Generates text from multiple seed phrases.
9. Saves the vocabulary, model, configuration and generated samples.
10. Supports the bonus experiment: different sequence lengths and LSTM depths.

Run:
    python lstm_text_generator.py

Quick run for testing:
    python lstm_text_generator.py --epochs 2 --max_tokens 15000 --num_sequences 5000

Full run:
    python lstm_text_generator.py --epochs 15

Notes:
- The full Shakespeare corpus can take substantial time and memory.
- For an interview/demo, start with --max_tokens 50000 or 100000.
- The model uses CrossEntropyLoss, which combines log-softmax with
  negative log-likelihood, so an explicit Softmax layer is not needed
  during training. The output is still a probability distribution.
"""

import argparse
import json
import random
import re
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Configuration
# -----------------------------
DATA_URL = "https://www.gutenberg.org/files/100/100-0.txt"
DATASET_PAGE = "https://www.gutenberg.org/ebooks/100"

SPECIAL_TOKENS = ["<PAD>", "<UNK>"]


# -----------------------------
# Dataset download / loading
# -----------------------------
def download_dataset(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size > 0:
        print(f"[INFO] Dataset already exists: {path}")
        return path

    print("[INFO] Downloading Shakespeare dataset...")
    try:
        urllib.request.urlretrieve(DATA_URL, path)
    except Exception as exc:
        raise RuntimeError(
            "Could not download the dataset. Download the plain-text file from "
            f"{DATASET_PAGE} and save it as {path}, then rerun."
        ) from exc

    print(f"[INFO] Saved dataset to: {path}")
    return path


def clean_text(text: str) -> str:
    """
    Task-required preprocessing:
    - lowercase
    - remove punctuation
    - normalize whitespace

    We intentionally keep only alphabetic/numeric word characters.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_tokens(path: Path, max_tokens=None):
    raw = path.read_text(encoding="utf-8", errors="ignore")
    cleaned = clean_text(raw)
    tokens = cleaned.split()

    if max_tokens is not None:
        tokens = tokens[:max_tokens]

    return tokens


# -----------------------------
# Vocabulary
# -----------------------------
class Vocabulary:
    def __init__(self, min_freq=2, max_vocab_size=20000):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.itos = []
        self.stoi = {}

    def build(self, tokens):
        counts = Counter(tokens)

        words = [
            word
            for word, count in counts.most_common()
            if count >= self.min_freq
        ]

        if self.max_vocab_size:
            words = words[: self.max_vocab_size - len(SPECIAL_TOKENS)]

        self.itos = SPECIAL_TOKENS + words
        self.stoi = {word: idx for idx, word in enumerate(self.itos)}

    @property
    def pad_idx(self):
        return self.stoi["<PAD>"]

    @property
    def unk_idx(self):
        return self.stoi["<UNK>"]

    def encode(self, tokens):
        return [self.stoi.get(token, self.unk_idx) for token in tokens]

    def decode(self, ids):
        words = []
        for idx in ids:
            if 0 <= idx < len(self.itos):
                word = self.itos[idx]
                if word not in SPECIAL_TOKENS:
                    words.append(word)
        return " ".join(words)

    def save(self, path: Path):
        payload = {
            "itos": self.itos,
            "min_freq": self.min_freq,
            "max_vocab_size": self.max_vocab_size,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        vocab = cls(
            min_freq=payload["min_freq"],
            max_vocab_size=payload["max_vocab_size"],
        )
        vocab.itos = payload["itos"]
        vocab.stoi = {word: idx for idx, word in enumerate(vocab.itos)}
        return vocab


# -----------------------------
# Input-output sequence creation
# -----------------------------
def make_sequences(token_ids, sequence_length):
    """
    For tokens:
        [w1, w2, w3, ..., w21]

    input  = [w1 ... w20]
    target = w21

    Then the window moves by one token.
    """
    if len(token_ids) <= sequence_length:
        raise ValueError("Not enough tokens for the selected sequence length.")

    X = []
    y = []

    for i in range(len(token_ids) - sequence_length):
        X.append(token_ids[i : i + sequence_length])
        y.append(token_ids[i + sequence_length])

    return np.asarray(X, dtype=np.int64), np.asarray(y, dtype=np.int64)


class TextSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# -----------------------------
# LSTM model
# -----------------------------
class LSTMTextGenerator(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        # Dropout is used only when num_layers > 1 by PyTorch's LSTM.
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # Dense output layer. During training, CrossEntropyLoss applies
        # log-softmax internally, so we do not add nn.Softmax here.
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)

        # We only need the final time-step representation to predict
        # the next word.
        last_output = output[:, -1, :]
        last_output = self.dropout(last_output)

        logits = self.fc(last_output)
        return logits


# -----------------------------
# Training / validation
# -----------------------------
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_items = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = X_batch.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=15,
    learning_rate=0.001,
    patience=3,
    checkpoint_path=None,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss = run_epoch(
            model, val_loader, criterion, optimizer=None, device=device, train=False
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            if checkpoint_path is not None:
                torch.save(model.state_dict(), checkpoint_path)
                print(f"[INFO] Best checkpoint saved: {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("[INFO] Early stopping triggered.")
                break

    return history


# -----------------------------
# Text generation
# -----------------------------
def sample_next_token(logits, temperature=1.0, top_k=20):
    """
    Temperature:
        < 1.0 -> more conservative
        = 1.0 -> normal sampling
        > 1.0 -> more random

    top_k:
        Keep only the k most likely next words before sampling.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        values, indices = torch.topk(logits, min(top_k, logits.numel()))
        filtered = torch.full_like(logits, float("-inf"))
        filtered[indices] = values
        logits = filtered

    probabilities = torch.softmax(logits, dim=-1)
    next_id = torch.multinomial(probabilities, num_samples=1).item()
    return next_id


@torch.no_grad()
def generate_text(
    model,
    vocab,
    seed_text,
    sequence_length,
    device,
    num_words=40,
    temperature=0.8,
    top_k=20,
):
    model.eval()

    cleaned_seed = clean_text(seed_text)
    seed_tokens = cleaned_seed.split()

    if not seed_tokens:
        raise ValueError("Seed text is empty after preprocessing.")

    generated = seed_tokens[:]

    for _ in range(num_words):
        context = generated[-sequence_length:]
        context_ids = vocab.encode(context)

        if len(context_ids) < sequence_length:
            context_ids = [vocab.pad_idx] * (
                sequence_length - len(context_ids)
            ) + context_ids

        x = torch.tensor([context_ids], dtype=torch.long, device=device)
        logits = model(x)[0]

        next_id = sample_next_token(
            logits,
            temperature=temperature,
            top_k=top_k,
        )

        next_word = vocab.itos[next_id]
        if next_word in SPECIAL_TOKENS:
            continue

        generated.append(next_word)

    return " ".join(generated)


# -----------------------------
# Utility functions
# -----------------------------
def perplexity(loss):
    return float(np.exp(min(loss, 20)))


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def build_loaders(
    tokens,
    vocab,
    sequence_length,
    train_ratio=0.9,
    batch_size=128,
    max_sequences=None,
):
    token_ids = vocab.encode(tokens)

    X, y = make_sequences(token_ids, sequence_length)

    if max_sequences is not None and len(X) > max_sequences:
        # Take a contiguous prefix for a reproducible smaller experiment.
        X = X[:max_sequences]
        y = y[:max_sequences]

    split = int(len(X) * train_ratio)

    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    train_dataset = TextSequenceDataset(X_train, y_train)
    val_dataset = TextSequenceDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, len(X_train), len(X_val)


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="LSTM word-level text generation using Shakespeare."
    )

    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--sequence_length", type=int, default=20)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_vocab_size", type=int, default=20000)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--num_sequences", type=int, default=None)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = data_dir / "shakespeare.txt"
    vocab_path = output_dir / "vocab.json"
    checkpoint_path = output_dir / "best_model.pt"
    history_path = output_dir / "training_history.json"
    samples_path = output_dir / "sample_generations.txt"
    config_path = output_dir / "config.json"

    # 1. Dataset
    download_dataset(dataset_path)

    # 2. Preprocessing
    tokens = load_tokens(dataset_path, max_tokens=args.max_tokens)
    print(f"[INFO] Number of tokens after preprocessing: {len(tokens):,}")

    # 3. Vocabulary
    vocab = Vocabulary(
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
    )
    vocab.build(tokens)
    vocab.save(vocab_path)

    print(f"[INFO] Vocabulary size: {len(vocab.itos):,}")

    # 4. Train/validation data
    train_loader, val_loader, n_train, n_val = build_loaders(
        tokens=tokens,
        vocab=vocab,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        max_sequences=args.num_sequences,
    )

    print(f"[INFO] Training sequences: {n_train:,}")
    print(f"[INFO] Validation sequences: {n_val:,}")

    # 5. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # 6. Model
    model = LSTMTextGenerator(
        vocab_size=len(vocab.itos),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable parameters: {parameter_count:,}")

    # 7. Train
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        checkpoint_path=checkpoint_path,
    )

    save_json(history_path, history)

    # Load best model before generation.
    if checkpoint_path.exists():
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )

    # 8. Generate sample text
    seeds = [
        "to be or not to be",
        "shall i compare thee",
        "the lady doth protest",
        "all the worlds a stage",
    ]

    samples = []
    print("\n========== GENERATED TEXT ==========\n")

    for seed_text in seeds:
        generated = generate_text(
            model=model,
            vocab=vocab,
            seed_text=seed_text,
            sequence_length=args.sequence_length,
            device=device,
            num_words=50,
            temperature=0.8,
            top_k=20,
        )

        result = f"SEED: {seed_text}\nGENERATED: {generated}\n"
        samples.append(result)
        print(result)

    samples_path.write_text(
        "\n".join(samples),
        encoding="utf-8",
    )

    # 9. Save configuration / experiment information
    final_val_loss = history[-1]["val_loss"] if history else None

    config = {
        "dataset_page": DATASET_PAGE,
        "dataset_download_url": DATA_URL,
        "preprocessing": {
            "lowercase": True,
            "remove_punctuation": True,
            "tokenization": "word-level",
        },
        "model": {
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
        },
        "training": {
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "learning_rate": args.learning_rate,
            "patience": args.patience,
            "train_sequences": n_train,
            "validation_sequences": n_val,
            "final_validation_loss": final_val_loss,
            "final_validation_perplexity": (
                perplexity(final_val_loss)
                if final_val_loss is not None
                else None
            ),
        },
        "device": str(device),
        "seed": args.seed,
    }

    save_json(config_path, config)

    print("\n========== FILES CREATED ==========")
    print(f"Dataset:       {dataset_path}")
    print(f"Vocabulary:    {vocab_path}")
    print(f"Checkpoint:    {checkpoint_path}")
    print(f"History:       {history_path}")
    print(f"Samples:       {samples_path}")
    print(f"Config:        {config_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
