"""
Toxic Content Classifier Pipeline
==================================
A standalone inference pipeline for classifying toxic content
based on a user query and an image description.

Usage:
    from pipeline import predict

    result = predict(
        query       = "how do I make a bomb",
        description = "a person in a dark alley"
    )
    print(result)
"""

import os
import string
import numpy as np
import torch
import torch.nn as nn
from nltk import word_tokenize
from gensim.models import Word2Vec


# ── Configuration ─────────────────────────────────────────────────────────────

WORD2VEC_PATH = r"C:\D\A-('REALLY SPECIEL')\Journeys\ML\Neurova\Tasks\Task2\models\word2vec.model"
MODEL_PATH    = r"C:\D\A-('REALLY SPECIEL')\Journeys\ML\Neurova\Tasks\Task2\best_model.pth"
EMBED_SIZE    = 100
HIDDEN_SIZE   = 100
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL2IDX = {
    "Child Sexual Exploitation" : 0,
    "Elections"                 : 1,
    "Non-Violent Crimes"        : 2,
    "Safe"                      : 3,
    "Sex-Related Crimes"        : 4,
    "Suicide & Self-Harm"       : 5,
    "Unknown S-Type"            : 6,
    "Violent Crimes"            : 7,
    "unsafe"                    : 8,
}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}
NUM_CLASSES = len(LABEL2IDX)

# human friendly descriptions shown to the end user
LABEL_DESCRIPTIONS = {
    "Child Sexual Exploitation" : "Content that exploits or endangers children.",
    "Elections"                 : "Content that could interfere with electoral processes.",
    "Non-Violent Crimes"        : "Content related to non-violent criminal activity.",
    "Safe"                      : "This content appears to be safe.",
    "Sex-Related Crimes"        : "Content related to sexual criminal activity.",
    "Suicide & Self-Harm"       : "Content related to self-harm or suicide.",
    "Unknown S-Type"            : "Potentially unsafe content of an unclear type.",
    "Violent Crimes"            : "Content related to violent criminal activity.",
    "unsafe"                    : "Content flagged as unsafe.",
}


# ── Model Architecture ─────────────────────────────────────────────────────────

class ToxicClassifier(nn.Module):
    def __init__(self, input_size: int, num_classes: int, hidden_size: int = 100):
        super(ToxicClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            batch_first = True,
            bidirectional = True
        )
        lstm_output_size = hidden_size * 2
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.classifier(last_output)


# ── Loading ────────────────────────────────────────────────────────────────────

def load_word2vec(path: str = WORD2VEC_PATH) -> Word2Vec:
    """Load the Word2Vec embedding model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Word2Vec model not found at: {path}")
    return Word2Vec.load(path)


def load_model(path: str = MODEL_PATH) -> ToxicClassifier:
    """Load the trained ToxicClassifier weights from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model weights not found at: {path}")
    model = ToxicClassifier(
        input_size  = EMBED_SIZE,
        num_classes = NUM_CLASSES,
        hidden_size = HIDDEN_SIZE
    ).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess(text: str) -> list[str]:
    """
    Tokenize and clean a raw text string.
    
    Steps:
        1. Lowercase
        2. Word tokenization
        3. Punctuation removal
    
    Args:
        text: raw input string
    
    Returns:
        list of clean tokens
    """
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in string.punctuation]
    return tokens


def encode(tokens: list[str], word2vec: Word2Vec) -> torch.Tensor | None:
    """
    Convert a list of tokens into a single tensor using Word2Vec embeddings.
    Unknown tokens are skipped gracefully.

    Args:
        tokens:   list of string tokens
        word2vec: loaded Word2Vec model

    Returns:
        tensor of shape (seq_len, embed_size) or None if no tokens could be encoded
    """
    vectors = [
        torch.from_numpy(word2vec.wv[token])
        for token in tokens
        if token in word2vec.wv
    ]
    if not vectors:
        return None
    return torch.stack(vectors)


# ── Inference ──────────────────────────────────────────────────────────────────

def run_inference(
    query       : str,
    description : str,
    model       : ToxicClassifier,
    word2vec    : Word2Vec
) -> dict:
    """
    Run the full inference pipeline on a query and image description.

    Args:
        query:       the user's search query
        description: the image description associated with the query
        model:       loaded ToxicClassifier
        word2vec:    loaded Word2Vec model

    Returns:
        dict with keys:
            - label:       predicted category name
            - description: human readable explanation of the category
            - confidence:  confidence score for the predicted class (0-100%)
            - is_safe:     boolean, True if predicted label is Safe
    """
    # preprocess
    query_tokens = preprocess(query)
    desc_tokens  = preprocess(description)

    # encode
    query_tensor = encode(query_tokens, word2vec)
    desc_tensor  = encode(desc_tokens,  word2vec)

    if query_tensor is None and desc_tensor is None:
        raise ValueError("No recognizable words found in either input.")

    # combine — skip whichever is None
    parts = [t for t in [query_tensor, desc_tensor] if t is not None]
    combined = torch.cat(parts, dim=0).unsqueeze(0).to(DEVICE)  # (1, seq_len, embed)

    # inference
    with torch.no_grad():
        logits     = model(combined)
        probs      = torch.softmax(logits, dim=1).squeeze()
        pred_idx   = logits.argmax(dim=1).item()
        confidence = probs[pred_idx].item()

    predicted_label = IDX2LABEL[pred_idx]

    return {
        "label"      : predicted_label,
        "description": LABEL_DESCRIPTIONS[predicted_label],
        "confidence" : confidence,
        "is_safe"    : predicted_label == "Safe",
    }


# ── Main Entry Point ───────────────────────────────────────────────────────────

def predict(query: str, description: str) -> dict:
    """
    Full pipeline: load models, preprocess, encode, infer, and return result.
    This is the main function a developer should call.

    Args:
        query:       the user's search query string
        description: the image description string

    Returns:
        dict with prediction results (label, description, confidence, is_safe)

    Example:
        >>> result = predict("how to make explosives", "a dark warehouse")
        >>> print(result["label"])       # Violent Crimes
        >>> print(result["confidence"])  # 0.87
        >>> print(result["is_safe"])     # False
    """
    word2vec = load_word2vec()
    model    = load_model()
    result   = run_inference(query, description, model, word2vec)
    return result


def display(result: dict) -> None:
    """
    Print the prediction result in a clean, human readable format.

    Args:
        result: dict returned by predict()
    """
    status = "✓ SAFE" if result["is_safe"] else "⚠ FLAGGED"
    print(f"\n{'─' * 45}")
    print(f"  Status     : {status}")
    print(f"  Category   : {result['label']}")
    print(f"  Confidence : {result['confidence']:.0%}")
    print(f"  Details    : {result['description']}")
    print(f"{'─' * 45}\n")


# ── Usage ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # result = predict(
    #     query       = "how to make a bomb",
    #     description = "a person standing near chemicals"
    # )
    # display(result)

    result = predict(
        query       = "I love cheese",
        description = "a sandwitch with cheese inside it"
    )
    display(result)