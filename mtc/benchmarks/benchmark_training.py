"""
End-to-End Training Benchmark: FP32 vs Balanced Ternary (BitNet-style QAT with STE).
Demonstrates real loss convergence and memory reduction during neural network training.
"""

import time
from typing import Tuple, List
import numpy as np

from mtc.nn.model import DeepMLP_FP32, DeepMLP_Ternary


def softmax_cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute softmax cross-entropy loss and gradient."""
    batch_size = logits.shape[0]
    # Stable softmax
    shift_logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shift_logits)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Loss: -log(p_target)
    correct_logprobs = -np.log(probs[np.arange(batch_size), targets] + 1e-12)
    loss = float(np.mean(correct_logprobs))

    # Gradient: (probs - 1_target) / batch_size
    grad = probs.copy()
    grad[np.arange(batch_size), targets] -= 1.0
    grad = grad / batch_size

    return loss, grad


def generate_classification_dataset(n_samples: int = 1000, in_dim: int = 256, n_classes: int = 16, seed: int = 42):
    """Generate synthetic non-linear classification dataset."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, in_dim).astype(np.float32)
    # Underlying non-linear transformation to assign classes
    W_true = rng.randn(in_dim, n_classes).astype(np.float32)
    scores = X @ W_true + 0.1 * rng.randn(n_samples, n_classes)
    y = np.argmax(scores, axis=-1)
    return X, y


def benchmark_training_convergence(epochs: int = 15, batch_size: int = 32):
    print("=" * 75)
    print("  NEURAL NETWORK TRAINING CONVERGENCE & THROUGHPUT BENCHMARK")
    print("  Task: 16-Class Classification with 4-Layer MLP (~528K parameters)")
    print("  Comparing: Full FP32 Baseline vs Balanced Ternary (BitNet QAT with STE)")
    print("=" * 75)

    X_train, y_train = generate_classification_dataset(1024, 256, 16, seed=42)
    n_batches = len(X_train) // batch_size

    m_fp32 = DeepMLP_FP32(256, 512, 16, seed=100)
    m_ternary = DeepMLP_Ternary(256, 512, 16, seed=100)

    print(f"\n[1. Model Parameter Memory Footprint During Training]")
    print(f"  - FP32 Model:          {m_fp32.total_bytes / 1024.0:.1f} KB (32 bits/weight)")
    print(f"  - Balanced Ternary:    {m_ternary.total_bytes / 1024.0:.1f} KB (1.6 bits/weight packed)")

    # Train FP32 Model
    print(f"\n[2. Training FP32 Reference Model ({epochs} Epochs)...]")
    t0 = time.perf_counter()
    loss_history_fp32 = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for b in range(n_batches):
            x_b = X_train[b * batch_size : (b + 1) * batch_size]
            y_b = y_train[b * batch_size : (b + 1) * batch_size]

            logits, cache = m_fp32.forward(x_b)
            loss, grad = softmax_cross_entropy_loss(logits, y_b)
            m_fp32.backward_and_step(grad, cache, lr=0.05)
            epoch_loss += loss

        avg_loss = epoch_loss / n_batches
        loss_history_fp32.append(avg_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f}")

    t_fp32_total = (time.perf_counter() - t0) * 1000.0

    # Train Balanced Ternary Model
    print(f"\n[3. Training Balanced Ternary Model via STE QAT ({epochs} Epochs)...]")
    t0 = time.perf_counter()
    loss_history_ternary = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for b in range(n_batches):
            x_b = X_train[b * batch_size : (b + 1) * batch_size]
            y_b = y_train[b * batch_size : (b + 1) * batch_size]

            logits, cache = m_ternary.forward(x_b)
            loss, grad = softmax_cross_entropy_loss(logits, y_b)
            m_ternary.backward_and_step(grad, cache, lr=0.05)
            epoch_loss += loss

        avg_loss = epoch_loss / n_batches
        loss_history_ternary.append(avg_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f}")

    t_ternary_total = (time.perf_counter() - t0) * 1000.0

    print("\n" + "=" * 75)
    print("  TRAINING SUMMARY & COMPARISON")
    print("=" * 75)
    print(f"FP32 Baseline:")
    print(f"  - Initial Loss:        {loss_history_fp32[0]:.4f} -> Final Loss: {loss_history_fp32[-1]:.4f} (Delta: -{loss_history_fp32[0] - loss_history_fp32[-1]:.4f})")
    print(f"  - Total Train Time:    {t_fp32_total:.2f} ms ({t_fp32_total / (epochs * n_batches):.2f} ms/step)")
    print(f"  - Parameter Storage:   {m_fp32.total_bytes / 1024.0:.1f} KB")

    print(f"\nBalanced Ternary QAT (BitNet-style STE):")
    print(f"  - Initial Loss:        {loss_history_ternary[0]:.4f} -> Final Loss: {loss_history_ternary[-1]:.4f} (Delta: -{loss_history_ternary[0] - loss_history_ternary[-1]:.4f})")
    print(f"  - Total Train Time:    {t_ternary_total:.2f} ms ({t_ternary_total / (epochs * n_batches):.2f} ms/step)")
    print(f"  - Parameter Storage:   {m_ternary.total_bytes / 1024.0:.1f} KB ")
    chance = float(np.log(16))
    learned = loss_history_ternary[-1] < 0.95 * chance
    print(f"  - Chance-level loss:   {chance:.4f} (ln 16)")
    print(f"  - Convergence Status:  {'LEARNING' if learned else 'NOT LEARNING (loss at chance level)'}")


if __name__ == "__main__":
    benchmark_training_convergence(epochs=15, batch_size=32)
