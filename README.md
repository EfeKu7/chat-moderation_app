# Real-Time Chat Moderation System (Use Case 4)

An end-to-end real-time chat moderation application built for the **SWE015 – Introduction to Large Language Models** course. This system flags offensive and toxic messages on a CPU-only server within a strict latency budget (<50ms).

---

## 🚀 Project Overview & Model Ladder
This project implements the required four-model ladder evaluated on the same held-out test dataset:
1. **Classical Baseline (Area B - Classical NLP):** Regex-based normalization (handling leetspeak, repeated characters, and Turkish/English characters), lexicon rule matching, and TF-IDF statistical baseline.
2. **Neural Network (Area A - Deep Learning):** Custom Feedforward Neural Network (FNN) trained from scratch using Word2Vec embeddings with SGD and backpropagation.
3. **Transformer (Area C - Transformers):** Fine-tuned BERTurk / Multilingual encoder classification head for deep semantic toxicity analysis.
4. **Quantized Transformer (Area D - Efficient Models):** Dynamic INT8 quantized transformer model optimized for CPU execution to meet the sub-50ms latency budget.

---

## 🏛️ Architecture & Data Flow
1. **User Input:** The user enters a live chat message via the interactive Gradio web interface.
2. **Preprocessing:** Text is normalized through regex rules to handle noise and character manipulation.
3. **Inference Pipeline:** The message is evaluated concurrently across all 4 models in the model ladder.
4. **Post-processing & Masking:** If the active quantized model flags the text as `OFFENSIVE`, the message is automatically filtered and masked as `[MASKED - Toxic Content Detected]`.

---

## 📊 Performance & Metrics Summary
| Model Layer | Technique Area | Task Metric (Macro-F1) | Model Size (MB) | Latency (CPU, avg) |
| :--- | :--- | :--- | :--- | :--- |
| **Classical Baseline** | Area B (Classical NLP) | ~0.68 | ~2 MB | ~0.1 ms |
| **Neural Network** | Area A (Deep Learning) | ~0.74 | ~15 MB | ~4.5 ms |
| **Transformer (BERTurk)** | Area C (Transformers) | ~0.92 | ~440 MB | ~35 ms |
| **Quantized Transformer** | Area D (Efficient Models) | ~0.91 | ~110 MB | ~12 ms |

---

## 🛠️ Tech Stack & Requirements
- **Python 3.8+**
- **Gradio** (Interactive Web UI)
- **NumPy** (Numerical computations)

---

## ⚙️ Setup and Run Instructions

Follow these steps to set up and run the project locally:

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd chat-moderation