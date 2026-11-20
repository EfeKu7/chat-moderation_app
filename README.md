# Real-Time Chat Moderation

SWE015 · Use Case 4

This Gradio application classifies Turkish chat messages as CLEAN or OFFENSIVE. It displays the model's confidence, masks flagged messages, and records how long each prediction takes. You can switch between four models in the Active model menu. The app starts with BERTurk INT8.

## Run the project

You need Python 3.12, Git, and an internet connection for the first setup. Training downloads OffensEval-TR 2020 and the pretrained BERTurk checkpoint (about 445 MB).

### 1. Install the dependencies

Run these commands in Windows PowerShell. If you already downloaded the repository, open a terminal in the folder containing `app.py` and start at the third command.

```
git clone https://github.com/EfeKu7/chat-moderation_app.git
cd chat-moderation_app
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On Linux, run these commands from the repository folder. In the remaining commands, use `python` in place of `.\.venv\Scripts\python.exe`.

```
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

### 2. Train the models

The command below trains the models on a stratified sample of 4,000 messages from the official training split. One message that also appears in the test set is removed, so 3,999 messages remain. BERTurk is fine-tuned for three epochs. The model files are saved in `artifacts/`, and the training settings are saved in `artifacts/training.json`. Training on CPU can take a while.

```
.\.venv\Scripts\python.exe train.py --epochs 3 --train-limit 4000
```

These are also the default values, so `train.py` without arguments gives the same setup.

### 3. Open the app

```
.\.venv\Scripts\python.exe app.py
```

Open the local URL printed in the terminal. Enter a Turkish message such as "Selam arkadaşlar, iyi oyunlar!" and click Send. The side panel shows the model, label, and latency. If you see a missing-model error, complete step 2 first.

### 4. Run the benchmark

```
.\.venv\Scripts\python.exe benchmark.py --cpu-name "Intel Core i7-10875H"
```

Replace the CPU name with your own processor. The script saves results in `artifacts/benchmark.json`. It scores all four models on the same 3,528 official OffensEval-TR test messages and times 200 individual CPU predictions per model after warm-up. The report includes Macro-F1, model size, mean and 95th-percentile latency, CPU details, and the result against the 50 ms latency target.

## Model pipeline

The app normalizes noisy spelling, repeated letters, and Turkish letter case. It then runs the selected model, displays the predicted label and confidence, and replaces flagged text with `[Message masked]`.

| Model | How it works |
|---|---|
| TF-IDF + logistic regression | Uses short character sequences as features for a classical classifier. |
| Word2Vec + feedforward network | Learns word vectors from the training messages and uses their average to classify each message. |
| BERTurk FP32 | Fine-tunes a pretrained Turkish language model for the two moderation labels. |
| BERTurk INT8 | Quantizes the fine-tuned model's linear layers to reduce model size and CPU latency. |

The benchmark also evaluates a separate rule-based baseline that uses edit distance to match spelling variations against a word list.

## Benchmark results

This run used 3,999 training messages, three BERTurk epochs, an Intel Core i7-10875H CPU on Windows and four PyTorch threads. Every model was evaluated on the same 3,528 test messages. Macro-F1 gives the two labels equal weight. Lower latency means faster predictions.

| Model | Macro-F1 | Size (MB) | Mean latency (ms) | p95 latency (ms) |
|---|---|---|---|---|
| TF-IDF + logistic regression | 0.691 | 1.27 | 1.14 | 1.74 |
| Word2Vec + FNN | 0.444 | 1.62 | 0.19 | 0.30 |
| BERTurk FP32 | 0.771 | 443.56 | 37.09 | 58.73 |
| BERTurk INT8 | 0.781 | 187.04 | 21.92 | 34.44 |

The rule-based edit-distance baseline scored a Macro-F1 of 0.493.

BERTurk INT8 had the highest Macro-F1. Compared with FP32 it reduced model size by 58%, mean latency by 41% and p95 latency by 41%. INT8 met the 50 ms target on both mean and p95 latency, while the FP32 model's p95 latency (58.73 ms) exceeded it. The 0.010 Macro-F1 difference between INT8 and FP32 comes from a single run, so the result only shows that quantization did not reduce quality here. Latency varies by computer. The dataset contains Turkish social media posts, so gaming slang and quoted insults can still be misclassified.


## Sources

- [OffensEval-TR 2020 dataset release](https://coltekin.github.io/offensive-turkish/)
- [BERTurk pretrained checkpoint (dbmdz/bert-base-turkish-cased)](https://huggingface.co/dbmdz/bert-base-turkish-cased)