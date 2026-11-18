Real-Time Chat Moderation
SWE015 · Use Case 4

This Gradio app moderates Turkish chat messages. It labels each message as CLEAN or OFFENSIVE, shows the selected model's confidence, and masks flagged messages. The side panel lets you switch models and records the inference time of each message. BERTurk INT8 is the default model.
1. Install
Install Git and Python 3.11 or 3.12. Run the commands for your operating system.
Windows (PowerShell)
git clone https://github.com/EfeKu7/chat-moderation_app.git
cd chat-moderation_app
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
If you installed Python 3.11, use py -3.11 to create the environment. Replace python in the following sections with .\.venv\Scripts\python.exe in PowerShell.
Linux
git clone https://github.com/EfeKu7/chat-moderation_app.git
cd chat-moderation_app
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
These commands install CPU-only PyTorch. For GPU training, install the PyTorch 2.8.0 wheel for your CUDA setup in place of the CPU wheel. The app and benchmark use the CPU.
2. Train the models
For a shorter first run, train on a stratified sample of 1,024 messages.
python train.py --train-limit 1024 --epochs 1
For full training, run python train.py. Full BERTurk fine-tuning can take several hours on CPU. The script downloads OffensEval-TR 2020 and the pretrained BERTurk checkpoint, then saves the four trained models under artifacts/. If you already have a trusted copy of the model files, place the artifacts/ folder alongside app.py and proceed to the next step.
3. Run the application
python app.py
Open the local URL printed in the terminal. Type a Turkish chat message and click Gönder. The side panel displays the model, classification and processing time for each message.
4. Measure the models
python benchmark.py
The benchmark evaluates all four models on the same 3,528 official OffensEval-TR test messages. Full training starts with 31,756 official training messages and excludes exact matches with test messages after normalization. The benchmark reports Macro-F1, saved model size in MB, CPU details, and mean and p95 latency over 200 single-message requests per model after warm-up. It checks the INT8 model against a 50 ms mean inference budget and reports the edit-distance baseline's Macro-F1 separately.

Find the results in artifacts/benchmark.json. Report the training size and CPU used with your results.
How it works
Technique	What the project does
Classical NLP	Normalizes leetspeak, repeated letters and Turkish case. Uses an edit-distance lexicon as a separate rule baseline. The trained classical model uses character TF-IDF and logistic regression.
Neural network	Trains Word2Vec on the training messages. The mean token vectors feed a feedforward network trained with SGD and backpropagation.
Transformer	Fine-tunes BERTurk for binary classification. The tokenizer truncates input to 64 tokens.
Quantization	Applies dynamic INT8 quantization to BERTurk's linear layers. Embeddings and other layers retain floating-point precision.
The latency log includes text normalization, feature extraction or tokenization, inference and label selection. Timing starts after model loading and ends before the interface updates.

The training data comes from Turkish social media. Gaming slang and quoted insults can lead to classification errors. Confidence is the selected model's class probability.
Sources
- OffensEval-TR 2020 dataset release
- BERTurk pretrained checkpoint