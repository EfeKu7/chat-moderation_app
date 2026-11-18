Real-Time Chat Moderation
SWE015 · Use Case 4

This Gradio application classifies Turkish chat messages as CLEAN or OFFENSIVE. It displays the model's confidence, masks flagged messages, and records how long each prediction takes. You can switch between four models in the Active model menu. The app starts with BERTurk INT8.
Run the project
You need Python 3.12, Git, and an internet connection for the first setup. Training downloads OffensEval-TR 2020 and the pretrained BERTurk checkpoint (about 445 MB).
1. Install the dependencies
Run these commands in Windows PowerShell. If you already downloaded the repository, open a terminal in the folder containing app.py and start at the third command.
git clone https://github.com/EfeKu7/chat-moderation_app.git
cd chat-moderation_app
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
On Linux, run these commands from the repository folder. In the remaining commands, use python in place of .\.venv\Scripts\python.exe.
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
2. Train the models
The first-run command trains on 1,024 messages and runs one BERTurk training epoch. It saves the model files in artifacts/.
.\.venv\Scripts\python.exe train.py --train-limit 1024 --epochs 1
For full training, run the same command without --train-limit 1024 --epochs 1. Full BERTurk training takes longer on CPU.
3. Open the app
.\.venv\Scripts\python.exe app.py
Open the local URL printed in the terminal. Enter a Turkish message such as “Selam arkadaşlar, iyi oyunlar!” and click Send. The side panel shows the model, label, and latency. If you see a missing-model error, complete step 2 first.
4. Run the benchmark
.\.venv\Scripts\python.exe benchmark.py
The script saves results in artifacts/benchmark.json. It scores all four models on the same 3,528 official OffensEval-TR test messages and times 200 individual CPU predictions per model after warm-up. The report includes Macro-F1, model size, mean and 95th-percentile latency, CPU details, and the result against the 50 ms mean latency target.
Model pipeline
The app normalizes noisy spelling, repeated letters, and Turkish letter case. It then runs the selected model, displays the predicted label and confidence, and replaces flagged text with [Message masked].
Model	How it works
TF-IDF + logistic regression	Uses short character sequences as features for a classical classifier.
Word2Vec + feedforward network	Learns word vectors from the training messages and uses their average to classify each message.
BERTurk FP32	Fine-tunes a pretrained Turkish language model for the two moderation labels.
BERTurk INT8	Quantizes the fine-tuned model's linear layers to reduce model size and CPU latency.


The benchmark also evaluates a separate rule-based baseline that uses edit distance to match spelling variations against a word list.
Example benchmark results
This run used 1,024 training messages and an AMD EPYC 9V74 CPU with four PyTorch threads. Macro-F1 gives the two labels equal weight. Lower latency means faster predictions.
Model	Macro-F1	Size (MB)	Mean latency (ms)
TF-IDF	0.640	0.50	0.56
Word2Vec + FNN	0.491	0.48	0.10
BERTurk FP32	0.529	443.56	45.55
BERTurk INT8	0.529	187.04	14.94


In this run, TF-IDF had the highest Macro-F1. BERTurk INT8 met the 50 ms mean latency target and retained BERTurk FP32's Macro-F1 to three decimal places. Latency varies by computer. The dataset contains Turkish social media posts, so gaming slang and quoted insults can still be misclassified.
Sources
- OffensEval-TR 2020 dataset release
- BERTurk pretrained checkpoint