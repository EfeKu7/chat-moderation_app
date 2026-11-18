import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sklearn.metrics import f1_score

from moderation import FILES, MODELS, Moderator, lexicon_flags, normalize
from train import SEED, load_data

CPU_BUDGET_MS = 50


def cpu_name():
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            name, _, value = line.partition(":")
            if name.strip() == "model name":
                return value.strip()
    return platform.processor() or platform.machine()


def benchmark(model_name, artifacts, texts, labels, indices):
    model = Moderator(model_name, artifacts)
    predictions = [
        prediction
        for start in range(0, len(texts), 32)
        for prediction in model.probabilities(texts[start:start + 32]).argmax(axis=1)
    ]
    for index in indices[:10]:
        model.classify(texts[index])
    latencies = []
    for index in indices:
        start = perf_counter()
        model.classify(texts[index])
        latencies.append((perf_counter() - start) * 1000)
    size = (artifacts / FILES[model_name]).stat().st_size
    if model_name in MODELS[2:]:
        size += sum(file.stat().st_size for file in (artifacts / "bert").iterdir()
                    if file.is_file())
    return {
        "model": model_name,
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "size_mb": size / 1_000_000,
        "mean_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/offenseval.zip"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.runs < 100 or args.threads < 1:
        parser.error("Use at least 100 requests and one CPU thread")
    torch.set_num_threads(args.threads)
    _, (texts, labels) = load_data(args.data)
    indices = np.random.default_rng(SEED).choice(len(texts), args.runs,
                                                  replace=args.runs > len(texts))
    report = {
        "training": json.loads((args.artifacts / "training.json").read_text()),
        "cpu": cpu_name(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "quantization_backend": torch.backends.quantized.engine,
        "torch_threads": args.threads,
        "test_messages": len(texts),
        "timed_requests_per_model": args.runs,
        "mean_cpu_budget_ms": CPU_BUDGET_MS,
        "lexicon_macro_f1": f1_score(
            labels, [lexicon_flags(normalize(text)) for text in texts], average="macro",
        ),
        "models": [],
    }
    for model_name in MODELS:
        metrics = benchmark(model_name, args.artifacts, texts, labels, indices)
        print(json.dumps(metrics), flush=True)
        report["models"].append(metrics)
    report["int8_mean_meets_budget"] = (
        report["models"][-1]["mean_latency_ms"] < CPU_BUDGET_MS
    )
    path = args.artifacts / "benchmark.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report written to {path}")


if __name__ == "__main__":
    main()