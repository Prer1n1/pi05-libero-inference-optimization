import atexit
import json
import statistics
import time
from pathlib import Path

import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch
from lerobot.scripts.lerobot_eval import main


records = []
current = None

original_chunk = PI05Policy.predict_action_chunk
original_sample = PI05Pytorch.sample_actions
original_denoise = PI05Pytorch.denoise_step


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_denoise(self, *args, **kwargs):
    global current
    sync()
    start = time.perf_counter()
    result = original_denoise(self, *args, **kwargs)
    sync()
    ms = (time.perf_counter() - start) * 1000

    if current is not None:
        current["denoise_ms"] += ms
        current["denoise_calls"] += 1

    return result


def timed_sample(self, *args, **kwargs):
    global current
    current = {"denoise_ms": 0.0, "denoise_calls": 0}
    sync()
    start = time.perf_counter()

    try:
        result = original_sample(self, *args, **kwargs)
        sync()
        current["sample_ms"] = (time.perf_counter() - start) * 1000
        current["other_sample_ms"] = max(
            0.0, current["sample_ms"] - current["denoise_ms"]
        )
        records.append(current)
        return result
    finally:
        current = None


def timed_chunk(self, *args, **kwargs):
    sync()
    start = time.perf_counter()
    result = original_chunk(self, *args, **kwargs)
    sync()
    if records:
        records[-1]["chunk_ms"] = (time.perf_counter() - start) * 1000
    return result


def summary(values):
    values = sorted(values)
    if not values:
        return None
    return {
        "mean_ms": round(statistics.mean(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "p95_ms": round(values[int(0.95 * (len(values) - 1))], 3),
    }


def save_results():
    if not records:
        print("No action chunks measured.")
        return

    # First chunk can include GPU warm-up overhead.
    measured = records[1:] if len(records) > 1 else records

    result = {
        "total_chunks": len(records),
        "measured_chunks_excluding_first": len(measured),
        "chunk_latency": summary([r["chunk_ms"] for r in measured]),
        "sampling_latency": summary([r["sample_ms"] for r in measured]),
        "denoising_latency": summary([r["denoise_ms"] for r in measured]),
        "other_sampling_latency": summary(
            [r["other_sample_ms"] for r in measured]
        ),
        "denoise_calls_per_chunk": [
            r["denoise_calls"] for r in measured
        ],
        "gpu_peak_allocated_gb": round(
            torch.cuda.max_memory_allocated() / 1024**3, 3
        ) if torch.cuda.is_available() else None,
        "raw_records": records,
    }

    output = Path(
        "/workspace/pi05-libero-research/"
        "pi05-libero-inference-optimization/results/"
        "inference_profile.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))

    print("\n=== PI05 INFERENCE PROFILE ===")
    print(json.dumps(
        {k: v for k, v in result.items() if k != "raw_records"},
        indent=2
    ))
    print("Saved:", output)


PI05Policy.predict_action_chunk = timed_chunk
PI05Pytorch.sample_actions = timed_sample
PI05Pytorch.denoise_step = timed_denoise

atexit.register(save_results)

if __name__ == "__main__":
    main()
