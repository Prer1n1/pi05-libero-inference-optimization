import json
import math
import statistics
import time
from pathlib import Path

import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.scripts.lerobot_eval import main

ROOT = Path("/workspace/pi05-libero-research/pi05-libero-inference-optimization")
WARMUP = 15
REPEATS = 120

original_predict = PI05Policy.predict_action_chunk
benchmark_completed = False


def percentile(values, fraction):
    values = sorted(values)
    position = math.ceil(len(values) * fraction) - 1
    return values[max(0, position)]


def benchmark_once(self, batch):
    device = next(self.model.parameters()).device
    batch_size = next(
        value.shape[0]
        for value in batch.values()
        if isinstance(value, torch.Tensor) and value.ndim > 0
    )

    # Use identical action noise across repetitions.
    generator = torch.Generator(device=device)
    generator.manual_seed(12345)

    noise = torch.randn(
        batch_size,
        self.config.chunk_size,
        self.config.max_action_dim,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    def predict():
        return original_predict(self, batch, noise=noise)

    # Warm up GPU kernels and allocations.
    for _ in range(WARMUP):
        predict()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    latencies = []

    for _ in range(REPEATS):
        torch.cuda.synchronize()
        start = time.perf_counter()

        predict()

        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    result = {
        "benchmark": "fixed_observation_action_chunk",
        "gpu": torch.cuda.get_device_name(device),
        "num_inference_steps": self.config.num_inference_steps,
        "n_action_steps": self.config.n_action_steps,
        "warmup_iterations": WARMUP,
        "measured_iterations": REPEATS,
        "noise_seed": 12345,
        "mean_ms": round(statistics.mean(latencies), 3),
        "p50_ms": round(statistics.median(latencies), 3),
        "p95_ms": round(percentile(latencies, 0.95), 3),
        "p99_ms": round(percentile(latencies, 0.99), 3),
        "min_ms": round(min(latencies), 3),
        "max_ms": round(max(latencies), 3),
        "peak_gpu_allocated_gib": round(
            torch.cuda.max_memory_allocated(device) / 1024**3, 3
        ),
        "latencies_ms": latencies,
    }

    output = ROOT / "results" / (
        f"benchmark_{self.config.num_inference_steps}steps.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))

    print("\n=== PI05 LOW-OVERHEAD BENCHMARK ===", flush=True)
    for key, value in result.items():
        if key != "latencies_ms":
            print(f"{key}: {value}", flush=True)
    print(f"Saved to: {output}", flush=True)


def patched_predict(self, batch, *args, **kwargs):
    global benchmark_completed

    if not benchmark_completed:
        benchmark_completed = True
        benchmark_once(self, batch)

    return original_predict(self, batch, *args, **kwargs)


PI05Policy.predict_action_chunk = patched_predict

if __name__ == "__main__":
    main()
