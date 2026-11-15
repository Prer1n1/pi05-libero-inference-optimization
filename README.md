# π0.5 LIBERO Inference Optimization

**Independent research in Vision-Language-Action (VLA) inference efficiency, GPU profiling, and closed-loop robot evaluation.**

> How far can we reduce the inference latency of an existing fine-tuned π0.5 robot policy while maintaining successful robot manipulation? Once flow-matching denoising is reduced to two steps, **which remaining operations dominate action-generation latency**, and can targeted runtime optimizations accelerate them further?

This repository studies the **inference system**, not a newly trained VLA model. It combines controlled GPU microbenchmarks, component-level profiling, and robot success measurements in **LIBERO Spatial**. The planned research contribution is to identify and optimize a measured bottleneck, then demonstrate benefits against a **ame-step, unmodified policy baseline** and validate the optimized policy in closed-loop simulation.

**Artifacts:** [Research code and results](https://github.com/Prer1n1/pi05-libero-inference-optimization) · [Hugging Face experiment dataset (currently private)](https://huggingface.co/datasets/Prerana9/pi05-libero-inference-benchmarks)

## 1. Key findings so far

| Metric | Original: 10 flow steps | Reduced: 2 flow steps |
|---|---:|---:|
| LIBERO Spatial episode success | **99/100 (99%)** | **97/100 (97%)** |
| Mean action-chunk inference latency | **170.670 ms** | **54.746 ms** |
| P50 latency | 170.166 ms | 53.316 ms |
| P95 latency | 173.567 ms | 61.610 ms |
| P99 latency | 174.993 ms | 64.800 ms |
| Peak PyTorch GPU memory allocated | 8.863 GiB | 8.863 GiB |

On an **NVIDIA H100 80GB**, using a fixed observation and fixed input action noise, the two-step setting produced an observed **3.12× reduction in mean action-chunk latency relative to ten steps** (about **67.9% less elapsed time**). Full-suite closed-loop success changed from **99/100 to 97/100 episodes** in the runs performed.

**Scope of the result:** The speedup is from an existing **inference configuration change**, not a new algorithm, a modified CUDA kernel, or model fine-tuning. The 2-point success difference is not enough to establish statistical equivalence. The microbenchmark measures repeated policy action generation on one observation per setting, **not** whole-robot control-cycle latency or production serving throughput.

Raw evidence: [10-step latency](results/benchmark_10steps.json) · [2-step latency](results/benchmark_2steps.json) · [10-step full-suite evaluation](results/eval_info.json) · [2-step full-suite evaluation](results/spatial_2steps_100ep.json).

## 2. Why this problem matters

A VLA policy converts visual observations and a language instruction into a chunk of robot actions. The policy must generate useful actions fast enough to support closed-loop interaction, while consuming a practical amount of GPU computation.

π0.5 uses a flow-matching process for action generation. It starts from action-space noise and iteratively updates the proposed trajectory. More integration steps require additional action-expert computation, but fewer steps can change action quality and task success.

**Reducing denoising steps is only the first experiment.** Even at two steps, the model still performs conditioning, prefix processing, attention/KV-cache operations, action-expert forward passes, and output processing. A meaningful **systems-level research contribution** requires measuring which of these operations actually costs time and then improving that path without changing the inference budget during the comparison.

### Research objectives

1. Establish an accurate **latency-versus-robot-success baseline** using the existing model.
2. Attribute inference cost to conditioning, prefix/KV-cache preparation, flow-matching denoising, attention, integration, and runtime overhead.
3. Implement **one targeted runtime optimization** justified by profiling.
4. Compare optimized and original implementations at **the same number of flow steps**, with matching inputs and checkpoints.
5. Confirm the speedup and identify any regressions through **closed-loop LIBERO evaluations**.

## 3. π0.5 action-generation path

The following is a conceptual view of the evaluated policy; it is not a measured per-component timing breakdown.

```text
Robot cameras / state + language task instruction
                    |
                    v
           Observation preprocessing
                    |
                    v
          Visual/language conditioning
                    |
                    v
          Prefix transformer computation
              + attention KV cache
                    |
                    v
           Sample action-space noise
                    |
                    v
           Flow-matching integration
             repeat N iterations:
             - action-expert forward
             - predict update/velocity
             - update action trajectory
                    |
                    v
          Predicted action chunk
                    |
                    v
      Execute selected actions on robot
                    |
                    v
         New observation -> replan
```

Two different parameters must not be confused:

- **`num_inference_steps`**: the flow-matching denoising/integration count (10, 5, or 2 in these experiments). More steps imply more repeated action-expert computations.
- **`n_action_steps = 10`**: how many predicted actions the controller executes before requesting another action chunk. The configured predicted action chunk contains **50 actions**. This execution horizon is **not** the denoising count.

Relevant pinned LeRobot code paths include `PI05Policy.predict_action_chunk`, `PI05Pytorch.sample_actions`, and `PI05Pytorch.denoise_step` in `src/lerobot/policies/pi05/modeling_pi05.py`, together with integration in `src/lerobot/policies/common/flow_matching.py`.

In the existing source, prefix computation and its KV cache are used during action sampling. This research **has not yet implemented a new prefix-cache optimization**. The cost of cache copying, attention, and fixed conditioning work is a candidate for detailed measurement.

## 4. Experiment setup

| Parameter | Recorded value |
|---|---|
| Policy | Existing fine-tuned π0.5 LIBERO policy (`--policy.path=lerobot/pi05_libero_finetuned`) |
| Task suite | LIBERO Spatial, 10 manipulation tasks |
| Full-suite trials | 10 episodes per task = 100 episodes per configuration |
| Episode maximum | 280 environment steps |
| Recorded baseline seed | 1000 |
| Flow steps | 10 and 2 in full-suite evaluation; 5 also in exploratory profiling/task-level checks |
| Executed actions per chunk | 10 |
| GPU | NVIDIA H100 80GB HBM3 |
| Driver / CUDA runtime | 595.91.07 / CUDA 13.2 |
| Python | 3.12.15 |
| PyTorch | 2.15.0+cu132 |
| LeRobot | 0.7.1 local checkout, commit `fe1c54594f6ba275fccf210b9f7b11f0e2324b89` |
| Transformers | 5.22.2 |
| MuJoCo / robosuite / hf-libero | 3.3.2 / 1.4.0 / 0.1.4 |
| Headless rendering | OSMesa: `MUJOCO_GL=osmesa`, `PYOPENGL_PLATFORM=osmesa` |

The original ten-step suite took approximately **4,068 seconds total evaluation wall time** (40.68 seconds/episode). This includes simulation/evaluation overhead and must **not** be interpreted as the model's inference latency.

### Checkpoint provenance

The model weights were sourced from an **existing third-party fine-tuned checkpoint**; no π0.5 training or fine-tuning was performed for these results. A matching recovery source was recorded as [`lerobot/pi05_libero_finetuned_v044`](https://huggingface.co/lerobot/pi05_libero_finetuned_v044). The local model was originally loaded using `lerobot/pi05_libero_finetuned`. The original pod verified the checkpoint's `model.safetensors` **SHA-256**:

```text
877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae
```

See [checkpoint_recovery.txt](checkpoint_recovery.txt). Do not assume the model files, original virtual environment, or complete simulator assets were committed to this repository. **Full restoration on a replacement GPU pod has not yet been validated.**

## 5. Experiment A: closed-loop robot success

The outcome is **successful episode completion in a simulator**, not offline classification accuracy.

| Configuration | Observed success | Source |
|---|---:|---|
| LIBERO Spatial, 10 flow steps | **99/100** | [`results/eval_info.json`](results/eval_info.json) |
| LIBERO Spatial, 2 flow steps | **97/100** | [`results/spatial_2steps_100ep.json`](results/spatial_2steps_100ep.json) |
| Standalone Task 3, 5 flow steps | **9/10** | [`results/task3_5steps_10ep.json`](results/task3_5steps_10ep.json) |
| Standalone Task 3, 2 flow steps | **9/10** | [`results/task3_2steps_10ep.json`](results/task3_2steps_10ep.json) |

**Interpretation:** Reducing to two steps retained high observed task success in this limited sample. The ten-step full-suite run recorded one unsuccessful episode; the two-step run recorded three. Standalone Task 3 experiments are separate runs and do not establish five-step performance for the entire ten-task suite. Additional seeds/trials are needed for a reliable non-inferiority assessment.

## 6. Experiment B: action-chunk inference latency

Implementation: [`scripts/benchmark_pi05.py`](scripts/benchmark_pi05.py).

The benchmark intercepts `PI05Policy.predict_action_chunk` during a real LIBERO evaluation to capture a **preprocessed observation batch**, then calls the existing prediction method repeatedly on that fixed observation with **identical seeded action noise**.

- **Warm-up:** 15 action-generation calls.
- **Timed calls:** 120 per configuration.
- **Timing:** `time.perf_counter()`, with `torch.cuda.synchronize()` before and after every timed call.
- **Statistics:** mean, median/P50, P95, P99, min, max, and peak **PyTorch allocated** GPU memory.
- **Held fixed:** H100 device, observation within a run, noise, action execution setting and benchmark harness; **flow steps changed**.

| Statistic | 10 flow steps | 2 flow steps |
|---|---:|---:|
| Mean | 170.670 ms | 54.746 ms |
| P50 | 170.166 ms | 53.316 ms |
| P95 | 173.567 ms | 61.610 ms |
| P99 | 174.993 ms | 64.800 ms |
| Min | 169.252 ms | 52.963 ms |
| Max | 199.133 ms | 67.121 ms |
| Peak GPU allocated | 8.863 GiB | 8.863 GiB |

**Derived:** `170.670 / 54.746 ≈ 3.12×` and `(170.670 − 54.746) / 170.670 ≈ 67.9%`.

This is **synchronized wall-clock action-chunk latency**, not GPU-only kernel time, whole-simulation time, robot servo frequency, or throughput under concurrent requests. Repeating one observation does not capture latency variation over all scenes or language instructions. PyTorch allocated memory is not equivalent to total GPU memory reserved or consumed.

## 7. Experiment C: exploratory component timing

Implementation: [`scripts/profile_pi05.py`](scripts/profile_pi05.py).

An exploratory profiler wraps `predict_action_chunk`, `sample_actions`, and `denoise_step` and synchronizes CUDA at nested timing boundaries. Each run recorded **8 action chunks** and excluded the first chunk from its summary (**7 measured chunks**).

| Flow steps | Mean instrumented chunk time | Mean instrumented denoising time | Mean residual sampling time |
|---|---:|---:|---:|
| 10 | 188.677 ms | 147.809 ms | 36.443 ms |
| 5 | 122.037 ms | 78.608 ms | 38.432 ms |
| 2 | 86.132 ms | 33.871 ms | 47.954 ms |

Data: [10-step](results/inference_profile_10steps.json) · [5-step](results/inference_profile_5steps.json) · [2-step](results/inference_profile_2steps.json).

**Important limitations:** These figures include instrumentation/synchronization overhead and are **not directly comparable** with the 120-call benchmark. The field labeled `other_sampling_latency` is calculated as `sample_actions - denoise_step` timing; it does **not** individually measure prefix processing or cache creation. The preliminary pattern motivates finer profiling, but does not prove that prefix computation is the dominant bottleneck.

## 8. Research plan and decision criteria

### Phase 1 — establish baselines and isolate bottlenecks

- [x] Verify model checkpoint integrity and pin the LeRobot revision.
- [x] Evaluate 10-step and 2-step LIBERO Spatial success.
- [x] Record 120-sample action-chunk latency distributions and exploratory component timings.
- [x] Back up scripts/results to GitHub and data to a Hugging Face Dataset.
- [ ] Reproduce the environment and run a smoke test on a replacement GPU pod.
- [ ] Measure diverse observations and control for warm-up, GPU clock/power, task, and seed effects.
- [ ] Separate conditioning, prefix/KV-cache preparation, denoising passes, integration and postprocessing timings; corroborate with PyTorch/CUDA profiling when possible.

### Phase 2 — implement a real runtime improvement

Potential directions, **subject to measured evidence**:

- Eliminate unnecessary KV-cache copying or memory movement while preserving cache semantics.
- Improve action-expert attention execution where mask/caching behavior and numerical accuracy permit.
- Reduce repeated preprocessing, tensor conversion, allocation, Python dispatch or framework overhead.
- Evaluate compile/graph techniques only if shape/dynamic-control constraints permit safe replay.

**Control experiment:** compare **original 2-step π0.5** versus **optimized 2-step π0.5** on identical observations, noise, model weights, GPU, warm-up, and measurement procedure. Report speed distributions, memory, numerical/action output differences, and a mechanistic explanation of the improvement.

### Phase 3 — validate the optimized policy

- [ ] Re-run the full LIBERO Spatial closed-loop suite on the optimized implementation.
- [ ] Repeat across seeds and investigate task-level failures.
- [ ] Measure whether predicted actions remain numerically equivalent or behaviorally close, and whether task success changes.
- [ ] Report trade-offs and regressions, not only best-case speedups.
- [ ] Optionally study an adaptive denoising budget **after** establishing a fair fixed-step optimized runtime comparison.

**Success criterion:** A genuine inference-systems contribution requires a reproducible **code-level optimization** that removes a **measured bottleneck**, provides a fair **same-step speedup**, and preserves acceptable robot-task performance. The current 3.12× configuration-level result is a necessary baseline, **not yet this contribution**.

## 9. Repository structure

```text
.
├── README.md
├── checkpoint_recovery.txt           # Checkpoint hash and recovery notes
├── gpu_environment.txt               # Original H100 / driver snapshot
├── lerobot_commit.txt                 # Pinned LeRobot source commit
├── working_environment.txt           # Dependency snapshot; local path must be adapted
├── scripts/
│   ├── benchmark_pi05.py              # Fixed-observation GPU action-chunk benchmark
│   └── profile_pi05.py                # Exploratory nested inference timers
└── results/
    ├── eval_info.json                 # 10-step full-suite evaluation
    ├── spatial_2steps_100ep.json      # 2-step full-suite evaluation
    ├── benchmark_10steps.json
    ├── benchmark_2steps.json
    ├── inference_profile_10steps.json
    ├── inference_profile_5steps.json
    ├── inference_profile_2steps.json
    ├── task3_5steps_10ep.json
    ├── task3_2steps_10ep.json
    └── task_descriptions.json
```

Some result JSONs point to **rollout videos saved in the original pod**; those paths are metadata and **do not imply the videos were uploaded**. The benchmark scripts currently contain absolute paths under `/workspace/pi05-libero-research`, and they import the LeRobot evaluation entry point, so their execution requires a compatible environment rather than standalone Python.

### Restoring on a replacement GPU pod

The previously working environment had Python **3.12.15**, PyTorch **2.15.0+cu132**, LeRobot commit **`fe1c54594f6ba275fccf210b9f7b11f0e2324b89`**, and software-rendered MuJoCo via OSMesa. The checkpoint was verified by SHA-256 as shown above. Recovery requires recreating the CUDA-compatible environment, checking out the source commit, downloading model and preprocessing files, installing LIBERO/simulator assets, and running a smoke test.

**A fully tested one-command recovery installer is not present yet.** In particular, [`working_environment.txt`](working_environment.txt) includes an editable `file:///workspace/.../lerobot` dependency and CUDA-specific wheel versions, so it must not be treated as a universally portable `pip install -r` file.

## 10. Attribution and current status

- Original pretrained/fine-tuned policy weights belong to their respective authors and maintainers; this repository **does not** claim to have trained them or redistribute the large checkpoint.
- The simulator and evaluation stack builds on [LeRobot](https://github.com/huggingface/lerobot) and [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).
- **Completed:** Baseline simulator runs, reduced-step success comparison, controlled H100 action-generation benchmark, exploratory profiling, and result backups.
- **In progress:** Recovery on a new GPU pod, rigorous component profiling, bottleneck attribution, and an original targeted runtime optimization.
- **Not yet demonstrated:** New kernels, new model weights, adaptive denoising, optimized-runtime speedup, production deployment, or statistical non-inferiority.

**Intended final contribution:** An evidence-based VLA inference engineering study linking **where GPU inference time is spent**, **which software change reduces it**, and **whether robots still complete their tasks**.
