"""Parse nvidia-smi output and engine output into structured metrics."""

from __future__ import annotations

import re
import statistics
import subprocess
from dataclasses import dataclass, field

from virt_runner.models import AggregateMetrics, IterationMetrics


# ---------------------------------------------------------------------------
# nvidia-smi snapshot
# ---------------------------------------------------------------------------


@dataclass
class NvidiaSmiSnapshot:
    """A point-in-time reading from nvidia-smi."""

    gpu_index: int = 0
    gpu_utilization_pct: float = 0.0
    memory_used_mib: float = 0.0
    memory_total_mib: float = 0.0
    power_draw_w: float = 0.0
    temperature_c: float = 0.0


def query_nvidia_smi() -> list[NvidiaSmiSnapshot]:
    """Query nvidia-smi for current GPU utilization, memory, power, temperature."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    snapshots: list[NvidiaSmiSnapshot] = []
    for line in result.stdout.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            snapshots.append(
                NvidiaSmiSnapshot(
                    gpu_index=int(parts[0]),
                    gpu_utilization_pct=float(parts[1]),
                    memory_used_mib=float(parts[2]),
                    memory_total_mib=float(parts[3]),
                    power_draw_w=float(parts[4]),
                    temperature_c=float(parts[5]),
                )
            )
        except (ValueError, IndexError):
            continue
    return snapshots


def parse_nvidia_smi_output(raw: str) -> list[NvidiaSmiSnapshot]:
    """Parse raw nvidia-smi CSV output (same format as query_nvidia_smi) from a string.

    Useful for parsing nvidia-smi output collected remotely over SSH.
    """
    snapshots: list[NvidiaSmiSnapshot] = []
    for line in raw.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            snapshots.append(
                NvidiaSmiSnapshot(
                    gpu_index=int(parts[0]),
                    gpu_utilization_pct=float(parts[1]),
                    memory_used_mib=float(parts[2]),
                    memory_total_mib=float(parts[3]),
                    power_draw_w=float(parts[4]),
                    temperature_c=float(parts[5]),
                )
            )
        except (ValueError, IndexError):
            continue
    return snapshots


# ---------------------------------------------------------------------------
# Engine output parsers
# ---------------------------------------------------------------------------


@dataclass
class EngineTimings:
    """Timings extracted from an inference engine's output."""

    tokens_generated: int = 0
    tokens_per_sec: float = 0.0
    time_to_first_token_ms: float = 0.0
    prompt_eval_rate_tokens_per_sec: float = 0.0
    wall_time_s: float = 0.0


def parse_llamacpp_output(output: str) -> EngineTimings:
    """Parse llama.cpp server/bench output for timing information.

    Looks for patterns like:
      - "eval time = ... ms / ... tokens (... tokens/s)"
      - "prompt eval time = ... ms / ... tokens"
      - "total time = ... ms"
      - "llama_print_timings: ... "
    """
    timings = EngineTimings()

    # llama.cpp timings format (from llama_print_timings):
    #   llama_print_timings: prompt eval time =   123.45 ms /    10 tokens (   12.35 ms per token,    81.00 tokens per second)
    #   llama_print_timings:        eval time =  5678.90 ms /   512 runs   (   11.09 ms per token,    90.17 tokens per second)
    #   llama_print_timings:       total time = 12345.67 ms /   522 tokens

    prompt_eval_match = re.search(
        r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens?\s*\("
        r"\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)",
        output,
    )
    if prompt_eval_match:
        prompt_time_ms = float(prompt_eval_match.group(1))
        timings.prompt_eval_rate_tokens_per_sec = float(prompt_eval_match.group(4))
        timings.time_to_first_token_ms = prompt_time_ms

    eval_match = re.search(
        r"(?<!prompt\s)eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*(?:runs|tokens)\s*\("
        r"\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)",
        output,
    )
    if eval_match:
        timings.tokens_generated = int(eval_match.group(2))
        timings.tokens_per_sec = float(eval_match.group(4))

    total_match = re.search(
        r"total time\s*=\s*([\d.]+)\s*ms",
        output,
    )
    if total_match:
        timings.wall_time_s = float(total_match.group(1)) / 1000.0

    return timings


def parse_vllm_output(output: str) -> EngineTimings:
    """Parse vLLM benchmark output for timing information.

    Looks for patterns like:
      - "Throughput: ... tokens/s"
      - "Total time: ... s"
      - "TTFT: ... ms"
    """
    timings = EngineTimings()

    throughput_match = re.search(r"[Tt]hroughput:\s*([\d.]+)\s*tokens?/s", output)
    if throughput_match:
        timings.tokens_per_sec = float(throughput_match.group(1))

    ttft_match = re.search(r"TTFT:\s*([\d.]+)\s*ms", output)
    if ttft_match:
        timings.time_to_first_token_ms = float(ttft_match.group(1))

    total_match = re.search(r"[Tt]otal time:\s*([\d.]+)\s*s", output)
    if total_match:
        timings.wall_time_s = float(total_match.group(1))

    tokens_match = re.search(r"[Gg]enerated\s+(\d+)\s+tokens?", output)
    if tokens_match:
        timings.tokens_generated = int(tokens_match.group(1))

    return timings


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class MetricsCollector:
    """Collects per-iteration metrics and gpu snapshots, then aggregates."""

    iteration_metrics: list[IterationMetrics] = field(default_factory=list)
    gpu_snapshots: list[list[NvidiaSmiSnapshot]] = field(default_factory=list)

    def record_iteration(
        self,
        iteration: int,
        engine_timings: EngineTimings,
        gpu_snapshot: list[NvidiaSmiSnapshot] | None = None,
    ) -> IterationMetrics:
        """Record metrics for a single iteration."""
        peak_vram = 0.0
        power = 0.0
        util = 0.0
        temp = 0.0
        if gpu_snapshot:
            peak_vram = max(s.memory_used_mib for s in gpu_snapshot)
            power = sum(s.power_draw_w for s in gpu_snapshot) / len(gpu_snapshot)
            util = sum(s.gpu_utilization_pct for s in gpu_snapshot) / len(gpu_snapshot)
            temp = max(s.temperature_c for s in gpu_snapshot)
            self.gpu_snapshots.append(gpu_snapshot)

        metrics = IterationMetrics(
            iteration=iteration,
            tokens_per_sec=engine_timings.tokens_per_sec,
            time_to_first_token_ms=engine_timings.time_to_first_token_ms,
            prompt_eval_rate_tokens_per_sec=engine_timings.prompt_eval_rate_tokens_per_sec,
            peak_vram_mib=peak_vram,
            power_draw_avg_w=power,
            power_draw_peak_w=max(s.power_draw_w for s in gpu_snapshot) if gpu_snapshot else 0.0,
            gpu_utilization_pct=util,
            gpu_temperature_c=temp,
            wall_time_s=engine_timings.wall_time_s,
            tokens_generated=engine_timings.tokens_generated,
        )
        self.iteration_metrics.append(metrics)
        return metrics

    def aggregate(self, warmup_iterations: int = 1) -> AggregateMetrics:
        """Compute aggregate metrics, excluding warmup iterations."""
        real = self.iteration_metrics[warmup_iterations:]
        if not real:
            return AggregateMetrics()

        tps = [m.tokens_per_sec for m in real if m.tokens_per_sec > 0]
        ttfts = [m.time_to_first_token_ms for m in real if m.time_to_first_token_ms > 0]
        prompt_rates = [
            m.prompt_eval_rate_tokens_per_sec
            for m in real
            if m.prompt_eval_rate_tokens_per_sec > 0
        ]

        return AggregateMetrics(
            tokens_per_sec_mean=statistics.mean(tps) if tps else 0.0,
            tokens_per_sec_std=statistics.stdev(tps) if len(tps) > 1 else 0.0,
            ttft_mean_ms=statistics.mean(ttfts) if ttfts else 0.0,
            ttft_std_ms=statistics.stdev(ttfts) if len(ttfts) > 1 else 0.0,
            prompt_eval_rate_mean=statistics.mean(prompt_rates) if prompt_rates else 0.0,
            peak_vram_mib=max(m.peak_vram_mib for m in real) if real else 0.0,
            power_draw_avg_w=(
                statistics.mean(m.power_draw_avg_w for m in real if m.power_draw_avg_w > 0)
                if any(m.power_draw_avg_w > 0 for m in real)
                else 0.0
            ),
            power_draw_peak_w=max(m.power_draw_peak_w for m in real) if real else 0.0,
            gpu_utilization_pct_mean=(
                statistics.mean(m.gpu_utilization_pct for m in real if m.gpu_utilization_pct > 0)
                if any(m.gpu_utilization_pct > 0 for m in real)
                else 0.0
            ),
            gpu_temperature_c_max=max(m.gpu_temperature_c for m in real) if real else 0.0,
            wall_time_total_s=sum(m.wall_time_s for m in real),
        )
