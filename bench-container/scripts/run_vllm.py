#!/usr/bin/env python3
"""Run a single vLLM inference iteration and print metrics to stdout.

Output format matches what virt-runner's parse_vllm_output expects:
  Throughput: <N> tokens/s
  TTFT: <N> ms
  Total time: <N> s
  Generated <N> tokens
"""

import os
import sys
import time

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("ERROR: vLLM is not installed in this container.", file=sys.stderr)
    sys.exit(1)

# Cache the LLM instance across calls so we don't reload on every iteration
_llm_cache = {}


def get_llm(model_dir: str) -> LLM:
    """Get or create a cached LLM instance."""
    if model_dir not in _llm_cache:
        print(f"Loading model from {model_dir}...", file=sys.stderr)
        _llm_cache[model_dir] = LLM(
            model=model_dir,
            trust_remote_code=True,
            dtype="float16",
        )
    return _llm_cache[model_dir]


def main():
    if len(sys.argv) < 2:
        print("Usage: run_vllm.py <prompt> [max_tokens] [temperature] [top_p]", file=sys.stderr)
        sys.exit(1)

    prompt = sys.argv[1]
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    temperature = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    top_p = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

    model_dir = os.environ.get("MODEL_DIR", "/models")

    llm = get_llm(model_dir)

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    print("Running inference...", file=sys.stderr)

    start = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params)
    end = time.perf_counter()

    wall_time_s = end - start
    output = outputs[0]
    num_tokens = len(output.outputs[0].token_ids)
    tokens_per_sec = num_tokens / wall_time_s if wall_time_s > 0 else 0

    # Compute TTFT from metrics if available
    ttft_ms = 0.0
    if hasattr(output, "metrics") and output.metrics:
        if hasattr(output.metrics, "first_token_time") and output.metrics.first_token_time:
            ttft_ms = output.metrics.first_token_time * 1000

    # Print in the format the orchestrator expects
    print(f"Throughput: {tokens_per_sec:.2f} tokens/s")
    print(f"TTFT: {ttft_ms:.2f} ms")
    print(f"Total time: {wall_time_s:.2f} s")
    print(f"Generated {num_tokens} tokens")


if __name__ == "__main__":
    main()
