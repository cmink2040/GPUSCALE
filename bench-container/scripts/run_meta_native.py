#!/usr/bin/env python3
"""Run inference using Meta's official llama-models library with native .pth format.

Output format matches what virt-runner's parse_vllm_output expects:
  Throughput: <N> tokens/s
  TTFT: <N> ms
  Total time: <N> s
  Generated <N> tokens
"""

import os
import sys
import time

MODEL_DIR = os.environ.get("MODEL_DIR", "/models")


def main():
    if len(sys.argv) < 2:
        print("Usage: run_meta_native.py <prompt> [max_tokens] [temperature] [top_p]", file=sys.stderr)
        sys.exit(1)

    prompt = sys.argv[1]
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    temperature = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    top_p = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

    try:
        from models.llama3.generation import Llama3
        from models.datatypes import RawMessage
    except ImportError:
        print("ERROR: llama-models not installed. Install with: pip install llama-models", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {MODEL_DIR} (Meta native .pth)...", file=sys.stderr)

    generator = Llama3.build(
        ckpt_dir=MODEL_DIR,
        max_seq_len=max_tokens + 512,
        max_batch_size=1,
        world_size=1,
        device="cuda",
    )

    print("Running inference...", file=sys.stderr)

    # Build message in Meta's format: List[List[RawMessage]]
    messages_batch = [[RawMessage(role="user", content=prompt)]]

    generated_tokens = 0
    first_token_time = None
    start = time.perf_counter()

    for token_results in generator.chat_completion(
        messages_batch=messages_batch,
        temperature=temperature if temperature > 0 else 0.6,
        top_p=top_p,
        max_gen_len=max_tokens,
    ):
        result = token_results[0]
        if first_token_time is None and result.text:
            first_token_time = time.perf_counter()
        generated_tokens += 1
        if result.finished:
            break

    end = time.perf_counter()

    wall_time_s = end - start
    tokens_per_sec = generated_tokens / wall_time_s if wall_time_s > 0 else 0
    ttft_ms = (first_token_time - start) * 1000 if first_token_time else 0

    print(f"Throughput: {tokens_per_sec:.2f} tokens/s")
    print(f"TTFT: {ttft_ms:.2f} ms")
    print(f"Total time: {wall_time_s:.2f} s")
    print(f"Generated {generated_tokens} tokens")


if __name__ == "__main__":
    main()
