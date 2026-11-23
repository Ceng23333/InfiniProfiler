#!/usr/bin/env python3
"""Bounded deterministic vLLM client used by the InfiniProfiler/Nsight gate."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Sequence


def parse_int_list(raw: str) -> List[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value < 0 for value in values):
        raise ValueError("expected a comma-separated list of non-negative integers")
    return values


def exact_prompt(tokenizer: Any, target_tokens: int, request_index: int) -> str:
    vocabulary = [
        "tensor parallel profiling",
        "vLLM deterministic decode window",
        "Nsight Systems CUDA interval check",
        "InfiniProfiler exclusive attribution",
        "mixed prefill and decode traffic",
    ]
    ids = tokenizer.encode(vocabulary[request_index % len(vocabulary)], add_special_tokens=False)
    while len(ids) < target_tokens:
        ids.extend(tokenizer.encode(" " + vocabulary[(len(ids) + request_index) % len(vocabulary)], add_special_tokens=False))
    ids = ids[:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def request_once(url: str, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_ns = time.time_ns()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            response_body = response.read().decode("utf-8")
            status = response.status
        response_doc = json.loads(response_body)
        return {
            "success": True,
            "status": status,
            "start_epoch_ns": started_ns,
            "end_epoch_ns": time.time_ns(),
            "request": payload,
            "response": response_doc,
            "error": None,
        }
    except Exception as exc:
        return {
            "success": False,
            "status": None,
            "start_epoch_ns": started_ns,
            "end_epoch_ns": time.time_ns(),
            "request": payload,
            "response": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_window(
    *,
    endpoint: str,
    model: str,
    prompts: Sequence[str],
    output_length: int,
    seed: int,
    concurrency: int,
    timeout_s: float,
) -> Dict[str, Any]:
    url = endpoint.rstrip("/") + "/v1/completions"
    start_epoch_ns = time.time_ns()
    workers = min(concurrency, len(prompts))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                request_once,
                url,
                {
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": output_length,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "seed": seed + index,
                    "ignore_eos": True,
                    "stream": False,
                },
                timeout_s,
            )
            for index, prompt in enumerate(prompts)
        ]
        results = [future.result() for future in futures]
    end_epoch_ns = time.time_ns()

    completed = sum(1 for item in results if item["success"])
    completion_tokens = 0
    prompt_tokens = 0
    for item in results:
        usage = (item.get("response") or {}).get("usage") or {}
        completion_tokens += int(usage.get("completion_tokens", 0))
        prompt_tokens += int(usage.get("prompt_tokens", 0))
    duration_ns = max(1, end_epoch_ns - start_epoch_ns)
    return {
        "endpoint": endpoint,
        "model": model,
        "concurrency": concurrency,
        "requested_requests": len(prompts),
        "completed_requests": completed,
        "failed_requests": len(prompts) - completed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "wall_duration_ns": duration_ns,
        "wall_duration_ms": duration_ns / 1_000_000.0,
        "tokens_per_second": completion_tokens / (duration_ns / 1_000_000_000.0),
        "start_epoch_ns": start_epoch_ns,
        "end_epoch_ns": end_epoch_ns,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True, help="local model path used to build exact prompt token counts")
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--prompt-lengths", type=parse_int_list, default=[256, 512, 1024])
    parser.add_argument("--output-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.requests <= 0 or args.concurrency <= 0 or args.output_length <= 0:
        parser.error("requests, concurrency, and output-length must be positive")
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError("transformers is required to construct exact token-length prompts") from exc
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompts = [
        exact_prompt(tokenizer, args.prompt_lengths[index % len(args.prompt_lengths)], index)
        for index in range(args.requests)
    ]
    summary = run_window(
        endpoint=args.endpoint,
        model=args.model,
        prompts=prompts,
        output_length=args.output_length,
        seed=args.seed,
        concurrency=args.concurrency,
        timeout_s=args.timeout_s,
    )
    summary["configured_prompt_lengths"] = args.prompt_lengths
    summary["configured_output_length"] = args.output_length
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "completed": summary["completed_requests"],
                "failed": summary["failed_requests"],
                "completion_tokens": summary["completion_tokens"],
                "tokens_per_second": summary["tokens_per_second"],
                "wall_duration_ms": summary["wall_duration_ms"],
            },
            sort_keys=True,
        )
    )
    return 0 if summary["failed_requests"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
