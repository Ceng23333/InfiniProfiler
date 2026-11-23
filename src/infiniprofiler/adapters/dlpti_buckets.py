"""Map DLPTI kernel names to InfiniLM / vLLM CU operation buckets.

Ported from c610 ``tools/dlpti/buckets.py`` for wrap-grade InfiniProfiler
normalize. First-match-wins; substrings are case-insensitive.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

# (bucket, substrings). First match wins.
BUCKET_RULES: List[Tuple[str, Tuple[str, ...]]] = [
	(
		"moe_fused",
		(
			"dl_fused_moe",
			"invoke_fused_moe",
			"fused_moe_opt",
			"moe_fused_dense",
			"moe_fused_gate",
			"fused_moe",
			"moe_align",
			"moe_sum",
			"prepare_moe",
		),
	),
	(
		"flash_attention",
		(
			"j_flash_attn",
			"flash_attn",
			"flashattention",
		),
	),
	(
		"fused_add_rmsnorm",
		(
			"fused_add_rms_norm",
			"fusedaddrmsnorm",
		),
	),
	(
		"rmsnorm",
		(
			"rmsnorm",
			"rms_norm",
		),
	),
	(
		"add",
		(
			"ops9addkernel",
			"addkernel",
		),
	),
	(
		"rope",
		(
			"rotary_embedding",
			"ropethreadperitem",
			"rope",
		),
	),
	(
		"silu_act",
		(
			"siluandmul",
			"act_and_mul",
			"silu",
		),
	),
	(
		"kv_cache_reshape",
		(
			"reshape_and_cache",
			"reshapeandcache",
		),
	),
	(
		"gemm",
		(
			"gemm",
			"cublas",
			"sbgemm",
			"matmul",
		),
	),
	(
		"bf16_copy",
		(
			"bf16copy",
			"bf16_copy",
			"copykernel",
		),
	),
]

# Buckets emitted as EngineMetric rows (includes residual Add+RMSNorm combine).
COMPARE_BUCKETS: Tuple[str, ...] = (
	"moe_fused",
	"flash_attention",
	"residual_add_rmsnorm",
	"rope",
	"silu_act",
	"kv_cache_reshape",
	"gemm",
	"bf16_copy",
	"other",
)

# Material buckets eligible for kernel.material ranking (exclude catch-alls).
MATERIAL_BUCKETS = frozenset(
	{
		"moe_fused",
		"flash_attention",
		"residual_add_rmsnorm",
		"rope",
		"silu_act",
		"kv_cache_reshape",
		"gemm",
	}
)

METRIC_PREFIX = "cu_bucket."


def bucket_for(kernel_name: str) -> str:
	lower = kernel_name.lower()
	for bucket, needles in BUCKET_RULES:
		if any(n in lower for n in needles):
			return bucket
	return "other"


def residual_add_rmsnorm_us(bucket_sums: Dict[str, Dict[str, float]]) -> Dict[str, float]:
	"""Combine fused residual-norm with separate Add + RMSNorm mass."""
	fused = bucket_sums.get("fused_add_rmsnorm", {"sum_us": 0.0, "count": 0.0})
	add = bucket_sums.get("add", {"sum_us": 0.0, "count": 0.0})
	rms = bucket_sums.get("rmsnorm", {"sum_us": 0.0, "count": 0.0})
	return {
		"sum_us": fused["sum_us"] + add["sum_us"] + rms["sum_us"],
		"count": fused["count"] + add["count"] + rms["count"],
	}


def select_bucket_stat(bucket_sums: Dict[str, Dict[str, float]], name: str) -> Dict[str, float]:
	if name == "residual_add_rmsnorm":
		return residual_add_rmsnorm_us(bucket_sums)
	return bucket_sums.get(name, {"sum_us": 0.0, "count": 0.0})


def empty_bucket_map(names: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, float]]:
	keys = list(names) if names is not None else [b for b, _ in BUCKET_RULES] + ["other"]
	return {k: {"sum_us": 0.0, "count": 0.0} for k in keys}


def metric_name(bucket: str) -> str:
	return f"{METRIC_PREFIX}{bucket}"
