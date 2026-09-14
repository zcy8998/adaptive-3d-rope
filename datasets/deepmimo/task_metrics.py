import math

import torch


def nmse(pred, target, valid=None, eps=1e-8):
    err = ((pred - target) ** 2).sum(dim=-1)
    ref = (target ** 2).sum(dim=-1).clamp_min(1e-12)
    if valid is not None and valid.shape == pred.shape:
        err = ((pred - target) ** 2 * valid.to(pred.dtype)).sum(dim=-1)
        ref = ((target ** 2) * valid.to(target.dtype)).sum(dim=-1).clamp_min(1e-12)
        valid = valid.any(dim=-1)
    value = err / ref
    sample_mask = ref > eps
    if valid is not None:
        sample_mask = sample_mask & valid
    value = value[sample_mask]
    linear = float(value.mean().detach().cpu()) if value.numel() else 0.0
    return linear, 10.0 * math.log10(max(linear, 1e-12))


def sgcs(pred, target, valid=None):
    if valid is not None and valid.shape == pred.shape:
        mask = valid.to(pred.dtype)
        pred = pred * mask
        target = target * mask
        valid = valid.any(dim=-1)
    dot = (pred * target).sum(dim=-1).abs()
    denom = torch.linalg.vector_norm(pred, dim=-1) * torch.linalg.vector_norm(target, dim=-1)
    value = dot / denom.clamp_min(1e-12)
    if valid is not None:
        value = value[valid]
    return float(value.mean().detach().cpu()) if value.numel() else 0.0


def csi_feedback_overhead(rank, payload_bits, periodicity_ms):
    bits = int(rank) * int(payload_bits)
    return {
        "csi_payload_bits": float(bits),
        "csi_feedback_periodicity_ms": float(periodicity_ms),
        "csi_feedback_overhead_bps": float(bits / max(float(periodicity_ms), 1e-9) * 1000.0),
    }


def throughput_proxy(nmse_linear, overhead_bps, bandwidth_mhz):
    quality = 1.0 / (1.0 + max(float(nmse_linear), 0.0))
    throughput = float(bandwidth_mhz) * 1e6 * quality
    return {
        "throughput_proxy_bps": throughput,
        "throughput_overhead_ratio": throughput / max(float(overhead_bps), 1.0),
    }


def topk_accuracy(logits, labels, ks=(1, 3, 5)):
    out = {}
    max_k = min(max(ks), logits.shape[-1])
    _, pred = logits.topk(max_k, dim=-1)
    labels = labels.view(-1, 1)
    for k in ks:
        kk = min(k, logits.shape[-1])
        out[f"top{k}_accuracy"] = float((pred[:, :kk] == labels).any(dim=1).float().mean().detach().cpu())
    return out


def one_db_margin_accuracy(logits, full_rsrp, labels):
    pred = logits.argmax(dim=-1)
    best = full_rsrp.gather(1, labels[:, None]).squeeze(1)
    chosen = full_rsrp.gather(1, pred[:, None]).squeeze(1)
    return float(((best - chosen) <= 1.0).float().mean().detach().cpu())


def rsrp_gap(logits, full_rsrp, labels):
    pred = logits.argmax(dim=-1)
    best = full_rsrp.gather(1, labels[:, None]).squeeze(1)
    chosen = full_rsrp.gather(1, pred[:, None]).squeeze(1)
    gap = best - chosen
    return {
        "avg_l1_rsrp_gap_db": float(gap.mean().detach().cpu()),
        "p50_l1_rsrp_gap_db": float(torch.quantile(gap, 0.50).detach().cpu()),
        "p90_l1_rsrp_gap_db": float(torch.quantile(gap, 0.90).detach().cpu()),
    }


def rs_overhead_reduction(set_a_size, set_b_size):
    return float(1.0 - (int(set_b_size) / max(1, int(set_a_size))))
