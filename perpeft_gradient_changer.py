import torch
from collections import defaultdict

import math
from typing import Dict, Any
from contextlib import contextmanager

def _find_adapters(model):
    adapters = set()
    for m in model.modules():
        # Works for PEFT LoRA layers
        if hasattr(m, "lora_A"):
            la = getattr(m, "lora_A")
            if isinstance(la, dict):
                adapters.update(la.keys())
            else:
                adapters.add("default")
    return sorted(adapters)

def _representative_param_for_adapter(model, adapter_name):
    """
    Return (qualified_name, tensor_param) for a representative LoRA param of the adapter.
    Prefer lora_A; fallback to any lora_ param that matches the adapter.
    """
    # Best-effort: get module.lora_A[adapter].weight
    for name, module in model.named_modules():
        if hasattr(module, "lora_A"):
            la = getattr(module, "lora_A")
            if isinstance(la, dict) and adapter_name in la:
                p = la[adapter_name].weight
                return f"{name}.lora_A.{adapter_name}.weight", p
            # Some older setups store a single tensor (no dict)
            if not isinstance(la, dict) and adapter_name == "default":
                p = la.weight if hasattr(la, "weight") else la
                return f"{name}.lora_A.default.weight", p

    # Fallback: scan named_parameters for anything with lora_ and adapter_name in path
    for n, p in model.named_parameters():
        if "lora_" in n and (f".{adapter_name}." in n or adapter_name == "default"):
            return n, p

    return None, None

def print_representative_param(model, adapter_name: str, max_values_preview: int = 6):
    """
    Print one representative LoRA param for a given adapter.
    Looks for ...lora_A.<adapter>.weight first; falls back to any lora_ param containing the adapter.
    """
    rep_name, rep_p = None, None

    # Best: find lora_A.<adapter>.weight
    for module_name, module in model.named_modules():
        if hasattr(module, "lora_A"):
            la = getattr(module, "lora_A")
            if isinstance(la, dict) and adapter_name in la:
                rep_name = f"{module_name}.lora_A.{adapter_name}.weight"
                rep_p = la[adapter_name].weight
                break

    # Fallback: scan params that include adapter_name and 'lora_'
    if rep_p is None:
        for n, p in model.named_parameters():
            if "lora_" in n and f".{adapter_name}." in n:
                rep_name, rep_p = n, p
                break

    print(f"[REP] adapter={adapter_name}")
    if rep_p is None:
        print("  - representative param: NOT FOUND")
        return

    with torch.no_grad():
        flat = rep_p.detach().flatten().cpu()
        preview = flat[:max_values_preview].tolist()
    print(f"  - name: {rep_name}")
    print(f"  - shape: {tuple(rep_p.shape)}  |  #params: {rep_p.numel()}")
    print(f"  - requires_grad: {rep_p.requires_grad}")
    print(f"  - sample values: {preview}{' ...' if rep_p.numel() > max_values_preview else ''}")
    print(f"  - has_grad: {rep_p.grad is not None}")
    if rep_p.grad is not None:
        g = rep_p.grad.detach().flatten().cpu()
        gprev = g[:max_values_preview].tolist()
        print(f"  - grad sample: {gprev}{' ...' if g.numel() > max_values_preview else ''}")


def _adapter_trainable_counts(model, adapters):
    # Count trainable parameters associated with each adapter
    counts = defaultdict(int)
    for n, p in model.named_parameters():
        if p.requires_grad and "lora_" in n:
            matched = False
            for ad in adapters:
                if f".{ad}." in n or (ad == "default" and ".default." in n or ad == "default"):
                    counts[ad] += p.numel()
                    matched = True
                    break
            if not matched:
                counts["(unattributed)"] += p.numel()
    return counts

def print_adapter_summary(model, max_values_preview=6):
    adapters = _find_adapters(model)
    if not adapters:
        print("No adapters found (no lora_A attributes detected).")
        return

    print(f"Detected adapters: {', '.join(adapters)}\n")

    # Summaries
    for ad in adapters:
        rep_name, rep_p = _representative_param_for_adapter(model, ad)
        print(f"[Adapter: {ad}]")
        if rep_p is None:
            print("  - Representative parameter: not found")
        else:
            with torch.no_grad():
                flat = rep_p.detach().flatten()
                preview = flat[:max_values_preview].cpu().tolist()
            print(f"  - Representative param: {rep_name}")
            print(f"  - Shape: {tuple(rep_p.shape)}  |  #params: {rep_p.numel()}")
            print(f"  - requires_grad: {rep_p.requires_grad}")
            print(f"  - sample values: {preview}{' ...' if rep_p.numel() > max_values_preview else ''}")
            # Guard grad safely
            has_grad = (rep_p.grad is not None)
            print(f"  - has_grad: {has_grad}")
            if has_grad:
                g = rep_p.grad.detach().flatten()
                gprev = g[:max_values_preview].cpu().tolist()
                print(f"  - grad sample: {gprev}{' ...' if g.numel() > max_values_preview else ''}")
        print()

    # Trainable counts by adapter
    counts = _adapter_trainable_counts(model, adapters)
    if counts:
        print("Trainable parameter counts by adapter (LoRA params only):")
        total = 0
        for ad in adapters:
            c = counts.get(ad, 0)
            total += c
            print(f"  - {ad}: {c:,}")
        # Any unmatched lora_ params
        if "(unattributed)" in counts:
            print(f"  - (unattributed): {counts['(unattributed)']:,}")
            total += counts['(unattributed)']
        print(f"  = Total trainable LoRA params: {total:,}")

# ---- Usage ----
# model = ...  # your PEFT-wrapped model
# print_adapter_summary(model)


@contextmanager
def use_adapter(model, name):
    prev = model.active_adapter
    if prev != name:
        model.set_adapter(name)
    try:
        yield
        # tripwire: ensure nothing changed it
        assert model.active_adapter == name, f"Adapter switched from {name} to {model.active_adapter} inside block"
    finally:
        if model.active_adapter != prev:
            model.set_adapter(prev)


def iter_lora_params(model):
    """Yield (name, param, adapter_name) for LoRA params only."""
    for n, p in model.named_parameters():
        if "lora_" not in n:
            continue
        # name patterns like: ...lora_A.adapter_7.weight
        parts = n.split(".")
        try:
            i = parts.index("lora_A")
        except ValueError:
            try:
                i = parts.index("lora_B")
            except ValueError:
                continue
        # adapter name is the next token after lora_A/lora_B
        if i + 1 < len(parts):
            adapter = parts[i + 1]
        else:
            adapter = "unknown"
        yield n, p, adapter

def adapter_grad_stats(model) -> Dict[str, Dict[str, Any]]:
    """
    Summarize gradient presence and magnitude per adapter.
    Call AFTER loss.backward(), BEFORE optimizer.step().
    """
    stats: Dict[str, Dict[str, Any]] = {}
    for n, p, adapter in iter_lora_params(model):
        s = stats.setdefault(adapter, {
            "num_params": 0,
            "with_grad": 0,
            "grad_abs_sum": 0.0,
            "grad_l2": 0.0,
            "nan_or_inf_grads": 0,
            "examples_with_grad": [],
            "examples_no_grad": [],
        })
        s["num_params"] += 1
        if p.grad is None:
            if len(s["examples_no_grad"]) < 3:
                s["examples_no_grad"].append(n)
            continue
        g = p.grad
        if not torch.isfinite(g).all():
            s["nan_or_inf_grads"] += 1
        s["with_grad"] += 1
        # use float CPU for consistent numbers
        ga = g.detach().abs().float().sum().item()
        s["grad_abs_sum"] += ga
        gl2 = float(torch.linalg.vector_norm(g.detach().float()).item())
        s["grad_l2"] += gl2
        if len(s["examples_with_grad"]) < 3:
            s["examples_with_grad"].append(n)
    return stats

def print_adapter_grad_report(model, intended_adapter: str = None):
    stats = adapter_grad_stats(model)
    lines = []
    for name, s in sorted(stats.items()):
        lines.append(
            f"[GRAD] adapter={name:>12}  "
            f"with_grad={s['with_grad']:3d}/{s['num_params']:3d}  "
            f"grad_abs_sum={s['grad_abs_sum']:.3e}  grad_l2≈{s['grad_l2']:.3e}  "
            f"nan_or_inf={s['nan_or_inf_grads']}"
            + ("" if intended_adapter is None else ("  <==" if name == intended_adapter else ""))
        )
    msg = "\n".join(lines) if lines else "[GRAD] No LoRA params found."
    print(msg)
    return stats


def collect_all_lora_params(model, peft_type):
    
    """Return a deduped list of ALL LoRA params from ALL adapters."""
    
    params = []
    seen = set()
    
    if peft_type == "lora" : 
    
        for n, p in model.named_parameters():
            if "lora_" in n:
                if id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))
                    
    else : 
        
        for n, p in model.named_parameters():
            if "ia3_" in n:
                if id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))
    return params

def safe_step(scaler, optimizer):
    if any(p.grad is not None for group in optimizer.param_groups for p in group["params"]):
        scaler.step(optimizer)
        
        
def adapter_fingerprint(model, adapter_name):
    model.set_adapter(adapter_name)
    h = hashlib.md5()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_" in n and adapter_name in n:
                h.update(p.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()

def fingerprints_all(model):
    return {name: adapter_fingerprint(model, name) for name in model.peft_config.keys()}
