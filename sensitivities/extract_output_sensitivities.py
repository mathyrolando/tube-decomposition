"""
Extractor de Sensibilidades de Salida (Gradientes de Fisher de Segundo Orden).
Mide \alpha_l = \mathbb{E}[\|\nabla_{Y_l} \mathcal{L}\|_F^2] para las 72 capas de OPT-125M.
Permite resolver el Water-Filling Global sin ninguna heuristica ni cota artificial.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import OPTForCausalLM

CURRENT_DIR = Path(__file__).resolve().parent
REPO_DIR = CURRENT_DIR.parent / "vendor" / "gptq"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from datautils import get_loaders, set_seed

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_opt(model_id: str = "facebook/opt-125m"):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = OPTForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    model.seqlen = 2048
    return model


def compute_output_sensitivities(
    model: OPTForCausalLM,
    dataloader,
    dev: torch.device,
    nsamples: int = 32,
) -> Dict[str, float]:
    print(f"\n[Extracción de Sensibilidades de Salida (Fisher Gradient Trace)] nsamples={nsamples}...")
    model.eval()
    model.zero_grad()
    # Solo hacen falta los gradientes de las salidas: congelar pesos ahorra su memoria (1.3B no entra si no).
    for prm in model.parameters():
        prm.requires_grad_(False)
    emb_hook = model.model.decoder.embed_tokens.register_forward_hook(lambda m, i, o: o.requires_grad_(True))

    stored_outs: Dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(module, inp, out):
            out.retain_grad()
            stored_outs[name] = out
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "decoder.layers" in name:
            h = mod.register_forward_hook(make_hook(name))
            handles.append(h)

    accumulated_energy: Dict[str, float] = {name: 0.0 for name in stored_outs.keys()}
    # Se llenará en el primer pase
    sample_count = 0

    for i, batch in enumerate(dataloader):
        if sample_count >= nsamples:
            break
        input_ids = batch[0].to(dev)
        batch_size, seqlen = input_ids.shape

        model.zero_grad()
        stored_outs.clear()

        logits = model(input_ids).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, model.config.vocab_size),
            shift_labels.view(-1),
        )
        loss.backward()

        for name, out in stored_outs.items():
            if out.grad is not None:
                # Energía promedio por elemento de salida de la capa
                g = out.grad.float()
                # Norma de Frobenius al cuadrado normalizada por número de tokens
                norm_sq = float(torch.sum(g ** 2).item()) / (batch_size * seqlen)
                if name not in accumulated_energy:
                    accumulated_energy[name] = 0.0
                accumulated_energy[name] += norm_sq

        sample_count += 1
        if (sample_count % 8 == 0) or (sample_count == nsamples):
            print(f"  Procesadas {sample_count}/{nsamples} muestras de calibración...")

    for h in handles:
        h.remove()
    emb_hook.remove()

    # Promediar sobre el total de muestras
    mean_energy = {name: accumulated_energy[name] / sample_count for name in accumulated_energy}

    # Normalizar para que la media global sea 1.0 (escala relativa invariante)
    global_mean = sum(mean_energy.values()) / len(mean_energy)
    normalized_sensitivities = {name: mean_energy[name] / global_mean for name in mean_energy}

    return normalized_sensitivities


def main():
    parser = argparse.ArgumentParser(description="Extraer sensibilidades de salida de Fisher")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--nsamples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    print("=" * 70)
    print(" EXTRACCIÓN DE SENSIBILIDADES ARQUITECTURALES (ALPHA_L) ")
    print("=" * 70)

    model = get_opt(args.model).to(DEV)
    dataloader, _ = get_loaders("wikitext2", nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=2048)

    t0 = time.time()
    sensitivities = compute_output_sensitivities(model, dataloader, DEV, nsamples=args.nsamples)
    elapsed = time.time() - t0

    tag = args.model.split("/")[-1].replace("-", "")
    out_path = CURRENT_DIR / f"{tag}_output_sensitivities.pt"
    torch.save(sensitivities, out_path)
    print(f"\n[+] Sensibilidades guardadas en: {out_path} ({elapsed:.1f}s)")

    print("\nResumen de Sensibilidades Normalizadas (Primeras 12 capas):")
    for name in list(sensitivities.keys())[:12]:
        val = sensitivities[name]
        print(f"  {name:45s}: alpha={val:.4f}")


if __name__ == "__main__":
    main()
