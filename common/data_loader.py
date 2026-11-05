#!/usr/bin/env python3
"""
Data loader modular y optimizado para validación cruzada entre WikiText-2 y C4.
Totalmente compatible con datasets >= 3.0 de Hugging Face y modelos causales (OPT, LLaMA, etc.).
"""
from __future__ import annotations
import random
from typing import Tuple, List, Any
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TokenizerWrapper:
    """Contenedor para exponer .input_ids de forma consistente con evaluate_ppl."""
    def __init__(self, input_ids: torch.Tensor):
        self.input_ids = input_ids


def get_wikitext2(
    nsamples: int = 128,
    seed: int = 0,
    seqlen: int = 2048,
    model: str = "facebook/opt-125m",
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], TokenizerWrapper]:
    """Carga y tokeniza WikiText-2 para calibración y evaluación."""
    traindata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, TokenizerWrapper(testenc.input_ids)


def get_c4(
    nsamples: int = 128,
    seed: int = 0,
    seqlen: int = 2048,
    model: str = "facebook/opt-125m",
    eval_chunks: int = 256,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], TokenizerWrapper]:
    """Carga y tokeniza C4 (shards cacheados) para calibración y evaluación estándar GPTQ."""
    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Construir test/val split de evaluación (256 secuencias de longitud seqlen = 524,288 tokens)
    val_texts = []
    target_tokens = eval_chunks * seqlen
    for item in valdata:
        val_texts.append(item["text"])
        if len(val_texts) >= 1100:
            break

    valenc = tokenizer(" ".join(val_texts), return_tensors="pt")
    valenc_ids = valenc.input_ids[:, :target_tokens]
    return trainloader, TokenizerWrapper(valenc_ids)


def get_dataset_loaders(
    name: str,
    nsamples: int = 128,
    seed: int = 0,
    seqlen: int = 2048,
    model: str = "facebook/opt-125m",
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], TokenizerWrapper]:
    name = name.lower()
    if "wikitext" in name:
        return get_wikitext2(nsamples=nsamples, seed=seed, seqlen=seqlen, model=model)
    elif "c4" in name:
        return get_c4(nsamples=nsamples, seed=seed, seqlen=seqlen, model=model)
    else:
        raise ValueError(f"Dataset desconocido: {name}. Opciones soportadas: 'wikitext2', 'c4'")
