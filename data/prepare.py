import argparse
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

import datasets
import numpy as np
import tokenizers
import yaml

SVG_COL = "Svg"

# Regexes for SVG normalisation.
XML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
WHITESPACE_RE  = re.compile(r"\s+")
NUMBER_RE      = re.compile(r"-?\d+\.\d+")


@dataclass
class DataConfig:
    dataset_name: str
    dataset_split: str
    output_dir: str
    seed: int
    val_frac: float
    test_frac: float
    min_chars: int
    max_chars: int
    token_budget: int
    num_proc: int
    hf_map_batch_size: int
    load_from_cache_file: bool
    dev_mode: bool
    dev_samples: int
    hf_cache_dir: str = ""
    # Cleaning / validation flags (defaulted for back-compat with older configs)
    clean_svgs: bool = True
    coord_precision: int = 1
    validate_xml: bool = True

    @property
    def cache_dir(self) -> Optional[str]:
        return self.hf_cache_dir or None


@dataclass
class TokenizerConfig:
    vocab_size: int
    save_path: str
    special_tokens: List[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    max_seq_len: int
    n_layer: Optional[int] = None
    n_head: Optional[int] = None
    n_embd: Optional[int] = None
    dropout: float = 0.0
    bias: bool = False


def _deep_merge(base, override):
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(config_path, override_path, dev_override):
    with open(config_path) as fp:
        cfg = yaml.safe_load(fp)

    if override_path:
        with open(override_path) as fp:
            cfg = _deep_merge(cfg, yaml.safe_load(fp))

    if dev_override:
        cfg["data"]["dev_mode"] = True
    if cfg["data"]["dev_mode"]:
        print("Warning! Running in dev mode.")

    data_cfg = DataConfig(**cfg["data"])
    tok_cfg = TokenizerConfig(**cfg["tokenizer"])
    model_cfg = ModelConfig(**cfg["model"])

    missing = []
    if not data_cfg.dataset_name:
        missing.append("data.dataset_name")
    if not data_cfg.output_dir:
        missing.append("data.output_dir")
    if not tok_cfg.save_path:
        missing.append("tokenizer.save_path")
    if missing:
        raise ValueError(f"Missing or None config values: {', '.join(missing)}")

    return data_cfg, tok_cfg, model_cfg


def clean_svg_text(text: str, coord_precision: int = 1) -> str:
    """Strip XML comments, collapse whitespace, round numeric coordinates.

    SVG coordinate precision rounding shrinks the BPE vocabulary by collapsing
    near-duplicate numeric strings (e.g. ``12.34567`` and ``12.34568``) into a
    single token.
    """
    text = XML_COMMENT_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text).strip()

    def _round(match):
        return f"{float(match.group(0)):.{coord_precision}f}"

    text = NUMBER_RE.sub(_round, text)
    return text


def is_valid_xml(text: str) -> bool:
    """True iff `text` parses as well-formed XML via lxml."""
    try:
        from lxml import etree
        etree.fromstring(text.encode("utf-8"))
        return True
    except Exception:
        return False


def clean_and_validate(ds, data_cfg: DataConfig, label: str = ""):
    """Apply optional SVG cleaning and XML validation as map/filter steps."""
    if not (data_cfg.clean_svgs or data_cfg.validate_xml):
        return ds

    n_before = len(ds)

    if data_cfg.clean_svgs:
        precision = data_cfg.coord_precision

        def _clean(example):
            example[SVG_COL] = clean_svg_text(example[SVG_COL], precision)
            return example

        ds = ds.map(
            _clean,
            num_proc=data_cfg.num_proc,
            load_from_cache_file=data_cfg.load_from_cache_file,
            desc=f"Cleaning SVGs ({label})" if label else "Cleaning SVGs",
        )

    if data_cfg.validate_xml:
        ds = ds.filter(
            lambda ex: is_valid_xml(ex[SVG_COL]),
            num_proc=data_cfg.num_proc,
            load_from_cache_file=data_cfg.load_from_cache_file,
            desc=f"Validating XML ({label})" if label else "Validating XML",
        )

    tag = f" [{label}]" if label else ""
    print(f"Clean+validate{tag}: {n_before} → {len(ds)} samples")
    return ds


def cap_train_split(splits, data_cfg: DataConfig, model_cfg: ModelConfig):
    """Cap the train split to roughly what's needed for the token budget.

    Each sample contributes at most ``max_seq_len`` tokens (longer ones get
    dropped in ``tokenize_and_save``), so ``token_budget / max_seq_len`` is the
    minimum sample count. Real SVGs average well below the cap, so multiply by
    a safety factor. Capping here avoids cleaning + XML-validating + training
    the tokenizer on data that will never be written to disk.
    """
    safety = 4
    max_samples = safety * data_cfg.token_budget // model_cfg.max_seq_len
    train = splits["train"]
    if len(train) > max_samples:
        n_before = len(train)
        splits["train"] = train.select(range(max_samples))
        print(
            f"Cap train: {n_before} → {len(splits['train'])} samples "
            f"({safety}× token_budget/max_seq_len)"
        )
    return splits


def load_and_filter(data_cfg: DataConfig):
    ds = datasets.load_dataset(
        data_cfg.dataset_name,
        split=data_cfg.dataset_split,
        cache_dir=data_cfg.cache_dir,
        trust_remote_code=True,
    )

    # In dev mode, slice the raw dataset *before* filtering so we don't pay the
    # full-corpus filter cost just to throw most rows away.
    if data_cfg.dev_mode:
        ds = ds.select(range(min(data_cfg.dev_samples * 4, len(ds))))

    n_before = len(ds)
    ds = ds.filter(
        lambda example: data_cfg.min_chars <= len(example[SVG_COL]) <= data_cfg.max_chars,
        num_proc=data_cfg.num_proc,
        load_from_cache_file=data_cfg.load_from_cache_file,
    )

    if data_cfg.dev_mode:
        ds = ds.select(range(min(data_cfg.dev_samples, len(ds))))

    print(f"Length filter: {n_before} → {len(ds)} samples")
    return ds


def split_dataset(ds, data_cfg: DataConfig):
    first_split = ds.train_test_split(
        test_size=data_cfg.val_frac + data_cfg.test_frac, seed=data_cfg.seed
    )
    train_split = first_split["train"]
    remainder = first_split["test"]

    second_split = remainder.train_test_split(
        test_size=data_cfg.test_frac / (data_cfg.test_frac + data_cfg.val_frac),
        seed=data_cfg.seed,
    )
    val_split = second_split["train"]
    test_split = second_split["test"]

    ds_split = datasets.DatasetDict(
        {"train": train_split, "val": val_split, "test": test_split}
    )
    print(
        f"Split sizes — train: {len(train_split)}, "
        f"val: {len(val_split)}, test: {len(test_split)}"
    )
    return ds_split


def train_tokenizer(train_split, tok_cfg: TokenizerConfig):
    def text_iterator():
        for example in train_split:
            yield example[SVG_COL]

    tokenizer = tokenizers.ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        text_iterator(),
        vocab_size=tok_cfg.vocab_size,
        special_tokens=tok_cfg.special_tokens,
    )

    os.makedirs(tok_cfg.save_path, exist_ok=True)
    tokenizer.save_model(tok_cfg.save_path)

    print(f"Trained tokenizer (vocab_size={tok_cfg.vocab_size}) saved to {tok_cfg.save_path}")
    return tokenizer


def tokenize_and_save(splits, tokenizer, data_cfg: DataConfig, model_cfg: ModelConfig):
    """Tokenise each split and save as flat uint16 arrays.

    Sequences whose token length exceeds ``max_seq_len`` are dropped entirely
    (per the PDF spec: filter long SVGs out). Truncation would silently bias
    the corpus toward partial SVGs.
    """
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    if eot_id is None:
        raise ValueError(
            "Tokenizer has no <|endoftext|> token. "
            "Add it to tokenizer.special_tokens in your config."
        )

    max_len = model_cfg.max_seq_len
    os.makedirs(data_cfg.output_dir, exist_ok=True)

    for split_name, split in splits.items():
        token_ids = []
        budget_hit = False
        kept = 0
        dropped_long = 0

        for start in range(0, len(split), data_cfg.hf_map_batch_size):
            end = min(start + data_cfg.hf_map_batch_size, len(split))
            texts = [split[i][SVG_COL] for i in range(start, end)]
            encodings = tokenizer.encode_batch(texts)

            for enc in encodings:
                if len(enc.ids) > max_len:
                    dropped_long += 1
                    continue
                token_ids.extend(enc.ids)
                token_ids.append(eot_id)
                kept += 1

            if split_name == "train" and len(token_ids) >= data_cfg.token_budget:
                token_ids = token_ids[:data_cfg.token_budget]
                budget_hit = True
                break

        arr = np.array(token_ids, dtype=np.uint16)
        out_path = os.path.join(data_cfg.output_dir, f"{split_name}.npy")
        np.save(out_path, arr)

        suffix = " (budget reached)" if budget_hit else ""
        print(
            f"{split_name}: {len(arr):,} tokens, "
            f"kept {kept:,}, dropped_long {dropped_long:,} → {out_path}{suffix}"
        )


def verify(data_cfg: DataConfig, tok_cfg: TokenizerConfig):
    split_names = ["train", "val", "test"]

    paths = {name: os.path.join(data_cfg.output_dir, f"{name}.npy") for name in split_names}
    missing = [name for name, p in paths.items() if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"Missing split files in {data_cfg.output_dir}: {missing}")

    arrays = {}
    for name, p in paths.items():
        arr = np.load(p)
        arrays[name] = arr
        print(f"{name}: {len(arr):,} tokens")
        if len(arr) == 0:
            raise ValueError(f"{name} split is empty: {p}")

    if len(arrays["train"]) > data_cfg.token_budget:
        raise ValueError(
            f"train has {len(arrays['train']):,} tokens, "
            f"exceeds budget {data_cfg.token_budget:,}"
        )

    vocab_file = os.path.join(tok_cfg.save_path, "vocab.json")
    merges_file = os.path.join(tok_cfg.save_path, "merges.txt")
    tokenizer = tokenizers.ByteLevelBPETokenizer.from_file(vocab_file, merges_file)

    sample_ids = arrays["train"][:50].tolist()
    decoded = tokenizer.decode(sample_ids)
    print(f"\nFirst 50 train tokens decoded:\n{decoded!r}")


def main():
    parser = argparse.ArgumentParser(prog="SVG LLM — data pipeline")
    parser.add_argument("-c", "--config")
    parser.add_argument("-o", "--override", default=None)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    data_cfg, tok_cfg, model_cfg = load_config(args.config, args.override, args.dev)
    ds = load_and_filter(data_cfg)
    splits = split_dataset(ds, data_cfg)
    splits = cap_train_split(splits, data_cfg, model_cfg)
    splits = datasets.DatasetDict(
        {name: clean_and_validate(s, data_cfg, label=name) for name, s in splits.items()}
    )
    tok = train_tokenizer(splits["train"], tok_cfg)
    tokenize_and_save(splits, tok, data_cfg, model_cfg)
    verify(data_cfg, tok_cfg)


if __name__ == "__main__":
    main()
