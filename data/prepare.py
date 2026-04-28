import argparse
import os

import datasets
import numpy as np
import tokenizers
import yaml

SVG_COL = "Svg"


def load_config(config_path, dev_override):
    with open(config_path) as fp:
        cfg = yaml.safe_load(fp)

    if dev_override:
        cfg["data"]["dev_mode"] = True
    if cfg["data"]["dev_mode"] == True:
        print("Warning! Running in dev mode.")

    required_fields = [
        ("data", "dataset_name"),
        ("data", "output_dir"),
        ("data", "seed"),
        ("tokenizer", "vocab_size"),
        ("tokenizer", "save_path"),
    ]

    missing_fields = [
        f"{section}.{key}"
        for section, key in required_fields
        if cfg.get(section, {}).get(key) is None
    ]

    if missing_fields:
        raise ValueError(f"Missing or None config values: {', '.join(missing_fields)}")

    return cfg


def load_and_filter(cfg):
    data_cfg = cfg["data"]
    dataset_name = data_cfg["dataset_name"]
    dataset_split = data_cfg["dataset_split"]
    cache_dir = data_cfg["hf_cache_dir"] or None
    min_chars = data_cfg["min_chars"]
    max_chars = data_cfg["max_chars"]
    dev_mode = data_cfg["dev_mode"]
    dev_samples = data_cfg["dev_samples"]
    num_proc = data_cfg["num_proc"]

    ds = datasets.load_dataset(
        dataset_name,
        split=dataset_split,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    n_before = len(ds)
    ds = ds.filter(
        lambda example: min_chars <= len(example[SVG_COL]) <= max_chars,
        num_proc=num_proc,
    )

    if dev_mode:
        ds = ds.select(range(min(dev_samples, len(ds))))

    print(f"Filtered {n_before} → {len(ds)} samples")
    return ds


def split_dataset(ds, cfg):
    data_cfg = cfg["data"]
    val_frac = data_cfg["val_frac"]
    test_frac = data_cfg["test_frac"]
    seed = data_cfg["seed"]

    first_split = ds.train_test_split(test_size=val_frac + test_frac, seed=seed)
    train_split = first_split["train"]
    remainder = first_split["test"]

    second_split = remainder.train_test_split(
        test_size=test_frac / (test_frac + val_frac), seed=seed
    )
    val_split = second_split["train"]
    test_split = second_split["test"]

    ds_split = datasets.DatasetDict(
        {
            "train": train_split,
            "val": val_split,
            "test": test_split,
        }
    )

    print(
        f"Split sizes — train: {len(train_split)}, "
        f"val: {len(val_split)}, test: {len(test_split)}"
    )
    return ds_split


def train_tokenizer(train_split, cfg):
    tok_cfg = cfg["tokenizer"]
    vocab_size = tok_cfg["vocab_size"]
    save_path = tok_cfg["save_path"]
    special_tokens = tok_cfg["special_tokens"]

    def text_iterator():
        for example in train_split:
            yield example[SVG_COL]

    tokenizer = tokenizers.ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        text_iterator(),
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    os.makedirs(save_path, exist_ok=True)
    tokenizer.save_model(save_path)

    print(f"Trained tokenizer (vocab_size={vocab_size}) saved to {save_path}")
    return tokenizer


def tokenize_and_save(splits, tokenizer, cfg):
    output_dir = cfg["data"]["output_dir"]
    token_budget = cfg["data"]["token_budget"]
    batch_size = cfg["data"]["hf_map_batch_size"]
    max_seq_len = cfg["model"]["max_seq_len"]

    tokenizer.enable_truncation(max_length=max_seq_len)
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    if eot_id is None:
        raise ValueError(
            "Tokenizer has no <|endoftext|> token. "
            "Add it to cfg['tokenizer']['special_tokens']."
        )

    os.makedirs(output_dir, exist_ok=True)

    for split_name, split in splits.items():
        token_ids = []
        budget_hit = False

        for start in range(0, len(split), batch_size):
            end = min(start + batch_size, len(split))
            texts = [split[i][SVG_COL] for i in range(start, end)]
            encodings = tokenizer.encode_batch(texts)

            for enc in encodings:
                token_ids.extend(enc.ids)
                token_ids.append(eot_id)

            if split_name == "train" and len(token_ids) >= token_budget:
                token_ids = token_ids[:token_budget]
                budget_hit = True
                break

        arr = np.array(token_ids, dtype=np.uint16)
        out_path = os.path.join(output_dir, f"{split_name}.npy")
        np.save(out_path, arr)

        suffix = " (budget reached)" if budget_hit else ""
        print(f"{split_name}: {len(arr):,} tokens → {out_path}{suffix}")


def verify(cfg):
    output_dir = cfg["data"]["output_dir"]
    token_budget = cfg["data"]["token_budget"]
    save_path = cfg["tokenizer"]["save_path"]

    split_names = ["train", "val", "test"]

    # 1. Files exist
    paths = {name: os.path.join(output_dir, f"{name}.npy") for name in split_names}
    missing = [name for name, p in paths.items() if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"Missing split files in {output_dir}: {missing}")

    # 2. Load and check shapes
    arrays = {}
    for name, p in paths.items():
        arr = np.load(p)
        arrays[name] = arr
        print(f"{name}: {len(arr):,} tokens")

        if len(arr) == 0:
            raise ValueError(f"{name} split is empty: {p}")

    if len(arrays["train"]) > token_budget:
        raise ValueError(
            f"train has {len(arrays['train']):,} tokens, "
            f"exceeds budget {token_budget:,}"
        )

    # 3. Spot-check a decode
    vocab_file = os.path.join(save_path, "vocab.json")
    merges_file = os.path.join(save_path, "merges.txt")
    tokenizer = tokenizers.ByteLevelBPETokenizer.from_file(vocab_file, merges_file)

    sample_ids = arrays["train"][:50].tolist()
    decoded = tokenizer.decode(sample_ids)
    print(f"\nFirst 50 train tokens decoded:\n{decoded!r}")


def main():
    parser = argparse.ArgumentParser(
        prog="SVG LLM Model", description="", epilog="Text at the bottom of help"
    )
    parser.add_argument("-c", "--config")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, args.dev)
    ds = load_and_filter(cfg)
    splits = split_dataset(ds, cfg)
    tok = train_tokenizer(splits["train"], cfg)
    tokenize_and_save(splits, tok, cfg)
    verify(cfg)


if __name__ == "__main__":
    main()
