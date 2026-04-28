import argparse

import datasets
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
    pass


def tokenize_and_save(splits, tokenizer, cfg):
    pass


def verify(cfg):
    pass


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
