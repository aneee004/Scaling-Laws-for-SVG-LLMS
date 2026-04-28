import argparse

import datasets
import yaml


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
    pass


def split_dataset(ds, cfg):
    pass


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
