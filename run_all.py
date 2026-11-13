"""End-to-end runner: train both models -> predict (val/test, 4 case variants) -> analysis -> results.json.

    python run_all.py                 # full run (3 seeds x 2 models), resumable
    python run_all.py --seeds 42      # single seed
    python run_all.py --quick         # smoke run: 500 train sentences, 1 epoch, small eval subsets (-> results_quick/)
    python run_all.py --stage analyze # only rebuild analysis/figures from saved predictions
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from ner.data import CASE_VARIANTS, case_transform, label_maps, load_conll  # noqa: E402
from ner.train import MODELS, TrainConfig, run_name, train_model  # noqa: E402

PRIMARY_SEED = 42
AUG_RUN = "bert-base-cased-caseaug"


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def predict_stage(ds, names, seeds, out_root, res_dir, use_cpu=False):
    import torch
    from ner.predict import NERPredictor
    pred_dir = os.path.join(res_dir, "preds")
    for split in ["validation", "test"]:
        save_json({"tokens": [list(t) for t in ds[split]["tokens"]], "tags": [[names[i] for i in s] for s in ds[split]["ner_tags"]]},
                  os.path.join(pred_dir, f"gold_{split}.json"))
    for model in MODELS:
        for seed in seeds:
            name = run_name(model, seed)
            jobs = [("validation", "original"), ("test", "original"), ("test", "lower")]
            if seed == seeds[0] or seed == PRIMARY_SEED:
                jobs += [("test", "upper"), ("test", "title")]
            todo = [(s, v) for s, v in jobs if not os.path.exists(os.path.join(pred_dir, f"{name}_{s}_{v}.json"))]
            if not todo:
                continue
            pred = NERPredictor(os.path.join(out_root, name, "best"), device="cpu" if use_cpu else None)
            for split, variant in todo:
                t0 = time.time()
                sents = [case_transform(t, variant) for t in ds[split]["tokens"]]
                tags = pred.predict_words(sents)
                save_json(tags, os.path.join(pred_dir, f"{name}_{split}_{variant}.json"))
                print(f"  predicted {name} {split}/{variant} in {time.time()-t0:.0f}s")
            del pred
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    aug = run_name(AUG_RUN, seeds[0])
    if os.path.exists(os.path.join(out_root, aug, "best", "config.json")):
        pred = NERPredictor(os.path.join(out_root, aug, "best"), device="cpu" if use_cpu else None)
        for variant in CASE_VARIANTS:
            path = os.path.join(pred_dir, f"{aug}_test_{variant}.json")
            if not os.path.exists(path):
                save_json(pred.predict_words([case_transform(t, variant) for t in ds["test"]["tokens"]]), path)
                print(f"  predicted {aug} test/{variant}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 13, 1234])
    ap.add_argument("--stage", choices=["all", "train", "predict", "analyze"], default="all")
    ap.add_argument("--results", default=None)
    ap.add_argument("--outputs", default=None)
    ap.add_argument("--skip-custom", action="store_true")
    ap.add_argument("--case-aug", action="store_true",
                    help="extra: also fine-tune bert-base-cased with case augmentation (seed 42)")
    a = ap.parse_args()

    res_dir = a.results or ("results_quick" if a.quick else "results")
    out_root = a.outputs or ("outputs_quick" if a.quick else "outputs")
    seeds = [42] if a.quick else a.seeds
    cfg_kw = {}
    ds = load_conll()
    names, id2label, label2id = label_maps(ds)
    if a.quick:
        ds["train"] = ds["train"].select(range(500))
        ds["validation"] = ds["validation"].select(range(200))
        ds["test"] = ds["test"].select(range(300))
        cfg_kw = {"num_train_epochs": 1, "logging_steps": 5}
    os.makedirs(res_dir, exist_ok=True)
    meta = {"dataset_source": getattr(ds, "source", "?"), "quick": a.quick, "seeds": seeds,
            "split_sizes": {k: len(v) for k, v in ds.items()}, "label_names": names}
    save_json(meta, os.path.join(res_dir, "run_meta.json"))

    if a.stage in ("all", "train"):
        for seed in seeds:
            for model in MODELS:  # same config object fields for both models -> fair comparison
                cfg = TrainConfig(seed=seed, **cfg_kw)
                print(f"=== training {model} seed={seed} ===", flush=True)
                s = train_model(model, ds, id2label, label2id, out_root, os.path.join(res_dir, "logs"), cfg,
                                use_cpu=a.cpu)
                print(f"    val F1={s['best_metric_val_f1']:.4f} time={s['train_time_sec']/60:.1f} min", flush=True)
        if a.case_aug:
            from datasets import DatasetDict
            from ner.data import case_augment
            ds_aug = DatasetDict({**ds, "train": case_augment(ds["train"], seed=seeds[0])})
            print("=== training bert-base-cased + case augmentation ===", flush=True)
            s = train_model("bert-base-cased", ds_aug, id2label, label2id, out_root, os.path.join(res_dir, "logs"),
                            TrainConfig(seed=seeds[0], **cfg_kw), use_cpu=a.cpu, run_label=AUG_RUN)
            print(f"    val F1={s['best_metric_val_f1']:.4f}", flush=True)
    if a.stage in ("all", "predict"):
        predict_stage(ds, names, seeds, out_root, res_dir, use_cpu=a.cpu)
    if a.stage in ("all", "analyze"):
        from ner.analysis import run_analysis
        run_analysis(ds, names, seeds, res_dir, out_root, quick=a.quick, skip_custom=a.skip_custom, use_cpu=a.cpu)
    print("done ->", res_dir)


if __name__ == "__main__":
    sys.exit(main())
