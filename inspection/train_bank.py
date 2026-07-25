"""Build a PatchCore memory bank for one dish from data/dishes/<dish>/normal.

Verified against the INSTALLED anomalib 2.5.1:
- image size lives on the model's PreProcessor (default 256), not the datamodule
- Folder with normal-only data carves val/test splits out of the normals itself,
  which the post-processor needs to fit its normalization stats
- Engine.fit writes results/_runs/Patchcore/<dish>/vN/...; the newest ckpt is
  then copied to results/<dish>/model.ckpt, the single path serve.py watches
  (its mtime is the hot-reload signal)
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description="Train a Seefu PatchCore bank")
    ap.add_argument("--dish", default="sesame-beef-bowl",
                    help="dish slug under data/dishes/")
    ap.add_argument("--image-size", type=int, default=320)
    args = ap.parse_args()

    dish_dir = ROOT / "data/dishes" / args.dish
    normal_dir = dish_dir / "normal"
    if not normal_dir.is_dir():
        sys.exit(f"no normal set at {normal_dir}")
    images = [p for p in normal_dir.iterdir()
              if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    if len(images) < 10:
        sys.exit(f"only {len(images)} images in {normal_dir}; need the normal set first")
    print(f"training {args.dish} on {len(images)} normals at {args.image_size}px, CPU")

    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    datamodule = Folder(
        name=args.dish,
        root=str(dish_dir),
        normal_dir="normal",
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,          # macOS: avoid dataloader fork overhead
        seed=42,
        # defaults keep 20% of normals as test, half of that as val -> the
        # post-processor fits its threshold/normalization on that val split
    )
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=("layer2", "layer3"),
        coreset_sampling_ratio=0.1,
        pre_processor=Patchcore.configure_pre_processor(
            image_size=(args.image_size, args.image_size)),
    )
    engine = Engine(accelerator="cpu", devices=1,
                    default_root_dir=str(ROOT / "results/_runs"), logger=False)

    t0 = time.time()
    engine.fit(model=model, datamodule=datamodule)
    print(f"\ntrained in {time.time() - t0:.0f}s")

    run_ckpt = (ROOT / f"results/_runs/Patchcore/{args.dish}/latest/"
                       "weights/lightning/model.ckpt")
    if not run_ckpt.exists():
        sys.exit(f"checkpoint missing at {run_ckpt} - inspect results/_runs layout")
    served = ROOT / f"results/{args.dish}/model.ckpt"
    served.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_ckpt, served)
    served.touch()              # fresh mtime = serve.py hot-reload signal
    print(f"serving checkpoint: {served}")


if __name__ == "__main__":
    main()
