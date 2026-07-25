"""Build the PatchCore memory bank from data/normal.

Verified against the INSTALLED anomalib 2.5.1:
- image size lives on the model's PreProcessor (default 256), not the datamodule
- Folder with normal-only data carves val/test splits out of the normals itself,
  which the post-processor needs to fit its normalization stats
- Engine.fit writes results/Patchcore/dish/vN/weights/lightning/model.ckpt
  with a `latest` symlink pointing at the newest vN
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "results/Patchcore/dish/latest/weights/lightning/model.ckpt"


def main():
    ap = argparse.ArgumentParser(description="Train the Seefu PatchCore bank")
    ap.add_argument("--normal-dir", default=str(ROOT / "data/normal"))
    ap.add_argument("--image-size", type=int, default=320)
    args = ap.parse_args()

    images = [p for p in Path(args.normal_dir).iterdir()
              if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    if len(images) < 10:
        sys.exit(f"only {len(images)} images in {args.normal_dir}; need the normal set first")
    print(f"training PatchCore on {len(images)} normals at {args.image_size}px, CPU")

    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    datamodule = Folder(
        name="dish",
        root=str(ROOT / "data"),
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
                    default_root_dir=str(ROOT / "results"), logger=False)

    t0 = time.time()
    engine.fit(model=model, datamodule=datamodule)
    print(f"\ntrained in {time.time() - t0:.0f}s")
    print(f"checkpoint: {CKPT}  exists={CKPT.exists()}")
    if not CKPT.exists():
        sys.exit("checkpoint missing where expected - inspect results/ layout")


if __name__ == "__main__":
    main()
