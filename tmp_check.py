from pathlib import Path
raw = Path("data/raw")
total = 0
for d in sorted(raw.iterdir()):
    if d.is_dir():
        imgs = [f for f in d.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        print(f"  {d.name}: {len(imgs)} images")
        total += len(imgs)
print(f"  TOTAL: {total}")
