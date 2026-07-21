"""Verify every neaq leaf dir holds same-size images.

deepforest's MultiImage dataset (dataloader_strategy="batch", which neaq takes
because it has no metadata) sizes its windowing grid from paths[0] alone, while
its docstring only warns "Images are expected to be the same size". A mixed dir
would not crash — it would silently mis-window every subsequent image, quietly
corrupting the prediction caches we are running to produce.

Reads EXIF/header dims only (no pixel decode), so it is cheap over 74k files.
"""

from collections import Counter
from pathlib import Path

from PIL import Image

MANIFEST = Path("/blue/ewhite/b.weinstein/src/BOEM/neaq_flights.txt")


def main():
    rows = [l.split("\t") for l in MANIFEST.read_text().splitlines() if l.strip()]
    bad = []
    for flight, d in rows:
        sizes = Counter()
        for f in sorted(Path(d).glob("*.JPG")):
            with Image.open(f) as im:   # lazy: header only, no decode
                sizes[im.size] += 1
        first = None
        files = sorted(Path(d).glob("*.JPG"))
        if files:
            with Image.open(files[0]) as im:
                first = im.size
        status = "OK " if len(sizes) == 1 else "MIXED"
        if len(sizes) != 1:
            bad.append((flight, dict(sizes)))
        top = ", ".join(f"{w}x{h}:{n}" for (w, h), n in sizes.most_common(4))
        print(f"{status} {flight:24s} first={first} | {top}")

    print()
    if bad:
        print(f"*** {len(bad)} MIXED-SIZE DIRS — MultiImage would mis-window these ***")
        for flight, sizes in bad:
            print(f"  {flight}: {sizes}")
    else:
        print("All dirs uniform: MultiImage's paths[0] sizing is safe for every task.")


if __name__ == "__main__":
    main()
