"""Create small catalogue previews from saved notebook image outputs.

This reads notebook JSON; it never executes notebook cells. Run manually after
reviewing notebook outputs. The generated files are committed with the site.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import yaml
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "catalog" / "recipes.yml"
PREVIEWS = ROOT / "web" / "previews"


def first_image(notebook: dict) -> Image.Image | None:
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            for mime in ("image/png", "image/jpeg"):
                if mime not in data:
                    continue
                encoded = data[mime]
                if isinstance(encoded, list):
                    encoded = "".join(encoded)
                try:
                    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
                except (OSError, ValueError):
                    continue
    return None


def main() -> None:
    import json

    records = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    made = 0
    for record in records:
        if not record["path"].endswith(".ipynb"):
            continue
        notebook = json.loads((ROOT / record["path"]).read_text(encoding="utf-8"))
        source = first_image(notebook)
        if source is None:
            record.pop("preview", None)
            continue
        image = ImageOps.contain(source, (640, 400), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (640, 400), "#f6f8fa")
        canvas.paste(image, ((640 - image.width) // 2, (400 - image.height) // 2))
        filename = f"{record['id']}.jpg"
        canvas.save(PREVIEWS / filename, "JPEG", quality=82, optimize=True)
        record["preview"] = f"previews/{filename}"
        made += 1
    CATALOG.write_text(
        "# Catalogue records for notebooks and runnable recipe YAML. See catalog/README.md.\n"
        + yaml.safe_dump(records, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
    )
    print(f"Created {made} notebook previews")


if __name__ == "__main__":
    main()
