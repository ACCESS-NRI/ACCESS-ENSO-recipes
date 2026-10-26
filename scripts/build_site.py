"""Validate catalogue metadata and build the static recipe gallery."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "catalog" / "recipes.yml"
LEGACY = ROOT / "catalog" / "legacy-paths.json"
WEB = ROOT / "web"
OUTPUT = ROOT / "_site"
CONTENT_ROOTS = ("recipes", "examples", "training")
KINDS = {"Notebook", "ESMValTool recipe", "Training", "Example"}


def validate(records: list[dict]) -> None:
    if not isinstance(records, list):
        raise ValueError("catalog/recipes.yml must contain a list")
    legacy = json.loads(LEGACY.read_text(encoding="utf-8"))
    old_by_new = {new: old for old, new in legacy.items()}
    ids: set[str] = set()
    paths: set[str] = set()
    for record in records:
        required = {"id", "title", "summary", "kind", "collection", "subjects", "realms", "tools", "path", "legacy_path"}
        missing = required - record.keys()
        if missing:
            raise ValueError(f"{record.get('id', 'record')}: missing {sorted(missing)}")
        item_id = record["id"]
        path = record["path"]
        if item_id in ids or path in paths:
            raise ValueError(f"Duplicate catalogue id or path: {item_id}, {path}")
        ids.add(item_id)
        paths.add(path)
        if record["kind"] not in KINDS:
            raise ValueError(f"{item_id}: unknown kind {record['kind']}")
        if not all(isinstance(record[key], str) and record[key].strip() for key in ("title", "summary", "collection")):
            raise ValueError(f"{item_id}: title, summary and collection must be nonempty")
        for key in ("subjects", "realms", "tools"):
            if not isinstance(record[key], list) or not record[key] or not all(isinstance(value, str) and value.strip() for value in record[key]):
                raise ValueError(f"{item_id}: {key} must be a nonempty list of strings")
        file_path = Path(path)
        if file_path.is_absolute() or ".." in file_path.parts or file_path.parts[0] not in CONTENT_ROOTS:
            raise ValueError(f"{item_id}: invalid source path {path}")
        if file_path.suffix not in {".ipynb", ".yml"} or not (ROOT / file_path).is_file():
            raise ValueError(f"{item_id}: source file missing: {path}")
        source_text = (ROOT / file_path).read_text(encoding="utf-8")
        if file_path.suffix == ".ipynb":
            notebook = json.loads(source_text)
            if not isinstance(notebook.get("cells"), list) or "nbformat" not in notebook:
                raise ValueError(f"{item_id}: invalid notebook structure")
        else:
            workflow = yaml.safe_load(source_text)
            if not isinstance(workflow, dict) or "documentation" not in workflow:
                raise ValueError(f"{item_id}: invalid ESMValTool recipe structure")
        if record["legacy_path"] != old_by_new.get(path):
            raise ValueError(f"{item_id}: legacy path does not match migration map")
        preview = record.get("preview")
        if preview:
            preview_path = Path(preview)
            if preview_path.is_absolute() or ".." in preview_path.parts or not (WEB / preview_path).is_file():
                raise ValueError(f"{item_id}: preview missing: {preview}")
    sources = {
        file.relative_to(ROOT).as_posix()
        for dirname in CONTENT_ROOTS
        for file in (ROOT / dirname).rglob("*")
        if file.suffix == ".ipynb" or (file.suffix == ".yml" and file.parent == ROOT / "recipes/enso/esmvaltool")
    }
    if paths != sources:
        raise ValueError(f"Catalogue coverage mismatch. Missing: {sorted(sources-paths)}; stale: {sorted(paths-sources)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate only; do not write _site")
    args = parser.parse_args()
    records = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    validate(records)
    print(f"Validated {len(records)} catalogue entries")
    if args.check:
        return
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    shutil.copytree(WEB, OUTPUT)
    (OUTPUT / "catalogue.json").write_text(json.dumps(records, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    (OUTPUT / ".nojekyll").touch()
    print(f"Built {OUTPUT}")


if __name__ == "__main__":
    main()
