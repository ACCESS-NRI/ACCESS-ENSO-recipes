"""Write local ESMValTool recipe copies with absolute diagnostic script paths.

ESMValTool resolves relative diagnostic paths from its installed diag_scripts
directory, rather than relative to this repository. Keep portable relative
paths in the source YAML and run this command on the machine executing them.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "recipes" / "enso" / "esmvaltool"
DESTINATION = ROOT / "build" / "esmvaltool"
SCRIPT_LINE = re.compile(r"^(\s*script:\s*)(diagnostic_scripts/[\w./-]+\.py)(\s*(?:#.*)?)$", re.MULTILINE)


def prepare(path: Path) -> Path:
    source = path.read_text(encoding="utf-8")
    yaml.safe_load(source)
    replaced = 0

    def absolute(match: re.Match[str]) -> str:
        nonlocal replaced
        relative = match.group(2)
        script = (SOURCE / relative).resolve()
        if not script.is_relative_to(SOURCE.resolve()) or not script.is_file():
            raise ValueError(f"{path.name}: missing diagnostic script {relative}")
        replaced += 1
        return f"{match.group(1)}{script}{match.group(3)}"

    output = SCRIPT_LINE.sub(absolute, source)
    if SCRIPT_LINE.search(output):
        raise ValueError(f"{path.name}: an unresolved diagnostic path remains")
    if replaced == 0:
        raise ValueError(f"{path.name}: no local diagnostic script references found")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    target = DESTINATION / path.name
    target.write_text(output, encoding="utf-8")
    yaml.safe_load(output)
    return target


def main() -> None:
    paths = sorted(SOURCE.glob("*.yml"))
    for path in paths:
        print(prepare(path).relative_to(ROOT))
    print(f"Prepared {len(paths)} local ESMValTool recipes")


if __name__ == "__main__":
    main()
