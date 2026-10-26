# Catalogue metadata

`recipes.yml` is the editable source for the recipe gallery. Each notebook and top-level ESMValTool YAML workflow has one record. The site builder checks that every source file has a record, all IDs and paths are unique, and referenced previews exist.

Required fields:

- `id`: stable, lowercase identifier. Keep it unchanged when editing a recipe.
- `title`, `summary`: plain language shown in search results.
- `kind`: `Notebook`, `ESMValTool recipe`, `Example`, or `Training`.
- `collection`: a named group shown on each card.
- `subjects`, `realms`, `tools`: controlled filter values. Use `General` when a realm does not apply.
- `path`: path from the repository root to the notebook or YAML file.
- `legacy_path`: former path in `ACCESS-ENSO-recipes`, or `null` for new work.
- `preview` (optional): image path relative to `web/`. Saved notebook output is illustrative; it is not a validation result.
- `status` (optional): a short label such as `Historical` for material retained for reference but known to need updating.

To add or update a preview, review the notebook's saved output, then run `python scripts/generate_previews.py` (requires PyYAML and Pillow). This reads notebook JSON without executing cells and creates small images in `web/previews/`. Check the generated image before committing it.

Run `python scripts/build_site.py --check` before submitting a pull request. Run `python scripts/build_site.py` to generate `_site/` for local preview.
