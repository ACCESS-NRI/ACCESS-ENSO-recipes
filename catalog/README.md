# Catalogue metadata

`recipes.yml` is the editable source for the recipe gallery. Each notebook and top-level ESMValTool YAML workflow has one record. The site builder checks that every source file has a record, all IDs and paths are unique, and referenced previews exist.

Required fields:

- `id`: stable, lowercase identifier. Keep it unchanged when editing a recipe.
- `title`, `summary`: plain language shown in search results.
- `kind`: `Notebook`, `ESMValTool recipe`, `Example`, or `Training`.
- `collection`: a named group shown on each card.
- `subjects`, `realms`, `tools`: controlled filter values. Use `General` when a realm does not apply.
- `models`: ACCESS models explicitly used or compared in the recipe. Use display names from `models.yml` (for example, `ACCESS-ESM1.5` for the CMIP dataset ID `ACCESS-ESM1-5`). Use `[]` when the recipe is model agnostic or only mentions a model as background. Do not tag a model solely because the method could be adapted to it.
- `path`: path from the repository root to the notebook or YAML file.
- `legacy_path`: former path in `ACCESS-ENSO-recipes`, or `null` for new work.
- `preview` (optional): image path relative to `web/`. Saved notebook output is illustrative; it is not a validation result.
- `status` (optional): a short label such as `Historical` for material retained for reference but known to need updating.

`models.yml` controls the model filter and the six featured model tiles. Tile counts are calculated from the recipe records; a zero means no recipe is explicitly tagged for that model yet. ACCESS-CM2 is available in the filter because existing recipes use it, although it is not one of the six featured models.

To add or update a preview, review the notebook's saved output, then run `python scripts/generate_previews.py` (requires PyYAML and Pillow). This reads notebook JSON without executing cells and creates small images in `web/previews/`. Check the generated image before committing it.

Run `python scripts/build_site.py` after editing metadata to refresh the checked-in `web/catalogue-data.js` used by direct file previews and build `_site/`. Run `python scripts/build_site.py --check` before submitting a pull request; it also checks that the preview data is current.
