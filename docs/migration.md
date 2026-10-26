# Migration from ACCESS-ENSO-recipes

This repository starts from `ACCESS-NRI/ACCESS-ENSO-recipes` at commit `d95e08d` (26 October 2026). It reorganises the same 91 notebooks and eight ESMValTool YAML recipes; ACCESS-Visualisation-Recipes is outside this project.

| Former location | New location |
| --- | --- |
| `*.ipynb` at repository root | `recipes/enso/` |
| `IOD_recipes/` | `recipes/iod/` |
| `MJO_BSISO/` | `recipes/mjo-bsiso/` |
| `recipe_diagnostics/` | `recipes/enso/esmvaltool/` |
| `cosima/` | `recipes/ocean/cosima/` |
| `russell18jgr/` | `recipes/ocean/russell2018/` |
| `basic_functions_examples/` | `examples/esmvalcore/` |
| `training2025/` | `training/2025/` |

The complete file map is in [`catalog/legacy-paths.json`](../catalog/legacy-paths.json). Because the original repository remains at its current URL during migration, existing links to its notebooks continue to work. Update links in ACCESS Hive and training material after the new repository and gallery are published.

Historical Zenodo citations should continue to use the original repository's DOI. Configure and verify a new Zenodo integration and citation for the broader collection before its first release.
