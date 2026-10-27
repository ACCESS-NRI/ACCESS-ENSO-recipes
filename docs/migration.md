# Migration from ACCESS-ENSO-recipes

This branch starts from `ACCESS-NRI/ACCESS-ENSO-recipes` at commit `d95e08d` (26 October 2026). It reorganises the same 91 notebooks and eight ESMValTool YAML recipes; ACCESS-Visualisation-Recipes is outside this project.

## Repository transition

The existing `ACCESS-NRI/ACCESS-ENSO-recipes` repository was renamed to `ACCESS-NRI/ACCESS-Evaluation-recipes`. Merge this branch there; the rename retains the history, issues, and contributors. GitHub redirects the old repository URL and Git remotes, but links to moved notebook files need explicit updates. Keep the former repository name available for GitHub's redirect.

Before releasing under the new name, check that Zenodo still receives releases from the renamed repository and review the citation metadata. Earlier Zenodo releases remain citable by their existing DOI.

GitHub Pages is configured to publish from GitHub Actions. Once this branch is merged into `main`, `.github/workflows/catalogue.yml` builds and deploys the gallery.

## Folder moves

| Former location | New location |
| --- | --- |
| `*.ipynb` at repository root | `recipes/enso/` |
| `IOD_recipes/` | `recipes/iod/` |
| `MJO_BSISO/` | `recipes/mjo-bsiso/` |
| `recipe_diagnostics/` | `recipes/enso/esmvaltool/` |
| `cosima/` | `recipes/ocean/cosima/` |
| `cosima/cmip6/` | `recipes/ocean/cmip6/` |
| `russell18jgr/` | `recipes/ocean/russell2018/` |
| `basic_functions_examples/` | `examples/esmvalcore/` |
| `training2025/` | `training/2025/` |

The complete file map is in [`catalog/legacy-paths.json`](../catalog/legacy-paths.json). Update links to moved files in ACCESS Hive and training material when the renamed repository and gallery are published. Historical Zenodo citations should continue to use the DOI of the release used.
