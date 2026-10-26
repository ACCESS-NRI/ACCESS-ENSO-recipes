# Migration from ACCESS-ENSO-recipes

This branch starts from `ACCESS-NRI/ACCESS-ENSO-recipes` at commit `d95e08d` (26 October 2026). It reorganises the same 91 notebooks and eight ESMValTool YAML recipes; ACCESS-Visualisation-Recipes is outside this project.

## Repository transition

Rename the existing `ACCESS-NRI/ACCESS-ENSO-recipes` repository to `ACCESS-NRI/ACCESS-Evaluation-Recipes`, then merge this branch there. A second repository is unnecessary for this expansion and would split the existing history, issues, and contributors. GitHub redirects the old repository URL and Git remotes after a rename, but individual moved notebook paths and an old GitHub Pages project URL need explicit link updates. Keep the former repository name available for GitHub's redirect.

Before releasing under the new name, check that Zenodo still receives releases from the renamed repository and review the citation metadata. Earlier Zenodo releases remain citable by their existing DOI.

This branch has not renamed the GitHub repository or changed its remote; that requires repository administration.

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
