# ACCESS Evaluation Recipes

A community collection of notebooks and ESMValTool workflows for evaluating ACCESS and other climate models. The collection began with the [CLIVAR ENSO metrics](https://doi.org/10.1175/BAMS-D-19-0337.1) and now includes the Indian Ocean Dipole, MJO/BSISO, ocean diagnostics, examples, and training material. Most notebooks use [ESMValCore](https://docs.esmvaltool.org/projects/ESMValCore/en/stable/) to discover, load, and preprocess climate data.

**Browse the catalogue:** the static gallery in [`web/`](web/) can be built locally with `python -m pip install -r requirements-site.txt` followed by `python scripts/build_site.py`. Once this repository is published, its GitHub Pages site will provide searchable cards, filters, and saved-output previews. Previews illustrate a notebook; they do not establish that its data or environment is current.

## Find a recipe

| Area | Contents |
| --- | --- |
| [ENSO metrics](recipes/enso/) | CLIVAR metrics notebooks and [ESMValTool YAML workflows](recipes/enso/esmvaltool/) |
| [Indian Ocean Dipole](recipes/iod/) | IOD characteristics and Indian Ocean climatology |
| [MJO and BSISO](recipes/mjo-bsiso/) | Intraseasonal patterns, life cycles, and precipitation biases |
| [Ocean](recipes/ocean/) | Native-model COSIMA examples, CMIP6 counterparts, and Russell et al. diagnostics |
| [ESMValCore examples](examples/esmvalcore/) | Regridding, ensembles, means, and data handling |
| [Training](training/2025/) | 2025 workshop examples and exercises |

The [catalogue metadata](catalog/recipes.yml) describes all 91 notebooks and eight ESMValTool YAML recipes. It is the source for the gallery. The `catalog/legacy-paths.json` file maps files from the former `ACCESS-ENSO-recipes` layout to their new locations.

## Run a notebook

Clone this repository on NCI Gadi, open an ARE JupyterLab session, and choose a notebook from the catalogue. The data collections and compute resources needed vary by recipe. See [running recipes](docs/running.md) for setup guidance, and read the first cells of each notebook before running it.

```bash
module use /g/data/xp65/public/modules
module load conda/analysis3
git clone https://github.com/ACCESS-NRI/ACCESS-Evaluation-Recipes.git
cd ACCESS-Evaluation-Recipes
```

For ESMValTool YAML workflows with local diagnostic scripts, first generate machine-specific copies as described in the [ENSO ESMValTool guide](recipes/enso/esmvaltool/README.md).

## Contribute

New notebooks and YAML workflows belong in the relevant subject folder. Add a record to [the catalogue](catalog/README.md), then run `python scripts/build_site.py --check`. Use a pull request so the scientific method, data requirements, preview, and metadata can be reviewed together. [Notebook review guidance](docs/github_notebooks.md) is available for contributors.

## Provenance and citation

This project reorganises material from [ACCESS-ENSO-recipes](https://github.com/ACCESS-NRI/ACCESS-ENSO-recipes), whose releases have a [Zenodo record](https://doi.org/10.5281/zenodo.14168217). Refer to that record when citing an earlier release. A new release and citation for this broader collection should be configured after the new repository is published. The original authorship remains in the notebook history and `.zenodo.json`.

Licensed under [Apache-2.0](LICENSE). Developed by the ACCESS-NRI Model Evaluation and Diagnostics team and contributors.
