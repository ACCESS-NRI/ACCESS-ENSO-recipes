# Running evaluation recipes

Most notebooks are intended for an [Australian Research Environment (ARE)](https://are.nci.org.au/) JupyterLab session on NCI Gadi. They use ESMValCore for data discovery and preprocessing, but each notebook may need different model, observation, project, storage, or compute resources. Check its first cells before running it.

## Start on Gadi

1. Join the NCI projects that hold the input data you intend to analyse. The ACCESS-NRI `xp65` project provides the ESMValTool workflow modules; other projects depend on the recipe.
2. In ARE, request a JupyterLab session with the needed `gdata` and `scratch` storage mounted. Use an appropriate CPU size for the notebook and its Dask workload.
3. In a terminal, load the available analysis environment and clone the repository:

   ```bash
   module use /g/data/xp65/public/modules
   module load conda/analysis3
   git clone https://github.com/ACCESS-NRI/ACCESS-Evaluation-Recipes.git
   cd ACCESS-Evaluation-Recipes
   ```

4. Open a notebook from `recipes/`, `examples/`, or `training/`. Inspect dataset names, time periods, and any local paths before executing cells.

The [ACCESS Hive evaluation documentation](https://docs.access-hive.org.au/model_evaluation/evaluation_on_gadi/recipes/) has broader setup guidance. Training-specific instructions remain in [training/2025/README.md](../training/2025/README.md).

## ESMValTool YAML workflows

The ENSO YAML workflows call diagnostic Python scripts stored with the recipes. ESMValTool requires an absolute path to a local script or a path in its installed diagnostic library. Generate machine-specific YAML copies with `scripts/prepare_esmvaltool_recipes.py` as described in the [ENSO workflow guide](../recipes/enso/esmvaltool/README.md).

## About the gallery previews

The gallery uses selected saved notebook images. These may reflect older model runs and do not prove that a recipe executes successfully with the current environment. The GitHub Pages build parses metadata and notebooks but does not execute analyses or access Gadi datasets.
