# ENSO ESMValTool workflows

Eight YAML workflows reproduce groups of CLIVAR ENSO metrics. Their diagnostic Python scripts are in [`diagnostic_scripts/`](diagnostic_scripts/). They complement the interactive ENSO notebooks in the parent directory.

## Run a workflow on Gadi

The tracked YAML files use portable `diagnostic_scripts/...` references. ESMValTool expects a local diagnostic either at an absolute path or inside its installed diagnostic library, so prepare runnable copies on the machine where you will run them:

```bash
module use /g/data/xp65/public/modules
module load esmvaltool-workflow
python scripts/prepare_esmvaltool_recipes.py
esmvaltool run build/esmvaltool/recipe_enso_climatology1.yml
```

The helper writes files to the ignored `build/esmvaltool/` directory and resolves each script to its absolute location in this checkout. Re-run it if you move the checkout. Review the selected YAML's datasets, observation access, and preprocessing settings before execution. The catalogue and GitHub Pages build validate file references but do not run scientific diagnostics.

## Groups

- Background climatology: `recipe_enso_climatology1.yml` and `recipe_enso_climatology_level2-3.yml`.
- ENSO characteristics and dive downs: `recipe_enso_characteristics.yml`, `recipe_enso_divedown9-10_14-15.yml`, and `recipe_enso_divedown11-13.yml`.
- Teleconnections: `recipe_enso_teleconnection.yml`.
- Feedbacks: `recipe_enso_feedback.yml` and `recipe_enso_feedbacknhf.yml`.

The `1_metrics_recipes.ipynb` notebook is retained as a historical exploration. Its original local paths and recipe names have not been brought up to date; use the YAML workflows above for the current catalogue entry.
