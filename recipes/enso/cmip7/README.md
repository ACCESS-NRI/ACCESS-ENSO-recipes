# ENSO recipes for ACCESS-ESM1-6 CMIP7

These 24 notebooks are separate CMIP7 versions of the numbered ENSO notebooks in ACCESS-Evaluation-Recipes. The diagnostic and plotting steps are retained. Model `Dataset` definitions and dataset labels use the published ACCESS-ESM1-6 `historical` `r1i1p1f1` output on Gadi. Existing observational comparisons are retained. Old cell outputs have been cleared so they cannot be mistaken for CMIP7 results.

Run in ARE with `conda/analysis3-26.10` and select its Python 3 kernel. Mount `gdata/im55+gdata/xp65+gdata/kj13+gdata/ct11` in the ARE Storage field. ESMValCore's Gadi configuration discovers the model files in `/g/data/im55/publications/MIP-DRS7/CMIP7/CMIP/ACCESS-Consortium/ACCESS-ESM1-6/historical/r1i1p1f1`. The notebooks also register the existing `/g/data/kj13/datasets/esmvaltool/obsdata-v2` OBS6 source using the ESMValCore 2.15 configuration API.

The model grids are `g115` for most atmospheric fields, `g109` for `tauu`, and `g118` for ocean fields. The published monthly and fixed fields were checked on Gadi on 27 October 2026. The requested historical time windows are inherited from the corresponding original notebooks.

Validation on Gadi used an ARE compute allocation with `conda/analysis3-26.10` and the published `/g/data/im55` files. File discovery succeeded for all 12 model variables and all 12 distinct observational `Dataset` definitions. Two-year samples of `pr`, `tos`, `tauu`, and `hfls` loaded; fixed land/ocean masks and cell-area fields resolved, and the area fields attached as cell measures. All 24 notebooks passed JSON and Python syntax checks.

Twenty-three notebooks (01–14 and 16–24) completed end to end with inline plotting on 27 October 2026. They saved 68 figures and no error outputs. Notebook 15 reached a 20-minute per-cell test limit in its model preprocessing step. A longer rerun was stopped before completion, so notebook 15 remains unverified. The duration guard in notebook 14 and the `float64` cast before Iris concatenation in notebook 24 are the two small diagnostic changes made after testing; both were approved for these CMIP7 copies. Some inherited detrending steps emit numerical warnings, although the affected notebooks completed and rendered populated figures.

The GPCP-SG observation files on Gadi use `type='sat'`; the six notebooks that use GPCP-SG set that Dataset facet accordingly.
