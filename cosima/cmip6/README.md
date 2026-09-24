# COSIMA CMIP6 diagnostics

This directory contains standalone CMIP6 counterparts of selected
[COSIMA advanced recipes](https://github.com/COSIMA/cosima-recipes/tree/main/03-Advanced-Recipes).
They complement the native ACCESS-OM/CICE examples in the parent
[`cosima/`](../) directory: input is loaded through ESMValCore's
`Dataset` interface and preprocessing utilities rather than the
ACCESS-NRI intake catalogue.

The notebooks are designed for interactive use on NCI Gadi in an
`esmvaltool-workflow` / `conda/analysis3` environment. Each opens a
Dask client, loads CMORised data, applies the required preprocessing,
and closes the client when it finishes. Set up an ARE JupyterLab session
as described in the repository README, then open the desired notebook.

## Diagnostics

| Notebook | Diagnostic |
| --- | --- |
| `01-Meridional_heat_transport.ipynb` | Ocean meridional heat transport |
| `02-Overturning_circulation.ipynb` | Depth- and density-space meridional overturning circulation |
| `03-Surface_water_mass_transformation.ipynb` | Surface water-mass transformation |
| `04-Sea_ice_seasonality.ipynb` | Antarctic sea-ice advance, retreat and duration |
| `05-Geostrophic_velocities.ipynb` | Surface geostrophic velocities from sea level |
| `06-Sea_ice_area_volume.ipynb` | Sea-ice area, extent and volume against observations |
| `07-Temperature_salinity_diagram.ipynb` | Volume-weighted temperature-salinity diagram |
| `08-Relative_vorticity.ipynb` | Relative vorticity and Rossby number |
| `09-Along_slope_velocities.ipynb` | Antarctic slope-current velocities |
| `10-Cross_slope_section.ipynb` | Cross-slope hydrographic section |
| `11-Along_isobath_average.ipynb` | Along-isobath averages |
| `12-Cross_contour_transport.ipynb` | Cross-contour transport |
| `13-Heaving_decomposition.ipynb` | Isopycnal heave and along-isopycnal decomposition |
| `14-Neutral_density.ipynb` | Neutral density |

## Scope and important limitations

These notebooks make CMIP6 diagnostics possible; they are not
numerically interchangeable with the native-grid COSIMA analyses.
CMOR data commonly lack staggered-grid geometry, online density-binned
transport, daily sea level and complete sea-ice/salt-flux observations.
Each notebook documents the applicable input, resolution and sampling
constraints. In particular, face-integrated transports are retained on
their native grids rather than regridded, and the relative-vorticity
notebook labels coarse-grid results as resolved-current shear rather
than mesoscale eddy fields.

The existing `cosima/Temperature_Salinity_Diagram.ipynb` remains a
native-model example. `07-Temperature_salinity_diagram.ipynb` is its
CMIP6-oriented companion, with CMOR inputs and a volume-weighted
histogram.
