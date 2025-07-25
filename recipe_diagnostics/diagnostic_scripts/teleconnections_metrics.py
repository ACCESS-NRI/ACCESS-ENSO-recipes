"""diagnostic script to plot ENSO teleconnections metrics"""

import os
import logging
import iris
from pprint import pformat
import numpy as np
from shapely import box
import shapely.vectorized as shp_vect
import matplotlib.pyplot as plt
import iris.plot as iplt
import cartopy.crs as ccrs

from esmvaltool.diag_scripts.shared import (
    run_diagnostic,
    save_figure,
    get_diagnostic_filename,
    group_metadata,
    select_metadata,
)
from esmvalcore.preprocessor import (
    extract_season,
    seasonal_statistics,
    mask_above_threshold,
    mask_below_threshold,
    climate_statistics,
)


# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))


def plot_level1(input_data, rmse, title):
    """Plot level 1, input data has model and obs"""
    figure = plt.figure(figsize=(20, 6), dpi=300)

    proj = ccrs.PlateCarree(central_longitude=180)
    figure.suptitle(title, y=0.95, size="x-large", weight="bold")
    i = 121

    for label, cube in input_data.items():
        ax1 = plt.subplot(i, projection=proj)
        ax1.coastlines()
        cf1 = iplt.contourf(
            cube, levels=np.arange(-1, 1, 0.1), extend="both", cmap="RdBu_r"
        )
        ax1.set_title(label)
        gl1 = ax1.gridlines(draw_labels=True, linestyle="--")
        gl1.top_labels = False
        gl1.right_labels = False
        i += 1

    plt.text(
        0.1,
        -0.25,
        f"RMSE: {rmse:.2f} ",
        fontsize=12,
        ha="left",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )
    # Add a single colorbar at the bottom
    cax = plt.axes([0.15, 0.08, 0.7, 0.05])
    cbar = figure.colorbar(
        cf1,
        cax=cax,
        orientation="horizontal",
        extend="both",
        ticks=np.arange(-1, 1.5, 0.5),
    )
    # get unit for reg
    if title.split(" ")[1] == "SST":  # SST or PR
        reg_unit = "°C"
    else:
        reg_unit = "mm/day"
    cbar.set_label(f"regression ({reg_unit}/°C)")
    logger.info(f"{title}, {label} : metric:{rmse}")
    figure.subplots_adjust(bottom=0.125, top=0.98, left=0.05, right=0.95)

    return figure


def lin_regress_matrix(cubeA, cubeB):  # array must not contain infs or NaNs
    """
    Calculate the linear regression of cubeA on cubeB using matrix operations.

    Parameters
    ----------
    cubeA: iris.cube.Cube
        The 2D input cube for which the regression is calculated.

    cubeB: iris.cube.Cube
        The cube used as the independent variable in the regression.

    Returns
    -------
    iris.cube.Cube
        A new cube containing the slope of the regression for each spatial point.
    """
    # Get data as flattened arrays
    A_data = cubeA.data.reshape(cubeA.shape[0], -1)  # Shape (time, spatial_points)
    B_data = cubeB.data.flatten()  # Shape (time,)
    logger.info("cubes: %s, %s", cubeA.name, cubeB.name)
    # Add intercept term by stacking a column of ones with cubeB
    B_with_intercept = np.vstack([B_data, np.ones_like(B_data)]).T

    # Solve the linear equations using least squares method
    coefs, _, _, _ = np.linalg.lstsq(B_with_intercept, A_data, rcond=None)
    logger.info("%s, %s", cubeA.coords(), cubeA.shape)
    # Extract slopes from coefficients
    slopes = coefs[0].reshape(cubeA.shape[1], cubeA.shape[2])

    # Create a new Iris Cube for the regression results
    result_cube = iris.cube.Cube(
        slopes,
        long_name="regression ENSO SSTA",
        dim_coords_and_dims=[
            (cubeA.coord("latitude"), 0),
            (cubeA.coord("longitude"), 1),
        ],
    )

    return result_cube


def mask_pacific(cube, seamask_cube):
    """Mask the Pacific region using a vectorized mask and sea mask."""
    region = box(110.0, -15.0, 280.0, 15)  # remove land
    x_p, y_p = np.meshgrid(
        cube.coord(axis="X").points,
        cube.coord(axis="Y").points,
    )

    mask = shp_vect.contains(region, x_p, y_p)
    # seamask get preproc_vargroup
    seamask = seamask_cube[0].data.mask
    # add sea mask to get land in pacific
    mask = mask.astype(int) + seamask.astype(int)
    use_mask = np.zeros_like(mask, bool)
    use_mask[mask == 2] = True

    cube.data = np.ma.MaskedArray(cube.data, mask=use_mask)
    return cube


def compute_telecon_metrics(input_pair, var_group, mask_cube, metric):
    """Compute teleconnection for level 1 and level 2 diagnostics."""
    if metric == "pr_telecon":
        title = "{} PR Teleconnection"  # both seasons
    elif metric == "ts_telecon":
        title = "{} SST Teleconnection"

    val, fig, fig2 = {}, {}, {}
    seasons = {"DJF": "MAMJJASON", "JJA": "SONDJFMAM"}
    for seas, rest in seasons.items():  # ['DJF','JJA']:
        data_values = []
        lvl2_dict, cubes = {}, {}
        for label, ds in input_pair.items():  # obs 0, mod 1
            preproc = {}
            for variable in var_group[:2]:  # slice[:2] enso_cube var_group[2]
                cube = extract_season(ds[variable].copy(), seas)
                preproc[variable] = seasonal_statistics(
                    cube, operator="mean", seasons=(seas, rest)
                )

            regcube = lin_regress_matrix(preproc[var_group[1]], preproc[var_group[0]])
            reg_masked = mask_pacific(regcube, mask_cube)

            data_values.append(reg_masked.data)
            cubes[label] = reg_masked

            # plot level 2 - enso events, enso_cube = ds[var_group[2]]
            lvl2_dict[label] = diagnostic_level_2(
                ds[var_group[2]], preproc[var_group[1]], mask_cube
            )

        fig2[seas] = plot_level2(lvl2_dict, seas)
        val[seas] = np.sqrt(np.mean((data_values[0] - data_values[1]) ** 2))
        fig[seas] = plot_level1(cubes, val[seas], title.format(seas))

    return val, fig, fig2


def mask_to_years(events):
    """Convert masked array to list of years."""
    maskedTime = np.ma.masked_array(events.coord("time").points, mask=events.data.mask)
    return [
        events.coord("time").units.num2date(time).year
        for time in maskedTime.compressed()
    ]


def enso_events(cube):
    """Identify ENSO events from the ssta cube data."""
    std = cube.data.std()
    a_events = mask_to_years(mask_above_threshold(cube.copy(), -0.75 * std))
    o_events = mask_to_years(mask_below_threshold(cube.copy(), 0.75 * std))
    return {"La Nina": a_events, "El Nino": o_events}


def diagnostic_level_2(enso_cube, glb_cube, mask_cube):
    """Compute teleconnection ENSO composites for level 2 plots."""
    events = enso_events(enso_cube)  # get enso events
    cubes_dict = {}
    for enso, years in events.items():
        year_enso = iris.Constraint(time=lambda cell: cell.point.year in years)
        cube_2 = glb_cube.extract(year_enso)  # extract from glb cube
        cube_2 = climate_statistics(cube_2, operator="mean")
        cubes_dict[enso] = mask_pacific(cube_2, mask_cube)

    return cubes_dict


def plot_level2(datads_dict, season):
    """Plot level 2 for ENSO composites."""
    fig = plt.figure(figsize=(20, 9))
    proj = ccrs.PlateCarree(central_longitude=180)
    i = 221
    for enso in ["La Nina", "El Nino"]:
        data_arrs = []
        for ds_name, ds in datads_dict.items():
            ax1 = plt.subplot(i, projection=proj)
            ax1.coastlines()
            cf1 = iplt.contourf(
                ds[enso], levels=np.arange(-2, 2, 0.1), extend="both", cmap="RdBu_r"
            )
            ax1.set_title(f"{ds_name} {enso} {season}", loc="left")
            gl1 = ax1.gridlines(draw_labels=True, linestyle="--")
            gl1.top_labels = False
            gl1.right_labels = False

            var_unit = f"{ds[enso].var_name.upper()}A ({ds[enso].units})"
            data_arrs.append(ds[enso].data)
            if i % 2 == 0:  # obs 0, mod 1
                rmse = np.sqrt(np.mean((data_arrs[0] - data_arrs[1]) ** 2))
                plt.text(
                    -0.05,
                    -0.2,
                    f"RMSE: {rmse:.2f} ",
                    fontsize=12,
                    ha="left",
                    transform=plt.gca().transAxes,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
                )

            i += 1
    cax = plt.axes([0.15, 0.05, 0.7, 0.03])
    cbar = fig.colorbar(
        cf1,
        cax=cax,
        orientation="horizontal",
        extend="both",
        ticks=np.arange(-2, 2.2, 0.5),
    )
    cbar.set_label(var_unit)

    fig.subplots_adjust(bottom=0.125, top=0.95, left=0.05, right=0.95)
    return fig


def get_provenance_record(caption, ancestor_files):
    """Create a provenance record describing the diagnostic data and plot."""

    record = {
        "caption": caption,
        "statistics": ["anomaly"],
        "domains": ["global"],
        "plot_types": ["map"],
        "authors": [
            "chun_felicity",
            # 'beucher_romain',
            # 'sullivan_arnold',
        ],
        "references": [
            "access-nri",
        ],
        "ancestors": ancestor_files,
    }
    return record


def main(cfg):
    """Run ENSO metrics."""

    input_data = cfg["input_data"].values()

    # iterate through each metric and get variable group, select_metadata, map to function call
    metrics = {
        "pr_telecon": ["tos_enso", "pr_global", "ts_enso_events"],
        "ts_telecon": ["tos_enso", "tos_global", "ts_enso_events"],
    }

    mask_grp = select_metadata(input_data, variable_group="enso_mask", project="CMIP6")
    mask_var = [
        iris.load_cube(dataset["filename"]) for dataset in mask_grp
    ]  # for multiple datasets do we need to iterate through?

    for metric, var_preproc in metrics.items():
        logger.info(f"{metric},{var_preproc}")
        obs, models = [], []
        for var_prep in var_preproc:
            obs += select_metadata(input_data, variable_group=var_prep, project="OBS")
            obs += select_metadata(input_data, variable_group=var_prep, project="OBS6")
            models += select_metadata(
                input_data, variable_group=var_prep, project="CMIP6"
            )

        # log
        msg = "{} : observation datasets {}, models {}".format(
            metric, len(obs), len(models)
        )
        logger.info(msg)

        # obs datasets for each model
        obs_datasets = {
            dataset["variable_group"]: iris.load_cube(dataset["filename"])
            for dataset in obs
        }
        obsds_label = f'{obs[0]["dataset"]}_{obs[1]["dataset"]}'
        # group models by dataset
        model_ds = group_metadata(models, "dataset", sort="project")

        # dataset name in models
        for dataset in model_ds:
            logger.info(
                f"{metric}, preprocessed cubes:{len(model_ds)}, dataset:{dataset}"
            )
            dt_files = [ds["filename"] for ds in obs] + [
                ds["filename"] for ds in model_ds[dataset]
            ]

            model_datasets = {
                attributes["variable_group"]: iris.load_cube(attributes["filename"])
                for attributes in model_ds[dataset]
            }

            input_pair = {obsds_label: obs_datasets, dataset: model_datasets}
            logger.info(pformat(model_datasets))
            # process function for each metric
            values, fig, fig_lvl2 = compute_telecon_metrics(
                input_pair, var_preproc, mask_var[0], metric
            )

            # save metric for each pair
            for seas, val in values.items():
                metricfile = get_diagnostic_filename("matrix", cfg, extension="csv")
                with open(metricfile, "a+") as f:
                    f.write(f"{dataset},{seas}_{metric},{val}\n")

                prov_record = get_provenance_record(
                    f"ENSO metrics {seas} {metric}", dt_files
                )
                save_figure(
                    f"{dataset}_{seas}_{metric}",
                    prov_record,
                    cfg,
                    figure=fig[seas],
                    dpi=300,
                )  #

                prov_record = get_provenance_record(
                    f"Dive down 2 {seas} {metric}", dt_files
                )
                save_figure(
                    f"{dataset}_{seas}_{metric}_lvl2",
                    prov_record,
                    cfg,
                    figure=fig_lvl2[seas],
                    dpi=300,
                )


if __name__ == "__main__":
    with run_diagnostic() as config:
        main(config)
