"""Compute and plot IOD pattern."""

import logging
import os

import iris
from iris.util import rolling_window
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import iris.plot as iplt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from esmvalcore.preprocessor import (
    climate_statistics,
    extract_season,
    mask_landsea,
)

from esmvaltool.diag_scripts.shared import (
    ProvenanceLogger,
    get_diagnostic_filename,
    group_metadata,
    run_diagnostic,
    save_data,
    save_figure,
    select_metadata,
)

logger = logging.getLogger(os.path.basename(__file__))

def iod_patternplot_level1(input_data, metricval, y_label, title, dtls, ax1):
    # with subplot ax1
    ax1.plot(*input_data[1], label=dtls[1])
    ax1.plot(*input_data[0], label=f"ref: {dtls[0]}", color="black")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_lon))

    ax1.set_yticks(np.arange(-2,3, step=1))
    ax1.axhline(y=0, color='black', linewidth=1)
    ax1.set_ylabel(y_label)
    ax1.set_title(title) #
    ax1.legend()
    ax1.grid(linestyle='--')

    ax1.text(
            0.5,
            0.95,
            f"RMSE: {metricval:.2f}",
            fontsize=12,
            ha="center",
            transform=plt.gca().transAxes,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

def plot_pattern_level2(eastwest):
        fig2 = plt.figure(figsize=(20, 12))
        proj = ccrs.Orthographic(central_longitude=80.0)
        i = 221
        for processed in eastwest:
            for label, cube in processed.items():
                
                ax1 = plt.subplot(i, projection=proj)
                cf1 = iplt.contourf(cube, levels=np.arange(-2,2,0.1), extend='both', cmap='RdBu_r')

                ax1.add_feature(cfeature.LAND, facecolor='gray')  # Add land feature with gray color
                ax1.coastlines()
                ax1.set_extent([40, 120, -15, 15], crs=ccrs.PlateCarree())
                ax1.set_title(label)
            
                # Add gridlines for latitude and longitude
                gl1 = ax1.gridlines(draw_labels=True, linestyle='--')
                gl1.top_labels = False
                gl1.right_labels = False
            
                i+=1
        cax = plt.axes([0.15,0.08,0.7,0.05])
        cbar = fig2.colorbar(cf1, cax=cax, orientation='horizontal', ticks=np.arange(-2, 2.5, 0.5))
        cbar.set_label('regression(IODE/W SSTA, SSTA) (°C/°C)')
        return fig2


#linear regression of sst_iod on sst_eq
def lin_regress(cube_ssta, cube_iod): #1d 
    A_data = cube_ssta.data  # Shape (time, spatial_points)
    B_data = cube_iod.data.flatten()  # Shape (time,)
    
    # Add intercept term by stacking a column of ones with cubeB
    B_with_intercept = np.vstack([B_data, np.ones_like(B_data)]).T
    # Solve the linear equations using least squares method
    coefs, _, _, _ = np.linalg.lstsq(B_with_intercept, A_data, rcond=None)
    return cube_ssta.coord('longitude').points, coefs[0]

def lin_regress_2d(cube_ssta, landmask, cube_iod): # cube_ssta from sst_eq2
   # Get data as flattened arrays
    A_data = cube_ssta.data.reshape(cube_ssta.shape[0], -1)  # Shape (time, spatial_points)
    B_data = cube_iod.data.flatten()  # Shape (time,)

    # Add intercept term by stacking a column of ones with cubeB
    B_with_intercept = np.vstack([B_data, np.ones_like(B_data)]).T

    # Solve the linear equations using least squares method
    coefs, _, _, _ = np.linalg.lstsq(B_with_intercept, A_data, rcond=None)
    
    # Extract slopes from coefficients #coefs 1
    slopes = coefs[0].reshape(cube_ssta.shape[1], cube_ssta.shape[2])
    
    # add land mask from ssta to slopes
    cube_data = np.ma.MaskedArray(slopes, mask=landmask)
    
    # Create a new Iris Cube for the regression results
    result_cube = iris.cube.Cube(cube_data, long_name='regression SSTA',
                                 dim_coords_and_dims=[(cube_ssta.coord('latitude'), 0),
                                                      (cube_ssta.coord('longitude'), 1)])

    return result_cube


def data_to_cube(line_point, in_cube, metric):
    """Translate computed data to cube to save."""
    if in_cube is None:
        cube = iris.cube.Cube(
            line_point,
            long_name=metric,
        )
    else:
        if isinstance(in_cube, iris.cube.Cube):
            coord = in_cube.coord("longitude")
        else:
            coord = iris.coords.DimCoord(
                in_cube,
                long_name="months 6 year ENSO epoch",
            )
        cube = iris.cube.Cube(
            line_point,
            long_name=metric,
            dim_coords_and_dims=[(coord, 0)],
        )
    return cube


def compute_enso_metrics(input_pair, dt_ls, var_group):
    """Compute values for each of the ENSO metrics.

    Takes groupings of datasets required for each ENSO metric sorted
    and iterated through in the main function to compute the metric.

    Args:
        input_pair: List of dictionaries [obs_datasets, model_datasets]
            where dictionary key of dataset is variable group(preprocessor).
        dt_ls: List of dataset names.
            Used for labels in plots.
        var_group: List referring to preprocessed group used for the metric;
            list length is 1,
            or 2 for pattern and diversity metrics for linear regrssion
    """
    metric_values = {}
    val = None

    regressed = {}
    east, west = {}, {}
    for i, data in enumerate(input_pair): #["pattern_iode", "pattern_iodw", "sst_eq"],
        reg_east = lin_regress(data[var_group[2]], data[var_group[0]])
        reg_west = lin_regress(data[var_group[2]], data[var_group[1]])

        regressed[dt_ls[i]] = [reg_east, reg_west]
    
        # level 2
        # from sst_eq get mask
        mask = mask_landsea(data[var_group[3]], mask_out="land")[0].data.mask
        regressed_2d_east = lin_regress_2d(data[var_group[3]], mask, data[var_group[0]])
        regressed_2d_west = lin_regress_2d(data[var_group[3]], mask, data[var_group[1]])
        # data_to_save.append(regressed_2d) 
        east[dt_ls[i]+" IODE"] = regressed_2d_east
        west[dt_ls[i]+" IODW"] = regressed_2d_west
    
    # create figure for subplots
    fig1 = plt.figure(figsize=(15, 7), dpi=300)
    for i, eastwest in enumerate(["East", "West"]):
        ax = fig1.add_subplot(121 + i)
        val = np.sqrt(
            np.mean((np.array(regressed[dt_ls[0]][i][1]) - np.array(regressed[dt_ls[1]][i][1])) ** 2),
        )
        iod_patternplot_level1(
            [regressed[dt_ls[0]][i], regressed[dt_ls[1]][i]],
            val,
            f"reg(IOD{eastwest[0]} SSTA, SSTA)",
            f"IOD {eastwest} pattern",
            dt_ls,
            ax,
        )
        metric_values[f"iod_pattern_{eastwest.lower()}"] = val

    # level 2 figure
    fig2 = plot_pattern_level2([east, west])

    return metric_values, fig1, fig2 #data_to_save


def format_lon(x_val, _):
    """Format longitude in plot axis."""
    if x_val > 180:
        return f"{(360 - x_val):.0f}°W"
    if x_val == 180:
        return f"{x_val:.0f}°"

    return f"{x_val:.0f}°E"


def group_obs_models(obs, models, var_preproc, cfg):
    """Group obs to models for metric computation."""
    metricfile = get_diagnostic_filename("matrix", cfg, extension="csv")
    prov_record = get_provenance_record(
        "iod_pattern",
        list(cfg["input_data"].keys()),
    )
    # obs datasets for each model
    obs_datasets = {
        dataset["variable_group"]: iris.load_cube(dataset["filename"])
        for dataset in obs
    }
    # group models by dataset
    for dataset, attributes in group_metadata(
        models,
        "dataset",
        sort="project",
    ).items():
        logger.info(
            "dataset:%s",
            dataset,
        )
        data_labels = [obs[0]["dataset"], dataset]
        output = compute_enso_metrics(
            [
                obs_datasets,
                {
                    attr["variable_group"]: iris.load_cube(attr["filename"])
                    for attr in attributes
                },
            ],
            data_labels,
            var_preproc,
            # metric,
        )
        # # save returned cubes
        # for i, cube in enumerate(output[2]):
        #     save_data(f"{data_labels[i]}_{metric}", prov_record, cfg, cube)

        if output[0]: # metric_values dict, iterate
            with open(metricfile, "a+", encoding="utf-8") as fileo:
                for metric_name, val in output[0].items():
                    fileo.write(f"{dataset},{metric_name},{val}\n")

            save_figure(
                f"{dataset}_iod_pattern",
                prov_record,
                cfg,
                figure=output[1],
                dpi=300,
            )

            save_figure(
                f"{dataset}_iod_pattern_level2",
                prov_record, # level 2 different record?
                cfg,
                figure=output[2],
                dpi=300,
            )
        # clear value,fig
        output = None


def get_provenance_record(metric, ancestor_files):
    """Create a provenance record describing the diagnostic data and plot."""
    caption = {
        "iod_pattern": (
            "Zonal structure of sea surface temperature anomalies in the "
            + "equatorial Pacific (averaged between 5°S and 5°N)."
        ),
        "values": "List of metric values.",
    }
    record = {
        "caption": caption[metric],
        "statistics": ["anomaly"],
        "domains": ["eq"],
        "plot_types": ["line"],
        "authors": [
            "chun_felicity",
            # "gillett_zoe", add to config_ref
        ],
        "references": [
            "planton2021",
        ],
        "ancestors": ancestor_files,
    }
    return record


def main(cfg):
    """Run metric iod_pattern."""
    input_data = cfg["input_data"].values()

    var_preproc = ["pattern_iode", "pattern_iodw", "sst_eq_meridional", "sst_eq"]

    obs, models = [], []
    for var_prep in var_preproc:
        obs += select_metadata(
            input_data,
            variable_group=var_prep,
            project="OBS",
        )
        obs += select_metadata(
            input_data,
            variable_group=var_prep,
            project="OBS6",
        )
        models += select_metadata(
            input_data,
            variable_group=var_prep,
            project="CMIP6",
        )

    group_obs_models(obs, models, var_preproc, cfg)

    # write provenance for csv metrics
    metricfile = get_diagnostic_filename("matrix", cfg, extension="csv")
    prov = get_provenance_record("values", list(cfg["input_data"].keys()))
    with ProvenanceLogger(cfg) as provenance_logger:
        provenance_logger.log(metricfile, prov)


if __name__ == "__main__":
    with run_diagnostic() as config:
        main(config)
