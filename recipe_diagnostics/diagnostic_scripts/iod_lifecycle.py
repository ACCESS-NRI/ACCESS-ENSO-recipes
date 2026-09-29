"""Compute and plot ENSO characteristics metrics."""

import logging
import os

import iris
from iris.util import rolling_window
import matplotlib.pyplot as plt
import iris.plot as iplt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import numpy as np
from matplotlib.lines import Line2D
from esmvalcore.preprocessor import (
    climate_statistics,
    extract_month,
    # rolling_window_statistics,
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


def plot_level1(input_data, metricval, y_label, title, dtls):
    """Create plots for output data."""
    figure = plt.figure(figsize=(10, 6), dpi=300)

    plt.scatter(
        range(len(input_data)),
        input_data,
        c=["black", "tab:blue"],
        marker="D",
        linewidth=2,
    )
    create_legend(dtls)
    plt.xlim(-0.5, 2)
    plt.xticks([])

    plt.text(
        0.75,
        0.8,
        f"metric(%): {metricval:.2f}",
        fontsize=12,
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    plt.title(title)  # metric name
    plt.grid(linestyle="--")
    plt.ylabel(y_label)

    logger.info("%s : metric: %f", dtls[1], metricval)

    return figure

def plot_line2(input_data, y_label, title, dtls):
    """Create plots for output data."""
    figure = plt.figure(figsize=(10, 6), dpi=300)

    iplt.plot(input_data[1], label=dtls[1])
    iplt.plot(input_data[0], label=f"ref: {dtls[0]}", color="black")
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_lon))

    # Adding labels and title
    plt.xlabel('Longitude')
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(linestyle='--')
    plt.legend()
    return figure

def plot_linemonths(input_data, title, dtls):
    figure = plt.figure(figsize=(6, 6))
    # Plot observation data in black
    iplt.plot(input_data[0], color='black', label=dtls[0], linewidth=4)
    # Plot model data in blue
    iplt.plot(input_data[1], color='blue', label=dtls[1], linewidth=4)

    # Define x-axis labels
    months = ['Jan', 'May', 'Sep']
    plt.xticks(range(1,13,4),labels=months)
    # Set the x and y axis labels
    plt.xlabel('Months')
    plt.ylabel('SSTA std (°C)')
    plt.title(title)
    plt.grid(linestyle='--')
    plt.legend()

    return figure

def plotmaps_level3(input_data, figsize=[20, 7]):
    """Create map plots for pair of input data."""
    fig = plt.figure(figsize=(figsize[0], figsize[1])) #20,10
    proj = ccrs.Orthographic(central_longitude=80)
    i=121
    for label, cube in input_data.items():

        ax1 = plt.subplot(i, projection=proj)
        ax1.add_feature(cfeature.LAND, facecolor="gray")
        ax1.coastlines()

        cf1 = iplt.contourf(
            cube,
            cmap="Reds",
            extend="both",
            levels=np.arange(0,2,0.1),
        )
        ax1.set_extent([40, 120, -15, 15], crs=ccrs.PlateCarree())
        ax1.set_title(label)

        # Add gridlines for latitude and longitude
        gl1 = ax1.gridlines(draw_labels=True, linestyle="--")
        gl1.top_labels = False
        gl1.right_labels = False
        i+= 1

    # Add a single colorbar at the bottom
    cax = plt.axes([0.15, 0.08, 0.7, 0.05])
    cbar = fig.colorbar(cf1, cax=cax, orientation="horizontal", extend="both")
    cbar.set_label('SSTA std (°C)') #

    return fig 
def plothovmoller_months(input_data):
    fig = plt.figure(figsize=(20, 7))
    i =121
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for label, cube in input_data.items():
        ax1 = plt.subplot(i)
        c1 = iplt.contourf(cube, cmap='Reds')
        ax1.set_yticks(range(1,13), labels=months)
        ax1.set_title(label)
        ax1.set_xlabel("Longitude")
        ax1.set_xlim([50,110]) #limit lon
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_lon))
        i+=1
    # Add a single colorbar at the bottom
    cax = plt.axes([0.15,-0.08,0.7,0.05])
    cbar = fig.colorbar(c1, cax=cax, orientation='horizontal', extend='both', ticks=np.arange(0,2,0.5))
    cbar.set_label('SSTA std (°C)')
    return fig

def iod_lifecycleplot_level1(input_data):
    # Define custom ticks for the y-axis (every 6 months)
    yticks = range(1, 73, 6)
    ytick_labels = ['Jan', 'Jul'] * (len(yticks) // 2)
    # Define shared color limits for both subplots
    vmin = -1.2
    vmax = 1.2

    # Create the subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 10), sharey=True)
    i = 0
    for label, data in input_data.items(): # input_data
        if i == 0:
            ax = ax1
        else:
            ax = ax2
        c1 = ax.contourf(data[1].coord('longitude').points, range(1, 73), data[0],
                  cmap='RdBu_r', levels=np.linspace(vmin, vmax, 14))
        ax.set_xlabel('Longitude')
        ax.set_title(f'{label}: reg(IOD SSTA, SSTA)')
        ax.set_xlim([50,110])
        i+= 1

    ax1.set_ylabel('Time (Months)')
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(ytick_labels)
    # Adjust the layout to add more space for the colorbar on the right
    plt.subplots_adjust(right=1.05)

    # Add a colorbar and position it slightly to the right of the plots
    ticks = np.arange(vmin, vmax + 0.2, 0.2)
    cbar = fig.colorbar(c1, ax=[ax1, ax2], label='Regression', orientation='vertical',
                        ticks=ticks, fraction=0.05, pad=0.04)
    return fig

def create_legend(dt_ls):
    """Create a legend for the scatter plots."""
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="black",
            markersize=8,
            label=dt_ls[0],
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="tab:blue",
            markersize=8,
            label=f"Ref: {dt_ls[1]}",
        ),
    ]
    plt.legend(handles=legend_elements)


def sst_regressed(dmi_cube, dmi_area): 
    """Regression function for both 1d and area for lifecycle epochs."""
    dmi_sep = extract_month(dmi_cube, 9)
    # 6 year epoch: [yr-2, yr-1, yr, yr+1, yr+2, yr+3] with monthly data (12) # exclude first year
    dmi_area_sel = rolling_window(dmi_area[12:].data, window=6*12, step=12, axis=0) #2d
    a_data = dmi_area_sel.reshape(dmi_area_sel.shape[0], -1)
    
    # not first 2 or last 3 years for epochs # also exclude first year, leadlagyr=3
    b_data = dmi_sep[3:-3].data
    b_with_intercept = np.vstack([b_data, np.ones_like(b_data)]).T    

    # 2 area linear regression
    coefs_area, _, _, _ = np.linalg.lstsq(b_with_intercept, a_data, rcond=None)
    slope_area = coefs_area[0].reshape(dmi_area_sel.shape[1], dmi_area_sel.shape[2])
    
    return slope_area


def compute_enso_metrics(input_pair, dt_ls, var_group, metric):
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
        metric: Name of metric to calculate.
            eg. 09pattern, 10lifecyle.
    """
    metric_comp = {}
    level2 = []
    val = None
    figures = []

    if metric == "iod_lifecycle": #["iod_west","iod_east", "meridional"],

        for i, data in enumerate(input_pair):
            cube_dmi = data[var_group[0]] - data[var_group[1]] #for amplitude, climate_stats
            sst_reg = sst_regressed(cube_dmi, data[var_group[2]])
            metric_comp[dt_ls[i]] = [sst_reg, data[var_group[2]]]

        fig = iod_lifecycleplot_level1(metric_comp)
        figures.append(fig)

    elif metric == "iod_amplitude": #["iod_west","iod_east", "meridional","trop"],
        values = []
        for data in input_pair:
            cube_dmi = data[var_group[0]] - data[var_group[1]]
            cube = climate_statistics(cube_dmi, operator="std_dev", period="full")
            values.append(cube.data.item()) #metrics
            
            level2.append(climate_statistics(data[var_group[2]], operator="std_dev", period="full"))
            metric_comp[dt_ls[input_pair.index(data)]] = climate_statistics(data[var_group[3]], operator="std_dev", period="full")

        val = compute(values[0], values[1])
        figures.append(plot_level1(values, val, "SSTA std (°C)", "DMI Amplitude", dt_ls))

        #level2 & 3
        figures.append(plot_line2(level2, "SSTA std (°C)", "SSTA standard deviation", dt_ls))
        figures.append(plotmaps_level3(metric_comp, figsize=[20, 7]))

    elif metric == "iod_seasonality": #["iod_west","iod_east","mer_climstd"],
        ssta_std = iod_seasonality(input_pair)
        val = compute(ssta_std[0], ssta_std[1])
        figures.append(plot_level1(ssta_std, val, "DMI std (MAM/SON) (°C/°C)", "IOD Seasonality", dt_ls))
        # level2
        for data in input_pair:
            cube_dmi = data[var_group[0]] - data[var_group[1]]
            cube = climate_statistics(cube_dmi, operator="std_dev", period="monthly")
            level2.append(cube)
        figures.append(plot_linemonths(level2, "DMI standard deviation", dt_ls))
        #level3
        level3 = {dt_ls[input_pair.index(data)] : data[var_group[2]] for data in input_pair}
        figures.append(plothovmoller_months(level3))

    return val, figures

def iod_seasonality(input_pair):
    ssta_std = []
    for data in input_pair:
        seas_values = []
        for season in ["MAM","SON"]:
            west_autumn = extract_season(data["iod_west"], season=season)
            east_autumn = extract_season(data["iod_east"], season=season)
            dmi = west_autumn - east_autumn
            clima_aut = climate_statistics(dmi, operator="std_dev", period="full")
            seas_values.append(clima_aut.data.item())
        
        ssta_std.append(seas_values[0]/seas_values[1])
        
    return ssta_std


def format_lon(x_val, _):
    """Format longitude in plot axis."""
    if x_val > 180:
        return f"{(360 - x_val):.0f}°W"
    if x_val == 180:
        return f"{x_val:.0f}°"

    return f"{x_val:.0f}°E"


def compute(obs, mod):
    """Compute percentage metric value."""
    return abs((mod - obs) / obs) * 100


def group_obs_models(obs, models, metric, var_preproc, cfg):
    """Group obs to models for metric computation."""
    metricfile = get_diagnostic_filename("matrix", cfg, extension="csv")
    prov_record = get_provenance_record(
        metric,
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
            "%s, dataset:%s",
            metric,
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
            metric,
        )
        # # save returned cubes
        # for i, cube in enumerate(output[2]):
        #     save_data(f"{data_labels[i]}_{metric}", prov_record, cfg, cube)

        if output[0]: # metric_values dict, iterate
            with open(metricfile, "a+", encoding="utf-8") as fileo:
                fileo.write(f"{dataset},{metric},{output[0]}\n")
        if output[1]: # figures list, iterate
            for i, figure in enumerate(output[1]):
                save_figure(
                    f"{dataset}_{metric}_level{i+1}",
                    prov_record, # level 1 different record?
                    cfg,
                    figure=figure,
                    dpi=300,
                )
        # clear value,fig
        output = None


def get_provenance_record(metric, ancestor_files):
    """Create a provenance record describing the diagnostic data and plot."""
    caption = {
        "iod_seasonality": (
            "Standard deviation of the DMI, illustrating the seasonal variability "
            + "and timing of SSTA."
        ),
        "iod_amplitude": (
            "Standard deviation of the Dipole Mode Index, representing the "
            + "variability of the Indian Ocean Dipole."
        ),
        "iod_lifecycle": (
            "spatial-temporal structure of SSTA in the equatorial Indian Ocean "
            + "(10°S-10°N average)."
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
            # "gillett_zoe",
        ],
        "references": [
            "planton2021",
        ],
        "ancestors": ancestor_files,
    }
    return record


def main(cfg):
    """Run ENSO metrics."""
    input_data = cfg["input_data"].values()

    # iterate through each metric and get variable group, select_metadata
    metrics = {
        "iod_lifecycle": ["iod_west","iod_east", "meridional"],
        "iod_amplitude": ["iod_west","iod_east", "meridional","trop"],
        "iod_seasonality": ["iod_west","iod_east","mer_climstd"],
    }

    # select twice with project to get obs, iterate through model selection
    for metric, var_preproc in metrics.items():
        logger.info("%s,%s", metric, var_preproc)
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

        group_obs_models(obs, models, metric, var_preproc, cfg)

    # write provenance for csv metrics
    metricfile = get_diagnostic_filename("matrix", cfg, extension="csv")
    prov = get_provenance_record("values", list(cfg["input_data"].keys()))
    with ProvenanceLogger(cfg) as provenance_logger:
        provenance_logger.log(metricfile, prov)


if __name__ == "__main__":
    with run_diagnostic() as config:
        main(config)
