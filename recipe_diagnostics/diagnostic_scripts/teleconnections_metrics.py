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

from esmvaltool.diag_scripts.shared import (run_diagnostic, 
                                            save_figure, 
                                            get_diagnostic_filename,
                                            group_metadata,
                                            select_metadata,
                                            )
from esmvalcore.preprocessor import (extract_season,
                                     anomalies)


# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))

def plot_level1(input_data, rmse, title): #input data is 2 - model and obs

    figure = plt.figure(figsize=(20, 7), dpi=300)

    proj = ccrs.PlateCarree(central_longitude=180)
    figure.suptitle(title)
    i =121

    for label, cube in input_data.items():
        
        ax1 = plt.subplot(i,projection=proj)
        ax1.coastlines()
        cf1 = iplt.contourf(cube, levels=np.arange(-1,1,0.1), extend='both',cmap='RdBu_r')
        ax1.set_title(label)
        gl1 = ax1.gridlines(draw_labels=True, linestyle='--')
        gl1.top_labels = False
        gl1.right_labels = False
        i+=1

    plt.text(0.1, -0.3, f'RMSE: {rmse:.2f} ', fontsize=12, ha='left',
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))    
    # Add a single colorbar at the bottom
    cax = plt.axes([0.15,0.08,0.7,0.05])
    cbar = figure.colorbar(cf1, cax=cax, orientation='horizontal', extend='both', ticks=np.arange(-1,1.5,0.5))
    cbar.set_label('regression (°C/°C)')
    logger.info(f"{title}, {label} : metric:{rmse}")
    plt.tight_layout
    
    return figure


def lin_regress_matrix(cubeA, cubeB): #array must not contain infs or NaNs
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
    logger.info("%s, %s",cubeA.coords(), cubeA.shape)
    # Extract slopes from coefficients #coefs 1
    slopes = coefs[0].reshape(cubeA.shape[1], cubeA.shape[2])

    # Create a new Iris Cube for the regression results
    result_cube = iris.cube.Cube(slopes, long_name='regression ENSO SSTA',
                                 dim_coords_and_dims=[(cubeA.coord('latitude'), 0),
                                                      (cubeA.coord('longitude'), 1)])

    return result_cube


def mask_pacific(cube):
    region = box(130.,-15.,270.,15) #remove land
    x_p, y_p = np.meshgrid(
        cube.coord(axis="X").points,
        cube.coord(axis="Y").points,
    )

    mask = shp_vect.contains(region, x_p, y_p)
    cube.data.mask = mask
    return cube

def compute_telecon_metrics(input_pair, var_group, metric):

    if metric =='pr_telecon':
        title = '{} PR Teleconnection' # both seasons
    elif metric == 'ts_telecon':
        title = '{} SST Teleconnection'

    val, fig = {}, {}
    for seas in ['DJF','JJA']:
        data_values = []
        cubes = {}
        for label, ds in input_pair.items(): #obs 0, mod 1
            preproc = {}
            for variable in var_group:
                cube = extract_season(ds[variable].copy(), seas)
                preproc[variable] = anomalies(cube, period="full")

            regcube = lin_regress_matrix(preproc[var_group[1]], preproc[var_group[0]])
            reg_masked = mask_pacific(regcube)

            data_values.append(reg_masked.data)
            cubes[label] = reg_masked

        val[seas] = np.sqrt(np.mean((data_values[0] - data_values[1]) ** 2))
        fig[seas] = plot_level1(cubes, val[seas], title.format(seas))

    return val, fig 



def compute(obs, mod):
    return abs((mod-obs)/obs)*100

def get_provenance_record(caption, ancestor_files):
    """Create a provenance record describing the diagnostic data and plot."""

    record = {
        'caption': caption,
        'statistics': ['anomaly'],
        'domains': ['eq'],
        'plot_types': ['map'],
        'authors': [
            'chun_felicity',
            'beucher_romain'
            # 'sullivan_arnold',
        ],
        'references': [
            'access-nri',
        ],
        'ancestors': ancestor_files,
    }
    return record

def main(cfg):
    """Run ENSO metrics."""

    input_data = cfg['input_data'].values() 

    # iterate through each metric and get variable group, select_metadata, map to function call
    metrics = {'pr_telecon': ['tos_enso', 'pr_global'],
                'ts_telecon':['tos_enso','tos_global']}
    
    # select twice with project to get obs, iterate through model selection
    for metric, var_preproc in metrics.items(): #if empty or try
        logger.info(f"{metric},{var_preproc}")
        obs, models = [], []
        for var_prep in var_preproc: #enumerate 1 or 2 length? if 2 append,
            obs += select_metadata(input_data, variable_group=var_prep, project='OBS')
            obs += select_metadata(input_data, variable_group=var_prep, project='OBS6')
            models += select_metadata(input_data, variable_group=var_prep, project='CMIP6')

        # log
        msg = "{} : observation datasets {}, models {}".format(metric, len(obs), len(models))
        logger.info(msg)
        
        # list dt_files
        dt_files = []
        for ds in models: #and obs?
            dt_files.append(ds['filename'])
        prov_record = get_provenance_record(f'ENSO metrics {metric}', dt_files)
        # obs datasets for each model
        obs_datasets = {dataset['variable_group']: iris.load_cube(dataset['filename']) for dataset in obs}
        
        # group models by dataset
        model_ds = group_metadata(models, 'dataset', sort='project')        
        # dataset name
        
        for dataset in model_ds:
            logger.info(f"{metric}, preprocessed cubes:{len(model_ds)}, dataset:{dataset}")
            
            model_datasets = {attributes['variable_group']: iris.load_cube(attributes['filename']) 
                              for attributes in model_ds[dataset]}
            input_pair = {obs[0]['dataset']:obs_datasets, dataset:model_datasets}
            logger.info(pformat(model_datasets))
            # process function for each metric - obs first.. if, else
            ### make one function, with the switches - same params
            values, fig = compute_telecon_metrics(input_pair, var_preproc, metric)

            # save metric for each pair, check not none teleconnection metric value djf, jja
            for seas, val in values.items():
                metricfile = get_diagnostic_filename('matrix', cfg, extension='csv')
                with open(metricfile, 'a+') as f:
                    f.write(f"{dataset},{seas}_{metric},{val}\n")

                save_figure(f'{dataset}_{seas}_{metric}', prov_record, cfg, figure=fig[seas], dpi=300)#


if __name__ == '__main__':

    with run_diagnostic() as config:
        main(config)
