"""diagnostic script to plot ENSO metrics

"""

import matplotlib.pyplot as plt
import iris.quickplot as qplt

import iris
import os
import logging
from pprint import pformat
import numpy as np

from esmvaltool.diag_scripts.shared import (run_diagnostic, 
                                            save_figure, 
                                            get_diagnostic_filename,
                                            group_metadata,
                                            select_metadata)
from esmvalcore.preprocessor import (convert_units, 
                                     zonal_statistics, 
                                     meridional_statistics,
                                     multi_model_statistics)


# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))

def plot_level1(obs_data, input_data, cfg):
    plt.clf()
    figure = plt.figure(figsize=(10, 6), dpi=300)
    var_units = {'tos': 'degC', 'pr': 'mm/day', 'tauu': '1e-3 N/m2', 'ts':'degC'}
    cbls5, cbls6 = [], []
    for dataset in input_data:  
        # Load the data
        fp, sn, dt, proj= (dataset['filename'], dataset['short_name'], dataset['dataset'], dataset['project'])
  
        logger.info(f"dataset: {dt} - {dataset['long_name']}")
    
        cube = iris.load_cube(fp) 
        #convert units for different variables 
        cube = convert_units(cube, units=var_units[sn])
        # func for sea_cycle, 
        title = f"Mean {dataset['long_name']}"
        if len(cube.coords('month_number')) == 1:
            cube = sea_cycle_month_stdev(cube, dataset['preprocessor'])
            #plot title 
            title = f"{dataset['long_name']} seasonal"
        # model_data = cube.data
        if proj == 'CMIP5':
            qplt.plot(cube, color='lightcoral', alpha=0.5, linewidth=0.5)
            #collect cubes for mean
            cbls5.append(cube)
            if dt == 'MultiModelMean':
                qplt.plot(cube, color='red', linewidth=3, label='CMIP5 MMM')
        elif proj == 'CMIP6':
            qplt.plot(cube, color='dodgerblue', alpha=0.5, linewidth=0.5)
            cbls6.append(cube)
            if dt == 'MultiModelMean':
                qplt.plot(cube, color='blue', linewidth=3, label='CMIP6 MMM')

        filename = [dataset['variable_group']]
    
    # multimodel mean for seasonal cycle - 
    if filename[0] == 'eq_sst_seacycle':
        mmm5 = multi_model_statistics(cbls5, span='full', statistics=['mean'], ignore_scalar_coords=True)['mean']
        qplt.plot(mmm5, color='red', linewidth=3, label='CMIP5 MMM')
        mmm6 = multi_model_statistics(cbls6, span='full', statistics=['mean'], ignore_scalar_coords=True)['mean']
        qplt.plot(mmm6,  color='blue', linewidth=3, label='CMIP6 MMM')

    fp, sn, dt= (obs_data['filename'], obs_data['short_name'], obs_data['dataset'])
    cube = iris.load_cube(fp)
    cube = convert_units(cube, units=var_units[sn])
    if len(cube.coords('month_number')) == 1:
        cube = sea_cycle_month_stdev(cube, dataset['preprocessor'])
    qplt.plot(cube, label=f'ref: {dt}', color='black', linewidth=4)
    # obs_data = cube.data

    # rmse = np.sqrt(np.mean((obs_data - model_data) ** 2))
    # metricfile = get_diagnostic_filename('matrix', cfg, extension='csv')
    # with open(metricfile, 'a+') as f:
    #     f.write(f"{filename[0]},{filename[1]},{rmse}\n")

    plt.title(title) #
    plt.legend()
    plt.grid(linestyle='--')
    plt.ylabel(f"{sn.upper()} ({cube.units})")
    
    #if latitude, zonal, long meridional
    if dataset['preprocessor'].startswith('ITCZ'):
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_latitude))
    else:
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_longitude))

    return figure, filename

def sea_cycle_month_stdev(cube, preproc):

    cube.coord('month_number').guess_bounds()
    cube = cube.collapsed('month_number', iris.analysis.STD_DEV)
    #ITCZ or zonal?
    if preproc.startswith('ITCZ'):
        cube = zonal_statistics(cube, 'mean')
    else:
        cube = meridional_statistics(cube, 'mean')

    return cube

def format_latitude(x, _pos):
    if x < 0:
        return f'{int(abs(x))}°S'
    elif x > 0:
        return f'{int(x)}°N'
    else:
        return '0°'

def format_longitude(x, _pos):
    if x > 180:
        return f'{int(360 - x)}°W'
    elif x == 180:
        return f'{int(x)}°'
    else:
        return f'{int(x)}°E'

def main(cfg):
    """Compute sea ice area for each input dataset."""
    provenance_record = {
        'caption': "ENSO metrics",
        'authors': [
            'chun_felicity',
        ],
        'references': [''],
        'ancestors': list(cfg['input_data'].keys()),
    }
    input_data = cfg['input_data'].values() 
    
    # group by variable groups
    variable_groups = group_metadata(input_data, 'variable_group', sort='project')

    for grp in variable_groups:
        msg = "{} : {}, {}".format(grp, len(variable_groups[grp]), pformat(variable_groups[grp]))
        logger.info(msg) 
        obs_data = variable_groups[grp][-1]
        model_data = variable_groups[grp][:-1]

        fig, filename = plot_level1(obs_data, model_data, cfg)

        save_figure('_'.join(filename), provenance_record, cfg, figure=fig, dpi=300)

if __name__ == '__main__':

    with run_diagnostic() as config:
        main(config)
