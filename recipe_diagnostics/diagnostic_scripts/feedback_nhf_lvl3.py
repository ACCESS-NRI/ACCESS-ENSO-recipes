"""diagnostic script to plot ENSO feedback metrics"""

import os
import logging
import iris
import numpy as np
import pandas as pd
import iris.quickplot as qplt
import matplotlib.pyplot as plt

from esmvaltool.diag_scripts.shared import (run_diagnostic, 
                                            save_figure, 
                                            get_diagnostic_filename,
                                            group_metadata,
                                            select_metadata,
                                            )
from esmvalcore.preprocessor import (mask_above_threshold, 
                                     mask_below_threshold,
                                     rolling_window_statistics,
                                     convert_units,
                                     )


# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))



def lin_regress_matrix(cubeA, cubeBsst):
    
    A_data = cubeA.data.reshape(cubeA.shape[0], -1)  # Shape (time, spatial_points)
    if cubeA.shape[0] == cubeBsst.shape[0]:
        B_data = cubeBsst.data.flatten() # or all
    else:
        B_data = cubeBsst.data.compressed() # masked threshold cube (time,) 

    # Add intercept term by stacking a column of ones with cubeB
    B_with_intercept = np.vstack([B_data, np.ones_like(B_data)]).T
    
    logger.info(f'least squares data shapes {B_with_intercept.shape}, {A_data.shape}')
    # Solve the linear equations using least squares method
    coefs, _, _, _ = np.linalg.lstsq(B_with_intercept, A_data, rcond=None)

    # Create a new Iris Cube for the regression results
    result_cube = iris.cube.Cube(coefs[0], long_name='regression A',
                                 dim_coords_and_dims=[(cubeA.coord('longitude'), 0)])

    return result_cube


def feedback_nonlin(sst_cube, tauu_cube):
    tauu_aux = tauu_cube.copy()
    sst_coord = iris.coords.AuxCoord(sst_cube.data, sst_cube.standard_name, sst_cube.long_name, sst_cube.var_name, sst_cube.units)
    tauu_aux.add_aux_coord(sst_coord, 0)
    logger.info(f'non linear shapes {sst_cube.shape}, {tauu_cube.shape}')
    logger.info(tauu_aux.summary())
    below0 = iris.Constraint(coord_values={sst_cube.standard_name:lambda cell: cell < 0})
    above0 = iris.Constraint(coord_values={sst_cube.standard_name:lambda cell: cell > 0})
    ssta_neg = mask_above_threshold(sst_cube.copy(), 0) #x<0
    ssta_pos = mask_below_threshold(sst_cube.copy(), 0) #x=>0
    xbelow0 = tauu_aux.extract(below0)
    xabove0 = tauu_aux.extract(above0)

    ky, cnts = np.unique(ssta_pos.data.mask, return_counts=True)
    msk_pos = dict(zip(ky.tolist(), cnts.tolist()))
    ky, cnts = np.unique(ssta_neg.data.mask, return_counts=True)
    msk_neg = dict(zip(ky.tolist(), cnts.tolist()))
    msk_dt = {'pos': msk_pos[False], 'neg':msk_neg[False]}
    logger.info(f'pos: {msk_pos[False]} cube: {xabove0.shape}, neg: {msk_neg[False]} cube: {xbelow0.shape}')
    outreg_cube = lin_regress_matrix(xbelow0, ssta_neg)
    posreg_cube = lin_regress_matrix(xabove0, ssta_pos)

    return outreg_cube, posreg_cube, msk_dt

def obs_extract_overlap(obs_1, obs_2):
    """Extract overlapping time range from two observation datasets."""
    start_1 = obs_1.coord('time').cell(0).point #
    end_1 = obs_1.coord('time').cell(-1).point
    start_2 = obs_2.coord('time').cell(0).point
    end_2 = obs_2.coord('time').cell(-1).point

    start_overlap = max(start_1, start_2)
    end_overlap = min(end_1, end_2)
    # convert to yymmdd? use extract time, num2date
    logger.info(f'{obs_1.standard_name}, {obs_2.standard_name} obs time overlap: {start_overlap} to {end_overlap}')
    obs1 = obs_1.extract(iris.Constraint(time=lambda t: start_overlap <= t.point <= end_overlap))
    obs2 = obs_2.extract(iris.Constraint(time=lambda t: start_overlap <= t.point <= end_overlap))

    return obs1, obs2

def format_longitude(x, pos):
    if x > 180:
        return f'{int(360 - x)}°W'
    elif x == 180:
        return f'{int(x)}°'
    else:
        return f'{int(x)}°E'
    
def plot_level3(obs_ds, model_ds, metric_varls, ds_labels, title): #edit for nhf
    """Plot level 3 diagnostics for ENSO feedback metrics."""
    figure = plt.figure(figsize=(10, 6), dpi=300)
    tau_modcube = rolling_window_statistics(model_ds[metric_varls[1]], 
                                     coordinate='longitude',operator='mean',window_length=30)
    tau_obcube = rolling_window_statistics(obs_ds[metric_varls[2]], # check index
                                     coordinate='longitude',operator='mean',window_length=30)
    # plot whole regression
    cb = lin_regress_matrix(tau_modcube, model_ds[metric_varls[0]])            
    qplt.plot(cb, color='black', linestyle='solid', label=ds_labels[1])

    # obs datasets can have different time range..
    obs1, obs2 = obs_extract_overlap(tau_obcube, obs_ds[metric_varls[0]])
    # obs_ds_label = f"{input_data['obs'][0]['dataset']}_{input_data['obs'][1]['dataset']}"
    cb2 = lin_regress_matrix(obs1, obs2)
    qplt.plot(cb2, color='black', linestyle='--', label=ds_labels[0])
    # process model data split
    neg, pos, cnts = feedback_nonlin(model_ds[metric_varls[0]], tau_modcube)
    
    xvar = metric_varls[0].split('_')[0] # ts, tauu, ssh
    yvar = metric_varls[2].split('_')[0]

    qplt.plot(neg, color='blue', linestyle='solid', label=f"{xvar.upper()}A<0")
    qplt.plot(pos, color='red', linestyle='solid', label=f"{xvar.upper()}A>0")
    # process obs data split
    neg, pos, cnts = feedback_nonlin(obs2, obs1)
    qplt.plot(neg, color='blue', linestyle='--')
    qplt.plot(pos, color='red', linestyle='--')

    plt.xlim(170, 250)
    plt.xlabel('longitude')
    plt.ylabel(f'reg({xvar.upper()}A, {yvar.upper()}A)') #
    plt.grid(linestyle='--')
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_longitude))
    plt.legend()
    plt.title(title)
    return figure

def get_provenance_record(caption, ancestor_files):
    """Create a provenance record describing the diagnostic data and plot."""

    record = {
        'caption': caption,
        'statistics': ['anomaly'],
        'domains': ['eq'],
        'plot_types': ['line'],
        'authors': [
            'chun_felicity',
            # 'beucher_romain',
            # 'sullivan_arnold',
        ],
        'references': [
            'access-nri',
        ],
        'ancestors': ancestor_files,
    }
    return record

def main(cfg):
    """Run ENSO feedback metrics."""

    input_data = cfg['input_data'].values() 

    SST_NHF = ['sst_eqp', 'nhf_eqp_mod', 'nhf_eqp_obs']
    metric = 'sst_nhf'
    # iterate and group just the variable groups? obs-[], mod[]
    obs, models = [], []

    obs += select_metadata(input_data, project='OBS')
    obs += select_metadata(input_data, project='OBS6')
    models += select_metadata(input_data, project='CMIP6')

    # log
    msg = "{} : observation datasets {}, models {}".format(metric, len(obs), len(models))
    logger.info(msg)

    # group models by dataset
    model_ds = group_metadata(models, 'dataset', sort='project')        
    
    # dataset name
    for dataset, mod_ds in model_ds.items():
        logger.info(f"{metric}, preprocessed cubes:{len(model_ds)}, dataset:{dataset}")
        dt_files = [ds['filename'] 
                    for ds in obs] + [ds['filename'] 
                                        for ds in model_ds[dataset]]

        
        obs_ds = {dataset['variable_group']: iris.load_cube(dataset['filename']) for dataset in obs}
        model = {attributes['variable_group']: iris.load_cube(attributes['filename']) 
                                for attributes in mod_ds}
        model['nhf_eqp_mod'] = - model['nhf_eqp_mod'] # make negative
        
        ds_labels = [f"{obs[0]['dataset']}_{obs[1]['dataset']}", mod_ds[0]['dataset']]
        title = f"net heat flux feedback"
        # plot level 3
        prov_record = get_provenance_record(f'ENSO metrics {metric} feedback level3', dt_files)
        fig = plot_level3(obs_ds, model, SST_NHF, ds_labels, title)
        save_figure(f'{dataset}_{metric}_lvl3', prov_record, cfg, figure=fig, dpi=300)#


if __name__ == '__main__':

    with run_diagnostic() as config:
        main(config)
