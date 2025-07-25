"""diagnostic script to plot ENSO metrics

"""

import matplotlib.pyplot as plt
import iris.quickplot as qplt
import iris.plot as iplt

import iris
import os
import logging
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from esmvaltool.diag_scripts.shared import (run_diagnostic, 
                                            save_figure, 
                                            get_diagnostic_filename,
                                            group_metadata,
                                            select_metadata,
                                            )
from esmvalcore.preprocessor import (extract_month,
                                     mask_above_threshold,
                                     mask_below_threshold,
                                     )


# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))

def mask_to_years(events):    # build time with mask
    maskedTime = np.ma.masked_array(events.coord('time').points, mask=events.data.mask)
    # return years
    return [events.coord('time').units.num2date(time).year for time in maskedTime.compressed()]


def enso_events_lc(cube): # get cube years min/max, remove 3:-3
    datayears = [cube.coord('time').units.num2date(time).year for time in cube.coord('time').points]
    leadlagyrs = datayears[:3] + datayears[-3:]
    
    cb_std = cube.data.std()
    a_events = mask_to_years(mask_above_threshold(cube.copy(), -0.5*cb_std))
    o_events = mask_to_years(mask_below_threshold(cube.copy(), 0.5*cb_std))
    events = {'la nina':a_events, 'el nino':o_events} 
    for key,yrls in events.items():
        events[key] = [yr for yr in yrls if yr not in leadlagyrs]
        
    return events

def enso_composite(n34):
    n34_dec = extract_month(n34, 12)
    events = enso_events(n34_dec) #check years not in first/last 3
    # print(events)
    enso_res = {}
    for enso, years in events.items(): 
        year_contraint = iris.Constraint(time=lambda cell: cell.point.year in years)
        # model_cb = model_n34_dec.extract(year_contraint) #values for regression
        
        years_of_interest=[]
        for yr in years:
            years_of_interest.append([yr - 2, yr - 1, yr, yr + 1, yr + 2, yr + 3])
        
        cube_data={}
        for enso_epoch in years_of_interest:
            year_enso = iris.Constraint(time=lambda cell: cell.point.year in enso_epoch)
            cube_2 = n34.extract(year_enso) #extract rolling 6
            yr = enso_epoch[2]
            cube_data[yr] = cube_2.data.data

        durations = [threshold_duration(line, 0.5, enso) for yr, line in cube_data.items()]
        enso_res[enso] = durations #ls of durations for each event
        
    return enso_res


def threshold_duration(line, value, enso):
    '''Count duration for each dataset and enso composite.'''
    if enso == 'el nino':
        cnt_month = line > value
    elif enso == 'la nina':
        cnt_month = line < - value
    cnt = 0
    durations = []
    for a in cnt_month:
        if a:
            cnt += 1
        else:
            if cnt != 0:
                durations.append(cnt)
            cnt = 0
    return max(durations)

def duration_composite_plot(obs_model, dt_ls):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7), sharey=True)

    for ninanino, ax in zip(['la nina', 'el nino'], [ax1, ax2]):
        ax.boxplot([obs_model[0][ninanino], obs_model[1][ninanino]], labels=dt_ls,
                   vert=False, patch_artist=True, notch=True,
                   boxprops=dict(facecolor='lightblue', color='blue'),
                   whiskerprops=dict(color='blue'),
                   capprops=dict(color='blue'),
                   medianprops=dict(color='red'))
        ax.set_title(f'{ninanino.title()} duration')
        ax1.set_ylabel(f'SSTA{symbol[ninanino]}0.5 (months)')
        ax.grid(linestyle='--', axis='y')
    
    return fig

def compute_enso_metrics(input_pair, dt_ls, var_group, metric): 

    # input_pair: obs first
    if metric =='14duration':
        #level 2
        mod = enso_composite(input_pair[1][var_group[0]])
        obs = enso_composite(input_pair[0][var_group[0]])

        fig3 = duration_composite_plot([obs, mod], dt_ls)


    return fig3

def format_longitude(x, pos):
    if x > 180:
        return f'{int(360 - x)}°W'
    elif x == 180:
        return f'{int(x)}°'
    else:
        return f'{int(x)}°E'


def get_provenance_record(caption, ancestor_files):
    """Create a provenance record describing the diagnostic data and plot."""

    record = {
        'caption': caption,
        'statistics': ['anomaly'],
        'domains': ['eq'],
        'plot_types': ['line'],
        'authors': [
            'chun_felicity',
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
    metrics = {'14diversity':['tos_lifdur1'],
                }
    
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
        obs_files = [ds['filename'] for ds in obs] #and models separate?

        # obs datasets for each model
        obs_datasets = {dataset['variable_group']: iris.load_cube(dataset['filename']) for dataset in obs}
        
        # group models by dataset
        model_ds = group_metadata(models, 'dataset', sort='project')        
        # dataset name
        
        for dataset in model_ds:
            logger.info(f"{metric}, preprocessed cubes:{len(model_ds)}, dataset:{dataset}")
            
            model_datasets = {attributes['variable_group']: iris.load_cube(attributes['filename']) 
                              for attributes in model_ds[dataset]}
            input_pair = [obs_datasets, model_datasets]

            # compute metric, get figure
            figs = compute_enso_metrics(input_pair, [dataset, obs[0]['dataset']], var_preproc, metric)
            
            dt_files = obs_files + [ds['filename'] for ds in models]

            # for i, fig in enumerate(figs):

            prov_record = get_provenance_record(f'ENSO metrics {metric} level {i+2}', dt_files)

            save_figure(f'{dataset}_{metric}_level_3', prov_record, cfg, figure=figs, dpi=300)


if __name__ == '__main__':

    with run_diagnostic() as config:
        main(config)
