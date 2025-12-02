"""diagnostic script to plot ENSO metrics matrix

"""

import matplotlib.pyplot as plt
import iris
import iris.plot as iplt

import os
import logging

import pandas as pd
import numpy as np
from esmvaltool.diag_scripts.shared import (run_diagnostic, 
                                            save_figure)

# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))

def collect_workfiles(diag_path):
    """Collect relevant work files from diagnostic path."""
    pattern, lifecycle = [], []
    for file in os.listdir(diag_path):
        # ends with 09pattern.nc & 10lifecycle.nc
        if file.endswith('09pattern.nc'):
            filepath = os.path.join(diag_path, file)
            if file.split('_')[0] == 'HadISST': #obs dataset define
                pattern.insert(0, filepath)  #obs first
            else:
                pattern.append(filepath)
        elif file.endswith('10lifecycle.nc'):
            filepath = os.path.join(diag_path, file)
            if file.split('_')[0] == 'HadISST': #obs dataset define
                lifecycle.insert(0, filepath)  #obs first
            else:
                lifecycle.append(filepath)
    return pattern, lifecycle


def plot_lines(input_data, y_label, title):
    """Create plots for output data."""
    figure = plt.figure(figsize=(10, 6), dpi=300)

    for mod_pth in input_data[1:]: #skip obs
        mod_name = mod_pth.split('/')[-1].split('_')[0]
        md_cb = iris.load_cube(mod_pth)
        iplt.plot(md_cb, label=mod_name)

    #obs 
    obs_cb = iris.load_cube(input_data[0])
    iplt.plot(obs_cb, label=f"ref: HadISST", color="black")

    plt.legend()
    plt.title(title)  # metric name
    plt.grid(linestyle="--")
    plt.ylabel(y_label)

    if title == "ENSO pattern":
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_lon))
    elif title == "ENSO lifecycle":
        plt.axhline(y=0, color="black", linewidth=2)
        xticks = np.arange(1, 73, 6) - 36  # Adjust for lead/lag months
        xtick_labels = ["Jan", "Jul"] * (len(xticks) // 2)
        plt.xticks(xticks, xtick_labels)
        plt.yticks(np.arange(-2, 2.5, step=1))

    return figure


def format_lon(x_val, _):
    """Format longitude in plot axis."""
    if x_val > 180:
        return f"{(360 - x_val):.0f}°W"
    if x_val == 180:
        return f"{x_val:.0f}°"

    return f"{x_val:.0f}°E"


def main(cfg):
    """Read metrics and plot matrix."""
    provenance_record = {
        'caption': "ENSO metrics",
        'authors': [
            'chun_felicity',
        ],
        'references': [''],
        'ancestors': cfg['diag_metrics']  #
    }
    # input_data = cfg['input_data'].values() 
    metrics = cfg['diag_metrics']
    diag_path = '/'.join(cfg['work_dir'].split('/')[:-2])
    diag_path = '/'.join([diag_path, metrics])
    logger.info(cfg)
    
    pattern, lifecycle = collect_workfiles(diag_path)
    figure = plot_lines(pattern, y_label="reg(ENSO SSTA, SSTA)", title="ENSO pattern")
    save_figure('ENSO_pattern', provenance_record, cfg, figure=figure, bbox_inches='tight')

    figure = plot_lines(lifecycle, y_label="Degree C / C", title="ENSO lifecycle")
    save_figure('ENSO_lifecycle', provenance_record, cfg, figure=figure, bbox_inches='tight')

if __name__ == '__main__':

    with run_diagnostic() as config:
        main(config)
