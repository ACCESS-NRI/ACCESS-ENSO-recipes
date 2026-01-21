"""diagnostic script to plot"""

import iris
import os
import logging
import numpy as np
import iris.plot as iplt
import matplotlib.pyplot as plt

import cmocean
import cartopy.crs as ccrs
import cartopy.feature as feature
import matplotlib.path as mpath
from esmvaltool.diag_scripts.shared import (
    run_diagnostic,
    save_figure,
    group_metadata,
)
from esmvalcore.preprocessor import (
    add_supplementary_variables,
    axis_statistics,
    regrid,
)


# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))

def kineticenergy(u_vel, v_vel, thkcel):
    """Calculate kinetic energy from u and v components."""
    # calculate KE = 0.5*(u^2 + v^2)
    KE = 0.5*(u_vel**2 + v_vel**2)
    KE = add_supplementary_variables(KE, [thkcel])
    KE = KE.collapsed('depth', iris.analysis.SUM, weights='cell_thickness')
    KE = axis_statistics(KE, operator='mean', axis='t')
    KE = regrid(KE, '0.5x0.5', scheme='linear')  # regrid to map
    return KE


def circumpolar_map():
    # fig = plt.figure(figsize = (12, 8))
    ax = plt.axes(projection = ccrs.SouthPolarStereo())
    ax.set_extent([-180, 180, -80, -50], crs = ccrs.PlateCarree())
    ax.set_facecolor('lightgrey')
    # Map the plot boundaries to a circle
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform = ax.transAxes)

    return fig, ax


# fig, axs = circumpolar_map()

def subplot_ke(ke_cube, axs, title):
    """Plot KE on circumpolar map."""

    iplt.contourf(ke_cube, levels=np.arange(0,26,0.5), extend='max', axes=axs, cmap = cmocean.cm.ice)
    axs.set_title(title)
    plt.colorbar(label='m$^3$ s$^{-2}$',ticks=np.arange(0,26,5), shrink=0.6)


def get_provenance_record(caption, ancestor_files):
    """Create a provenance record describing the diagnostic data and plot."""

    record = {
        "caption": caption,
        "statistics": ["other"],
        "domains": ["shpolar"],
        "plot_types": ["vert"],
        "authors": [
            "chun_felicity",
        ],
        "references": [
            "access-nri",
        ],
        "ancestors": ancestor_files,
    }
    return record


def main(cfg):
    """Create Hovmoller diagrams."""

    input_data = cfg["input_data"].values()

    # group by dataset
    ds_groups = group_metadata(
        input_data,
        "dataset",
    )

    levels_tempsal = {'thetao':np.arange(-0.3, 0.31, 0.01), 'so':np.arange(-0.03, 0.031, 0.001)}
    cmapcm = {'thetao':cmocean.cm.balance, 'so':cmocean.cm.curl}
    axi = {'thetao':0, 'so':1}
    # for each ds plt ts, sal 
    for grp, var_attr in ds_groups.items():
        fig, ax = plot_hovmoller(fsize = 14)
        files=[]
        for metadata in var_attr:  #[thetao, so]
            name = metadata["dataset"]
            shortname = metadata["short_name"]
            input_file = metadata['filename']
            cube = iris.load_cube(metadata['filename'])
            hov_cube = hovmoller(cube, metadata['start_year'])

            # plot TKE, MKE, EKE (3subplts)
            for i in [axi[shortname], axi[shortname]+2]:
                p1 = iplt.contourf(hov_cube,
                            levels = levels_tempsal[shortname],
                            extend = 'both',
                            cmap = cmapcm[shortname],
                            axes = ax[i], #0,2 temp
                            )
            files.append(input_file)

            if shortname == 'so':
                cf_salt = p1
            else:
                cf_temp = p1
        
        plot_details(ax, cf_temp, cf_salt)
        # Save output
        prov_record = get_provenance_record(
            f'Depth-Time Temperature and Salinity from {name}.',
            files,
        )
        save_figure(
            f"hovmoller_{name}",
            prov_record,
            cfg,
            figure=fig,
            dpi=300,
            )


if __name__ == "__main__":
    with run_diagnostic() as config:
        main(config)
