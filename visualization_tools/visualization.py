import accessvis
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import os
import pandas as pd
import os
import xarray as xr
import lavavu

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import cmocean
from skimage.transform import resize

from tqdm import tqdm
from scipy.ndimage import gaussian_filter

class Visualization():
    """
    A class to generate, process, and render visualizations of geospatial data using contour plots and Lavavu.

    This class supports generating RGBA images from geospatial data, applying resizing and padding,
    handling opacity layers, and displaying results in Lavavu either as a contour field or smoothed gradient.

    Attributes:
        data (iris.cube.Cube): Input geospatial data (2D).
        size (tuple): Desired image size as (width, height).
        lat_range (tuple): Latitude range to display (min_lat, max_lat).
        lon_range (tuple): Longitude range to display (min_lon, max_lon).
        title (str): Title of the visualization.
        colormap (str or callable): Colormap used for the visualization.
        tickvalues (list): Tick values to show on colorbar.
        vmax (float): Maximum value for normalization.
        vmin (float): Minimum value for normalization.
        resolution (tuple, optional): Output resolution of the Lavavu window. Default is (700, 700).

    Methods:
        generate_rgba(): Generate RGBA image from the input data using matplotlib.
        resize_rgba(rgba, width, height): Resize an RGBA image to given dimensions.
        pad_rgba(data, pad_width, pad_height, pad_depth=None, constant_values=255): Pad an RGBA or data array.
        opacity_rgba(padded_array, opacity_array): Apply opacity to the alpha channel of an RGBA image.
        generate_lv(): Set up and configure the Lavavu Earth visualization.
        calculate_pad(): Compute the padding sizes based on latitude and longitude ranges.
        visualise_contourf(window=False): Generate and display contour visualization in Lavavu.
        visualise_gradient(window=False): Generate and display a smoothed gradient visualization in Lavavu.
    """

    
    def __init__(self, data, size, lat_range, lon_range, title, colormap, tickvalues, vmax, vmin, resolution=(700, 700), region=None):
        self.data = data
        self.size = size
        self.width = size[0]
        self.height = size[1]
        self.title = title
        self.colormap = colormap
        self.tickvalues = tickvalues
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.resolution =  resolution
        self.vmax = vmax
        self.vmin = vmin
        self.region = region
        

    def generate_rgba(self):
        data = self.data.data  # 2D numpy array
        lon = self.data.coord('longitude').points
        lat = self.data.coord('latitude').points
        
        lon2d, lat2d = np.meshgrid(lon, lat)
        
        cmap = cmocean.cm.balance
        norm = mcolors.Normalize(vmin=self.vmin, vmax=self.vmax)
        
        fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
        cf = ax.contourf(lon2d, lat2d, data, cmap=cmap, norm=norm, levels=20)
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
        fig.canvas.draw()
        rgba = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        rgba = rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # (H, W, 4)
    
        rgba = rgba[:, :, [1, 2, 3, 0]]
        plt.close(fig)
        
        return rgba


    def resize_rgba(self, rgba, width, height):
        #If the original image is of type uint8, it needs to be converted to float32 before resizing.
        rgba_float = rgba.astype(np.float32) / 255.0
    
        # Resize the image (while preserving the number of channels).
        rgba_resized = resize(rgba_float, (width, height, 4), preserve_range=True, anti_aliasing=True)
        
        # change back to uint8
        rgba_resized = np.clip(rgba_resized * 255, 0, 255).astype(np.uint8)
    
        return rgba_resized


    def normalise_array(self, values, minimum=None, maximum=None):
        """
        Normalize an array to the range [0,1]
    
        Parameters
        ----------
        values : numpy.ndarray
            Values to convert, numpy array
        minimum: number
            Use a fixed minimum bound, default is to use the data minimum
        maximum: number
            Use a fixed maximum bound, default is to use the data maximum
        """
    
        # Ignore nan when getting min/max
        if not minimum:
            minimum = np.nanmin(values)
        if not maximum:
            maximum = np.nanmax(values)
    
        # Normalise
        array = (values - minimum) / (maximum - minimum)
        # Clip out of [0,1] range - in case defined range is not the global minima/maxima
        array = np.clip(array, 0, 1)
    
        return array

        
    def pad_rgba(self, data, pad_width, pad_height, pad_depth=None, constant_values=255):
        if pad_depth:
            padded_rgba = np.pad(
                        data,
                        pad_width=(pad_width, pad_height, pad_depth),
                        mode='constant',
                        constant_values=constant_values 
            )
        else:
             padded_rgba = np.pad(
                        data,
                        pad_width=(pad_width, pad_height),  
                        mode='constant',
                        constant_values=constant_values  
            )
            
        return padded_rgba


    def rotate_region(self, lv):
        cam_africa={
            'translate': [0.0, 0.0, -22.071718],
            'rotate': [0.015515, -0.213022, 0.000855, 0.976923],
            'xyzrotate': [1.887586, -24.597353, -0.311234],
            'fov': 45.0,
            'focus': [-0.014273, -0.037801, 0.001495]
        }

        cam_euro={
            'translate': [0.0, 0.0, -22.071718],
            'rotate': [0.355491, -0.340382, -0.278646, 0.824683],
            'xyzrotate': [56.403061, -21.303127, -48.855858],
            'fov': 45.0,
             'focus': [-0.014273, -0.037801, 0.001495]
        }

        cam_aisa={
            'translate': [0.0, 0.0, -22.071718],
            'rotate': [0.102419, -0.760827, -0.207076, 0.606436],
            'xyzrotate': [112.134262, -61.686878, -120.888702],
            'fov': 45.0,
            'focus': [-0.014273, -0.037801, 0.001495]
        }

        cam_aust={
            'translate': [0.0, 0.0, -22.071718],
            'rotate': [0.003478, -0.91839, 0.218037, 0.330141],
            'xyzrotate': [-149.899796, -37.438725, 170.021637],
            'fov': 45.0,
            'focus': [-0.014273, -0.037801, 0.001495]
        }

        cam_north_am={
            'translate': [0.0, 0.0, -22.071718],
            'rotate': [-0.165649, -0.746298, -0.326733, -0.5557],
            'xyzrotate': [104.104782, 46.152786, 118.210793],
            'fov': 45.0,
            'focus': [-0.014273, -0.037801, 0.001495]
        }

        cam_south_am={
            'translate': [0.0, 0.0, -22.071718],
            'rotate': [0.130201, -0.517479, 0.102816, -0.839431],
            'xyzrotate': [-37.048805, 57.352139, -34.736542],
            'fov': 45.0,
            'focus': [-0.014273, -0.037801, 0.001495]
        }

        cam_anta={
            'translate': [0.0, 0.0, -22.071718],
            'rotate': [-0.207506, -0.678652, 0.653672, 0.262741],
            'xyzrotate': [-90.41713, -4.895439, 141.134872],
            'fov': 45.0,
            'focus': [-0.014273, -0.037801, 0.001495]
        }

        cam_arct={
            'translate': [0.0, 0.0, -22.032022],
            'rotate': [0.248796, -0.694516, -0.63765, 0.221573],
            'xyzrotate': [95.078087, 0.545349, -141.07901],
            'fov': 45.0,
            'focus': [-0.014273, -0.037801, 0.001495]
        }

        cam_paci={
            'translate': [0.0, 0.0, -22.032022],
            'rotate': [0.036096, -0.990832, 0.024548, -0.127577],
            'xyzrotate': [-176.572876, 14.539167, -175.389648],
            'fov': 45.0,
            'focus': [-0.014273, -0.037801, 0.001495]
        }

        cam_indi={
            'translate': [0.0, 0.0, -22.032022],
            'rotate': [-0.102844, -0.613056, 0.101087, 0.77672],
            'xyzrotate': [-51.315109, -68.678268, 51.170242],
            'fov': 45.0,
            'focus': [-0.014273, -0.037801, 0.001495]
        }

        cam_atla={
            'translate': [0.0, 0.0, -22.071718],
            'rotate': [-0.004712, 0.202786, -0.002818, 0.979204],
            'xyzrotate': [-0.647486, 23.397743, -0.463794],
            'fov': 45.0,
            'focus': [-0.014273, -0.037801, 0.001495]
        }

        match self.region:
            case 'Africa'|'afri':
                lv.camera(cam_africa)
            case 'Europe'|'euro':
                lv.camera(cam_euro)
            case 'Aisa'|'aisa':
                lv.camera(cam_aisa)
            case 'Australia'|'aust':
                lv.camera(cam_aust)
            case 'North_America'|'noam':
                lv.camera(cam_north_am)
            case 'South_America'|'soam':
                lv.camera(cam_south_am)
            case 'Antarctica'|'anta':
                lv.camera(cam_anta)
            case 'Arctic'|'arct':
                lv.camera(cam_arct)
            case 'Pacific'|'paci':
                lv.camera(cam_paci)
            case 'Atlantic'|'atla':
                lv.camera(cam_atla)
            case 'Indian'|'indi':
                lv.camera(cam_indi)

        return lv
                            

    def generate_lv(self):
        lv = accessvis.plot_earth(texture='bluemarble', background="white", vertical_exaggeration=20)
        if self.region is None:
            lv.rotation(15.0, -200.0, 0.0) #Rotate to Pacific
        else:
            lv=self.rotate_region(lv)
        lv.set_properties(diffuse=0.8, ambient=0.1, specular=0.35, shininess=0.03, light=[1,1,0.98]) # make pretty
        lv.brightness_contrast_saturation(0.5, 0.5, 0.65)

        cbar=lv.colourbar(size=[0.95,15], align="bottom", tickvalues=self.tickvalues)
        cbar.colourmap(self.colormap, range=[self.vmin,self.vmax])
        lv.title(self.title)
        
        return lv


    def calculate_pad(self):
        self.pad_width = ((self.lat_range[0]-(-90))*5, (90-self.lat_range[1])*5)
        self.pad_height = (self.lon_range[0]*5, (360 - self.lon_range[1])*5)


    def visualise_contourf(self, window=False, Alpha=0.7):
        self.calculate_pad()
        rgba=self.generate_rgba()
      
        resized_rgba=self.resize_rgba(rgba,width=self.width,height=self.height)
        opacity_array = resize(self.data.data, self.size, order=1, preserve_range=True, anti_aliasing=True)
        padded_rgba=self.pad_rgba(resized_rgba,pad_width=self.pad_width,pad_height=self.pad_height,pad_depth=(0, 0), constant_values=0)
        padded_rgba[:,:,3]=padded_rgba[:,:,3]*Alpha

        lv = self.generate_lv()
        accessvis.update_earth_values(lv, dataMode=0, data=padded_rgba)
        

        if window:
            lv.window(resolution=self.resolution)
        else:
            lv.display(resolution=self.resolution)


    def visualise_gradient(self, window=False):
        lv = self.generate_lv()

        pad_width = ((self.lat_range[0]-(-90)), (90-self.lat_range[1]))
        pad_height = (self.lon_range[0], (360 - self.lon_range[1]))

        padded_data = np.pad(self.data.data, pad_width=(pad_width, pad_height), mode='constant', constant_values=0)

        smoothed = gaussian_filter(padded_data, sigma=1.0)
        colours_model = accessvis.array_to_rgba(smoothed, flip=True, colourmap=self.colormap, opacitymap=True, minimum=self.vmin, maximum=self.vmax)
        accessvis.update_earth_values(lv, dataMode=0, data=colours_model)
        if window:
            lv.window(resolution=self.resolution)
        else:
            lv.display(resolution=self.resolution)