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

    
    def __init__(self, data, size, lat_range, lon_range, title, colormap, tickvalues, vmax, vmin, resolution=(700, 700)):
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

    # def opacity_rgba(self, padded_array, opacity_array):
    #     array = self.normalise_array(opacity_array)
    #     oarray = array
    #     oarray = np.nan_to_num(oarray)
    #     oarray = (oarray * 255).astype(np.uint8)
    #     padded_array[::, ::, 3] = oarray
        
    #     return padded_array

    def generate_lv(self):
        lv = accessvis.plot_earth(texture='bluemarble', background="white", vertical_exaggeration=20)
        lv.rotation(15.0, -200.0, 0.0) #Rotate to Pacific
        lv.set_properties(diffuse=0.8, ambient=0.1, specular=0.35, shininess=0.03, light=[1,1,0.98]) # make pretty
        lv.brightness_contrast_saturation(0.5, 0.5, 0.65)

        cbar=lv.colourbar(size=[0.95,15], align="bottom", tickvalues=self.tickvalues)
        cbar.colourmap(self.colormap, range=[self.vmin,self.vmax])
        lv.title(self.title)
        
        return lv

    def calculate_pad(self):
        self.pad_width = ((self.lat_range[0]-(-90))*5, (90-self.lat_range[1])*5)
        self.pad_height = (self.lon_range[0]*5, (360 - self.lon_range[1])*5)

    # def visualise_contourf(self, window=False):
    #     self.calculate_pad()
    #     rgba=self.generate_rgba()
    #     resized_rgba=self.resize_rgba(rgba,width=self.width,height=self.height)
    #     padded_rgba=self.pad_rgba(resized_rgba,pad_width=self.pad_width,pad_height=self.pad_height,pad_depth=(0, 0))
    #     opacity_array = resize(self.data.data, self.size, order=1, preserve_range=True, anti_aliasing=True)
    #     padded_opacity_array=self.pad_rgba(opacity_array,pad_width=self.pad_width,pad_height=self.pad_height,constant_values=0)
    #     opacitied_rgba=self.opacity_rgba(padded_rgba, padded_opacity_array)

    #     lv = self.generate_lv()
    #     accessvis.update_earth_values(lv, dataMode=0, data=opacitied_rgba)

    #     if window:
    #         lv.window(resolution=self.resolution)
    #     else:
    #         lv.display(resolution=self.resolution)

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