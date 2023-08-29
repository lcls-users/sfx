import numpy as np
import h5py

from btx.interfaces.ipsana import (
    PsanaInterface,
    retrieve_pixel_index_map,
    assemble_image_stack_batch,
)

import holoviews as hv
hv.extension('bokeh')
from holoviews.streams import Params

import panel as pn
pn.extension(template='fast')
pn.state.template.param.update()
import panel.widgets as pnw

def display_dashboard(filename):
    """
    Displays an interactive dashboard with a PC plot, scree plot, 
    and intensity heatmaps of the selected source image as well as 
    its reconstructed image using the model obtained by pipca.
    """
    (
        exp, run, det_type,
        start_img, loadings,
        U, S, V
    ) = unpack_pipca_model_file(filename)

    psi = PsanaInterface(exp=exp, run=run, det_type=det_type)
    psi.counter = start_img

    # Create PC dictionary and widgets
    PCs = {f'PC{i}' : v for i, v in enumerate(loadings, start=1)}
    PC_options = list(PCs)

    PCx = pnw.Select(name='X-Axis', value='PC1', options=PC_options)
    PCy = pnw.Select(name='Y-Axis', value='PC2', options=PC_options)
    widgets_scatter = pn.WidgetBox(PCx, PCy, width=150)

    PC_scree = pnw.Select(name='Component Cut-off', value=f'PC{len(PCs)}', options=PC_options)
    widgets_scree = pn.WidgetBox(PC_scree, width=150)

    tap_source = None
    posxy = hv.streams.Tap(source=tap_source, x=0, y=0)

    # Create PC scatter plot
    @pn.depends(PCx.param.value, PCy.param.value)
    def create_scatter(PCx, PCy):
        img_index_arr = np.arange(start_img, start_img + len(PCs[PCx]))
        scatter_data = {**PCs, 'Image': img_index_arr}

        opts = dict(width=400, height=300, color='Image', cmap='rainbow', colorbar=True,
                    show_grid=True, shared_axes=False, toolbar='above', tools=['hover'])
        scatter = hv.Points(scatter_data, kdims=[PCx, PCy], vdims=['Image'], 
                            label="%s vs %s" % (PCx.title(), PCy.title())).opts(**opts)
            
        posxy.source = scatter
        return scatter
        
    # Create scree plot
    @pn.depends(PC_scree.param.value)
    def create_scree(PC_scree):
        q = int(PC_scree[2:])
        components = np.arange(1, q + 1)
        singular_values = S[:q]
        bars_data = np.stack((components, singular_values)).T

        opts = dict(width=400, height=300, show_grid=True, show_legend=False,
                    shared_axes=False, toolbar='above', default_tools=['hover','save','reset'],
                    xlabel='Components', ylabel='Singular Values')
        scree = hv.Bars(bars_data, label="Scree Plot").opts(**opts)

        return scree
        
    # Define function to compute heatmap based on tap location
    def tap_heatmap(x, y, pcx, pcy, pcscree):
        # Finds the index of image closest to the tap location
        img_source = closest_image_index(x, y, PCs[pcx], PCs[pcy])

        counter = psi.counter
        psi.counter = start_img + img_source
        img = psi.get_images(1)
        psi.counter = counter
        img = img.squeeze()

        # Downsample so heatmap is at most 100 x 100
        hm_data = construct_heatmap_data(img, 100)

        opts = dict(width=400, height=300, cmap='plasma', colorbar=True, shared_axes=False, toolbar='above')
        heatmap = hv.HeatMap(hm_data, label="Source Image %s" % (start_img+img_source)).aggregate(function=np.mean).opts(**opts)

        return heatmap
        
    # Define function to compute reconstructed heatmap based on tap location
    def tap_heatmap_reconstruct(x, y, pcx, pcy, pcscree):
        # Finds the index of image closest to the tap location
        img_source = closest_image_index(x, y, PCs[pcx], PCs[pcy])

        # Calculate and format reconstructed image
        p, x, y = psi.det.shape()
        pixel_index_map = retrieve_pixel_index_map(psi.det.geometry(psi.run))

        q = int(pcscree[2:])
        img = U[:, :q] @ np.diag(S[:q]) @ np.array([V[img_source][:q]]).T
        img = img.reshape((p, x, y))
        img = assemble_image_stack_batch(img, pixel_index_map)

        # Downsample so heatmap is at most 100 x 100
        hm_data = construct_heatmap_data(img, 100)

        opts = dict(width=400, height=300, cmap='plasma', colorbar=True, shared_axes=False, toolbar='above')
        heatmap_reconstruct = hv.HeatMap(hm_data, label="PiPCA Image %s" % (start_img+img_source)).aggregate(function=np.mean).opts(**opts)

        return heatmap_reconstruct
        
    # Connect the Tap stream to the heatmap callbacks
    stream1 = [posxy]
    stream2 = Params.from_params({'pcx': PCx.param.value, 'pcy': PCy.param.value, 'pcscree': PC_scree.param.value})
    tap_dmap = hv.DynamicMap(tap_heatmap, streams=stream1+stream2)
    tap_dmap_reconstruct = hv.DynamicMap(tap_heatmap_reconstruct, streams=stream1+stream2)
        
    return pn.Column(pn.Row(widgets_scatter, create_scatter, tap_dmap),
                     pn.Row(widgets_scree, create_scree, tap_dmap_reconstruct)).servable('PiPCA Dashboard')

def display_eigenimages(filename):
    """
    Displays a PC selector widget and a heatmap of the
    eigenimage corresponding to the selected PC.
    """
    (
        exp, run, det_type,
        start_img, _,
        U, _, _
    ) = unpack_pipca_model_file(filename)

    psi = PsanaInterface(exp=exp, run=run, det_type=det_type)
    psi.counter = start_img
    
    # Create eigenimage dictionary and widget
    eigenimages = {f'PC{i}' : v for i, v in enumerate(U.T, start=1)}
    PC_options = list(eigenimages)
    
    component = pnw.Select(name='Components', value='PC1', options=PC_options)
    widget_heatmap = pn.WidgetBox(component, width=150)
    
    # Define function to compute heatmap
    @pn.depends(component.param.value)
    def create_heatmap(component):
        # Reshape selected eigenimage to work with construct_heatmap_data()
        p, x, y = psi.det.shape()
        pixel_index_map = retrieve_pixel_index_map(psi.det.geometry(psi.run))
        
        img = eigenimages[component]
        img = img.reshape((p, x, y))
        img = assemble_image_stack_batch(img, pixel_index_map)
        
        # Downsample so heatmap is at most 100 x 100
        hm_data = construct_heatmap_data(img, 100)
    
        opts = dict(width=400, height=300, cmap='BrBG', colorbar=True,
                    symmetric=True, shared_axes=False, toolbar='above')
        heatmap = hv.HeatMap(hm_data, label="%s Eigenimage" % (component.title())).aggregate(function=np.mean).opts(**opts)
        
        return heatmap
    
    return pn.Row(widget_heatmap, create_heatmap).servable('PiPCA Eigenimages')

def unpack_pipca_model_file(filename):
    """
    Reads PiPCA model information from h5 file and returns its contents

    Parameters
    ----------
    filename: str
        name of h5 file you want to unpack

    Returns
    -------
    exp: str
        experiment name
    run: int
        run number
    det_type: str
        detector type
    start_img: int
        index f starting image in the run
    loadings: ndarray, shape (q x n)
        loadings matrix containing PC data
    U: ndarray, shape (d x q)
        principal components or eigenimages.
        U matrix from pipca model
    S: ndarray, shape (q,)
        singular values.
        S matrix from pipca model
    V: ndarray, shape (n x q)
        V matrix from pipca model
    """
    with h5py.File(filename, 'r') as f:
        exp = str(np.asarray(f.get('exp')))[2:-1]
        run = int(np.asarray(f.get('run')))
        det_type = str(np.asarray(f.get('det_type')))[2:-1]
        start_img = int(np.asarray(f.get('start_offset')))
        loadings = np.asarray(f.get('loadings'))
        U = np.asarray(f.get('U'))
        S = np.asarray(f.get('S'))
        V = np.asarray(f.get('V'))

    return exp, run, det_type, start_img, loadings, U, S, V

def closest_image_index(x, y, PCx_vector, PCy_vector):
    """
    Finds the index of image closest to the tap location
    
    Parameters:
    -----------
    x : float
        tap location on x-axis of tap source plot
    y : float
        tap location on y-axis of tap source plot
    PCx_vector : ndarray, shape (d,)
        principle component chosen for x-axis
    PCx_vector : ndarray, shape (d,)
        principle component chosen for y-axis
        
    Returns:
    --------
    img_source:
        index of the image closest to tap location
    """
    img_source = None
    min_diff = None
    square_diff = None
    
    for i, (xv, yv) in enumerate(zip(PCx_vector, PCy_vector)):    
        square_diff = (x - xv) ** 2 + (y - yv) ** 2
        if (min_diff is None or square_diff < min_diff):
            min_diff = square_diff
            img_source = i
    
    return img_source

def construct_heatmap_data(img, max_pixels):
    """
    Formats img to properly be displayed by hv.Heatmap()
    
    Parameters:
    -----------
    img : ndarray, shape (x_pixels x y_pixels)
        single image we want to display on a heatmap
    max_pixels: ing
        max number of pixels on x and y axes of heatmap
    
    Returns:
    --------
    hm_data : ndarray, shape ((x_pixels*y__pixels) x 3)
        coordinates to be displayed by hv.Heatmap (row, col, color)
    """
    x_pixels, y_pixels = img.shape
    bin_factor_x = int(x_pixels / max_pixels)
    bin_factor_y = int(y_pixels / max_pixels)
    
    while x_pixels % bin_factor_x != 0:
        bin_factor_x += 1
    while y_pixels % bin_factor_y != 0:
        bin_factor_y += 1
    
    img = img.reshape((x_pixels, y_pixels))
    binned_img = img.reshape(int(x_pixels / bin_factor_x),
                                bin_factor_x,
                                int(y_pixels / bin_factor_y),
                                bin_factor_y).mean(-1).mean(1)
    
    # Creates hm_data array for heatmap
    bin_x_pixels, bin_y_pixels = binned_img.shape
    rows = np.tile(np.arange(bin_x_pixels).reshape((bin_x_pixels, 1)), bin_y_pixels).flatten()
    cols = np.tile(np.arange(bin_y_pixels), bin_x_pixels)
    
    hm_data = np.stack((rows, cols, binned_img.flatten()))
    hm_data = hm_data.T.reshape((bin_x_pixels * bin_y_pixels, 3))
    
    return hm_data