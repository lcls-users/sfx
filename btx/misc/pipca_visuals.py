import numpy as np
import h5py
import random
import math
import time
import json
import pickle

import csv 
from sklearn.cluster import DBSCAN

from btx.interfaces.ipsana import (
    PsanaInterface,
    retrieve_pixel_index_map,
    assemble_image_stack_batch,
)

from sklearn.manifold import TSNE
from sklearn.manifold import trustworthiness
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

import holoviews as hv
hv.extension('bokeh')
from holoviews.streams import Params

import panel as pn
pn.extension(template='fast')
pn.state.template.param.update()
import panel.widgets as pnw

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

def display_dashboard_pytorch(filename):
    """
    OUTDATED: Waiting for Deeban's dashboard
    Displays an interactive dashboard with a PC plot, scree plot, 
    and intensity heatmaps of the selected source image as well as 
    its reconstructed image using the model obtained by pipca.

    Parameters
    -----------------
    filename : str
            Name of the read document to display figures
    """
    data = unpack_ipca_pytorch_model_file(filename)

    exp = data['exp']
    run = data['run']
    det_type = data['det_type']
    start_img = data['start_offset']
    transformed_images = data['transformed_images']
    mu = data['mu']
    S = data['S']
    V = data['V']

    psi = PsanaInterface(exp=exp, run=run, det_type=det_type)
    psi.counter = start_img

    # Create PC dictionary and widgets
    PCs = {f'PC{i}' : v for i, v in enumerate(np.concatenate(transformed_images,axis=1).T, start=1)}
    PC_options = list(PCs)

    PCx = pnw.Select(name='X-Axis', value='PC1', options=PC_options)
    PCy = pnw.Select(name='Y-Axis', value='PC2', options=PC_options)
    widgets_scatter = pn.WidgetBox(PCx, PCy, width=150)

    PC_scree = pnw.Select(name='First Component Cut-off', value=f'PC{1}', options=PC_options)
    PC_scree2 = pnw.Select(name='Last Component Cut-off', value=f'PC{len(PCs)}', options=PC_options)
    widgets_scree = pn.WidgetBox(PC_scree,PC_scree2, width=150)

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
                            label=f"{PCx.title()} vs {PCy.title()}").opts(**opts)
            
        posxy.source = scatter
        return scatter
        
    # Create scree plot
    @pn.depends(PC_scree.param.value, PC_scree2.param.value)
    def create_scree(PC_scree, PC_scree2):
        first_compo = int(PC_scree[2:])
        last_compo = int(PC_scree2[2:])

        if first_compo > last_compo:
            raise ValueError("Error: First component cut-off cannot be greater than last component cut-off.")

        components = np.arange(first_compo,last_compo+1)
        singular_values = S[0][first_compo-1:last_compo] ################# Only first rank

        bars_data = np.stack((components, singular_values)).T

        opts = dict(width=400, height=300, show_grid=True, show_legend=False,
                    shared_axes=False, toolbar='above', default_tools=['hover','save','reset'],
                    xlabel='Components', ylabel='Singular Values')
        scree = hv.Bars(bars_data, label="Scree Plot").opts(**opts)
        
        return scree
        
    # Define function to compute heatmap based on tap location
    def tap_heatmap(x, y, pcx, pcy, pcscree, pcscree2):
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

        hv.save(heatmap, f"/sdf/data/lcls/ds/mfx/mfxp23120/scratch/test_btx/pipca/heatmap.png")

        return heatmap
        
    # Define function to compute reconstructed heatmap based on tap location
    def tap_heatmap_reconstruct(x, y, pcx, pcy, pcscree, pcscree2):
        # Finds the index of image closest to the tap location
        img_source = closest_image_index(x, y, PCs[pcx], PCs[pcy])

        # Calculate and format reconstructed image
        p, x, y = psi.det.shape()
        pixel_index_map = retrieve_pixel_index_map(psi.det.geometry(psi.run))

        first_compo = int(pcscree[2:])
        last_compo = int(pcscree2[2:])

        imgs = []
        for rank in range(len(S)):
            img = np.dot(transformed_images[rank][img_source, first_compo-1:last_compo], V[rank][:, first_compo-1:last_compo].T)+mu[rank]
            img = img.reshape((p, x, y))
            img = assemble_image_stack_batch(img, pixel_index_map)
            imgs.append(img)
        img = np.concatenate(imgs, axis=0)

        # Downsample so heatmap is at most 100 x 100
        hm_data = construct_heatmap_data(img, 100)

        opts = dict(width=400, height=300, cmap='plasma', colorbar=True, shared_axes=False, toolbar='above')
        heatmap_reconstruct = hv.HeatMap(hm_data, label="iPCA Pytorch Image %s" % (start_img+img_source)).aggregate(function=np.mean).opts(**opts)

        hv.save(heatmap_reconstruct, f"/sdf/data/lcls/ds/mfx/mfxp23120/scratch/test_btx/pipca/heatmap_reconstruct.png")
        return heatmap_reconstruct
    
    def compute_loss(x, y, pcx, pcy, pcscree, pcscree2):
        # Finds the index of image closest to the tap location
        img_source = closest_image_index(x, y, PCs[pcx], PCs[pcy])

        #Get geometry
        p, x, y = psi.det.shape()
        pixel_index_map = retrieve_pixel_index_map(psi.det.geometry(psi.run))

        #Get image
        counter = psi.counter
        psi.counter = start_img + img_source
        img = psi.get_images(1)
        img = img.squeeze()

        #Get reconstructed image
        first_compo = int(pcscree[2:])
        last_compo = int(pcscree2[2:])
        img_reconstructed = np.dot(transformed_images[img_source, first_compo-1:last_compo], V[:, first_compo-1:last_compo].T)+mu
        img_reconstructed = img_reconstructed.reshape((p, x, y))
        img_reconstructed = assemble_image_stack_batch(img_reconstructed, pixel_index_map)

        #Compute loss
        diff = np.abs(img - img_reconstructed)
        loss = np.linalg.norm(diff, 'fro') / np.linalg.norm(img, 'fro') * 100
        print(f"Loss: {loss:.3f}%")
        return loss

    # Connect the Tap stream to the heatmap callbacks
    stream1 = [posxy]
    stream2 = Params.from_params({'pcx': PCx.param.value, 'pcy': PCy.param.value, 'pcscree': PC_scree.param.value, 'pcscree2': PC_scree2.param.value})
    tap_dmap = hv.DynamicMap(tap_heatmap, streams=stream1+stream2)
    tap_dmap_reconstruct = hv.DynamicMap(tap_heatmap_reconstruct, streams=stream1+stream2)

    PC_scree.param.watch(lambda event: compute_loss(posxy.x, posxy.y,PCx.value, PCy.value, PC_scree.value, PC_scree2.value), 'value')
    PC_scree2.param.watch(lambda event: compute_loss(posxy.x, posxy.y,PCx.value, PCy.value, PC_scree.value, PC_scree2.value), 'value')
    
    return pn.Column(pn.Row(widgets_scatter, create_scatter, tap_dmap),
                     pn.Row(widgets_scree, create_scree, tap_dmap_reconstruct)).servable('PiPCA Dashboard')

def display_image_pypca(model_filename, projection_filename, image_to_display=None,num_pixels=100,first_compo=0,last_compo=10**8,clim_original=(0,1000),clim_rec=(0,1000)):
    start_idx = image_to_display
    end_idx = image_to_display+1
    data = unpack_ipca_pytorch_model_file(model_filename,start_idx=start_idx, end_idx=end_idx)
    exp = data['exp']
    run = data['run']
    det_type = data['det_type']
    start_img = data['start_offset']
    mu = data['mu']
    S = data['S']

    with h5py.File(projection_filename, 'r') as f:
        projected_images = f['projected_images']
        projected_images_list = []
        for rank in range(S.shape[0]):
            projected_images_list.append(projected_images[rank,start_idx:end_idx,:])
        projected_images = np.array(projected_images_list)

    print(projected_images.shape)
    psi = PsanaInterface(exp=exp, run=run, det_type=det_type)
    if image_to_display is None:
        counter = start_img
    else:
        counter = start_img+image_to_display

    psi.counter = counter
    img = psi.get_images(1)
    img = img.squeeze()
    pixel_index_map = retrieve_pixel_index_map(psi.det.geometry(psi.run))
    a,b,c = psi.det.shape()
    
    
    # Downsample so heatmap is at most 100 x 100
    """hm_data = construct_heatmap_data(img, num_pixels)"""
    opts = dict(width=800, height=600, cmap='plasma', colorbar=True, shared_axes=False, toolbar='above',clim=clim_original)
    """heatmap = hv.HeatMap(hm_data, label="Original Source Image %s" % (counter)).aggregate(function=np.mean).opts(**opts).opts(title="Original Source Image")"""
    original_image = hv.Image(img).opts(**opts).opts(title="Original Source Image")

    rec_imgs = []
    
    for rank in range(len(S)):
        rec_img = np.zeros((1, mu.shape[1]))
        with h5py.File(model_filename, 'r') as f:
            V = f['V']
            batch_component_size = min(20, S.shape[1])
            for i in range(first_compo, min(S.shape[1],last_compo), batch_component_size):
                rec_img += np.dot(projected_images[rank,:,i:i+batch_component_size], V[rank,:,i:i+batch_component_size].T)
            print("Reconstructed image on rank %d" % (rank))
        rec_img = rec_img.reshape((int(a/len(S)), b, c))
        rec_imgs.append(rec_img)
    rec_img = np.concatenate(rec_imgs, axis=0)
    rec_img = assemble_image_stack_batch(rec_img, pixel_index_map)
    """hm_rec_data = construct_heatmap_data(rec_img, num_pixels)
    heatmap_reconstruct = hv.HeatMap(hm_rec_data, label="PyPCA Reconstructed Image %s" % (counter)).aggregate(function=np.mean).opts(**opts).opts(title="PyPCA Reconstructed Image")"""
    opts = dict(width=800, height=600, cmap='plasma', colorbar=True, shared_axes=False, toolbar='above',clim=clim_rec)
    reconstructed_image = hv.Image(rec_img).opts(**opts).opts(title="PyPCA Reconstructed Image")
    """layout = (heatmap + heatmap_reconstruct).cols(2)"""
    img = img.assign_coords(x=np.arange(img.shape[1]), y=np.arange(img.shape[0]))
    rec_img = rec_img.assign_coords(x=np.arange(rec_img.shape[1]), y=np.arange(rec_img.shape[0]))   
    hextiles = hv.HexTiles((img, rec_img)).opts(width=800, height=600, cmap='plasma',clim=(0,1000),tools=['hover'],title="HexTiles")
    layout_combined = (original_image + reconstructed_image + hextiles).cols(3)
    hv.save(layout, 'heatmaps_layout.html')
    return layout

def display_eigenimages_pypca(filename,nb_eigenimages=3,sklearn_test=False,classic_pca_test=False,num_images=10,num_pixels=100,compute_diff=False):
    data = unpack_ipca_pytorch_model_file(filename)

    exp = data['exp']
    run = data['run']
    det_type = data['det_type']
    start_img = data['start_offset']
    transformed_images = data['transformed_images']
    mu = data['mu']
    S = data['S']
    V = data['V']
    num_components = S.shape[0]
    num_gpus = len(V)
    psi = PsanaInterface(exp=exp, run=run, det_type=det_type)
    counter = start_img

    psi.counter = counter

    a,b,c = psi.det.shape()
    pixel_index_map = retrieve_pixel_index_map(psi.det.geometry(psi.run))

    heatmaps=[]
    eigen_images_pypca = []
    eigenimages_pca = []

    for k in range(nb_eigenimages):
        eigenimages = []
        for rank in range(num_gpus):
            eigenimage = V[rank].T[k]
            eigenimage = eigenimage.reshape((int(a/len(S)), b, c))/np.linalg.norm(eigenimage.reshape(-1,1), 'fro')
            eigenimages.append(eigenimage)
        eigenimages = np.concatenate(eigenimages, axis=0)
        eigen_images_pypca.append(eigenimages)
        eigenimages = assemble_image_stack_batch(eigenimages, pixel_index_map)
        hm_data = construct_heatmap_data(eigenimages, num_pixels)

        opts = dict(width=400, height=300, cmap='plasma', colorbar=True, shared_axes=False, toolbar='above')

        heatmap =hv.HeatMap(hm_data, label="Eigen Image %s" % (k)).aggregate(function=np.mean).opts(**opts).opts(title=f"PyPCA Eigen Image {k}")
        heatmaps.append(heatmap)
    
    layout = hv.Layout(heatmaps).cols(nb_eigenimages)
    print('PyPCA done')
    
    if classic_pca_test:
        heatmaps_pca = []
        imgs = psi.get_images(num_images,assemble=False)
        imgs = imgs[
            [i for i in range(num_images) if not np.isnan(imgs[i : i + 1]).any()]
        ]
        imgs = np.reshape(imgs, (imgs.shape[0], a,b,c))
        
        """pca = PCA(n_components=num_components+1)"""

        ##
        list_V = []
        list_imgs = np.split(imgs,num_gpus,axis=1)
        for rank in range(num_gpus):
            pca = PCA(n_components=num_components+1)
            pca.fit(list_imgs[rank].reshape(list_imgs[rank].shape[0], -1))
            list_V.append(pca.components_)
        ##
        """pca.fit(imgs.reshape(imgs.shape[0], -1))
        V = pca.components_"""

        for k in range(nb_eigenimages):
            eigenimages = []
            for rank in range(num_gpus):
                eigenimage = list_V[rank][k]
                eigenimage = eigenimage.reshape((int(a/num_gpus),b,c))/np.linalg.norm(eigenimage.reshape(-1,1), 'fro')
                eigenimages.append(eigenimage)
            eigenimages = np.concatenate(eigenimages,axis=0)
            eigenimages_pca.append(eigenimages)
            eigenimages = assemble_image_stack_batch(eigenimages, pixel_index_map)
            hm_data = construct_heatmap_data(eigenimages, num_pixels)

            opts = dict(width=400, height=300, cmap='plasma', colorbar=True, shared_axes=False, toolbar='above')

            heatmap =hv.HeatMap(hm_data, label="Eigen Image Classic PCA %s" % (k)).aggregate(function=np.mean).opts(**opts).opts(title=f"Classic PCA Eigen Image {k}")
            heatmaps_pca.append(heatmap)
        
        layout_pca = hv.Layout(heatmaps_pca).cols(nb_eigenimages)
        print('PCA done')
        
        layout_combined = layout + layout_pca
    
    if classic_pca_test:
        layout_combined.cols(nb_eigenimages)
    else:
        layout.cols(nb_eigenimages)

    if compute_diff and classic_pca_test:
        list_norm_diff = []
        for k in range(nb_eigenimages):
            print('Computing loss for eigenimage : ',k)
            diff = np.abs(eigen_images_pypca[k]) - np.abs(eigenimages_pca[k])
            diff = diff.reshape(1,-1)
            norm_diff = np.linalg.norm(diff, 'fro') * 100 / np.linalg.norm(eigen_images_pypca[k].reshape(-1,1), 'fro') ##PEUT ETRE LE FAIRE PANEL PAR PANEL COMME DANS LE CALCUL DE LA LOSS DE RECONSTRUCTION
            list_norm_diff.append(norm_diff)
        print(list_norm_diff)
        
    return layout_combined if classic_pca_test else layout

def display_umap(filename,num_images):
    data = unpack_ipca_pytorch_model_file(filename)

    exp = data['exp']
    run = data['run']
    det_type = data['det_type']
    start_img = data['start_offset']
    transformed_images = data['transformed_images']
    mu = data['mu']
    S = data['S']
    V = data['V']
    num_components = S.shape[0]
    num_gpus = len(V)
    psi = PsanaInterface(exp=exp, run=run, det_type=det_type)
    counter = start_img

    psi.counter = counter

    a,b,c = psi.det.shape()
    pixel_index_map = retrieve_pixel_index_map(psi.det.geometry(psi.run))
    imgs = psi.get_images(num_images,assemble=False)
    imgs = imgs[
        [i for i in range(num_images) if not np.isnan(imgs[i : i + 1]).any()]
    ]
    imgs = np.reshape(imgs, (imgs.shape[0], a,b,c))

    imgs = np.split(imgs,num_gpus,axis=1)
    print("Images gathered and split")
    fig = sp.make_subplots(rows=2, cols=2, subplot_titles=[f't-SNE projection (GPU {rank})' for rank in range(num_gpus)])

    for rank in range(num_gpus):
        U = np.dot(np.dot(imgs[rank].reshape(num_images,-1), V[rank]), np.diag(1.0 / S[rank]))
        U = np.array([u.flatten() for u in U])
        print("Projectors reconstructed")

        tsne = TSNE(n_components=2, init='random', learning_rate='auto')
        embedding = tsne.fit_transform(U)
        trustworthiness_score = trustworthiness(U, embedding)

        # Print the trustworthiness score
        print(f"Trustworthiness score: {trustworthiness_score:.4f}")
        print(f"Fitting of the t-SNE {rank} done")
        
        df = pd.DataFrame({
            't-SNE1': embedding[:, 0],
            't-SNE2': embedding[:, 1],
            'Index': np.arange(len(embedding)),
        })
        
        scatter = px.scatter(df, x='t-SNE1', y='t-SNE2', 
                            hover_data={'Index': True},
                            labels={'t-SNE1': 't-SNE1', 't-SNE2': 't-SNE2'},
                            title=f't-SNE projection (GPU {rank})')
        
        fig.add_trace(scatter.data[0], row=(rank // 2) + 1, col=(rank % 2) + 1)

    fig.update_layout(height=800, width=800, showlegend=False, title_text="t-SNE Projections Across GPUs")
    fig.show()

def plot_t_sne_scatters(filename, type_of_embedding='t-SNE', eps=0.5, min_samples=3):
    with open(filename, "rb") as f:
        data = pickle.load(f)

    embedding_tsne = np.array(data["embeddings_tsne"])
    embedding_umap = np.array(data["embeddings_umap"])
    S = np.array(data["S"])
    num_gpus = len(S)

    if type_of_embedding == 't-SNE':
        embedding = embedding_tsne
        projection_title = 't-SNE projection'
    else:
        embedding = embedding_umap
        projection_title = 'UMAP projection'
    
    # Initialize dictionaries to store indices of each cluster
    clusters_per_gpu = [defaultdict(set) for _ in range(num_gpus)]

    fig = make_subplots(rows=2, cols=2, subplot_titles=[f'{projection_title} (GPU {rank})' for rank in range(num_gpus)])

    for rank in range(num_gpus):
        df = pd.DataFrame({
            'Projection1': embedding[rank][:, 0],
            'Projection2': embedding[rank][:, 1],
            'Index': np.arange(len(embedding[rank])),
        })

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        df['Cluster'] = dbscan.fit_predict(df[['Projection1', 'Projection2']])
        
        # Check if there are valid clusters
        if len(set(df['Cluster'])) > 1 or (len(set(df['Cluster'])) == 1 and -1 not in df['Cluster']):
            # Store indices in each cluster
            for cluster, indices in df.groupby('Cluster')['Index']:
                clusters_per_gpu[rank][cluster].update(indices)
            
            # Separate noise points from clustered points
            noise_points = df[df['Cluster'] == -1]
            clustered_points = df[df['Cluster'] != -1]

            # Create scatter plot for clustered points
            scatter = px.scatter(clustered_points, x='Projection1', y='Projection2', 
                                color='Cluster',
                                color_discrete_sequence=px.colors.qualitative.Dark24,
                                hover_data={'Index': True},
                                title=f'{projection_title} (GPU {rank})')

            # Add noise points in light grey
            scatter.add_trace(go.Scatter(
                x=noise_points['Projection1'],
                y=noise_points['Projection2'],
                mode='markers',
                marker=dict(color='lightgrey', size=5),
                hoverinfo='text',
                text=noise_points['Index'],
                name='Noise'
            ))

            for trace in scatter.data:
                fig.add_trace(trace, row=(rank // 2) + 1, col=(rank % 2) + 1)
        else:
            print(f"Warning: No valid clusters found for GPU {rank}")
            fig.add_annotation(
                text="No valid clusters",
                xref="x domain", yref="y domain",
                x=0.5, y=0.5, showarrow=False,
                row=(rank // 2) + 1, col=(rank % 2) + 1
            )

    fig.update_layout(
        height=800, width=800, showlegend=False, 
        title_text=f"{projection_title} Across GPUs",
        plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black')
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig.show()

    # Calculate similarity between clusters of each GPU pair
    def jaccard_index(set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0

    # Initialize similarity matrices
    cluster_similarity = defaultdict(lambda: defaultdict(float))

    for i in range(num_gpus):
        for cluster_i, indices_i in clusters_per_gpu[i].items():
            for j in range(num_gpus):
                if i != j:
                    for cluster_j, indices_j in clusters_per_gpu[j].items():
                        similarity = jaccard_index(indices_i, indices_j)
                        cluster_similarity[(i, cluster_i)][(j, cluster_j)] = similarity

    # Prepare data for heatmap
    clusters_pairs = sorted(set([(i, cluster) for i in range(num_gpus) for cluster in clusters_per_gpu[i].keys()]))
    cluster_pairs_idx = {pair: idx for idx, pair in enumerate(clusters_pairs)}

    num_pairs = len(clusters_pairs)
    similarity_matrix = np.zeros((num_pairs, num_pairs))

    # Ensure keys are properly unpacked
    for key in cluster_similarity:
        i, cluster_i = key
        for sub_key in cluster_similarity[key]:
            j, cluster_j = sub_key
            idx_i = cluster_pairs_idx[(i, cluster_i)]
            idx_j = cluster_pairs_idx[(j, cluster_j)]
            similarity_matrix[idx_i, idx_j] = cluster_similarity[key].get(sub_key, 0)

    # Check if similarity matrix contains valid data
    if np.any(similarity_matrix) and len(clusters_pairs) > 0:
        # Display similarity matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(similarity_matrix, annot=True, cmap="YlGnBu", 
                    xticklabels=[f'GPU {p[0]}-Cluster {p[1]}' for p in clusters_pairs], 
                    yticklabels=[f'GPU {p[0]}-Cluster {p[1]}' for p in clusters_pairs])
        plt.title('Cluster Similarity Matrix')
        plt.show()
    else:
        print("Warning: No valid clusters found to create similarity heatmap.")

def ipca_execution_time(num_components,num_images,batch_size,filename):
    data = unpack_ipca_pytorch_model_file(filename)

    exp = data['exp']
    run = data['run']
    det_type = data['det_type']
    start_img = data['start_offset']

    psi = PsanaInterface(exp=exp, run=run, det_type=det_type)
    counter = start_img
    psi.counter = counter
    images = psi.get_images(num_images)

    ipca = IncrementalPCA(n_components=num_components, batch_size=batch_size)
    start_time = time.time()
    for start in range(0, num_images, batch_size):
        print('Processing batch %d' % (start // batch_size))
        end = min(start + batch_size, num_images)
        batch_imgs = images[start:end]
        ipca.partial_fit(batch_imgs.reshape(batch_imgs.shape[0], -1))
    end_time = time.time()
    execution_time = end_time - start_time
    print(execution_time)
    return execution_time
 
def display_dashboard(filename):
    """
    Displays an interactive dashboard with a PC plot, scree plot, 
    and intensity heatmaps of the selected source image as well as 
    its reconstructed image using the model obtained by pipca.

    Parameters
    -----------------
    filename : str
            Name of the read document to display figures
    """
    data = unpack_pipca_model_file(filename)

    exp = data['exp']
    run = data['run']
    loadings = data['loadings']
    det_type = data['det_type']
    start_img = data['start_img']
    U = data['U']
    S = data['S']
    V = data['V']

    psi = PsanaInterface(exp=exp, run=run, det_type=det_type)
    psi.counter = start_img

    # Create PC dictionary and widgets
    PCs = {f'PC{i}' : v for i, v in enumerate(loadings, start=1)}
    PC_options = list(PCs)

    PCx = pnw.Select(name='X-Axis', value='PC1', options=PC_options)
    PCy = pnw.Select(name='Y-Axis', value='PC2', options=PC_options)
    widgets_scatter = pn.WidgetBox(PCx, PCy, width=150)

    PC_scree = pnw.Select(name='First Component Cut-off', value=f'PC{1}', options=PC_options)
    PC_scree2 = pnw.Select(name='Last Component Cut-off', value=f'PC{len(PCs)}', options=PC_options)
    widgets_scree = pn.WidgetBox(PC_scree,PC_scree2, width=150)

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
                            label=f"{PCx.title()} vs {PCy.title()}").opts(**opts)
            
        posxy.source = scatter
        return scatter
        
    # Create scree plot
    @pn.depends(PC_scree.param.value, PC_scree2.param.value)
    def create_scree(PC_scree, PC_scree2):
        first_compo = int(PC_scree[2:])
        last_compo = int(PC_scree2[2:])

        if first_compo > last_compo:
            raise ValueError("Error: First component cut-off cannot be greater than last component cut-off.")

        components = np.arange(first_compo,last_compo+1)
        singular_values = S[first_compo-1:last_compo]

        bars_data = np.stack((components, singular_values)).T

        opts = dict(width=400, height=300, show_grid=True, show_legend=False,
                    shared_axes=False, toolbar='above', default_tools=['hover','save','reset'],
                    xlabel='Components', ylabel='Singular Values')
        scree = hv.Bars(bars_data, label="Scree Plot").opts(**opts)

        return scree
        
    # Define function to compute heatmap based on tap location
    def tap_heatmap(x, y, pcx, pcy, pcscree, pcscree2):
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
    def tap_heatmap_reconstruct(x, y, pcx, pcy, pcscree, pcscree2):
        # Finds the index of image closest to the tap location
        img_source = closest_image_index(x, y, PCs[pcx], PCs[pcy])

        # Calculate and format reconstructed image
        p, x, y = psi.det.shape()
        pixel_index_map = retrieve_pixel_index_map(psi.det.geometry(psi.run))

        first_compo = int(pcscree[2:])
        last_compo = int(pcscree2[2:])

        img = U[:, first_compo-1:last_compo] @ np.diag(S[first_compo-1:last_compo]) @ np.array([V[img_source][first_compo-1:last_compo]]).T
        img = img.reshape((p, x, y))
        img = assemble_image_stack_batch(img, pixel_index_map)

        # Downsample so heatmap is at most 100 x 100
        hm_data = construct_heatmap_data(img, 100)

        opts = dict(width=400, height=300, cmap='plasma', colorbar=True, shared_axes=False, toolbar='above')
        heatmap_reconstruct = hv.HeatMap(hm_data, label="PiPCA Image %s" % (start_img+img_source)).aggregate(function=np.mean).opts(**opts)

        return heatmap_reconstruct
    
    # Connect the Tap stream to the heatmap callbacks
    stream1 = [posxy]
    stream2 = Params.from_params({'pcx': PCx.param.value, 'pcy': PCy.param.value, 'pcscree': PC_scree.param.value, 'pcscree2': PC_scree2.param.value})
    tap_dmap = hv.DynamicMap(tap_heatmap, streams=stream1+stream2)
    tap_dmap_reconstruct = hv.DynamicMap(tap_heatmap_reconstruct, streams=stream1+stream2)
        
    return pn.Column(pn.Row(widgets_scatter, create_scatter, tap_dmap),
                     pn.Row(widgets_scree, create_scree, tap_dmap_reconstruct)).servable('PiPCA Dashboard')

def unpack_ipca_pytorch_model_file(filename,start_idx=0,end_idx=-1):
    """
    Reads PiPCA model information from h5 file and returns its contents

    Parameters
    ----------
    filename: str
        name of h5 file you want to unpack

    Returns
    -------
    data: dict
        A dictionary containing the extracted data from the h5 file.
    """
    data = {}
    with h5py.File(filename, 'r') as f:
        """data['exp'] = str(np.asarray(f.get('exp')))[2:-1]
        data['run'] = int(np.asarray(f.get('run')))
        data['det_type'] = str(np.asarray(f.get('det_type')))[2:-1]
        data['start_offset'] = int(np.asarray(f.get('start_offset')))
        data['transformed_images'] = np.asarray(f.get('transformed_images'))
        data['S'] = np.asarray(f.get('S'))
        data['V'] = np.asarray(f.get('V'))
        data['mu'] = np.asarray(f.get('mu'))"""
        metadata = f['metadata']
        data['exp'] = str(np.asarray(metadata.get('exp')))[2:-1]
        data['run'] = int(np.asarray(metadata.get('run')))
        data['det_type'] = str(np.asarray(metadata.get('det_type')))[2:-1]
        data['start_offset'] = int(np.asarray(metadata.get('start_offset')))
        data['S'] = np.asarray(f.get('S'))
        data['mu'] = np.asarray(f.get('mu'))

    return data

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
    
    square_diff = (PCx_vector - x)**2 + (PCy_vector - y)**2
    img_source = square_diff.argmin()
    
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
    
    y_pixels, x_pixels = img.shape
    bin_factor_y = int(y_pixels / max_pixels)
    bin_factor_x = int(x_pixels / max_pixels)

    while y_pixels % bin_factor_y != 0:
        bin_factor_y += 1
    while x_pixels % bin_factor_x != 0:
        bin_factor_x += 1

    img = img.reshape((y_pixels, x_pixels))
    binned_img = img.reshape(int(y_pixels / bin_factor_y),
                            bin_factor_y,
                            int(x_pixels / bin_factor_x),
                            bin_factor_x).mean(-1).mean(1)

    # Create hm_data array for heatmap
    bin_y_pixels, bin_x_pixels = binned_img.shape
    rows = np.tile(np.arange(bin_y_pixels).reshape((bin_y_pixels, 1)), bin_x_pixels).flatten()
    cols = np.tile(np.arange(bin_x_pixels), bin_y_pixels)

    # Create hm_data array for heatmap
    hm_data = np.stack((rows, cols, binned_img.flatten())).T

    return hm_data

def compute_compression_loss(filename, num_components, random_images=False, num_images=10, type_of_pca='pipca', write_results=False,training_percentage=1.0):
    """
    Compute the average frobenius norm between images in an experiment run and their reconstruction. 
    The reconstructed images and their metadata (experiment name, run, detector, ...) are assumed to be found in the input file created by PiPCA."

    Parameters:
    -----------
    filename : string
        name of the h5 file
    num_components: int
        number of components used
    random_images: bool, optional
        whether to choose random images or all images (default is False)
    num_images: int, optional
        number of random images to select if random_images is True (default is 10)

    Returns:
    --------
    compression loss between initial dataset and pipca projected images
    averaged compression loss
    run number
    Eventually write the results in a file
    """

    if type_of_pca == 'pipca':
        data = unpack_pipca_model_file(filename)

        exp, run, loadings, det_type, start_img, U, S, V, mu = data['exp'], data['run'], data['loadings'], data['det_type'], data['start_img'], data['U'], data['S'], data['V'], data['mu']

        model_rank = S.shape[0]
        if num_components > model_rank:
            raise ValueError("Error: num_components cannot be greater than the maximum model rank.")
        
        psi = PsanaInterface(exp=exp, run=run, det_type=det_type)
        psi.counter = start_img

        PCs = {f'PC{i}': v for i, v in enumerate(loadings, start=1)}

        compression_losses = []

        p, x, y = psi.det.shape()
        pixel_index_map = retrieve_pixel_index_map(psi.det.geometry(psi.run))

        # Compute the projection matrix outside the loop
        projection_matrix = U[:, :num_components] @ np.diag(S[:num_components])

        image_indices = random.sample(range(len(PCs['PC1'])), num_images) if random_images else range(len(PCs['PC1']))

        for img_source in image_indices:
            counter = psi.counter
            psi.counter = start_img + img_source
            img = psi.get_images(1).squeeze()

            reconstructed_img = projection_matrix @ np.array([V[img_source][:num_components]]).T
            reconstructed_img = reconstructed_img.reshape((p, x, y))
            reconstructed_img = assemble_image_stack_batch(reconstructed_img, pixel_index_map)

            # Compute the Frobenius norm of the difference between the original image and the reconstructed image
            difference = np.subtract(img_normalized, reconstructed_img_normalized)
            norm = np.linalg.norm(difference, 'fro')
            original_norm = np.linalg.norm(img, 'fro')

            compression_loss = norm / original_norm * 100
            compression_losses.append(compression_loss)

            psi.counter = counter  # Reset counter for the next iteration

        average_loss = np.mean(compression_losses)

    elif type_of_pca == 'pytorch':
        data = unpack_ipca_pytorch_model_file(filename)

        exp, run, det_type, start_img, reconstructed_images, S, V, mu = data['exp'], data['run'], data['det_type'], data['start_img'], data['reconstructed_images'], data['S'], data['V'], data['mu']

        model_rank = S.shape[0]
        if max(num_components) > model_rank:
            raise ValueError("Error: num_components cannot be greater than the maximum model rank.")
        
        psi = PsanaInterface(exp=exp, run=run, det_type=det_type)
        psi.counter = start_img

        training_compression_losses = [[] for _ in range(len(num_components))]
        eval_compression_losses = [[] for _ in range(len(num_components))]

        p, x, y = psi.det.shape()
        pixel_index_map = retrieve_pixel_index_map(psi.det.geometry(psi.run))
    
        num_training_images = math.ceil(len(reconstructed_images) * training_percentage)
        print(f"Number of training images: {num_training_images}")
        print(f"Number of validation images: {len(reconstructed_images)-num_training_images}")
        if num_training_images <= num_components[-1]:
            num_training_images = max(num_components)

        training_image_indices = random.sample(range(num_training_images), num_images) if random_images else range(num_training_images)

        nb_images_treated = 0
        
        for img_source in training_image_indices:
            counter = psi.counter
            psi.counter = start_img + img_source
            img = psi.get_images(1).squeeze()

            count_compo=0
            for k in num_components:
                reconstructed_img = np.dot(reconstructed_images[:,:k], V[:,:k].T)[img_source]+mu
                reconstructed_img = reconstructed_img.reshape((p, x, y))
                reconstructed_img = assemble_image_stack_batch(reconstructed_img, pixel_index_map)
                # Compute the Frobenius norm of the difference between the original image and the reconstructed image
                difference = np.subtract(img, reconstructed_img)
                norm = np.linalg.norm(difference, 'fro')
                original_norm = np.linalg.norm(img, 'fro')

                compression_loss = norm / original_norm * 100
                training_compression_losses[count_compo].append(compression_loss)
                if count_compo % 3 == 0:
                    print(f"Computed compression loss for {count_compo+1} different numbers of components")
                count_compo+=1

            nb_images_treated+=1
            if nb_images_treated % 3 == 0:
                print(f"Processed {nb_images_treated} training images out of {len(training_image_indices)}")
            
            psi.counter = counter
        
        if num_training_images < (len(reconstructed_images)-num_images):
            eval_image_indices = random.sample(range(num_training_images, len(reconstructed_images)), num_images) if random_images else range(num_training_images, len(reconstructed_images))
            for img_source in eval_image_indices:
                counter = psi.counter
                psi.counter = start_img + img_source
                img = psi.get_images(1).squeeze()

                count_compo=0
                for k in num_components:
                    reconstructed_img = np.dot(reconstructed_images[:,:k], V[:,:k].T)[img_source]+mu
                    reconstructed_img = reconstructed_img.reshape((p, x, y))
                    reconstructed_img = assemble_image_stack_batch(reconstructed_img, pixel_index_map)
                    # Compute the Frobenius norm of the difference between the original image and the reconstructed image
                    difference = np.subtract(img, reconstructed_img)
                    norm = np.linalg.norm(difference, 'fro')
                    original_norm = np.linalg.norm(img, 'fro')

                    compression_loss = norm / original_norm * 100
                    eval_compression_losses[count_compo].append(compression_loss)
                    if count_compo % 3 == 0:
                        print(f"Computed compression loss for {count_compo+1} different numbers of components")
                    count_compo+=1

                nb_images_treated+=1
                if nb_images_treated % 3 == 0:
                    print(f"Processed {nb_images_treated} validation images out of {len(eval_image_indices)}")
                
                psi.counter = counter
        else:
            eval_compression_losses = [0]
        
        if len(training_image_indices) == 0:
            training_compression_losses = [0]
        
        average_loss = ([np.mean(training_compression_losses[k]) for k in range(len(training_compression_losses))],[np.mean(eval_compression_losses[k]) for k in range(len(eval_compression_losses))])

    elif type_of_pca == 'sklearn':
        raise NotImplementedError("Error: Sklearn PCA is not yet implemented.")
    
    else:
        raise ValueError("Error: type_of_pca must be either 'pipca' or 'pytorch' or 'sklearn'.")

    if write_results:
        results_file = "/sdf/data/lcls/ds/mfx/mfxp23120/scratch/test_btx/pipca/results_loss.txt"

        with open(results_file, "a") as f:
            f.write("\n============================================\nNUMBER OF COMPONENTS:\n============================================\n" + str(num_components) + "\n")
            f.write("Average Loss:\n")
            f.write(str(average_loss) + "\n")
            f.write("Training Compression Losses:\n")
            f.write(str(training_compression_losses) + "\n")
            f.write("Eval Compression Losses:\n")
            f.write(str(eval_compression_losses) + "\n")
            f.write("Run:\n")
            f.write(str(run) + "\n")
        
        print("Results written in file")
    
    print("Loss computation done")

    return average_loss, training_compression_losses, eval_compression_losses, run

