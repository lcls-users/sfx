import sys
sys.path.append("/sdf/home/w/winnicki/btx/")
from btx.processing.dimRed import *

import os, csv, argparse
import math
import time
import random
from collections import Counter
import h5py

import numpy as np
from numpy import zeros, sqrt, dot, diag
from numpy.linalg import svd, LinAlgError
from scipy.linalg import svd as scipy_svd
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
import heapq

from mpi4py import MPI

from matplotlib import pyplot as plt
from matplotlib import colors

from PIL import Image
from io import BytesIO
import base64

from datetime import datetime

import umap
import hdbscan
from sklearn.cluster import OPTICS, cluster_optics_dbscan

from matplotlib import colors
import matplotlib as mpl
from matplotlib import cm

from bokeh.plotting import figure, show, output_file, save
from bokeh.models import HoverTool, CategoricalColorMapper, LinearColorMapper, ColumnDataSource, CustomJS, Slider, RangeSlider, Toggle, RadioButtonGroup, Range1d, Label
from bokeh.palettes import Viridis256, Cividis256, Turbo256, Category20, Plasma3
from bokeh.layouts import column, row

import cProfile
import string

class visualizeFD:
    """
    Visualize FD Dimension Reduction using UMAP and DBSCAN
    """
    def __init__(self, inputFile, outputFile, numImgsToUse, nprocs, includeABOD, userGroupings, 
            skipSize, umap_n_neighbors, umap_random_state, hdbscan_min_samples, hdbscan_min_cluster_size,
            optics_min_samples, optics_xi, optics_min_cluster_size, outlierQuantile):
        self.inputFile = inputFile
        self.outputFile = outputFile
        output_file(filename=outputFile, title="Static HTML file")
        self.viewResults = None
        self.numImgsToUse = numImgsToUse
        self.nprocs = nprocs
        self.includeABOD = includeABOD
        self.userGroupings = userGroupings
        self.skipSize = skipSize
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_random_state = umap_random_state
        self.hdbscan_min_samples=hdbscan_min_samples
        self.hdbscan_min_cluster_size=hdbscan_min_cluster_size
        self.optics_min_samples=optics_min_samples
        self.optics_xi = optics_xi
        self.optics_min_cluster_size = optics_min_cluster_size
        self.outlierQuantile = outlierQuantile

    def embeddable_image(self, data):
        img_data = np.uint8(cm.jet(data/max(data.flatten()))*255)
#        image = Image.fromarray(img_data, mode='RGBA').resize((75, 75), Image.Resampling.BICUBIC)
        image = Image.fromarray(img_data, mode='RGBA')
        buffer = BytesIO()
        image.save(buffer, format='png')
        for_encoding = buffer.getvalue()
        return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode('utf-8')

    def random_unique_numbers_from_range(self, start, end, count):
        all_numbers = list(range(start, end + 1))
        random.shuffle(all_numbers)
        return all_numbers[:count]

    def compute_medoid(self, points):
        return points[np.argmin(euclidean_distances(points).sum(axis=0))]

    def genMedoids(self, medoidLabels, clusterPoints):
        dictMe = {}
        for j in set(medoidLabels):
            dictMe[j] = []
        for index, class_name in enumerate(medoidLabels):
            dictMe[class_name].append((index, clusterPoints[index, 0], clusterPoints[index, 1]))
        medoid_lst = []
        for k, v in dictMe.items():
            lst = [(x[1], x[2]) for x in v]
            medoid_point = self.compute_medoid(lst)
            for test_index, test_point in enumerate(lst):
                if math.isclose(test_point[0],medoid_point[0]) and math.isclose(test_point[1], medoid_point[1]):
                    fin_ind = test_index
            medoid_lst.append((k, v[fin_ind][0]))
        return medoid_lst

    def relabel_to_closest_zero(self, labels):
        unique_labels = sorted(set(labels))
        relabel_dict = {label: new_label for new_label, label in enumerate(unique_labels)}
        relabeled = [relabel_dict[label] for label in labels]
        return relabeled

    def regABOD(self, pts):
        abofs = []
        for a in range(len(pts)):
            test_list = [x for x in range(len(pts)) if x != a]
            otherPts = [(d, e) for idx, d in enumerate(test_list) for e in test_list[idx + 1:]]
            outlier_factors = []
            for b, c in otherPts:
                apt = pts[a]
                bpt = pts[b]
                cpt = pts[c]
                ab = bpt - apt
                ac = cpt - apt
                outlier_factors.append(np.dot(ab, ac)/((np.linalg.norm(ab)**2) * (np.linalg.norm(ac))))
            abofs.append(np.var(np.array(outlier_factors)))
        return abofs

    def fastABOD(self, pts, nsamples):
        nbrs = NearestNeighbors(n_neighbors=nsamples, algorithm='ball_tree').fit(pts)
        k_inds = nbrs.kneighbors(pts)[1]
        abofs = []
        count = 0
        for a in range(len(pts)):
            test_list = k_inds[a][1:]
            otherPts = [(d, e) for idx, d in enumerate(test_list) for e in test_list[idx + 1:]]
            outlier_factors = []
            for (b, c) in otherPts:
                apt = pts[a]
                bpt = pts[b]
                cpt = pts[c]
                ab = bpt - apt
                ac = cpt - apt
                if math.isclose(np.linalg.norm(ab), 0.0) or math.isclose(np.linalg.norm(ac), 0.0):
                    count += 1
#                    print("TOO CLOSE")
                    continue
                outlier_factors.append(np.dot(ab, ac)/((np.linalg.norm(ab)**2) * (np.linalg.norm(ac))))
#            print("CURRENT POINT: ", pts[a], test_list, outlier_factors, np.var(np.array(outlier_factors)))
            if(len(outlier_factors)==0):
                abofs.append(np.inf)
            else:
                abofs.append(np.var(np.array(outlier_factors)))
        return abofs

    def getOutliers(self, lst):
#        lstCopy = lst.copy()
#        lstCopy.sort()
#        quart10 = lstCopy[len(lstCopy)//divBy]

        lstQuant = np.quantile(np.array(lst), self.outlierQuantile)
#        print("AIDJWOIJDAOWIDJWAOIDJAWOIDWJA", lstQuant, lst)
        outlierInds = []
        notOutlierInds = []
        for j in range(len(lst)):
            if lst[j]>lstQuant:
                outlierInds.append(j)
            else:
                notOutlierInds.append(j)
#        print("OUTLIER INDS: ", outlierInds)
#        print("NOT OUTLIER INDS: ", notOutlierInds)
        return np.array(outlierInds), np.array(notOutlierInds)

    def genHist(self, vals, endClass):
        totNum = endClass + 1
        countVals = Counter(vals)
        hist = [0]*(totNum)
        for val in set(countVals):
            hist[val] = countVals[val]
        maxval = max(countVals.values())
        return hist, maxval

    def genLeftRight(self, endClass):
        return [*range(endClass+1)], [*range(1, endClass+2)]

    def genUMAP(self):
        imgs = None
        projections = None
        for currRank in range(self.nprocs):
            with h5py.File(self.inputFile+"_"+str(currRank)+".h5", 'r') as hf:
                if imgs is None:
                    imgs = hf["SmallImages"][:]
                    projections = hf["ProjectedData"][:]
                else:
                    imgs = np.concatenate((imgs, hf["SmallImages"][:]), axis=0)
                    projections = np.concatenate((projections, hf["ProjectedData"][:]), axis=0)

        print("AOIDWJOIAWDJ", len(imgs), len(projections))

        intensities = []
        for img in imgs:
            intensities.append(np.sum(img.flatten()))
        intensities = np.array(intensities)

        self.imgs = imgs[:self.numImgsToUse:self.skipSize]
        self.projections = projections[:self.numImgsToUse:self.skipSize]
        self.intensities = intensities[:self.numImgsToUse:self.skipSize]

        self.numImgsToUse = int(self.numImgsToUse/self.skipSize)

        if len(self.imgs)!= self.numImgsToUse:
            raise TypeError("NUMBER OF IMAGES REQUESTED ({}) EXCEEDS NUMBER OF DATA POINTS PROVIDED ({})".format(len(self.imgs), self.numImgsToUse))

        self.clusterable_embedding = umap.UMAP(
            n_neighbors=self.umap_n_neighbors,
            random_state=self.umap_random_state,
            n_components=2,
#            min_dist=0.25,
            min_dist=0.1,
        ).fit_transform(self.projections)

        self.labels = hdbscan.HDBSCAN(
            min_samples = self.hdbscan_min_samples,
            min_cluster_size = self.hdbscan_min_cluster_size
        ).fit_predict(self.clusterable_embedding)
        exclusionList = np.array([])
        self.clustered = np.isin(self.labels, exclusionList, invert=True)

        self.opticsClust = OPTICS(min_samples=self.optics_min_samples, xi=self.optics_xi, min_cluster_size=self.optics_min_cluster_size)
        self.opticsClust.fit(self.clusterable_embedding)
        self.opticsLabels = cluster_optics_dbscan(
            reachability=self.opticsClust.reachability_,
            core_distances=self.opticsClust.core_distances_,
            ordering=self.opticsClust.ordering_,
            eps=2.5,
        )
#        self.opticsLabels = self.opticsClust.labels_

        self.experData_df = pd.DataFrame({'x':self.clusterable_embedding[self.clustered, 0],'y':self.clusterable_embedding[self.clustered, 1]})
        self.experData_df['image'] = list(map(self.embeddable_image, self.imgs[self.clustered]))
        self.experData_df['imgind'] = np.arange(self.numImgsToUse)*self.skipSize

    def genABOD(self):
        if self.includeABOD:
            abod = self.fastABOD(self.projections, 10)
            outliers, notOutliers = self.getOutliers(abod)
        else:
            outliers = []
            notOutliers = []
        outlierLabels = []
        for j in range(self.numImgsToUse):
            if j in outliers:
                outlierLabels.append(str(6))
            else:
                outlierLabels.append(str(0))
        self.experData_df['anomDet'] = outlierLabels
        self.experData_df['anom_backgroundColor'] = [Category20[20][int(x)] for x in outlierLabels]

    def setUserGroupings(self, userGroupings):
        """
        Set User Grouping. An adjustment is made at the beginning of this function,
        whereby 1 is added to each label. This is because internally, the clusters are stored
        starting at -1 rather than 0.
        """
        self.userGroupings = [[x-1 for x in grouping] for grouping in userGroupings]

    def genLabels(self):
        newLabels = []
        for j in self.labels[self.clustered]:
            doneChecking = False
            for grouping in self.userGroupings:
                if j in grouping and not doneChecking:
                    newLabels.append(min(grouping))
                    doneChecking=True
            if not doneChecking:
                newLabels.append(j)
        newLabels = list(np.array(newLabels) + 1)
        self.newLabels = np.array(self.relabel_to_closest_zero(newLabels))
        self.experData_df['cluster'] = [str(x) for x in self.newLabels[self.clustered]]
        self.experData_df['ptColor'] = [x for x in self.experData_df['cluster']]
        self.experData_df['dbscan_backgroundColor'] = [Category20[20][x] for x in self.newLabels]
        self.experData_df['backgroundColor'] = [Category20[20][x] for x in self.newLabels]
        medoid_lst = self.genMedoids(self.newLabels, self.clusterable_embedding)
        self.medoidInds = [x[1] for x in medoid_lst]
        medoidBold = []
        for ind in range(self.numImgsToUse):
            if ind in self.medoidInds:
                medoidBold.append(12)
            else:
                medoidBold.append(4)
        self.experData_df['medoidBold'] = medoidBold

        opticsNewLabels = []
        for j in self.opticsLabels[self.clustered]:
            doneChecking = False
            for grouping in self.userGroupings:
                if j in grouping and not doneChecking:
                    opticsNewLabels.append(min(grouping))
                    doneChecking=True
            if not doneChecking:
                opticsNewLabels.append(j)
        opticsNewLabels = list(np.array(opticsNewLabels) + 1)
        self.opticsNewLabels = np.array(self.relabel_to_closest_zero(opticsNewLabels))
#        self.experData_df['optics_backgroundColor'] = [Category20[20][x] for x in self.opticsNewLabels[self.opticsClust.ordering_]]
        self.experData_df['optics_backgroundColor'] = [Category20[20][x] for x in self.opticsNewLabels]

    def genHTML(self):
        datasource = ColumnDataSource(self.experData_df)
        color_mapping = CategoricalColorMapper(factors=[str(x) for x in list(set(self.newLabels))],palette=Category20[20])
        plot_figure = figure(
            title='UMAP projection with DBSCAN clustering of the LCLS dataset',
            tools=('pan, wheel_zoom, reset'),
            width = 2000, height = 600
        )
        plot_figure.add_tools(HoverTool(tooltips="""
        <div style="width: 170; height: 64; background-color:@backgroundColor; margin: 5px 0px 5px 0px">
            <div style='width: 64; height: 64; float: left;'>
                <img src='@image'; float: left;'/>
            </div>
            <div style="height: 64;">
                <div style='margin-left: 75; margin-top: 10'>
                    <span style='font-size: 15px; color: #224499'>Cluster </span>
                    <span style='font-size: 15px'>@cluster</span>
                </div>
                <div style='margin-left: 75; margin-top: 10'>
                    <span style='font-size: 15px; color: #224499'>Image </span>
                    <span style='font-size: 15px'>@imgind</span>
                </div>
            </div>
        </div>
        """))
        plot_figure.circle(
            'x',
            'y',
            source=datasource,
            color=dict(field='ptColor', transform=color_mapping),
            line_alpha=0.6,
            fill_alpha=0.6,
            size='medoidBold',
            legend_field='cluster'
        )
        plot_figure.sizing_mode = 'scale_both'
        plot_figure.legend.location = "bottom_right"
        plot_figure.legend.title = "Clusters"

        vals = [x for x in self.newLabels]
        trueSource = ColumnDataSource(data=dict(vals = vals))
        hist, maxCount = self.genHist(vals, max(vals))
        left, right = self.genLeftRight(max(vals))
        histsource = ColumnDataSource(data=dict(hist=hist, left=left, right=right))
        p = figure(width=2000, height=450, toolbar_location=None,
                   title="Histogram Testing")
        p.quad(source=histsource, top='hist', bottom=0, left='left', right='right',
                 fill_color='skyblue', line_color="white")
        p.y_range = Range1d(0, maxCount)
        p.x_range = Range1d(0, max(vals)+1)
        p.xaxis.axis_label = "Cluster Label"
        p.yaxis.axis_label = "Count"

        indexCDS = ColumnDataSource(dict(
            index=[*range(0, self.numImgsToUse, 2)]
            )
        )
        cols = RangeSlider(title="ET",
                start=0,
                end=self.numImgsToUse,
                value=(0, self.numImgsToUse-1),
                step=1, sizing_mode="stretch_width")
        callback = CustomJS(args=dict(cols=cols, trueSource = trueSource,
                                      histsource = histsource, datasource=datasource, indexCDS=indexCDS), code="""
        function countNumbersAtIndices(numbers, startInd, endInd, smallestVal, largestVal) {
            let counts = new Array(largestVal-smallestVal); for (let i=0; i<largestVal-smallestVal; ++i) counts[i] = 0;
            for (let i = Math.round(startInd); i <= Math.round(endInd); i++) {
                let numMe = numbers[i];
                if (typeof counts[numMe] === 'undefined') {
                  counts[numMe] = 1;
                } else {
                  counts[numMe]++;
                }
            }
            return counts;
            }
        const vals = trueSource.data.vals
        const leftVal = cols.value[0]
        const rightVal = cols.value[1]
        const oldhist = histsource.data.hist
        const left = histsource.data.left
        const right = histsource.data.right
        const hist = countNumbersAtIndices(vals, leftVal, rightVal, left[0], right.slice(-1))
        histsource.data = { hist, left, right }
        let medoidBold = new Array(datasource.data.medoidBold.length); for (let i=0; i<datasource.data.medoidBold.length; ++i) medoidBold[i] = 0;
                for (let i = Math.round(leftVal); i < Math.round(rightVal); i++) {
            medoidBold[i] = 5
        }
        const x = datasource.data.x
        const y = datasource.data.y
        const image = datasource.data.image
        const cluster = datasource.data.cluster
        const ptColor = datasource.data.ptColor
        const anomDet = datasource.data.anomDet
        const imgind = datasource.data.imgind
        const backgroundColor = datasource.data.backgroundColor
        const dbscan_backgroundColor = datasource.data.dbscan_backgroundColor
        const anom_backgroundColor = datasource.data.anom_backgroundColor
        const optics_backgroundColor = datasource.data.optics_backgroundColor
        datasource.data = { x, y, image, cluster, medoidBold, ptColor, anomDet, imgind, backgroundColor, dbscan_backgroundColor, anom_backgroundColor, optics_backgroundColor}
        """)
        cols.js_on_change('value', callback)


        imgsPlot = figure(width=2000, height=150, toolbar_location=None)
        imgsPlot.image(image=[self.imgs[imgindMe][::-1] for imgindMe in self.medoidInds],
                x=[0.25+xind for xind in range(len(self.medoidInds))],
                y=0,
                dw=0.5, dh=1,
                palette="Turbo256", level="image")
        imgsPlot.axis.visible = False
        imgsPlot.grid.visible = False
        for xind in range(len(self.medoidInds)):
            mytext = Label(x=0.375+xind, y=-0.25, text='Cluster {}'.format(xind))
            imgsPlot.add_layout(mytext)
        imgsPlot.y_range = Range1d(-0.3, 1.1)
        imgsPlot.x_range = Range1d(0, max(vals)+1)

        toggl = Toggle(label='► Play',active=False)
        toggl_js = CustomJS(args=dict(slider=cols,indexCDS=indexCDS),code="""
        // https://discourse.bokeh.org/t/possible-to-use-customjs-callback-from-a-button-to-animate-a-slider/3985/3
            var check_and_iterate = function(index){
                var slider_val0 = slider.value[0];
                var slider_val1 = slider.value[1];
                var toggle_val = cb_obj.active;
                if(toggle_val == false) {
                    cb_obj.label = '► Play';
                    clearInterval(looop);
                    }
                else if(slider_val1 >= index[index.length - 1]) {
//                    cb_obj.label = '► Play';
                    slider.value = [0, slider_val1-slider_val0];
//                   cb_obj.active = false;
//                    clearInterval(looop);
                    }
                else if(slider_val1 !== index[index.length - 1]){
                    slider.value = [index.filter((item) => item > slider_val0)[0], index.filter((item) => item > slider_val1)[0]];
                    }
                else {
                clearInterval(looop);
                    }
            }
            if(cb_obj.active == false){
                cb_obj.label = '► Play';
                clearInterval(looop);
            }
            else {
                cb_obj.label = '❚❚ Pause';
                var looop = setInterval(check_and_iterate, 0.1, indexCDS.data['index']);
            };
        """)
        toggl.js_on_change('active',toggl_js)

        reachabilityDiag = figure(
            title='OPTICS Reachability Diag',
            tools=('pan, wheel_zoom, reset'),
            width = 2000, height = 400
        )
        space = np.arange(self.numImgsToUse)
        reachability = self.opticsClust.reachability_[self.opticsClust.ordering_]
#        reachability = self.opticsClust.reachability_
        opticsData_df = pd.DataFrame({'x':space,'y':reachability})
        opticsData_df['clusterForScatterPlot'] = [str(x) for x in self.opticsNewLabels]
        opticsData_df['cluster'] = [str(x) for x in self.opticsNewLabels[self.opticsClust.ordering_]]
        opticsData_df['ptColor'] = [x for x in opticsData_df['cluster']]
        color_mapping2 = CategoricalColorMapper(factors=[str(x) for x in list(set(self.opticsNewLabels))],
                                               palette=Category20[20])
        opticssource = ColumnDataSource(opticsData_df)
        reachabilityDiag.circle(
            'x',
            'y',
            source=opticssource,
            color=dict(field='ptColor', transform=color_mapping2),
            line_alpha=0.6,
            fill_alpha=0.6,
            legend_field='cluster'
        )
        reachabilityDiag.line([0, len(opticsData_df['ptColor'])], [2, 2], line_width=2, color="black", line_dash="dashed")
        reachabilityDiag.y_range = Range1d(-1, 10)

        LABELS = ["DBSCAN Clustering", "OPTICS Clustering", "Anomaly Detection"]
        radio_button_group = RadioButtonGroup(labels=LABELS, active=0)
        radioGroup_js = CustomJS(args=dict(datasource=datasource, opticssource=opticssource), code="""
            const x = datasource.data.x
            const y = datasource.data.y
            const image = datasource.data.image
            const medoidBold = datasource.data.medoidBold
            const cluster = datasource.data.cluster
            const anomDet = datasource.data.anomDet
            const imgind = datasource.data.imgind
            const dbscan_backgroundColor = datasource.data.dbscan_backgroundColor
            const anom_backgroundColor = datasource.data.anom_backgroundColor
            const optics_backgroundColor = datasource.data.optics_backgroundColor

            const opticsClust = opticssource.data.clusterForScatterPlot

            let ptColor = null
            let backgroundColor = null

            if (cb_obj.active==0){
                ptColor = cluster
                backgroundColor = dbscan_backgroundColor
            }
            else if (cb_obj.active==1){
                ptColor = opticsClust
                backgroundColor = optics_backgroundColor
            }
            else{
                ptColor = anomDet
                backgroundColor = anom_backgroundColor
            }
            datasource.data = { x, y, image, cluster, medoidBold, ptColor, anomDet, imgind, backgroundColor, dbscan_backgroundColor, anom_backgroundColor, optics_backgroundColor}
        """)
        radio_button_group.js_on_change("active", radioGroup_js)

        self.viewResults = column(plot_figure, p, imgsPlot, row(cols, toggl, radio_button_group), reachabilityDiag)

    def fullVisualize(self):
        self.genUMAP()
        self.genABOD()
        self.genLabels()
        self.genHTML()

    def updateLabels(self):
        self.genLabels()
        self.genHTML()

    def userSave(self):
        save(self.viewResults)

    def userShow(self):
        from IPython.display import display, HTML
        display(HTML("<style>.container { width:100% !important; }</style>"))
        display(HTML("<style>.output_result { max-width:100% !important; }</style>"))
        display(HTML("<style>.container { height:100% !important; }</style>"))
        display(HTML("<style>.output_result { max-height:100% !important; }</style>"))
        from bokeh.io import output_notebook
        output_notebook()
        show(self.viewResults)
