import numpy as np
import scipy as sp
import os,re,sys,optparse,glob,time,datetime,collections, socket
import os.path as osp
import TimeTool as TT
import psana
import matplotlib as mpl
import matplotlib.pyplot as plt
from psana import *

def relative_time(edge_pos,a,b,c):
    """
    Translate edge position into fs (for cxij8816)
     
    from docs >> fs_result = a + b*x + c*x^2, x is edge position
    """
    x = edge_pos
    tt_correction = a + b*x + c*x**2
    return  tt_correction

def rel_time(edge_pos,model):
    """
    Translate edge position into time
    
    edge_pos= from TT analysis or psana (see ttdata.position_pixel() or ds.env().epicsStore().value('CXI:TIMETOOL:FLTPOS'))
    model= output from 'TTcalib' 
     
    from docs >> ns_result = a + b*x + c*x^2, x is edge position
    """
    if len(model) == 2:
        a = model[1]
        b = model[0]
        c = 0
    elif len(model) ==3:
        a = model[2]
        b = model[1] 
        c = model[0]
    x = edge_pos
    tt_correction = a + b*x + c*x**2
    return  tt_correction


def absolute_time(rel_time, nom_time):
    """
    Calculate actual delay from nominal time and TT correction
     
    """

    delay = -(nom_time + rel_time)*1e06
    return delay

def TTcalib(calib_run, exp, beamline, make_plot=False, poly=2, parallel=False):
    """
    Calibration of time tool:
    roi = region of interest on detector that is being used to determine the edge
    calib_run = run number of the calibration run
    exp = experiment number including instrument (e.g. 'cxilz0720' for run LZ0720 at CXI)
    beamline = 'CXI', 'MFX',.... 
    make_plot: if true automatically plots the calibration curve and fit
    poly = polynomial used for fitting calibration curve, 1 or 2, default 2 as in confluence documentation
    returns model that can be used to determine the delay using the function 'rel_time'
    parallel = set tru to parallelise using MPI
    """

    psana_keyword=f'exp={exp}:run={calib_run}'

    if parallel:
        ds = MPIDataSource(f'{psana_keyword}:smd')
    else:
        ds = psana.DataSource(f'{psana_keyword}:smd')
    
    edge_pos = []
    amp = []
    time = []

    for idx,evt in enumerate(ds.events()):
        edge_pos = np.append(edge_pos, ds.env().epicsStore().value(f'{beamline}:TIMETOOL:FLTPOS'))
        amp = np.append(amp,ds.env().epicsStore().value(f'{beamline}:TIMETOOL:AMPL'))
        time = np.append(time,ds.env().epicsStore().value('LAS:FS45:VIT:FS_TGT_TIME_DIAL'))

    model = np.polyfit(edge_pos, time, int(poly))

    if make_plot:
        if poly == 1:
            model_time = model[0]*edge_pos+model[1]
        elif poly == 2:
            model_time = model[0]**2*edge_pos + model[1]*edge_pos + model[2]
        else:
            print('polynomial not defined, use 1st or 2nd order')
            
   
        plt.plot(edge_pos,time, 'o', color='black',label='edge position')
        plt.plot(edge_pos, model_time, color='red',label = 'calibration fit')
        plt.xlabel('pixel edge')
        plt.ylabel('laser delay')
        plt.legend()

    return model, time, edge_pos, amp

def get_diagnostics(run, direct=True,roi=[]):
    if not direct:
        ttOptions = TT.AnalyzeOptions(get_key='Timetool', eventcode_nobeam=13, sig_roi_y=roi)
        ttAnalyze = TT.PyAnalyze(ttOptions)
    
    ds = psana.DataSource('exp=cxilz0720:run=' + str(run), module=ttAnalyze)
    evr_det = psana.Detector('evr1')
    edge_pos = []
    amp = []
    time = []
    evt = []
    stamp = []
    tt_delay = []
    abs_delay = []

    if not direct:
        ttOptions = TT.AnalyzeOptions(get_key='Timetool', eventcode_nobeam=13, sig_roi_y=roi)
        ttAnalyze = TT.PyAnalyze(ttOptions)
        for idx,evt in enumerate(ds.events()):
            ec = evr_det.eventCodes(evt)
            if ec is None: continue
            ttdata = ttAnalyze.process(evt)
            if ttdata is None: continue
            edge_pos = np.append(edge_pos, ttdata.position_pixel())
            edge_fwhm = np.append(edge_fwhm, ttdata.position_fwhm())
            edge_amp = np.append(edge_amp,ttdata.amplitude())
    if direct:
        for idx,evt in enumerate(ds.events()):
            ec = evr_det.eventCodes(evt)
            if ec is None: continue
            edge_pos = np.append(edge_pos,ds.env().epicsStore().value(f'{beamline}:TIMETOOL:FLTPOS'))
            edge_fwhm = np.append(edge_fwhm,ds.env().epicsStore().value(f'{beamline}:TIMETOOL:FLTPOSFWHM'))
            edge_amp = np.append(edge_amp,ds.env().epicsStore().value(f'{beamline}:TIMETOOL:AMPL'))
            
    return edge_pos, edge_fwhm, edge_amp


def get_delay(run_start, run_end,
              expID, outDir, beamline, event_on,
              roi='30 50', redoTT=False,
              calib_model=[], diagnostics = False,
              parallel=False, TTfilter = True):
    """
    Function to determine the delay using the time tool:
    run_start, run_end: first and last run to analyze
    roi = region of interest on detector that is being used to determine the edge
    expID = experiment number including instrument (e.g. 'cxilz0720' for run LZ0720 at CXI)
    outDir = directory where output should be saved
    beamline = 'MFX', 'CXI' or similar
    redoTT = true if you need to redo the TT analysis (edge finding)
    calib_model = output from time tool calibration (using 'TTcalib'), if not empty TTanalysis is performed again (direct = False)
    diagnostics = set true if you want amplitude and FWHM of TT edge as output
    parallel = set tru to parallelise using MPI
    TTfilter = if true creates a mask to filter out low ampltudes/large FWHM

    saves .txt files linking a delay time to each shot, identified by a stamp 
    each row in the output file: ['644172952-167590310-79638','-1275.255309579068']
    """

    if not os.path.exists(outDir):
        os.makedirs(outDir)
      
    if redoTT:
        ttOptions = TT.AnalyzeOptions(get_key='Timetool', eventcode_nobeam=13, sig_roi_y=roi)
        ttAnalyze = TT.PyAnalyze(ttOptions)

    runs = np.arange(run_start,run_end+1)
    for run_number in runs:
        psana_keyword=f'exp={expID}:run={run_number}'
        print(psana_keyword)

        if not redoTT:
            if parallel:
                ds = MPIDataSource(f'{psana_keyword}:smd')
            else:
                ds = psana.DataSource(f'{psana_keyword}:smd')
        else:
            if parallel:
                ds = MPIDataSource(psana_keyword, module=ttAnalyze)
            else:
                ds = psana.DataSource(psana_keyword, module=ttAnalyze)
        evr_det = psana.Detector('evr0')
        edge_pos = []
        edge_amp = []
        edge_fwhm = []
        time = []
        stamp = []
        tt_delay = []

        for idx,evt in enumerate(ds.events()):
            ec = evr_det.eventCodes(evt)
            if ec is None: continue
            if event_on in ec:
                
                if redoTT:
                    ttdata = ttAnalyze.process(evt)
                    if ttdata is None: continue
                eid = evt.get(EventId)
                fid = eid.fiducials()
                sec = eid.time()[0]
                nsec = eid.time()[1]
                stamp = np.append(stamp, str(sec) + "-" + str(nsec) + "-" + str(fid))
                if not redoTT:
                    edge_pos = np.append(edge_pos, ds.env().epicsStore().value(f'{beamline}:TIMETOOL:FLTPOS'))
                    if len(calib_model) == 0:
                        tt_delay = np.append(tt_delay, ds.env().epicsStore().value(f'{beamline}:TIMETOOL:FLTPOS_PS')/1e6)
                    else:
                        tt_delay = np.append(tt_delay, rel_time(edge_pos[-1], calib_model))
                    edge_fwhm = np.append(edge_fwhm, ds.env().epicsStore().value(f'{beamline}:TIMETOOL:FLTPOSFWHM'))
                    edge_amp = np.append(edge_amp, ds.env().epicsStore().value(f'{beamline}:TIMETOOL:AMPL'))
                else:
                    edge_pos = np.append(edge_pos, ttdata.position_pixel())
                    tt_delay = np.append(tt_delay, rel_time(edge_pos[-1], calib_model))
                    edge_fwhm = np.append(edge_fwhm, ttdata.position_fwhm())
                    edge_amp = np.append(edge_amp,ttdata.amplitude())
                time = np.append(time, ds.env().epicsStore().value('LAS:FS45:VIT:FS_TGT_TIME_DIAL'))
        abs_delay = absolute_time(time, tt_delay)
        
        if TTfilter:
            mask = np.where((edge_amp > 0.05) & (edge_fwhm < 300), True, False)
            stamp = stamp[mask]
            abs_delay = abs_delay[mask]
            edge_pos = edge_pos[mask]
            edge_fwhm = edge_fwhm[mask]
            edge_amp = edge_amp[mask]
            
            
        if diagnostics:
            output = np.column_stack([stamp, abs_delay, edge_pos, edge_fwhm, edge_amp])
        else:
            output = np.column_stack([stamp, abs_delay])
        fn = f'{outDir}/{run_number}'
        #fOn = np.savetxt(fn, output, delimiter=',', fmt = '%s')
        np.save(fn, output)
        
def get_histo(run_start, run_end, expID, outDir, beamline, roi='30 50', redoTT=False, calib_model=[], diagnostics = False):
    delays = []
    for run in np.arange(run_start, run_end+1):
        if not os.path.exists(f'{outDir}/{run}.npy'):
            get_delay(run, run, expID, outDir, beamline, roi='30 50', redoTT=False, calib_model=[], diagnostics = False)
    
        tmp = np.load(f'{outDir}/{run}.npy')
        delays = np.append(delays, tmp[:,1].astype('float'))
        
    counts, bins = np.histogram(delays, bins=30)
    
    plt.hist(bins[:-1], bins, weights=counts)
    plt.ylabel('#shots')
    plt.xlabel('time delay (fs)')
    plt.show()
        
        
        
