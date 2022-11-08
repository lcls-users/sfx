import os
import numpy as np
import shutil
from glob import glob

class eLogInterface:

    def __init__(self, setup):
        self.exp = setup.exp
        self.root_dir = setup.root_dir

    def update_summary(self):
        for run in self.list_processed_runs():
            self.update_run(run)
        for sample in self.list_processed_samples():
            self.update_sample(sample)

    def update_run(self, run):
        os.makedirs(self.target_dir(subdir=f"runs/{run}"), exist_ok=True)
        self.update_png(run, 'geom', 'powder')
        self.update_png(run, 'powder', 'powder')
        self.update_png(run, 'powder', 'stats')
        self.update_html(['geom', 'powder', 'stats'], f"runs/{run}/")

    def update_sample(self, sample):
        os.makedirs(self.target_dir(subdir=f"samples/stats_{sample}"), exist_ok=True)
        self.update_png(sample, 'index', 'peakogram')
        self.update_png(sample, 'index', 'cell')
        self.update_png(sample, 'merge', 'Rsplit')
        self.update_png(sample, 'merge', 'CCstar')
        self.update_html(['peakogram', 'cell', 'Rsplit', 'CCstar'], f"samples/stats_{sample}/")
        self.update_uglymol(sample)

    def update_uglymol(self, sample):
        source_dir = f'{self.source_dir(subdir=f"solve/{sample}/")}'
        target_dir = f'{self.target_dir(subdir=f"samples/maps_{sample}/")}'
        if os.path.isfile(f'{source_dir}dimple.out'):
            if os.path.isdir(target_dir):
                shutil.rmtree(target_dir)
            shutil.copytree(self.btx_dir(subdir='misc/uglymol/'), target_dir)
        for filetype in ['pdb', 'mtz']:
            if os.path.isfile(f'{source_dir}final.{filetype}'):
                os.makedirs(target_dir, exist_ok=True)
                shutil.copy2(f'{source_dir}final.{filetype}', f'{target_dir}final.{filetype}')

    def update_html(self, png_list, subdir):
        with open(f'{self.target_dir(subdir=f"{subdir}")}report.html', 'w') as hfile:
            hfile.write('<!doctype html><html><head></head><body>')
            for png in png_list:
                if os.path.isfile(f'{self.target_dir(subdir=f"{subdir}")}{png}.png'):
                    hfile.write(f"<img src='{png}.png' width=1000><br>")
            hfile.write('</body></html>')

    def update_png(self, item, task, image):
        if task == 'powder':
            source_subdir = 'powder/figs/'
            target_subdir = f'runs/{item}/'
            source_filename = f'{image}_{item}.png'
        elif task == 'geom':
            source_subdir = 'geom/figs/'
            target_subdir = f'runs/{item}/'
            source_filename = f'{item}.png'
        elif task == 'index':
            source_subdir = 'index/figs/'
            target_subdir = f'samples/stats_{item}/'
            source_filename = f'{image}_{item}.png'
        elif task == 'merge':
            source_subdir = f'merge/{item}/figs/'
            target_subdir = f'samples/stats_{item}/'
            source_filename = f'{item}_{image}.png'
        source_path = f'{self.source_dir(subdir=f"{source_subdir}")}{source_filename}'
        target_path = f'{self.target_dir(subdir=f"{target_subdir}")}{image}.png'
        if os.path.isfile(source_path):
            shutil.copy2(source_path, target_path)

    def btx_dir(self, subdir=''):
        import btx
        return f'{os.path.dirname(btx.__file__)}/{subdir}'

    def source_dir(self, subdir=''):
        return f'{self.root_dir}/{subdir}'

    def target_dir(self, subdir=''):
        import os
        return f'{os.environ.get("SIT_PSDM_DATA")}/{self.exp[:3]}/{self.exp}/stats/summary/{subdir}'

    def list_processed_runs(self):
        run_list = []
        for task in ['geom', 'powder']:
            for png in glob(f'{self.source_dir(subdir=f"{task}/figs/")}*'):
              run_list.append(png.split('/')[-1].split('_')[-1].split('.')[0])
        return np.unique(run_list)

    def list_processed_samples(self):
        sample_list = []
        for png in glob(f'{self.source_dir(subdir="index/figs/")}*.png'):
            sample_list.append(png.split('/')[-1].split('_')[-1].split('.')[0])
        for task in ['merge', 'solve']:
            for sample in glob(f'{self.source_dir(subdir=f"{task}/")}*'):
                if os.path.isdir(sample):
                    sample_list.append(sample.split('/')[-1])
        return np.unique(sample_list)