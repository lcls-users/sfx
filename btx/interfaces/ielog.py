import os
import numpy as np
import shutil
from glob import glob
import json
import requests
from typing import Optional, TextIO
import logging

logger = logging.getLogger(__name__)

def elog_report_post(summary_file: str, update_url: Optional[str] = None):
    """! Post a summary file to the eLog's run report section.

    @param summary_file (str) Path to the summary file to post.
    @param update_url (str | None) URL to post to. Attempts to retrieve if None.
    """
    try:
        with open(summary_file, 'r') as f:
            data: dict = json.load(f) # load for files instead of loads
            post_list: list = [ { 'key': f'{key}', 'value': f'{data[key]}' }
                                for key in data ]
            if not update_url:
                url = os.environ.get('JID_UPDATE_COUNTERS')
                if url:
                    requests.post(url, json = post_list)
                else:
                    logger.warning('WARNING: JID_UPDATE_COUNTERS url not found.')
            else:
                requests.post(update_url, json = post_list)
    except FileNotFoundError as err:
        logger.warning('WARNING: No task summary file found!')

def update_summary(summary_file: str, data: dict):
    """! Append summary data to a JSON file.

    @param summary_file (str) Path to the summary file to update.
    @param data (dict) Key/value pairs to be stored in the JSON summary.
    """
    if os.path.isfile(summary_file):
        with open(summary_file, 'r+') as f:
            try:
                summary_data: dict = json.load(f)
            except json.decoder.JSONDecodeError:
                summary_data: dict = {}
    else:
        with open(summary_file, 'w') as f:
            summary_data: dict = {}

    summary_data.update(data)

    with open(summary_file, 'w') as f:
        json.dump(summary_data, f) # dump for files instead of dumps

class eLogInterface:

    def __init__(self, setup):
        self.exp = setup.exp
        self.root_dir = setup.root_dir

    def update_summary(self, plot_type: str = 'pyplot'):
        """! Update the eLog summaries for runs and samples.

        @param plot_type (str) Choose to move `pyplot` or `holoviews` plots.
        """
        for run in self.list_processed_runs():
            self.update_run(run, plot_type=plot_type)
        for sample in self.list_processed_samples():
            self.update_sample(sample)

    def update_run(self, run, plot_type: str = 'pyplot'):
        """! Update eLog summaries by moving files to the stats folder.

        @param plot_type (str) Choose to move `pyplot` or `holoviews` plots.
        """
        os.makedirs(self.target_dir(subdir=f"runs/{run}"), exist_ok=True)
        self.update_img(run, 'geom', 'geom')
        self.update_img(run, 'powder', 'powder')

        if plot_type == 'pyplot':
            self.update_img(run, 'powder', 'stats')
            self.update_html(['geom', 'powder', 'stats'], f"runs/{run}/")
        elif plot_type == 'holoviews':
            self.update_img(run, 'powder', 'stats', ext='html')
            self.update_html(['geom', 'powder', 'stats'],
                             f"runs/{run}/",
                             merge_html=True)

    def update_sample(self, sample):
        os.makedirs(self.target_dir(subdir=f"samples/stats_{sample}"), exist_ok=True)
        self.update_img(sample, 'index', 'peakogram')
        self.update_img(sample, 'index', 'cell')
        self.update_img(sample, 'merge', 'Rsplit')
        self.update_img(sample, 'merge', 'CCstar')
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
                shutil.copyfile(f'{source_dir}final.{filetype}', f'{target_dir}final.{filetype}')

    def update_html(self, img_list, subdir, *, merge_html=False):
        if merge_html:
            hv_head: str = ''
            hv_body: str = ''
            png_body: str = ''
            for img in img_list:
                path = f'{self.target_dir(subdir=f"{subdir}")}{img}'
                if os.path.isfile(f'{path}.html'):
                    hvparser = HTMLParser(path=f'{path}.html')
                    head, body = hvparser.extract_holoviews_img()
                    hv_head += f'{head}\n'
                    hv_body += f'{body}\n'
                elif os.path.isfile(f'{path}.png'):
                    png_body += f"<img src='{img}.png' width=1000>\n<br>\n"

            report_path: str = f'{self.target_dir(subdir=f"{subdir}")}report.html'
            with open(report_path, 'w') as hfile:
                hfile.write(('<!doctype html>\n'
                             '<html>\n'
                             '<head>\n'))
                hfile.write(hv_head)
                hfile.write('</head>')
                hfile.write(hv_body)
                hfile.write(png_body)
                hfile.write('</body></html>')
        else:
            with open(f'{self.target_dir(subdir=f"{subdir}")}report.html', 'w') as hfile:
                hfile.write('<!doctype html><html><head></head><body>')
                for png in img_list:
                    if os.path.isfile(f'{self.target_dir(subdir=f"{subdir}")}{png}.png'):
                        hfile.write(f"<img src='{png}.png' width=1000><br>")
                hfile.write('</body></html>')

    def update_img(self, item, task, image, ext: str = 'png'):
        if task == 'powder':
            source_subdir = 'powder/figs/'
            target_subdir = f'runs/{item}/'
            source_filename = f'{image}_{item}.{ext}'
        elif task == 'geom':
            source_subdir = 'geom/figs/'
            target_subdir = f'runs/{item}/'
            source_filename = f'{item}.{ext}'
        elif task == 'index':
            source_subdir = 'index/figs/'
            target_subdir = f'samples/stats_{item}/'
            source_filename = f'{image}_{item}.{ext}'
        elif task == 'merge':
            source_subdir = f'merge/{item}/figs/'
            target_subdir = f'samples/stats_{item}/'
            source_filename = f'{item}_{image}.{ext}'
        source_path = f'{self.source_dir(subdir=f"{source_subdir}")}{source_filename}'
        target_path = f'{self.target_dir(subdir=f"{target_subdir}")}{image}.{ext}'
        if os.path.isfile(source_path):
            shutil.copyfile(source_path, target_path)

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

class HTMLParser:
    """! Parse HTML files produced by various tasks."""

    def __init__(self, path: str):
        """! Initialize a parser.

        @param path (str) Path to the HTML file to parse.
        """
        self._path = path

    def extract_between(self, *, fptr: TextIO, START_FLAG: str, END_FLAG: str):
        """! Extract the HTML between two flags.

        @param fptr (TextIO) File TextIO wrapper.
        @param START_FLAG (str) Begin extraction at the line after this word.
        @param END_FLAG (str) The line before this word is the final extracted.
        @return html_out (str) The extracted HTML.
        """
        append_data: bool = False
        html_out: str = ''

        for line in fptr:
            if START_FLAG in line:
                append_data = True
                continue
            if END_FLAG in line:
                # Simply break the loop so extract_between can be called again
                # with different flags
                break
            if append_data:
                html_out += line
        return html_out

    def extract_holoviews_img(self):
        """! Extract the <head> from the HTML file produced by HoloViews.

        @return (head, body) (Tuple[str..]) HoloViews <head> and <body> content.
        """
        head: str = ''
        body: str = ''

        with open(self._path, 'r') as f:
            head += self.extract_between(fptr=f,
                                         START_FLAG='<head>',
                                         END_FLAG='</head>')

            body += self.extract_between(fptr=f,
                                         START_FLAG='<body>',
                                         END_FLAG='</body>')
        return head, body
