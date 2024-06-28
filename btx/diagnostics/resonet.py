import argparse
import os
import numpy as np
from matplotlib import pyplot as plt


class ResoNet:
    """
    Class to handle ResoNet post-processing.
    """
    def __init__(self, input_file, output_dir):
        self.input_file = input_file
        self.output_name = self.input_file.split('.')[0]
        self.output_dir = output_dir

    def save_trace(self):
        with open(self.input_file, 'r') as f:
            evt_list = []
            reso_list = []
            for ln in f:
                if ln.startswith("Resolution"):
                    evt_list.append(ln[0:].split()[6].split('/')[0])
                    reso_list.append(ln[0:].split()[2])
        self.reso = np.column_stack((np.array(evt_list).astype('int'),
                                     np.array(reso_list).astype('float')))
        np.save(os.path.join(self.output_dir, f"{self.output_name}.npy"), self.reso)

    def visualize_resolution(self, savefig=False):
        fig = plt.figure(figsize=(6, 3), dpi=180, )
        gs = fig.add_gridspec(1, 3)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.scatter(self.reso[:, 0], self.reso[:, 1], s=1, c='gray')
        ax1.set_xlabel('event #')
        ax1.set_ylabel('Resolution (A)')
        ax1.set_title('ResoNet inference')
        ax2 = fig.add_subplot(gs[0, 2], sharey=ax1)
        ax2.hist(self.reso[:, 1], bins=50, log=True, orientation='horizontal', color='gray')
        ax2.set_xlabel('# of events')
        if savefig:
            plt.savefig(os.path.join(self.output_dir, f"{self.output_name}.png"))


def main():
    """
    Perform ResoNet post-processing and display results.
    """
    params = parse_input()
    os.makedirs(os.path.join(params.outdir, 'figs'), exist_ok=True)
    resonet = ResoNet(input_file=params.infile, output_dir=params.outdir)
    resonet.save_trace()
    resonet.visualize_resolution(savefig=True)


def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', help="Input file", required=True, type=str)
    parser.add_argument('-o', '--outdir', help='Output directory', required=True, type=str)
    return parser.parse_args()


if __name__ == '__main__':
    main()
