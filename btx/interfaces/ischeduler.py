import shutil
import os

""" Helper methods for job scheduling. """

class JobScheduler:

    def __init__(self, jobfile, logdir='./', jobname='btx',
                 queue='ffbh3q', ncores=1, time='0:30:00'):
        self.manager = 'SLURM'
        self.jobfile = jobfile
        self.logdir = logdir
        self.jobname = jobname
        self.queue = queue
        self.ncores = ncores
        self.time = time

    def _find_python_path(self):
        """ Determine the relevant python path. """
        pythonpath=None
        possible_paths = ["/cds/sw/ds/ana/conda2/inst/envs/ps-4.5.10/bin/python"]
    
        try:
            pythonpath = os.environ['WHICHPYTHON']
        except KeyError:
            pass
    
        for ppath in possible_paths:
            if os.path.exists(ppath):
                pythonpath = ppath

        return pythonpath            

    def write_header(self):
        """ Write resource specification to submission script. """
        if(self.manager == 'SLURM'):
            template = ("#!/bin/bash\n"
                        "#SBATCH -p {queue}\n"
                        "#SBATCH --job-name={jobname}\n"
                        "#SBATCH --output={output}\n"
                        "#SBATCH --error={error}\n"
                        "#SBATCH --ntasks={ncores}\n"
                        "#SBATCH --time={time}\n"
                        "#SBATCH --exclusive\n\n"
                        "source /cds/home/a/apeck/btx/btx/conversion/env_psana2.sh\n")
        else:
            raise NotImplementedError('JobScheduler not implemented.')

        context = {
            "queue": self.queue,
            "jobname": self.jobname,
            "output": os.path.join(self.logdir, f"{self.jobname}.out"),
            "error": os.path.join(self.logdir, f"{self.jobname}.err"),
            "ncores": self.ncores,
            "time": self.time
        }

        with open(self.jobfile, 'w') as jfile:
            jfile.write(template.format(**context))

    def _write_dependencies(self, dependencies):
        """ Source dependencies."""
        dep_paths = ""
        if "crystfel" in dependencies:
            dep_paths += "export PATH=/cds/sw/package/crystfel/crystfel-dev/bin:$PATH\n"
        if "ccp4" in dependencies:
            dep_paths += "source /cds/sw/package/ccp4/ccp4-7.0/setup-scripts/ccp4.setup-sh\n"
        if "phenix" in dependencies:
            dep_paths += "source /cds/sw/package/phenix-1.13-2998/phenix_env.sh\n"
        if "xds" in dependencies:
            dep_paths += "export PATH=/reg/common/package/XDS-INTEL64_Linux_x86_64:$PATH\n"
        if "xgandalf" in dependencies:
            dep_paths += "export PATH=/reg/g/cfel/crystfel/indexers/xgandalf/include/:$PATH\n"
            dep_paths += "export PATH=/reg/g/cfel/crystfel/indexers/xgandalf/include/eigen3/Eigen/:$PATH"
        dep_paths += "\n"
        
        with open(self.jobfile, 'a') as jfile:
            jfile.write(dep_paths)

    def write_main(self, application, dependencies=[]):
        """ Write application and source requested dependencies. """
        if dependencies:
            self._write_dependencies(dependencies)

        pythonpath = self._find_python_path()
        with open(self.jobfile, 'a') as jfile:
            jfile.write(application.replace("python", pythonpath))

    def submit(self):
        """ Submit to queue. """
        os.system(f"sbatch {self.jobfile}")

    def clean_up(self):
        """ Add a line to delete submission file."""
        with open(self.jobfile, 'a') as jfile:
            jfile.write(f"if [ -f {self.jobfile} ]; then rm -f {self.jobfile}; fi")
