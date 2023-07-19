import shutil
import logging
import os

""" Helper methods for job scheduling. """

logger = logging.getLogger(__name__)

class JobScheduler:

    def __init__(self, jobfile, logdir='./', jobname='btx',
                 account='lcls', queue='ffbh3q', ncores=1, time='0:30:00'):
        self.manager = 'SLURM'
        self.jobfile = jobfile
        self.logdir = logdir
        self.jobname = jobname
        self.account = account
        self.queue = queue
        self.ncores = ncores
        self.time = time
        self._data_systems_management()

    def _data_systems_management(self):
        """ List the Data Systems group folder paths. """
        try:
            computing_facility = os.environ['FACILITY']
        except KeyError:
            print('FACILITY environment variable not defined.')

        if(computing_facility == 'SLAC'):
            self.ana_conda_dir = '/cds/sw/ds/ana/'
            self.ana_tools_dir = '/cds/sw/package/'
        elif(computing_facility == 'SRCF_FFB'):
            self.ana_conda_dir = '/cds/sw/ds/ana/'
            self.ana_tools_dir = '/cds/sw/package/'
        elif(computing_facility == 'S3DF'):
            self.ana_conda_dir = '/sdf/group/lcls/ds/ana/sw/'
            self.ana_tools_dir = '/sdf/group/lcls/ds/tools/'
        else:
            raise NotImplementedError('Unknown computing facility.')

        self.ana_conda_manage = f'{self.ana_conda_dir}conda1/manage/bin/'
        self.ana_conda_bin = f'{self.ana_conda_dir}conda1/inst/envs/ana-4.0.47-py3/bin/'

    def _find_python_path(self):
        """ Determine the relevant python path. """
        pythonpath=None
        possible_paths = [f"{self.ana_conda_bin}python"]
    
        try:
            pythonpath = os.environ['WHICHPYTHON']
        except KeyError:
            pass
    
        for ppath in possible_paths:
            if os.path.exists(ppath):
                pythonpath = ppath
                if self.ncores > 1:
                    pythonpath = f"{os.path.split(ppath)[0]}/mpirun -n {self.ncores} {ppath}"

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
                        "#SBATCH --exclusive\n\n")
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

        facility = os.environ['FACILITY']
        if self.account is not None and facility == 'S3DF':
            with open(self.jobfile, 'a') as jfile:
                jfile.write(f"#SBATCH -A {self.account}\n\n")

    def _write_dependencies(self, dependencies):
        """ Source dependencies."""
        dep_paths = ""
        if "psana" in dependencies:
            dep_paths += f"source {self.ana_conda_manage}psconda.sh \n"
        if "crystfel" in dependencies:
            if (os.environ['FACILITY'] == 'S3DF'):
                dep_paths += f"export PATH={self.ana_tools_dir}crystfel/0.10.2/bin:$PATH\n"
            else:
                dep_paths += f"export PATH={self.ana_tools_dir}crystfel/crystfel-dev/bin:$PATH\n"
        if "mosflm" in dependencies:
            if(os.environ['FACILITY'] == 'S3DF'):
                dep_paths += f"export PATH={self.ana_tools_dir}:$PATH\n"
            else:
                dep_paths += f"export PATH={self.ana_tools_dir}autosfx:$PATH\n"
        if "ccp4" in dependencies:
            if (os.environ['FACILITY'] == 'S3DF'):
                dep_paths += f"source {self.ana_tools_dir}ccp4-8.0/bin/ccp4.setup-sh\n"
            else:
                dep_paths += f"source {self.ana_tools_dir}ccp4/ccp4-8.0/bin/ccp4.setup-sh\n"
        if "phenix" in dependencies:
            dep_paths += f"source {self.ana_tools_dir}phenix-1.13-2998/phenix_env.sh\n"
        if "xds" in dependencies:
            dep_paths += f"export PATH={self.ana_tools_dir}XDS-INTEL64_Linux_x86_64:$PATH\n"
        if "xgandalf" in dependencies:
            dep_paths += "export PATH=/reg/g/cfel/crystfel/indexers/xgandalf/include/:$PATH\n"
            dep_paths += "export PATH=/reg/g/cfel/crystfel/indexers/xgandalf/include/eigen3/Eigen/:$PATH"
        dep_paths += "\n"
        
        with open(self.jobfile, 'a') as jfile:
            jfile.write(dep_paths)
            if 'SIT_PSDM_DATA' in os.environ:
                jfile.write(f"export SIT_PSDM_DATA={os.environ['SIT_PSDM_DATA']}\n")

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
        logger.info(f"sbatch {self.jobfile}")

    def clean_up(self):
        """ Add a line to delete submission file."""
        with open(self.jobfile, 'a') as jfile:
            jfile.write(f"if [ -f {self.jobfile} ]; then rm -f {self.jobfile}; fi")
