unset LD_LIBRARY_PATH
unset PYTHONPATH
source /cds/sw/ds/ana/conda2/manage/bin/psconda.sh
conda deactivate
conda activate ps-4.5.10
RELDIR=/cds/home/a/apeck/lcls2   
export PATH=$RELDIR/install/bin:${PATH}
pyver=$(python -c "import sys; print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))")
export PYTHONPATH=$RELDIR/install/lib/python$pyver/site-packages
# for procmgr
export TESTRELDIR=$RELDIR/install

# needed by Ric to get correct libfabric man pages
export MANPATH=$CONDA_PREFIX/share/man${MANPATH:+:${MANPATH}} 

export PYTHONPATH="${PYTHONPATH}:/cds/home/a/apeck/btx"
