#!/bin/bash

usage()
{
cat << EOF
$(basename "$0"):
    Script to prepare btx directories and eLog jobs for an experiment.
    OPTIONS:
      -a | --account
         Optionally specify an account to run with.
      -e | --experiment
         Experiment to setup btx for.
      -h | --help
         Display this help message.
      -n | --ncores
         Number of cores to use for btx jobs.
      -q | --queue
         Which S3DF queue to run on.
      -r | --reservation
         Optionally specifiy a reservation to run on.
      -v | --verbose
         Optionally enable verbose logging.
      -w | --workflow
         Specify which workflow to setup.
EOF
}

ARGS=()
while [[ $# -gt 0 ]]
do
    arg="$1"

    case $arg in
        -a|--account)
            ACCOUNT="$2"
            shift
            shift
            ;;
        -e|--experiment)
            EXP="$2"
            shift
            shift
            ;;
        -h|--help)
            usage
            exit
            ;;
        -n|--ncores)
            CORES="$2"
            shift
            shift
            ;;
        -q|--queue)
            QUEUE="$2"
            shift
            shift
            ;;
        -r|--reservation)
            RESERVATION="$2"
            shift
            shift
            ;;
        -v|--verbose)
            VERBOSE=1
            shift
            shift
            ;;
        -w|--workflow)
            WORKFLOW="$2"
            shift
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            shift
            ;;
    esac
done
set -- "${ARGS[@]}"

if [[ -z ${EXP} ]]; then
    echo "You must provide an experiment name!"
    usage
    exit 1
fi
if [[ -z ${QUEUE} ]]; then
    echo "No queue provided, defaulting to milano. Can abort with ctrl-c."
    sleep 0.5
fi
if [[ -z ${CORES} ]]; then
    echo "Number of cores not provided. Will use 64."
fi
if [[ -z ${WORKFLOW} ]]; then
    echo "No workflow provided. Defaulting to SFX experiment."
fi

ACCOUNT=${ACCOUNT:="lcls:${EXP}"}
CORES=${CORES:=64}
QUEUE=${QUEUE:="milano"}
RESERVATION=${RESERVATION:="lcls:onshift"}
WORKFLOW=${WORKFLOW:="sfx"}

BTX_DIR="/sdf/group/lcls/ds/tools/btx/stable"
EXP_DIR="/sdf/data/lcls/ds/${EXP:0:3}/${EXP}"

if [ ! -d "${EXP_DIR}" ]; then
    echo "Experiment folder does not exist yet."
    exit 1
fi


umask 002
echo "Preparing btx work directories."
mkdir -p ${EXP_DIR}/scratch/btx/yamls
mkdir -p ${EXP_DIR}/scratch/btx/launchpad
cp ${BTX_DIR}/config/config.yaml ${EXP_DIR}/scratch/btx/yamls
sed -i "s|root_dir: ''|root_dir: \'${EXP_DIR}/scratch/btx/\'|g" ${EXP_DIR}/scratch/btx/yamls/config.yaml
sed -i "s|exp: ''|exp: \'${EXP}\'|g" ${EXP_DIR}/scratch/btx/yamls/config.yaml
sed -i "s|queue: ''|queue: \'${QUEUE}\'|g" ${EXP_DIR}/scratch/btx/yamls/config.yaml
sed -i "s|account: ''|account: \'${ACCOUNT}\'|g" ${EXP_DIR}/scratch/btx/yamls/config.yaml

chmod -R o+r ${EXP_DIR}/scratch/btx
chmod -R o+w ${EXP_DIR}/scratch/btx

echo "Preparing eLog jobs."
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh
PARAMS="-a ${ACCOUNT} -e ${EXP} -n ${CORES} -q ${QUEUE} -w ${WORKFLOW}"

if [[ $RESERVATION ]]; then
    PARAMS+=" -r ${RESERVATION}"
fi
if [[ $VERBOSE ]]; then
    PARAMS+=" --verbose"
fi

python ${BTX_DIR}/scripts/setup_btx.py $PARAMS

echo "Finished setup"
