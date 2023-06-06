#!bin/bash

set -e
 
SUITE_NAME=test_nl2sql

PATH_HELM=/home/data/gaodawei.gdw/helm
PATH_WORKDIR=${PATH_HELM}/evaluation


cd ${PATH_WORKDIR}

export PYTHONPATH=$PYTHONPATH:${PATH_HELM}/src/

/usr/bin/python ${PATH_HELM}/src/helm/benchmark/presentation/summarize.py \
        --suite ${SUITE_NAME}

/usr/bin/python ${PATH_HELM}/src/helm/benchmark/server.py -o ${PATH_WORKDIR}/benchmark_output
