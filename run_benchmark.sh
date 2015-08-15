#!/usr/bin/env bash

set -e
set -o nounset

if [ $# -ne 2 ]; then
    echo "Usage: $0 <hidden size> <num threads>"
    exit 1
fi

basedir=$(dirname $0)/benchmarks
data=$basedir/simple-examples/data
hidden_size=$1
threads=$2

taskset_cmd="taskset -c $(seq -s, 0 $(( $threads - 1 )))"
mkdir -p $basedir

if ! type svn ; then
    echo "svn is required"
    exit 1
fi

function fat_echo() {
    echo "############################################"
    echo "########## $1"
}

function wget_or_curl() {
  [ $# -eq 2 ] || { echo "Usage: wget_or_curl <url> <fpath>" && exit 1; }
  if type wget &> /dev/null; then
    local download_cmd="wget -T 10 -t 3 -O"
  else
    local download_cmd="curl -L -o"
  fi
  $download_cmd "$2" "$1"
}

function run_test() {
    time $taskset_cmd $1 -rnnlm $basedir/models/$2 -train $data/ptb.train.txt -valid $data/ptb.valid.txt -hidden $hidden_size -threads $threads ${3:-}
    $1 -rnnlm $basedir/models/$2 -test $data/ptb.test.txt -nce-accurate-test 1 2>&1 > /dev/null | grep "Test entropy" | cat
}

fat_echo "Downloading Penn Tree Bank corpora"
if [ ! -d "$basedir/simple-examples" ]; then
    pushd $basedir > /dev/null
    wget_or_curl http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz simple-examples.tgz
    tar -xf simple-examples.tgz
    rm simple-examples.tgz
    popd > /dev/null
fi


fat_echo "Downloading and building C-RNNLM from rnnlm.org"
if [ ! -f "$basedir/crnnlm/rnnlm" ]; then
    rm -rf $basedir/crnnlm
    mkdir -p $basedir/crnnlm
    (
        cd $basedir/crnnlm
        wget_or_curl https://f25ea9ccb7d3346ce6891573d543960492b92c30.googledrive.com/host/0ByxdPXuxLPS5RFM5dVNvWVhTd0U/rnnlm-0.4b.tgz rnnlm-0.4b.tgz
        tar -xf rnnlm-0.4b.tgz
        cd rnnlm-0.4b
        sed -i -- 's/x86_64-linux-g++-4.6/g++/g' makefile
        make
        cd ..
        ln -s rnnlm-0.4b/rnnlm
    )
fi

fat_echo "Downloading and building RNNLM-HS from kaldi svn"
if [ ! -f "$basedir/rnnlm-hs-0.1b/rnnlm" ]; then
    mkdir -p $basedir/rnnlm-hs-0.1b
    (
        cd $basedir/rnnlm-hs-0.1b
        svn checkout https://svn.code.sf.net/p/kaldi/code/trunk/tools/rnnlm-hs-0.1b/ .
        make
    )
fi

fat_echo "Building Faster-RNNLM"
$(dirname $0)/build.sh

rm -rf $basedir/models
mkdir -p $basedir/models

fat_echo "Training Faster RNNLM on ptb"
run_test $(dirname $0)/faster-rnnlm/rnnlm fasterrnnlm

fat_echo "Training Faster RNNLM on ptb (NCE mode)"
run_test $(dirname $0)/faster-rnnlm/rnnlm fasterrnnlm-nce "-nce 15"

fat_echo "Training RNNLM-HS on ptb"
run_test $basedir/rnnlm-hs-0.1b/rnnlm rnnlm-hs

fat_echo "Training C-RNNLM on ptb"
run_test $basedir/crnnlm/rnnlm crnnlm
$basedir/crnnlm/rnnlm -rnnlm $basedir/models/crnnlm -test $data/ptb.test.txt  2>&1 | awk '$0 ~ /PPL/ {print "Test entropy", log($3) / log(2)}'
