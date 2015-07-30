#!/usr/bin/env bash

# This script downloads and unpacks a specific version of Eigen library
# http://eigen.tuxfamily.org/

set -e

cd $(dirname $0)

# settings start
version="3.2.4"
dst_folder="eigen3"
# settings end

if [ -d "eigen3" ]; then
    echo "Folder eigen3 already exists. Exiting"
else
    fname="$version.tar.bz2"
    if [ ! -f "eigen.$fname" ]; then
      if ! wget -T 10 -t 3 "http://bitbucket.org/eigen/eigen/get/$fname" -O eigen.$fname; then
          echo "Failed to download $fname"
          rm -f eigen.$fname
          exit 1
      fi
    fi

    tar -xvjf eigen.$fname  || exit 1;
    folder_name=$(tar -tjf eigen.$fname | sed 's/^[./]*//' | head -n 1 | cut -d/ -f 1)
    mv -v $folder_name $dst_folder
    rm -f eigen.$fname
fi

cd faster-rnnlm
make -j
