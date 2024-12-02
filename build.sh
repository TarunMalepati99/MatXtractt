#!/bin/bash

rm -rf build
mkdir build
cd build
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 cmake ..
make -j
