#!/bin/bash

sed -i -e '/USE_CUDNN/s/#//' Makefile.config
sed -i -e '/WITH_PYTHON_LAYER/s/#//' Makefile.config
sed -i -e '/BLAS := /s/atlas/open/' Makefile.config
sed -i -e '/BLAS_INCLUDE :=/s#.*#BLAS_INCLUDE :=/home/xyz/code/OpenBLAS/include #' Makefile.config
sed -i -e '/BLAS_LIB :=/s#.*#BLAS_LIB :=/home/xyz/code/OpenBLAS/lib#' Makefile.config
sed -i -e '/# USE_LEVELDB := 0/s/#//' Makefile.config
sed -i -e '/# USE_LMDB := 0/s/#//' Makefile.config
sed -i -e '/# OPENCV_VERSION := 3/s/#//' Makefile.config
sed -i -e '/INCLUDE_DIRS :=/s#$# /usr/include/hdf5/serial#' Makefile.config
sed -i -e '/LIBRARY_DIRS :=/s#$# /usr/lib/x86_64-linux-gnu/hdf5/serial#' Makefile.config

