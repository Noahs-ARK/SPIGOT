#!/bin/bash
set -euo pipefail  # strict mode
IFS=$'\n\t'
git submodule add https://github.com/clab/dynet.git dynet

BASE_DIR=$(cd $(dirname "$0"); pwd)
LOCAL_DEPS_DIR="${BASE_DIR}/deps/local"

rm -rf "${LOCAL_DEPS_DIR}"

cd deps
mkdir -p "${LOCAL_DEPS_DIR}/lib"
mkdir -p "${LOCAL_DEPS_DIR}/include"

# Add the local lib folder to the search path. This is important because glog
# searches for libgflags.
export LD_LIBRARY_PATH="${LOCAL_DEPS_DIR}/lib${LD_LIBRARY_PATH+:$LD_LIBRARY_PATH}"

# Set to true to run autoconf and automake (sometimes necessary in Mac OS-X).
RUN_AUTOTOOLS=true

# Install gflags.
echo ""
echo "Installing gflags..."
tar -zxf gflags-2.0-no-svn-files.tar.gz
cd gflags-2.0
if ${RUN_AUTOTOOLS}
then
    # rm missing
    aclocal
    autoconf
    automake --add-missing
fi
./configure --prefix=${LOCAL_DEPS_DIR} && make && make install
cd ..
echo "Done."


# Install glog.
echo ""
echo "Installing glog..."
tar -zxf glog.tar.gz
cd glog
	./autogen.sh
./configure --prefix=${LOCAL_DEPS_DIR} && make && make install
cd ..
echo "Done."
# Install eigen
echo ""
echo "Installing Eigen..."
rm -rf eigen
hg clone https://bitbucket.org/eigen/eigen/ -r 346ecdb
cd eigen
mkdir -p build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX="${LOCAL_DEPS_DIR}"
make install
cd ../..
rm -rf eigen
echo "Done."

