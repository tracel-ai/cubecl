#!/bin/bash

set -e

mkdir -p .llvm
cd .llvm

rm -rf llvm llvm-project

git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout llvmorg-21.1.0-rc3

rm -rf build
mkdir -p build

OS_NAME=$(uname -s)
if [[ "$OS_NAME" == "Linux" ]]; then
  OS="linux"
  ARCH="x64"
elif [[ "$OS_NAME" == "Darwin" ]]; then
  OS="macos"
  ARCH="AArch64"
elif [[ "$OS_NAME" == MINGW* ]]; then
  OS="windows"
  ARCH="x64"
else
  echo "Unsupported OS: $OS_NAME"
  exit 1
fi

PLATFORM="${OS}-${ARCH}"

EXTRA_CMAKE_FLAGS=""
if [ "$OS" == "windows" ]; then
  EXTRA_CMAKE_FLAGS="-DCMAKE_CXX_FLAGS=-Wa,-mbig-obj -DCMAKE_C_FLAGS=-Wa,-mbig-obj"
fi

cmake -S llvm -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_BUILD_TESTS=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_BUILD_DOCS=OFF \
  -DLLVM_ENABLE_DOXYGEN=OFF \
  -DLLVM_ENABLE_LTO=OFF \
  -DLLVM_ENABLE_SPHINX=OFF \
  -DLLVM_STATIC_LINK_CXX_STDLIB=ON \
  -DLLVM_ENABLE_ZLIB=OFF \
  -DLLVM_ENABLE_LIBXML2=OFF \
  -DLLVM_ENABLE_LIBEDIT=OFF \
  -DCMAKE_INSTALL_PREFIX=../llvm $EXTRA_CMAKE_FLAGS

ninja -C build install

cd ../llvm

CONFIG="llvm-config"
if [[ "$OS_NAME" == MINGW* ]]; then
  CONFIG="llvm-config.exe"
fi

mv bin/$CONFIG .
rm bin/*
mv $CONFIG bin/
cd ..

if [ "$OS" == "windows" ]; then
  tar -cf $PLATFORM.tar llvm
  7z a -txz $PLATFORM.tar.xz $PLATFORM.tar
  rm $PLATFORM.tar
else
  tar -cJf $PLATFORM.tar.xz llvm
fi

echo "LLVM build and packaging completed successfully!"
