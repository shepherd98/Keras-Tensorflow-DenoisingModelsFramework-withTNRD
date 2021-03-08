#!/bin/bash
bazel build --jobs 8  --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --config opt //tensorflow/core/user_ops:icg.so


