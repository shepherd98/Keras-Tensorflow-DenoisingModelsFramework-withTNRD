# Description:
# Custom operators for ICGVN module.

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "tf_kernel_library",
)
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

tf_custom_op_library(
    name = "icg.so",
    srcs = ["ops/ffttools.cc",
            "ops/ffttools.h",
            "common/definitions.h",
            "ops/activations.h",
            "ops/activations.cc",
            ],
    gpu_srcs = ["kernels/ffttools.cu.cc",
		"ops/ffttools.h",
                "common/definitions.cu.cc",
                "common/definitions.h",
                "ops/activations.h",
                "kernels/activations.cu.cc",
                ],
)

tf_custom_op_library(
    name = "INRF2d.so",
    srcs = ["INRF2d.cc",
            "INRF2d.h",
            ],
    gpu_srcs = ["INRF2d.cu.cc",
		"INRF2d.h",
                ],
)


tf_custom_op_library(
    name = "INRF2d_gpu0.so",
    srcs = ["INRF2d_gpu0.cc",
            "INRF2d_gpu0.h",
            ],
    gpu_srcs = ["INRF2d_gpu0.cu.cc",
		"INRF2d_gpu0.h",
                ],
)

tf_custom_op_library(
    name = "INRF2d_gpu1.so",
    srcs = ["INRF2d_gpu1.cc",
            "INRF2d_gpu1.h",
            ],
    gpu_srcs = ["INRF2d_gpu1.cu.cc",
		"INRF2d_gpu1.h",
                ],
)

tf_custom_op_library(
    name = "INRF2d_gpu.so",
    srcs = ["INRF2d_gpu.cc",
            "INRF2d_gpu.h",
            ],
    gpu_srcs = ["INRF2d_gpu.cu.cc",
		"INRF2d_gpu.h",
                ],
)

tf_custom_op_library(
    name = "test_INRF2d_gpu.so",
    srcs = ["test_INRF2d_gpu.cc",
            "test_INRF2d_gpu.h",
            ],
    gpu_srcs = ["test_INRF2d_gpu.cu.cc",
		"test_INRF2d_gpu.h",
                ],
)


tf_custom_op_library(
    name = "INRF2d_cpu.so",
    srcs = ["INRF2d_cpu.cc",
            "INRF2d_cpu.h",
            ],
)

tf_custom_op_library(
    name = "extract_pixels.so",
    srcs = ["ExtractPixels.cc",
            "ExtractPixels.h",
            ],
    gpu_srcs = ["ExtractPixels.cu.cc",
		"ExtractPixels.h",
                ],
)

tf_custom_op_library(
    name = "threshold.so",
    srcs = ["Threshold.cc",
            "Threshold.h",
            ],
    gpu_srcs = ["Threshold.cu.cc",
		"Threshold.h",
                ],
)

tf_custom_op_library(
    name = "tents.so",
    srcs = ["tents.cc",
            "tents.h",
            ],
    gpu_srcs = ["tents.cu.cc",
		"tents.h",
                ],
)

tf_custom_op_library(
    name = "relulinear.so",
    srcs = ["relu_linear_layer.cc",
            "relu_linear_layer.h",
            ],
    gpu_srcs = ["relu_linear_layer.cu.cc",
		"relu_linear_layer.h",
                ],
)


tf_custom_op_library(
    name = "test_INRF2d_cpu.so",
    srcs = ["test_INRF2d_cpu.cc",
            "test_INRF2d_cpu.h",
            ],
)

tf_custom_op_library(
    name = "nrf2d_gpu.so",
    srcs = ["nrf2d_gpu.cc",
            "nrf2d_gpu.h",
            ],
    gpu_srcs = ["nrf2d_gpu.cu.cc",
		"nrf2d_gpu.h",
                ],
)
