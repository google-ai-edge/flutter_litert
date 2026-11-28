// Copyright 2018 The TensorFlow Authors.
// Copyright 2019 The MediaPipe Authors.
// Copyright 2025 flutter_litert authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Standalone implementation of MediaPipe's Convolution2DTransposeBias custom op.
// Based on the original MediaPipe implementation but adapted to use only the
// public TFLite C API structures.

#include "transpose_conv_bias.h"
// common.h is already included via transpose_conv_bias.h's platform-specific includes
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// --- Windows CRT heap fix ---
// On Windows, each DLL has its own CRT heap. TfLiteIntArray allocated with
// malloc() in this DLL will be freed by TFLite's free() in a different DLL,
// causing heap corruption. We dynamically resolve TfLiteIntArrayCreate from
// the already-loaded TFLite DLL so allocations use TFLite's heap.
// On Linux/macOS all .so/.dylib share one allocator, so plain malloc is fine.
#if defined(_WIN32)
#include <windows.h>
#include <stdio.h>
typedef TfLiteIntArray* (*TfLiteIntArrayCreateFn)(int size);
static TfLiteIntArrayCreateFn g_intArrayCreate = NULL;
static int g_intArrayCreateResolved = 0;

static TfLiteIntArrayCreateFn ResolveTfLiteIntArrayCreate(void) {
    HMODULE mod = GetModuleHandleA("libtensorflowlite_c-win.dll");
    if (!mod) {
        mod = GetModuleHandleA("tensorflowlite_c-win.dll");
    }

    if (!mod) {
        HMODULE self_mod = NULL;
        if (GetModuleHandleExA(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                    GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                (LPCSTR)(void*)&ResolveTfLiteIntArrayCreate,
                &self_mod) &&
            self_mod) {
            char self_path[MAX_PATH];
            DWORD len = GetModuleFileNameA(self_mod, self_path, MAX_PATH);
            if (len > 0 && len < MAX_PATH) {
                char* slash = strrchr(self_path, '\\');
                if (slash) {
                    *(slash + 1) = '\0';
                    char candidate[MAX_PATH];
                    int wrote = snprintf(
                        candidate,
                        MAX_PATH,
                        "%slibtensorflowlite_c-win.dll",
                        self_path);
                    if (wrote > 0 && wrote < MAX_PATH) {
                        mod = LoadLibraryA(candidate);
                    }
                }
            }
        }
    }

    if (!mod) {
        return NULL;
    }

    return (TfLiteIntArrayCreateFn)GetProcAddress(mod, "TfLiteIntArrayCreate");
}
#endif

static TfLiteIntArray* CreateIntArray(int size) {
#if defined(_WIN32)
    if (!g_intArrayCreateResolved) {
        g_intArrayCreate = ResolveTfLiteIntArrayCreate();
        g_intArrayCreateResolved = 1;
    }
    if (g_intArrayCreate) {
        return g_intArrayCreate(size);
    }

    // Fallback when TfLiteIntArrayCreate is not exported by the runtime DLL.
    // Keep layout identical to TfLiteIntArray and set size explicitly.
    TfLiteIntArray* arr = (TfLiteIntArray*)malloc(sizeof(int) + sizeof(int) * size);
    if (arr) arr->size = size;
    return arr;
#else
    // Linux/macOS/Android share allocator boundaries for this usage.
    TfLiteIntArray* arr = (TfLiteIntArray*)malloc(sizeof(int) + sizeof(int) * size);
    if (arr) arr->size = size;
    return arr;
#endif
}

// Tensor indices for the custom op
#define kDataInputTensor 0
#define kWeightsTensor 1
#define kBiasTensor 2
#define kOutputTensor 0

// Padding types (matching TFLite internal values)
#define PADDING_UNKNOWN 0
#define PADDING_SAME 1
#define PADDING_VALID 2

// Parameters structure matching MediaPipe's TfLiteTransposeConvParams
typedef struct {
    int padding;
    int stride_width;
    int stride_height;
} TransposeConvBiasParams;

// Helper to compute tensor offset for NHWC layout
static inline int Offset(const int* dims, int batch, int height, int width, int channel) {
    return ((batch * dims[1] + height) * dims[2] + width) * dims[3] + channel;
}

// Helper to get max of two ints
static inline int max_int(int a, int b) {
    return a > b ? a : b;
}

// Returns the tensor that `indices` names at `position`, or NULL when the
// position or the tensor index is out of range. The index comes from the
// model file, which can hold any value there, including kTfLiteOptionalTensor
// (-1), so it is checked before `context->tensors` is indexed.
static TfLiteTensor* GetTensorAt(TfLiteContext* context, const TfLiteIntArray* indices,
                                 int position) {
    if (context == NULL || context->tensors == NULL || indices == NULL) return NULL;
    if (position < 0 || position >= indices->size) return NULL;
    const int index = indices->data[position];
    if (index < 0 || (size_t)index >= context->tensors_size) return NULL;
    return &context->tensors[index];
}

// Returns 1 when every dimension is positive and the element count fits in an
// int, which Offset() and the loops in Eval rely on.
static int HasValidElementCount(const TfLiteIntArray* dims) {
    int64_t count = 1;
    for (int i = 0; i < dims->size; ++i) {
        if (dims->data[i] <= 0) return 0;
        count *= dims->data[i];
        if (count > INT32_MAX) return 0;
    }
    return 1;
}

// Parse the custom options to get parameters
static void* Init(TfLiteContext* context, const char* buffer, size_t length) {
    (void)context;

    TransposeConvBiasParams* params = (TransposeConvBiasParams*)malloc(sizeof(TransposeConvBiasParams));
    if (!params) return NULL;

    // Default values - MediaPipe selfie segmentation uses stride 2x2 with SAME padding
    params->padding = PADDING_SAME;
    params->stride_width = 2;
    params->stride_height = 2;

    // The custom_options in MediaPipe models are typically a flexbuffer
    // MediaPipe selfie segmentation uses these standard params
    (void)buffer;
    (void)length;

    return params;
}

static void Free(TfLiteContext* context, void* buffer) {
    (void)context;
    free(buffer);
}

static TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
    // Validate number of inputs/outputs
    if (node->inputs == NULL || node->inputs->size != 3) {
        context->ReportError(context, "Convolution2DTransposeBias requires 3 inputs, got %d",
                           node->inputs ? node->inputs->size : 0);
        return kTfLiteError;
    }
    if (node->outputs == NULL || node->outputs->size != 1) {
        context->ReportError(context, "Convolution2DTransposeBias requires 1 output, got %d",
                           node->outputs ? node->outputs->size : 0);
        return kTfLiteError;
    }

    // Get tensors
    TfLiteTensor* input = GetTensorAt(context, node->inputs, kDataInputTensor);
    TfLiteTensor* weights = GetTensorAt(context, node->inputs, kWeightsTensor);
    TfLiteTensor* bias = GetTensorAt(context, node->inputs, kBiasTensor);
    TfLiteTensor* output = GetTensorAt(context, node->outputs, kOutputTensor);
    if (input == NULL || weights == NULL || bias == NULL || output == NULL) {
        context->ReportError(
            context, "Convolution2DTransposeBias has a missing or out-of-range tensor index");
        return kTfLiteError;
    }
    if (input->dims == NULL || weights->dims == NULL || bias->dims == NULL) {
        context->ReportError(context, "Convolution2DTransposeBias inputs must have a shape");
        return kTfLiteError;
    }

    // Validate dimensions
    if (input->dims->size != 4) {
        context->ReportError(context, "Input must be 4D, got %dD", input->dims->size);
        return kTfLiteError;
    }
    if (weights->dims->size != 4) {
        context->ReportError(context, "Weights must be 4D, got %dD", weights->dims->size);
        return kTfLiteError;
    }
    if (bias->dims->size != 1) {
        context->ReportError(context, "Bias must be 1D, got %dD", bias->dims->size);
        return kTfLiteError;
    }
    if (!HasValidElementCount(input->dims) || !HasValidElementCount(weights->dims) ||
        !HasValidElementCount(bias->dims)) {
        context->ReportError(
            context, "Input, weights and bias dimensions must be positive and fit in an int");
        return kTfLiteError;
    }

    // Validate types - currently only float32
    if (input->type != kTfLiteFloat32) {
        context->ReportError(context, "Input must be float32");
        return kTfLiteError;
    }
    if (weights->type != kTfLiteFloat32) {
        context->ReportError(context, "Weights must be float32");
        return kTfLiteError;
    }
    if (bias->type != kTfLiteFloat32) {
        context->ReportError(context, "Bias must be float32");
        return kTfLiteError;
    }
    if (output->type != kTfLiteFloat32) {
        context->ReportError(context, "Output must be float32");
        return kTfLiteError;
    }

    // Validate channel dimensions match
    // Weights format: OHWI (output_channels, height, width, input_channels)
    if (input->dims->data[3] != weights->dims->data[3]) {
        context->ReportError(context, "Input channels (%d) != weight input channels (%d)",
                           input->dims->data[3], weights->dims->data[3]);
        return kTfLiteError;
    }
    if (weights->dims->data[0] != bias->dims->data[0]) {
        context->ReportError(context, "Weight output channels (%d) != bias size (%d)",
                           weights->dims->data[0], bias->dims->data[0]);
        return kTfLiteError;
    }

    // Get parameters
    TransposeConvBiasParams* params = (TransposeConvBiasParams*)node->user_data;
    int stride_height = 2, stride_width = 2, padding = PADDING_SAME;
    if (params) {
        stride_height = params->stride_height;
        stride_width = params->stride_width;
        padding = params->padding;
    }

    const int filter_height = weights->dims->data[1];
    const int filter_width = weights->dims->data[2];
    const int in_height = input->dims->data[1];
    const int in_width = input->dims->data[2];

    // Calculate output dimensions
    int output_height, output_width;
    int padding_height = 0, padding_width = 0;

    if (padding == PADDING_SAME) {
        padding_height = max_int(0, filter_height - (in_height - 1) % stride_height - 1);
        padding_width = max_int(0, filter_width - (in_width - 1) % stride_width - 1);
    }

    const int64_t output_height64 =
        (int64_t)stride_height * (in_height - 1) + filter_height - padding_height;
    const int64_t output_width64 =
        (int64_t)stride_width * (in_width - 1) + filter_width - padding_width;
    if (output_height64 <= 0 || output_height64 > INT32_MAX ||
        output_width64 <= 0 || output_width64 > INT32_MAX) {
        context->ReportError(context, "Output height or width is out of range");
        return kTfLiteError;
    }
    output_height = (int)output_height64;
    output_width = (int)output_width64;
    // Avoid ResizeTensor ownership/allocation in this custom op path.
    // The model's graph already defines output dims; validate them here.
    if (output->dims == NULL || output->dims->size != 4) {
        context->ReportError(context, "Output must be 4D");
        return kTfLiteError;
    }
    if (output->dims->data[0] != input->dims->data[0] ||
        output->dims->data[1] != output_height ||
        output->dims->data[2] != output_width ||
        output->dims->data[3] != weights->dims->data[0]) {
        context->ReportError(
            context,
            "Unexpected output shape [%d,%d,%d,%d], expected [%d,%d,%d,%d]",
            output->dims->data[0], output->dims->data[1],
            output->dims->data[2], output->dims->data[3],
            input->dims->data[0], output_height, output_width,
            weights->dims->data[0]);
        return kTfLiteError;
    }
    if (!HasValidElementCount(output->dims)) {
        context->ReportError(context, "Output element count must fit in an int");
        return kTfLiteError;
    }

    return kTfLiteOk;
}

static TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
    // Get tensors
    const TfLiteTensor* input = GetTensorAt(context, node->inputs, kDataInputTensor);
    const TfLiteTensor* weights = GetTensorAt(context, node->inputs, kWeightsTensor);
    const TfLiteTensor* bias = GetTensorAt(context, node->inputs, kBiasTensor);
    TfLiteTensor* output = GetTensorAt(context, node->outputs, kOutputTensor);
    if (input == NULL || weights == NULL || bias == NULL || output == NULL) {
        context->ReportError(
            context, "Convolution2DTransposeBias has a missing or out-of-range tensor index");
        return kTfLiteError;
    }
    if (input->dims == NULL || weights->dims == NULL || output->dims == NULL ||
        input->dims->size != 4 || weights->dims->size != 4 || output->dims->size != 4) {
        context->ReportError(context, "Convolution2DTransposeBias tensors must be 4D");
        return kTfLiteError;
    }

    // Get data pointers
    const float* input_data = input->data.f;
    const float* filter_data = weights->data.f;
    const float* bias_data = bias->data.f;
    float* output_data = output->data.f;
    if (input_data == NULL || filter_data == NULL || bias_data == NULL || output_data == NULL) {
        context->ReportError(context, "Convolution2DTransposeBias tensor has no data");
        return kTfLiteError;
    }

    // Get parameters
    TransposeConvBiasParams* params = (TransposeConvBiasParams*)node->user_data;
    int stride_height = 2, stride_width = 2, padding = PADDING_SAME;
    if (params) {
        stride_height = params->stride_height;
        stride_width = params->stride_width;
        padding = params->padding;
    }

    // Get dimensions
    const int batches = input->dims->data[0];
    const int input_height = input->dims->data[1];
    const int input_width = input->dims->data[2];
    const int input_depth = input->dims->data[3];

    const int filter_height = weights->dims->data[1];
    const int filter_width = weights->dims->data[2];
    const int output_depth = weights->dims->data[0];

    const int output_height = output->dims->data[1];
    const int output_width = output->dims->data[2];

    // Calculate padding for SAME mode
    int pad_height = 0, pad_width = 0;
    if (padding == PADDING_SAME) {
        int padding_height = max_int(0, filter_height - (input_height - 1) % stride_height - 1);
        int padding_width = max_int(0, filter_width - (input_width - 1) % stride_width - 1);
        pad_height = padding_height / 2;
        pad_width = padding_width / 2;
    }

    const int input_dims[4] = {batches, input_height, input_width, input_depth};
    const int filter_dims[4] = {output_depth, filter_height, filter_width, input_depth};
    const int output_dims[4] = {batches, output_height, output_width, output_depth};

    // Execute transposed convolution with bias
    for (int batch = 0; batch < batches; ++batch) {
        // Initialize output with bias
        for (int out_y = 0; out_y < output_height; out_y++) {
            for (int out_x = 0; out_x < output_width; out_x++) {
                for (int out_channel = 0; out_channel < output_depth; out_channel++) {
                    output_data[Offset(output_dims, batch, out_y, out_x, out_channel)] =
                        bias_data[out_channel];
                }
            }
        }

        // Transposed convolution
        for (int in_y = 0; in_y < input_height; ++in_y) {
            for (int in_x = 0; in_x < input_width; ++in_x) {
                for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                    const int out_x_origin = (in_x * stride_width) - pad_width;
                    const int out_y_origin = (in_y * stride_height) - pad_height;

                    for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                        for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                            for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
                                const int out_x = out_x_origin + filter_x;
                                const int out_y = out_y_origin + filter_y;

                                // Check bounds
                                if ((out_x >= 0) && (out_x < output_width) &&
                                    (out_y >= 0) && (out_y < output_height)) {
                                    float input_value = input_data[Offset(input_dims, batch, in_y, in_x, in_channel)];
                                    float filter_value = filter_data[Offset(filter_dims, out_channel, filter_y, filter_x, in_channel)];
                                    output_data[Offset(output_dims, batch, out_y, out_x, out_channel)] +=
                                        input_value * filter_value;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return kTfLiteOk;
}

// Registration - using only the basic fields that exist in all versions
static TfLiteRegistration g_registration = {
    Init,                           // init
    Free,                           // free
    Prepare,                        // prepare
    Eval,                           // invoke
    NULL,                           // profiling_string
    kTfLiteBuiltinCustom,          // builtin_code
    "Convolution2DTransposeBias",  // custom_name
    1,                             // version
    NULL,                          // registration_external
};

TFLITE_CUSTOM_OPS_EXPORT TfLiteRegistration* TfLiteFlutter_RegisterConvolution2DTransposeBias(void) {
    return &g_registration;
}

