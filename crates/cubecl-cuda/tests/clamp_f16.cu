#include <cuda_fp16.h>
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint;
typedef unsigned long long int uint64;
typedef long long int int64;

extern "C" __global__ void clamp_kernel_f16(__half input_0[], __half output_0[],
                                            uint info[]) {

  int3 absoluteIdx = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                               blockIdx.y * blockDim.y + threadIdx.y,
                               blockIdx.z * blockDim.z + threadIdx.z);

  uint idxGlobal =
      (absoluteIdx.z * gridDim.x * blockDim.x * gridDim.y * blockDim.y) +
      (absoluteIdx.y * gridDim.x * blockDim.x) + absoluteIdx.x;
  __half l_0_0;
  uint l_0_1;
  bool l_0_2;
  __half l_0_3;
  l_0_1 = info[uint(0)];
  l_0_2 = uint(0) < l_0_1;
  l_0_3 = input_0[uint(0)];
  l_0_0 = (l_0_2) ? l_0_3 : __half(0.0);
  l_0_0 = __hmax(__half(0.0), __hmin(__half(2.0), l_0_0));
  uint l_0_4;
  bool l_0_5;
  l_0_4 = info[uint(1)];
  l_0_5 = idxGlobal < l_0_4;
  if (l_0_5) {
    output_0[idxGlobal] = l_0_0;
  }
}