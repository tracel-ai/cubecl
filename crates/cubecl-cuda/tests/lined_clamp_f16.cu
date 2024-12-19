#include <cuda_fp16.h>
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint;
typedef unsigned long long int uint64;
typedef long long int int64;

struct __align__(8) __half_4 {
  __half i_0;
  __half i_1;
  __half i_2;
  __half i_3;
};

struct __align__(8) __half2_2 {
  __half2 i_0;
  __half2 i_1;
};

extern "C" __global__ void
lined_clamp_kernel_f16(__half_4 input_0[], __half_4 output_0[], uint info[]) {

  int3 absoluteIdx = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                               blockIdx.y * blockDim.y + threadIdx.y,
                               blockIdx.z * blockDim.z + threadIdx.z);

  uint idxGlobal =
      (absoluteIdx.z * gridDim.x * blockDim.x * gridDim.y * blockDim.y) +
      (absoluteIdx.y * gridDim.x * blockDim.x) + absoluteIdx.x;
  __half_4 l_0_0;
  uint l_0_1;
  l_0_1 = info[uint(0)];
  l_0_0 = (uint(0) < l_0_1) ? input_0[uint(0)] : __half_4{};
  l_0_0 = __half2_2{
      __hmax2(__half(0.0),
              __hmin2(__half(2.0), (reinterpret_cast<__half2_2 &>(l_0_0)).i_0)),
      __hmax2(__half(0.0),
              __hmin2(__half(2.0), (reinterpret_cast<__half2_2 &>(l_0_0)).i_1)),
  };
  uint l_0_2;
  bool l_0_3;
  l_0_2 = info[uint(1)];
  l_0_3 = idxGlobal < l_0_2;
  if (l_0_3) {
    output_0[idxGlobal] = l_0_0;
  }
}