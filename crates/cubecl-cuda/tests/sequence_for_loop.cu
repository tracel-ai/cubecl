#include <mma.h>
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint;
typedef unsigned long long int uint64;
typedef long long int int64;

extern "C" __global__ void sequence_for_loop_kernel(float output_0[], 
                                                    uint info[]) { 

  int threadIdxGlobal = threadIdx.x + threadIdx.y * blockDim.x +
                        threadIdx.z * (blockDim.x * blockDim.y);
  bool l_0_0;
  float l_0_1;
  l_0_0 = threadIdxGlobal != uint(0);
  if (l_0_0) {
    return;
  }
  uint l_0_2;
  l_0_2 = info[uint(0)];
  l_0_1 = (uint(0) < l_0_2) ? output_0[uint(0)] : float(0);
  l_0_1 = l_0_1 + float(1.0);
  uint l_0_3;
  bool l_0_4;
  l_0_3 = info[uint(0)];
  l_0_4 = uint(0) < l_0_3;
  if (l_0_4) {
    output_0[uint(0)] = l_0_1;
  }
  uint l_0_5;
  l_0_5 = info[uint(0)];
  l_0_1 = (uint(0) < l_0_5) ? output_0[uint(0)] : float(0);
  l_0_1 = l_0_1 + float(4.0);
  uint l_0_6;
  bool l_0_7;
  l_0_6 = info[uint(0)];
  l_0_7 = uint(0) < l_0_6;
  if (l_0_7) {
    output_0[uint(0)] = l_0_1;
  }
}