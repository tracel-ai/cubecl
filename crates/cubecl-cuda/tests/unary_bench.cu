#include <mma.h>
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint;
typedef unsigned long long int uint64;
typedef long long int int64;

struct __align__(16) float_4 {
  float i_0;
  float i_1;
  float i_2;
  float i_3;
};

extern "C" __global__ void execute_unary_kernel(float_4 input_0[],
                                                float_4 input_1[],
                                                float_4 output_0[],
                                                uint info[]) {

  int3 absoluteIdx = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                               blockIdx.y * blockDim.y + threadIdx.y,
                               blockIdx.z * blockDim.z + threadIdx.z);

  uint idxGlobal =
      (absoluteIdx.z * gridDim.x * blockDim.x * gridDim.y * blockDim.y) +
      (absoluteIdx.y * gridDim.x * blockDim.x) + absoluteIdx.x;
  uint l_0_0;
  bool l_0_1;
  bool l_0_2;
  float_4 l_0_3;
  float_4 l_0_4;
  l_0_0 = info[uint(5)];
  l_0_1 = idxGlobal < l_0_0;
  if (l_0_1) {

    for (uint l_2_0 = uint(0); l_2_0 < uint(256); ++l_2_0) {
      l_0_0 = l_2_0 % uint(2);
      l_0_2 = l_0_0 == uint(0);
      if (l_0_2) {
        uint l_3_0;
        l_3_0 = info[uint(0)];
        l_0_3 = (idxGlobal < l_3_0) ? input_0[idxGlobal] : float_4{};
        uint l_3_1;
        l_3_1 = info[uint(1)];
        l_0_4 = (idxGlobal < l_3_1) ? input_1[idxGlobal] : float_4{};
        l_0_4 = float_4{
            l_0_3.i_0 * l_0_4.i_0,
            l_0_3.i_1 * l_0_4.i_1,
            l_0_3.i_2 * l_0_4.i_2,
            l_0_3.i_3 * l_0_4.i_3,
        };
        l_0_4 = float_4{
            cos(l_0_4.i_0),
            cos(l_0_4.i_1),
            cos(l_0_4.i_2),
            cos(l_0_4.i_3),
        };
        uint l_3_2;
        l_3_2 = info[uint(2)];
        l_0_3 = (idxGlobal < l_3_2) ? output_0[idxGlobal] : float_4{};
        l_0_3 = float_4{
            l_0_3.i_0 - l_0_4.i_0,
            l_0_3.i_1 - l_0_4.i_1,
            l_0_3.i_2 - l_0_4.i_2,
            l_0_3.i_3 - l_0_4.i_3,
        };
        uint l_3_3;
        bool l_3_4;
        l_3_3 = info[uint(2)];
        l_3_4 = idxGlobal < l_3_3;
        if (l_3_4) {
          output_0[idxGlobal] = l_0_3;
        }
      } else {
        uint l_3_0;
        l_3_0 = info[uint(0)];
        l_0_4 = (idxGlobal < l_3_0) ? input_0[idxGlobal] : float_4{};
        uint l_3_1;
        l_3_1 = info[uint(1)];
        l_0_3 = (idxGlobal < l_3_1) ? input_1[idxGlobal] : float_4{};
        l_0_4 = float_4{
            l_0_4.i_0 * l_0_3.i_0,
            l_0_4.i_1 * l_0_3.i_1,
            l_0_4.i_2 * l_0_3.i_2,
            l_0_4.i_3 * l_0_3.i_3,
        };
        l_0_4 = float_4{
            cos(l_0_4.i_0),
            cos(l_0_4.i_1),
            cos(l_0_4.i_2),
            cos(l_0_4.i_3),
        };
        uint l_3_2;
        l_3_2 = info[uint(2)];
        l_0_3 = (idxGlobal < l_3_2) ? output_0[idxGlobal] : float_4{};
        l_0_3 = float_4{
            l_0_3.i_0 + l_0_4.i_0,
            l_0_3.i_1 + l_0_4.i_1,
            l_0_3.i_2 + l_0_4.i_2,
            l_0_3.i_3 + l_0_4.i_3,
        };
        uint l_3_3;
        bool l_3_4;
        l_3_3 = info[uint(2)];
        l_3_4 = idxGlobal < l_3_3;
        if (l_3_4) {
          output_0[idxGlobal] = l_0_3;
        }
      }
    }
  }
}