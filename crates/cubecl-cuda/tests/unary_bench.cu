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

extern "C" __global__ void execute_unary_kernel_f32(float_4 input_0[],
                                                    float_4 input_1[],
                                                    float_4 output_0[],
                                                    uint info[]) {

  int3 absoluteIdx = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                               blockIdx.y * blockDim.y + threadIdx.y,
                               blockIdx.z * blockDim.z + threadIdx.z);

  uint idxGlobal =
      (absoluteIdx.z * gridDim.x * blockDim.x * gridDim.y * blockDim.y) +
      (absoluteIdx.y * gridDim.x * blockDim.x) + absoluteIdx.x;
  const uint l_0_0 = info[uint(5)];
  const bool l_0_1 = idxGlobal < l_0_0;
  if (l_0_1) {

    for (uint l_mut_2_0 = uint(0); l_mut_2_0 < uint(256); ++l_mut_2_0) {
      const uint l_2_1 = l_mut_2_0 % uint(2);
      const bool l_2_2 = l_2_1 == uint(0);
      if (l_2_2) {
        uint l_mut_3_6;
        bool l_mut_3_7;
        float_4 l_mut_3_8;
        l_mut_3_6 = info[uint(0)];
        l_mut_3_7 = idxGlobal < l_mut_3_6;
        l_mut_3_8 = input_0[idxGlobal];
        const float_4 l_3_0 = float_4{
            (l_mut_3_7) ? l_mut_3_8.i_0 : float(0.0),
            (l_mut_3_7) ? l_mut_3_8.i_1 : float(0.0),
            (l_mut_3_7) ? l_mut_3_8.i_2 : float(0.0),
            (l_mut_3_7) ? l_mut_3_8.i_3 : float(0.0),
        };
        uint l_mut_3_9;
        bool l_mut_3_10;
        float_4 l_mut_3_11;
        l_mut_3_9 = info[uint(1)];
        l_mut_3_10 = idxGlobal < l_mut_3_9;
        l_mut_3_11 = input_1[idxGlobal];
        const float_4 l_3_1 = float_4{
            (l_mut_3_10) ? l_mut_3_11.i_0 : float(0.0),
            (l_mut_3_10) ? l_mut_3_11.i_1 : float(0.0),
            (l_mut_3_10) ? l_mut_3_11.i_2 : float(0.0),
            (l_mut_3_10) ? l_mut_3_11.i_3 : float(0.0),
        };
        const float_4 l_3_2 = float_4{
            l_3_0.i_0 * l_3_1.i_0,
            l_3_0.i_1 * l_3_1.i_1,
            l_3_0.i_2 * l_3_1.i_2,
            l_3_0.i_3 * l_3_1.i_3,
        };
        const float_4 l_3_3 = float_4{
            cos(l_3_2.i_0),
            cos(l_3_2.i_1),
            cos(l_3_2.i_2),
            cos(l_3_2.i_3),
        };
        uint l_mut_3_12;
        bool l_mut_3_13;
        float_4 l_mut_3_14;
        l_mut_3_12 = info[uint(2)];
        l_mut_3_13 = idxGlobal < l_mut_3_12;
        l_mut_3_14 = output_0[idxGlobal];
        const float_4 l_3_4 = float_4{
            (l_mut_3_13) ? l_mut_3_14.i_0 : float(0.0),
            (l_mut_3_13) ? l_mut_3_14.i_1 : float(0.0),
            (l_mut_3_13) ? l_mut_3_14.i_2 : float(0.0),
            (l_mut_3_13) ? l_mut_3_14.i_3 : float(0.0),
        };
        const float_4 l_3_5 = float_4{
            l_3_4.i_0 - l_3_3.i_0,
            l_3_4.i_1 - l_3_3.i_1,
            l_3_4.i_2 - l_3_3.i_2,
            l_3_4.i_3 - l_3_3.i_3,
        };
        uint l_mut_3_15;
        bool l_mut_3_16;
        l_mut_3_15 = info[uint(2)];
        l_mut_3_16 = idxGlobal < l_mut_3_15;
        if (l_mut_3_16) {
          output_0[idxGlobal] = reinterpret_cast<float_4 const &>(l_3_5);
        }
      } else {
        uint l_mut_3_6;
        bool l_mut_3_7;
        float_4 l_mut_3_8;
        l_mut_3_6 = info[uint(0)];
        l_mut_3_7 = idxGlobal < l_mut_3_6;
        l_mut_3_8 = input_0[idxGlobal];
        const float_4 l_3_0 = float_4{
            (l_mut_3_7) ? l_mut_3_8.i_0 : float(0.0),
            (l_mut_3_7) ? l_mut_3_8.i_1 : float(0.0),
            (l_mut_3_7) ? l_mut_3_8.i_2 : float(0.0),
            (l_mut_3_7) ? l_mut_3_8.i_3 : float(0.0),
        };
        uint l_mut_3_9;
        bool l_mut_3_10;
        float_4 l_mut_3_11;
        l_mut_3_9 = info[uint(1)];
        l_mut_3_10 = idxGlobal < l_mut_3_9;
        l_mut_3_11 = input_1[idxGlobal];
        const float_4 l_3_1 = float_4{
            (l_mut_3_10) ? l_mut_3_11.i_0 : float(0.0),
            (l_mut_3_10) ? l_mut_3_11.i_1 : float(0.0),
            (l_mut_3_10) ? l_mut_3_11.i_2 : float(0.0),
            (l_mut_3_10) ? l_mut_3_11.i_3 : float(0.0),
        };
        const float_4 l_3_2 = float_4{
            l_3_0.i_0 * l_3_1.i_0,
            l_3_0.i_1 * l_3_1.i_1,
            l_3_0.i_2 * l_3_1.i_2,
            l_3_0.i_3 * l_3_1.i_3,
        };
        const float_4 l_3_3 = float_4{
            cos(l_3_2.i_0),
            cos(l_3_2.i_1),
            cos(l_3_2.i_2),
            cos(l_3_2.i_3),
        };
        uint l_mut_3_12;
        bool l_mut_3_13;
        float_4 l_mut_3_14;
        l_mut_3_12 = info[uint(2)];
        l_mut_3_13 = idxGlobal < l_mut_3_12;
        l_mut_3_14 = output_0[idxGlobal];
        const float_4 l_3_4 = float_4{
            (l_mut_3_13) ? l_mut_3_14.i_0 : float(0.0),
            (l_mut_3_13) ? l_mut_3_14.i_1 : float(0.0),
            (l_mut_3_13) ? l_mut_3_14.i_2 : float(0.0),
            (l_mut_3_13) ? l_mut_3_14.i_3 : float(0.0),
        };
        const float_4 l_3_5 = float_4{
            l_3_4.i_0 + l_3_3.i_0,
            l_3_4.i_1 + l_3_3.i_1,
            l_3_4.i_2 + l_3_3.i_2,
            l_3_4.i_3 + l_3_3.i_3,
        };
        uint l_mut_3_15;
        bool l_mut_3_16;
        l_mut_3_15 = info[uint(2)];
        l_mut_3_16 = idxGlobal < l_mut_3_15;
        if (l_mut_3_16) {
          output_0[idxGlobal] = reinterpret_cast<float_4 const &>(l_3_5);
        }
      }
    }
  }
}