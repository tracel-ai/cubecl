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

extern "C" __global__ void
lined_clamp_kernel_f32(float_4 input_0[], float_4 output_0[], uint info[]) {

  int3 absoluteIdx = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                               blockIdx.y * blockDim.y + threadIdx.y,
                               blockIdx.z * blockDim.z + threadIdx.z);

  uint idxGlobal =
      (absoluteIdx.z * gridDim.x * blockDim.x * gridDim.y * blockDim.y) +
      (absoluteIdx.y * gridDim.x * blockDim.x) + absoluteIdx.x;
  float_4 l_0_0;
  uint l_0_1;
  bool l_0_2;
  float_4 l_0_3;
  l_0_1 = info[uint(0)];
  l_0_2 = uint(0) < l_0_1;
  l_0_3 = input_0[uint(0)];
  l_0_0 = float_4{
      (l_0_2) ? l_0_3.i_0 : float(0.0),
      (l_0_2) ? l_0_3.i_1 : float(0.0),
      (l_0_2) ? l_0_3.i_2 : float(0.0),
      (l_0_2) ? l_0_3.i_3 : float(0.0),
  };
  l_0_0 = float_4{
      max(float(0.0), min(float(2.0), l_0_0.i_0)),
      max(float(0.0), min(float(2.0), l_0_0.i_1)),
      max(float(0.0), min(float(2.0), l_0_0.i_2)),
      max(float(0.0), min(float(2.0), l_0_0.i_3)),
  };
  uint l_0_4;
  bool l_0_5;
  l_0_4 = info[uint(1)];
  l_0_5 = idxGlobal < l_0_4;
  if (l_0_5) {
    output_0[idxGlobal] = l_0_0;
  }
}