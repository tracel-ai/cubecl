typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint;
typedef unsigned long long int uint64;
typedef long long int int64;

extern "C" __global__ void sequence_for_loop_kernel(float output_0[],
                                                    uint info[]) {

  int threadIdxGlobal = threadIdx.x + threadIdx.y * blockDim.x +
                        threadIdx.z * (blockDim.x * blockDim.y);
  const bool l_0_0 = threadIdxGlobal != uint(0);
  if (l_0_0) {
    return;
  }
  uint l_mut_0_5;
  bool l_mut_0_6;
  float l_mut_0_7;
  l_mut_0_5 = info[uint(0)];
  l_mut_0_6 = uint(0) < l_mut_0_5;
  l_mut_0_7 = output_0[uint(0)];
  const float l_0_1 = (l_mut_0_6) ? l_mut_0_7 : float(0.0);
  const float l_0_2 = l_0_1 + float(1.0);
  uint l_mut_0_8;
  bool l_mut_0_9;
  l_mut_0_8 = info[uint(0)];
  l_mut_0_9 = uint(0) < l_mut_0_8;
  if (l_mut_0_9) {
    output_0[uint(0)] = l_0_2;
  }
  uint l_mut_0_10;
  bool l_mut_0_11;
  float l_mut_0_12;
  l_mut_0_10 = info[uint(0)];
  l_mut_0_11 = uint(0) < l_mut_0_10;
  l_mut_0_12 = output_0[uint(0)];
  const float l_0_3 = (l_mut_0_11) ? l_mut_0_12 : float(0.0);
  const float l_0_4 = l_0_3 + float(4.0);
  uint l_mut_0_13;
  bool l_mut_0_14;
  l_mut_0_13 = info[uint(0)];
  l_mut_0_14 = uint(0) < l_mut_0_13;
  if (l_mut_0_14) {
    output_0[uint(0)] = l_0_4;
  }
}