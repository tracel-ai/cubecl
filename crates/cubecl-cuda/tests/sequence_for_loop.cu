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
  bool l_0_3;
  float l_0_4;
  l_0_2 = info[uint(0)];
  l_0_3 = uint(0) < l_0_2;
  l_0_4 = output_0[uint(0)];
  l_0_1 = (l_0_3) ? l_0_4 : float(0.0);
  l_0_1 = l_0_1 + float(1.0);
  uint l_0_5;
  bool l_0_6;
  l_0_5 = info[uint(0)];
  l_0_6 = uint(0) < l_0_5;
  if (l_0_6) {
    output_0[uint(0)] = l_0_1;
  }
  uint l_0_7;
  bool l_0_8;
  float l_0_9;
  l_0_7 = info[uint(0)];
  l_0_8 = uint(0) < l_0_7;
  l_0_9 = output_0[uint(0)];
  l_0_1 = (l_0_8) ? l_0_9 : float(0.0);
  l_0_1 = l_0_1 + float(4.0);
  uint l_0_10;
  bool l_0_11;
  l_0_10 = info[uint(0)];
  l_0_11 = uint(0) < l_0_10;
  if (l_0_11) {
    output_0[uint(0)] = l_0_1;
  }
}
