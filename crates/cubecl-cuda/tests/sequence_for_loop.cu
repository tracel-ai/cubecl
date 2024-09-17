typedef unsigned int uint;

extern "C" __global__ void kernel(float output_0[], uint info[]) {

  int threadIdxGlobal = threadIdx.x + threadIdx.y * blockDim.x +
                        threadIdx.z * (blockDim.x * blockDim.y);
  uint rank = info[0];
  uint rank_2 = rank * 2;
  bool l_0_0;
  float l_0_1;
  l_0_0 = threadIdxGlobal != uint(0);
  if (l_0_0) {
    return;
  }
  uint l_0_2;
  bool l_0_3;
  l_0_2 = info[(1 * 2 * info[0]) + 1];
  l_0_3 = uint(0) < l_0_2;
  if (l_0_3) {
    l_0_1 = output_0[uint(0)];
  } else {
    l_0_1 = float(0.0);
  }
  l_0_1 = l_0_1 + float(1.0);
  uint l_0_4;
  bool l_0_5;
  l_0_4 = info[(1 * 2 * info[0]) + 1];
  l_0_5 = uint(0) < l_0_4;
  if (l_0_5) {
    output_0[uint(0)] = l_0_1;
  }
  uint l_0_6;
  bool l_0_7;
  l_0_6 = info[(1 * 2 * info[0]) + 1];
  l_0_7 = uint(0) < l_0_6;
  if (l_0_7) {
    l_0_1 = output_0[uint(0)];
  } else {
    l_0_1 = float(0.0);
  }
  l_0_1 = l_0_1 + float(4.0);
  uint l_0_8;
  bool l_0_9;
  l_0_8 = info[(1 * 2 * info[0]) + 1];
  l_0_9 = uint(0) < l_0_8;
  if (l_0_9) {
    output_0[uint(0)] = l_0_1;
  }
}