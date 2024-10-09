typedef unsigned int uint;

extern "C" __global__ void kernel(float output_0[], uint info[]) {

  int3 absoluteIdx = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                               blockIdx.y * blockDim.y + threadIdx.y,
                               blockIdx.z * blockDim.z + threadIdx.z);

  uint idxGlobal =
      (absoluteIdx.z * gridDim.x * blockDim.x * gridDim.y * blockDim.y) +
      (absoluteIdx.y * gridDim.x * blockDim.x) + absoluteIdx.x;
  uint rank = info[0];
  uint rank_2 = rank * 2;
  const float arrays_0[3] = {
      float(3),
      float(5),
      float(1),
  };
  uint l_0_0;
  bool l_0_1;
  float l_0_2;
  l_0_0 = info[(1 * 2 * info[0]) + 1];
  l_0_1 = idxGlobal < l_0_0;
  if (l_0_1) {
    l_0_2 = arrays_0[idxGlobal];
    uint l_1_0;
    bool l_1_1;
    l_1_0 = info[(1 * 2 * info[0]) + 1];
    l_1_1 = idxGlobal < l_1_0;
    if (l_1_1) {
      output_0[idxGlobal] = l_0_2;
    }
  }
}