typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint;
typedef unsigned long long int uint64;
typedef long long int int64;

extern "C" __global__ void constant_array_kernel_f32(float output_0[],
                                                     uint info[]) {

  int3 absoluteIdx = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                               blockIdx.y * blockDim.y + threadIdx.y,
                               blockIdx.z * blockDim.z + threadIdx.z);

  uint idxGlobal =
      (absoluteIdx.z * gridDim.x * blockDim.x * gridDim.y * blockDim.y) +
      (absoluteIdx.y * gridDim.x * blockDim.x) + absoluteIdx.x;
  const float arrays_0[3] = {
      float(3),
      float(5),
      float(1),
  };
  const uint l_0_0 = info[uint(1)];
  const bool l_0_1 = idxGlobal < l_0_0;
  if (l_0_1) {
    const float l_1_0 = arrays_0[idxGlobal];
    uint l_mut_1_1;
    bool l_mut_1_2;
    l_mut_1_1 = info[uint(0)];
    l_mut_1_2 = idxGlobal < l_mut_1_1;
    if (l_mut_1_2) {
      output_0[idxGlobal] = l_1_0;
    }
  }
}