typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint;
typedef unsigned long long int uint64;
typedef long long int int64;

extern "C" __global__ void naming_kernel_f32_u8_bf16_i64(float output_0[],
                                                         uint info[]) {

  int3 absoluteIdx = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                               blockIdx.y * blockDim.y + threadIdx.y,
                               blockIdx.z * blockDim.z + threadIdx.z);

  uint idxGlobal =
      (absoluteIdx.z * gridDim.x * blockDim.x * gridDim.y * blockDim.y) +
      (absoluteIdx.y * gridDim.x * blockDim.x) + absoluteIdx.x;
  const uint l_0_0 = info[uint(1)];
  const bool l_0_1 = idxGlobal < l_0_0;
  if (l_0_1) {
    uint l_mut_1_0;
    bool l_mut_1_1;
    l_mut_1_0 = info[uint(0)];
    l_mut_1_1 = idxGlobal < l_mut_1_0;
    if (l_mut_1_1) {
      output_0[idxGlobal] = float(0.0);
    }
  }
}