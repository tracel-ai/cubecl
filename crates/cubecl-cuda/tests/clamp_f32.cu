typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint;
typedef unsigned long long int uint64;
typedef long long int int64;

extern "C" __global__ void clamp_kernel_f32(float input_0[], float output_0[],
                                            uint info[]) {

  int3 absoluteIdx = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                               blockIdx.y * blockDim.y + threadIdx.y,
                               blockIdx.z * blockDim.z + threadIdx.z);

  uint idxGlobal =
      (absoluteIdx.z * gridDim.x * blockDim.x * gridDim.y * blockDim.y) +
      (absoluteIdx.y * gridDim.x * blockDim.x) + absoluteIdx.x;
  float l_0_0;
  uint l_0_1;
  l_0_1 = info[uint(0)];
  l_0_0 = (uint(0) < l_0_1) ? input_0[uint(0)] : float(0);
  l_0_0 = max(float(0.0), min(float(2.0), l_0_0));
  uint l_0_2;
  bool l_0_3;
  l_0_2 = info[uint(1)];
  l_0_3 = idxGlobal < l_0_2;
  if (l_0_3) {
    output_0[idxGlobal] = l_0_0;
  }
}