typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint;
typedef unsigned long long int uint64;
typedef long long int int64;

extern "C" __global__ void slice_assign_kernel(float input_0[],
                                               float output_0[], uint info[]) {

  int threadIdxGlobal = threadIdx.x + threadIdx.y * blockDim.x +
                        threadIdx.z * (blockDim.x * blockDim.y);
  bool l_0_0;
  float l_0_1;
  l_0_0 = threadIdxGlobal == uint(0);
  if (l_0_0) {
    uint l_1_0;
    l_1_0 = info[uint(1)];
    const uint slice_1_0_length = min(l_1_0, uint(3)) - uint(2);
    float *slice_1_0 = output_0 + uint(2);
    uint l_1_1;
    l_1_1 = info[uint(0)];
    l_0_1 = (uint(0) < l_1_1) ? input_0[uint(0)] : float(0);
    uint l_1_2;
    bool l_1_3;
    l_1_2 = slice_1_0_length;
    l_1_3 = uint(0) < l_1_2;
    if (l_1_3) {
      slice_1_0[uint(0)] = l_0_1;
    }
  }
}
