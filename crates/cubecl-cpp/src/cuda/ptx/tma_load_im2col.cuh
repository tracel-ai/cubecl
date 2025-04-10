inline __device__ void
tma_load_im2col_3d(void const *desc_ptr,
                   ::cuda::barrier<::cuda::thread_scope_block> &__bar,
                   void *smem_ptr, int32 const &coord_c, int32 const &coord_w,
                   int32 const &coord_n, uint16 const &offset_w) {
  uint64 *mbar_ptr = ::cuda::device::barrier_native_handle(__bar);

  uint64 gmem_int_desc = reinterpret_cast<uint64>(desc_ptr);
  uint32 smem_int_mbar =
      static_cast<uint32>(__cvta_generic_to_shared(mbar_ptr));
  uint32 smem_int_ptr = static_cast<uint32>(__cvta_generic_to_shared(smem_ptr));
  // Copy from global to shared::cluster.
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.im2col.mbarrier:"
               ":complete_tx::bytes"
               " [%0], [%1, {%3, %4, %5}], [%2], {%6};"
               :
               : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                 "r"(coord_c), "r"(coord_w), "r"(coord_n), "h"(offset_w)
               : "memory");
}

inline __device__ void
tma_load_im2col_4d(void const *desc_ptr,
                   ::cuda::barrier<::cuda::thread_scope_block> &__bar,
                   void *smem_ptr, int32 const &coord_c, int32 const &coord_w,
                   int32 const &coord_h, int32 const &coord_n,
                   uint16 const &offset_w, uint16 const &offset_h) {
  uint64 *mbar_ptr = ::cuda::device::barrier_native_handle(__bar);

  uint64 gmem_int_desc = reinterpret_cast<uint64>(desc_ptr);
  uint32 smem_int_mbar =
      static_cast<uint32>(__cvta_generic_to_shared(mbar_ptr));
  uint32 smem_int_ptr = static_cast<uint32>(__cvta_generic_to_shared(smem_ptr));

  // Copy from global to shared::cluster.
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.im2col.mbarrier:"
               ":complete_tx::bytes"
               " [%0], [%1, {%3, %4, %5, %6}], [%2], {%7, %8};"
               :
               : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                 "r"(coord_c), "r"(coord_w), "r"(coord_h), "r"(coord_n),
                 "h"(offset_w), "h"(offset_h)
               : "memory");
}

inline __device__ void tma_load_im2col_5d(
    void const *desc_ptr, ::cuda::barrier<::cuda::thread_scope_block> &__bar,
    void *smem_ptr, int32 const &coord_c, int32 const &coord_w,
    int32 const &coord_h, int32 const &coord_d, int32 const &coord_n,
    uint16 const &offset_w, uint16 const &offset_h, uint16 const &offset_d) {
  uint64 *mbar_ptr = ::cuda::device::barrier_native_handle(__bar);

  uint64 gmem_int_desc = reinterpret_cast<uint64>(desc_ptr);
  uint32 smem_int_mbar =
      static_cast<uint32>(__cvta_generic_to_shared(mbar_ptr));
  uint32 smem_int_ptr = static_cast<uint32>(__cvta_generic_to_shared(smem_ptr));
  // Copy from global to shared::cluster.
  asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.im2col.mbarrier:"
               ":complete_tx::bytes"
               " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], {%8, %9, %10};"
               :
               : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                 "r"(coord_c), "r"(coord_w), "r"(coord_h), "r"(coord_d),
                 "r"(coord_n), "h"(offset_w), "h"(offset_h), "h"(offset_d)
               : "memory");
}