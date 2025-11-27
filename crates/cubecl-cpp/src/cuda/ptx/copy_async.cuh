inline __device__ void
__cp_async_arrive(::cuda::barrier<::cuda::thread_scope_block> &__bar) {
  uint64 *mbar_ptr = ::cuda::device::barrier_native_handle(__bar);

  uint32 smem_int_mbar =
      static_cast<uint32>(__cvta_generic_to_shared(mbar_ptr));
  asm volatile("cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n"
               :: "r"(smem_int_mbar));
}

// Only 16 byte size allows the `cg` modifier, so this is a more general version with `ca`. Executes
// a partial copy of `src_size` bytes.
template <size_t _Copy_size>
inline __device__ void
__cp_async_shared_global(void const *src_ptr, void *smem_ptr, int32 const &src_size) {
  static_assert(_Copy_size == 4 || _Copy_size == 8 || _Copy_size == 16,
                "cp.async.shared.global requires a copy size of 4, 8, or 16.");

  uint64 gmem_int_desc = reinterpret_cast<uint64>(src_ptr);
  uint32 smem_int_ptr = static_cast<uint32>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.ca.shared::cta.global [%0], [%1], %2, %3;\n"
               :: "r"(smem_int_ptr), "l"(gmem_int_desc), "n"(_Copy_size), "r"(src_size) : "memory");
}

// Specialized on copy size 16 to use `cg`. Executes a partial copy of `src_size` bytes.
template <>
inline __device__ void
__cp_async_shared_global<16>(void const *src_ptr, void *smem_ptr, int32 const &src_size) {
  uint64 gmem_int_desc = reinterpret_cast<uint64>(src_ptr);
  uint32 smem_int_ptr = static_cast<uint32>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared::cta.global [%0], [%1], %2, %3;\n"
               :: "r"(smem_int_ptr), "l"(gmem_int_desc), "n"(16), "r"(src_size) : "memory");
}

// Only 16 byte size allows the `cg` modifier, so this is a more general version with `ca`. Executes
// a full copy.
template <size_t _Copy_size>
inline __device__ void
__cp_async_shared_global(void const *src_ptr, void *smem_ptr) {
  static_assert(_Copy_size == 4 || _Copy_size == 8 || _Copy_size == 16,
                "cp.async.shared.global requires a copy size of 4, 8, or 16.");

  uint64 gmem_int_desc = reinterpret_cast<uint64>(src_ptr);
  uint32 smem_int_ptr = static_cast<uint32>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.ca.shared::cta.global [%0], [%1], %2, %2;\n"
               :: "r"(smem_int_ptr), "l"(gmem_int_desc), "n"(_Copy_size) : "memory");
}

// Specialized on copy size 16 to use `cg`. Executes a full copy.
template <>
inline __device__ void
__cp_async_shared_global<16>(void const *src_ptr, void *smem_ptr) {
  uint64 gmem_int_desc = reinterpret_cast<uint64>(src_ptr);
  uint32 smem_int_ptr = static_cast<uint32>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared::cta.global [%0], [%1], %2, %2;\n"
               :: "r"(smem_int_ptr), "l"(gmem_int_desc), "n"(16) : "memory");
}