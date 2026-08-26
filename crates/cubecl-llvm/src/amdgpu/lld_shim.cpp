// C ABI over LLD's C++ entry point, so the Rust side needs no mangled names.
//
// `LLD_HAS_DRIVER` declares `lld::elf::link` and commits us to linking the ELF
// driver, which `build.rs` does via `lldELF`.

#include <cstddef>
#include <lld/Common/Driver.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/raw_ostream.h>

LLD_HAS_DRIVER(elf)

extern "C" bool cubecl_lld_elf_link(const char *const *argv, size_t argc) {
  // exitEarly=false: a link error must return here, not call exit() and take
  // the compiling process with it. Diagnostics go to the real stderr; capturing
  // them would need a raw_ostream subclass and a buffer to own.
  return lld::elf::link(llvm::ArrayRef<const char *>(argv, argc), llvm::nulls(),
                        llvm::errs(), /*exitEarly=*/false,
                        /*disableOutput=*/false);
}
