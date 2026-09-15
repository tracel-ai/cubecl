// C bindings for the LLD ELF driver.

#include <cstddef>
#include <lld/Common/Driver.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/raw_ostream.h>

LLD_HAS_DRIVER(elf)

extern "C" bool cubecl_lld_elf_link(const char *const *argv, size_t argc) {
  return lld::elf::link(llvm::ArrayRef<const char *>(argv, argc), llvm::nulls(),
                        llvm::errs(), /*exitEarly=*/false,
                        /*disableOutput=*/false);
}
