// C bindings for the perf support plugin of ORC. The LLVM C API has no
// binding for it.

#include <llvm-c/Error.h>
#include <llvm-c/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/AbsoluteSymbols.h>
#include <llvm/ExecutionEngine/Orc/Debugging/DebugInfoSupport.h>
#include <llvm/ExecutionEngine/Orc/Debugging/PerfSupportPlugin.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/Shared/PerfSharedStructs.h>
#include <llvm/ExecutionEngine/Orc/TargetProcess/JITLoaderPerf.h>
#include <llvm/Support/Error.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

using namespace llvm;
using namespace llvm::orc;

namespace {

// The plugin adds 0x40 to each line address, for the ELF header that perf
// puts before the code. Since Linux 5.8 (commit 1e4bd2ae4564), `perf inject`
// adds this offset itself. Thus the lines are 0x40 bytes too far, unless the
// perf on the host is older than 5.8.
constexpr uint64_t PERF_HEADER_OFFSET = 0x40;
std::atomic<bool> remove_header_offset{true};

constexpr uint8_t DW_EH_PE_OMIT = 0xff;
constexpr uint8_t DW_EH_PE_UDATA4 = 0x03;
constexpr uint8_t DW_EH_PE_SDATA4 = 0x0b;
constexpr uint8_t DW_EH_PE_PCREL = 0x10;
constexpr uint8_t DW_EH_PE_DATAREL = 0x30;

uint64_t align8(uint64_t value) { return (value + 7) & ~uint64_t{7}; }

uint32_t read_u32(const std::string &data, size_t offset) {
  uint32_t value;
  std::memcpy(&value, data.data() + offset, sizeof(value));
  return value;
}

void write_i32(std::string &data, size_t offset, int64_t value) {
  auto narrow = static_cast<int32_t>(value);
  std::memcpy(data.data() + offset, &narrow, sizeof(narrow));
}

void append_i32(std::string &data, int64_t value) {
  auto narrow = static_cast<int32_t>(value);
  data.append(reinterpret_cast<const char *>(&narrow), sizeof(narrow));
}

bool fits_i32(int64_t value) {
  return value >= INT32_MIN && value <= INT32_MAX;
}

// Reads a LEB128 number at `offset` and moves `offset` past it.
std::optional<int64_t> read_leb128(const std::string &data, size_t &offset,
                                   bool is_signed) {
  uint64_t value = 0;
  unsigned shift = 0;
  uint8_t byte;
  do {
    if (offset >= data.size() || shift >= 64) {
      return std::nullopt;
    }
    byte = static_cast<uint8_t>(data[offset++]);
    value |= uint64_t{byte & 0x7fu} << shift;
    shift += 7;
  } while (byte & 0x80);
  if (is_signed && shift < 64 && (byte & 0x40)) {
    value |= ~uint64_t{0} << shift;
  }
  return static_cast<int64_t>(value);
}

// The pointer encoding of the FDEs of the CIE at `offset`, from its `R`
// augmentation. Only the encodings that LLVM gives on x86-64 are supported.
std::optional<uint8_t> fde_encoding(const std::string &data, size_t offset) {
  size_t cursor = offset + 8; // length, CIE id
  if (cursor >= data.size()) {
    return std::nullopt;
  }
  uint8_t version = static_cast<uint8_t>(data[cursor++]);
  size_t end = data.find('\0', cursor);
  if (end == std::string::npos) {
    return std::nullopt;
  }
  std::string augmentation = data.substr(cursor, end - cursor);
  cursor = end + 1;
  if (augmentation.empty() || augmentation[0] != 'z') {
    return std::nullopt;
  }
  if (!read_leb128(data, cursor, false) || !read_leb128(data, cursor, true)) {
    return std::nullopt;
  }
  if (version == 1) {
    cursor++;
  } else if (!read_leb128(data, cursor, false)) {
    return std::nullopt;
  }
  if (!read_leb128(data, cursor, false)) {
    return std::nullopt;
  }
  for (char letter : augmentation.substr(1)) {
    if (cursor >= data.size()) {
      return std::nullopt;
    }
    switch (letter) {
    case 'R':
      return static_cast<uint8_t>(data[cursor]);
    case 'L':
      cursor++;
      break;
    case 'S':
      break;
    default:
      return std::nullopt;
    }
  }
  return std::nullopt;
}

// An unwind record, and the `.eh_frame_hdr` that its `EHFrameAddr` must point
// to while it is written.
struct Unwinding {
  PerfJITCodeUnwindingInfoRecord record;
  std::string header;
};

// The `.eh_frame` and `.eh_frame_hdr` of `code` as perf lays them out in the
// ELF file that `perf inject` writes for it. Returns nothing when the
// `.eh_frame` has a form this does not support, or no FDE covers `code`.
//
// perf puts the code at a fixed offset `T`, the `.eh_frame` at
// `align8(T + size)`, and the `.eh_frame_hdr` after it. In the plugin's copy,
// the PC-relative addresses of the FDEs are from the JIT memory, and the
// header comes first. Both are wrong for perf.
std::optional<Unwinding>
unwinding_for(const PerfJITCodeUnwindingInfoRecord &original,
              const PerfJITCodeLoadRecord &code) {
  uint64_t frame_size = original.UnwindDataSize - original.EHFrameHdrSize;
  std::string frame(reinterpret_cast<const char *>(original.EHFrameAddr),
                    frame_size);

  // With `T` a multiple of 8, the distances below do not depend on `T`.
  // Thus they hold for each version of perf.
  auto code_start = static_cast<int64_t>(align8(code.CodeSize));
  std::vector<std::pair<int64_t, int64_t>> table; // (code offset, FDE offset)
  size_t offset = 0;
  while (offset + 4 <= frame.size()) {
    uint32_t length = read_u32(frame, offset);
    if (length == 0) {
      break;
    }
    if (length == 0xffffffff || offset + 4 + length > frame.size()) {
      return std::nullopt;
    }
    uint32_t cie_pointer = read_u32(frame, offset + 4);
    if (cie_pointer != 0) {
      size_t cie = offset + 4 - cie_pointer;
      auto encoding = fde_encoding(frame, cie);
      if (encoding != (DW_EH_PE_PCREL | DW_EH_PE_SDATA4)) {
        return std::nullopt;
      }
      size_t field = offset + 8;
      auto relative = static_cast<int32_t>(read_u32(frame, field));
      uint64_t target = original.EHFrameAddr + field + relative;
      // The FDE of another function of the object is kept, but points out of
      // this code.
      int64_t into_code = static_cast<int64_t>(target - code.CodeAddr);
      int64_t moved = into_code - code_start - static_cast<int64_t>(field);
      if (!fits_i32(moved)) {
        return std::nullopt;
      }
      write_i32(frame, field, moved);
      if (target >= code.CodeAddr && target < code.CodeAddr + code.CodeSize) {
        table.emplace_back(into_code, static_cast<int64_t>(offset));
      }
    }
    offset += 4 + length;
  }
  if (table.empty()) {
    return std::nullopt;
  }

  std::string header;
  header.push_back(1); // version
  header.push_back(static_cast<char>(DW_EH_PE_PCREL | DW_EH_PE_SDATA4));
  header.push_back(static_cast<char>(DW_EH_PE_UDATA4));
  header.push_back(static_cast<char>(DW_EH_PE_DATAREL | DW_EH_PE_SDATA4));
  // The plugin's writer puts `EHFrameHdr` first and the bytes at
  // `EHFrameAddr` after it. perf reads the `.eh_frame` first. With the two
  // parts padded to one size `size`, the `.eh_frame` goes in `EHFrameHdr`.
  // Zero bytes end an `.eh_frame`, and perf ignores the end of the header.
  uint64_t size = std::max(align8(frame.size() + 4),
                           align8(header.size() + 8 + 8 * table.size()));
  auto header_start = static_cast<int64_t>(size);
  append_i32(header, -(header_start + 4)); // the `.eh_frame`
  append_i32(header, static_cast<int64_t>(table.size()));
  std::sort(table.begin(), table.end());
  for (auto [into_code, fde] : table) {
    int64_t location = into_code - code_start - header_start;
    if (!fits_i32(location)) {
      return std::nullopt;
    }
    append_i32(header, location);
    append_i32(header, fde - header_start);
  }
  frame.resize(size, '\0');
  header.resize(size, '\0');

  PerfJITCodeUnwindingInfoRecord record;
  record.Prefix.Id = PerfJITRecordType::JIT_CODE_UNWINDING_INFO;
  record.EHFrameHdr = std::move(frame);
  record.EHFrameHdrAddr = 0;
  record.EHFrameHdrSize = size;
  record.UnwindDataSize = 2 * size;
  record.MappedSize = 0;
  // The caller points `EHFrameAddr` to `header` once `header` has its place.
  record.EHFrameAddr = 0;
  record.Prefix.TotalSize = static_cast<uint32_t>(
      2 * sizeof(uint32_t) + sizeof(uint64_t) + 3 * sizeof(uint64_t) +
      record.UnwindDataSize);
  return Unwinding{std::move(record), std::move(header)};
}

using RegisterPerfImpl = shared::WrapperFunction<shared::SPSError(
    shared::SPSPerfJITRecordBatch)>;

// Gives `batch` to the jitdump writer of LLVM.
Error write_batch(const PerfJITRecordBatch &batch) {
  auto caller = [](const char *data, size_t size) {
    return shared::WrapperFunctionBuffer(
        llvm_orc_registerJITLoaderPerfImpl(data, size));
  };
  Error result = Error::success();
  if (auto err = RegisterPerfImpl::call(caller, result, batch)) {
    consumeError(std::move(result));
    return err;
  }
  return result;
}

// Fixes the records of the plugin for perf, and writes them.
//
// perf gives a debug record and an unwind record to the next code record
// only. The plugin writes all debug records, then all code records. Thus each
// code record is written in a batch of its own, after its own records.
Error write_fixed_batch(const PerfJITRecordBatch &batch) {
  bool remove_offset = remove_header_offset.load(std::memory_order_relaxed);
  bool has_unwinding = batch.UnwindingRecord.Prefix.TotalSize > 0;
  for (const auto &code : batch.CodeLoadRecords) {
    PerfJITRecordBatch one;
    one.UnwindingRecord.Prefix.TotalSize = 0;
    one.CodeLoadRecords.push_back(code);
    for (const auto &debug : batch.DebugInfoRecords) {
      if (debug.CodeAddr != code.CodeAddr) {
        continue;
      }
      auto fixed = debug;
      if (remove_offset) {
        for (auto &entry : fixed.Entries) {
          entry.Addr -= PERF_HEADER_OFFSET;
        }
      }
      one.DebugInfoRecords.push_back(std::move(fixed));
    }
    // Owns the header that `EHFrameAddr` points to until the write is done.
    std::optional<Unwinding> unwinding;
    if (has_unwinding) {
      unwinding = unwinding_for(batch.UnwindingRecord, code);
    }
    if (unwinding) {
      unwinding->record.EHFrameAddr =
          reinterpret_cast<uint64_t>(unwinding->header.data());
      one.UnwindingRecord = unwinding->record;
    }
    if (auto err = write_batch(one)) {
      return err;
    }
  }
  return Error::success();
}

} // namespace

/// Gives the records of the perf support plugin to the jitdump writer of
/// LLVM, after `write_fixed_batch` fixes them for perf.
extern "C" shared::CWrapperFunctionBuffer
cubecl_orc_register_jit_loader_perf_impl(const char *data, size_t size) {
  return RegisterPerfImpl::handle(data, size,
                                  [](const PerfJITRecordBatch &batch) {
                                    return write_fixed_batch(batch);
                                  })
      .release();
}

/// Adds the perf support plugin to `jit`, as `LLVMOrcLLJITEnableDebugSupport`
/// adds the debugger plugin. The plugin writes `jit-<pid>.dump` in
/// `$JITDUMPDIR/.debug/jit` for `perf inject --jit`. It records the code of
/// each symbol, the line table when `emit_debug_info` is true, and the
/// `.eh_frame` when `emit_unwind_info` is true. `remove_offset` is false only
/// for a perf older than 5.8, which needs the 0x40 that the plugin adds to
/// each line address.
///
/// The JIT must link with JITLink. The plugin supports ELF only, and the
/// jitdump writer supports Linux only. Returns null on success, or an error
/// that the caller owns.
extern "C" LLVMErrorRef
cubecl_orc_lljit_enable_perf_support(LLVMOrcLLJITRef jit_ref,
                                     bool emit_debug_info,
                                     bool emit_unwind_info,
                                     bool remove_offset) {
  LLJIT &jit = *reinterpret_cast<LLJIT *>(jit_ref);
  auto *layer = dyn_cast<ObjectLinkingLayer>(&jit.getObjLinkingLayer());
  if (layer == nullptr) {
    return wrap(make_error<StringError>(
        "perf support requires JITLink", inconvertibleErrorCode()));
  }
  remove_header_offset.store(remove_offset, std::memory_order_relaxed);

  // The plugin finds the jitdump writer by a lookup of exported symbols in a
  // JITDylib. A static LLVM does not export the writer to the dynamic linker,
  // so this defines its functions from their addresses. The write function
  // is cubecl's, which fixes the records before LLVM writes them.
  auto &session = jit.getExecutionSession();
  JITDylibSP process_symbols = jit.getProcessSymbolsJITDylib();
  JITDylib &dylib =
      process_symbols ? *process_symbols : jit.getMainJITDylib();
  if (auto err = dylib.define(absoluteSymbols({
          {session.intern("llvm_orc_registerJITLoaderPerfStart"),
           ExecutorSymbolDef::fromPtr(&llvm_orc_registerJITLoaderPerfStart,
                                     JITSymbolFlags::Exported)},
          {session.intern("llvm_orc_registerJITLoaderPerfEnd"),
           ExecutorSymbolDef::fromPtr(&llvm_orc_registerJITLoaderPerfEnd,
                                     JITSymbolFlags::Exported)},
          {session.intern("llvm_orc_registerJITLoaderPerfImpl"),
           ExecutorSymbolDef::fromPtr(
               &cubecl_orc_register_jit_loader_perf_impl,
               JITSymbolFlags::Exported)},
      }))) {
    return wrap(std::move(err));
  }

  auto plugin =
      PerfSupportPlugin::Create(session.getExecutorProcessControl(), dylib,
                                emit_debug_info, emit_unwind_info);
  if (!plugin) {
    return wrap(plugin.takeError());
  }
  // JITLink drops the debug sections before the perf plugin reads the line
  // table, unless this plugin keeps them. `llvm-jitlink -perf-support` does
  // the same.
  if (emit_debug_info) {
    auto preservation = DebugInfoPreservationPlugin::Create();
    if (!preservation) {
      return wrap(preservation.takeError());
    }
    layer->addPlugin(std::move(*preservation));
  }
  layer->addPlugin(std::move(*plugin));
  return nullptr;
}
