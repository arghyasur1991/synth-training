// Stub implementations for JIT serialization functions not compiled in mobile builds.
// These are referenced by the C++ API's output-archive.cpp (inside libtorch_cpu.a)
// but the actual export.cpp is excluded from INTERN_BUILD_MOBILE builds.
// On Android we never call Module::save() or ExportModule(), so these are safe stubs.

#include <torch/csrc/jit/api/module.h>
#include <stdexcept>

namespace torch::jit {

void ExportModule(
    const Module& /*module*/,
    const std::string& /*filename*/,
    const std::unordered_map<std::string, std::string>& /*extra_files*/,
    bool /*bytecode_format*/,
    bool /*save_mobile_debug_info*/,
    bool /*use_flatbuffer*/) {
  throw std::runtime_error("JIT ExportModule not available on Android");
}

void ExportModule(
    const Module& /*module*/,
    std::ostream& /*out*/,
    const std::unordered_map<std::string, std::string>& /*extra_files*/,
    bool /*bytecode_format*/,
    bool /*save_mobile_debug_info*/,
    bool /*use_flatbuffer*/) {
  throw std::runtime_error("JIT ExportModule not available on Android");
}

void ExportModule(
    const Module& /*module*/,
    const std::function<size_t(const void*, size_t)>& /*writer_func*/,
    const std::unordered_map<std::string, std::string>& /*extra_files*/,
    bool /*bytecode_format*/,
    bool /*save_mobile_debug_info*/,
    bool /*use_flatbuffer*/) {
  throw std::runtime_error("JIT ExportModule not available on Android");
}

void Module::save(
    const std::string& /*filename*/,
    const std::unordered_map<std::string, std::string>& /*extra_files*/) const {
  throw std::runtime_error("JIT Module::save not available on Android");
}

void Module::save(
    std::ostream& /*out*/,
    const std::unordered_map<std::string, std::string>& /*extra_files*/) const {
  throw std::runtime_error("JIT Module::save not available on Android");
}

}  // namespace torch::jit
