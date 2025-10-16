// Minimal C shim to expose DialectRegistry::getDialectNames() to ctypes.
//
// Provides two entry points:
//  - mlir_collect_all_dialect_names: creates a fresh registry, registers all
//    upstream dialects, returns names.
//  - mlir_collect_dialect_names_from_registry: takes an existing
//    MlirDialectRegistry handle and returns names from it.

#include <cstdlib>
#include <cstring>
#include <vector>

#include "mlir-c/IR.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"

extern "C" {

// Returns 0 on success; on success the caller must free via
// mlir_free_dialect_names(names, count).
int mlir_collect_all_dialect_names(char ***outNames, size_t *count) {
  if (!outNames || !count)
    return -1;
  *outNames = nullptr;
  *count = 0;

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  std::vector<const char *> cstrs;
  for (mlir::StringRef name : registry.getDialectNames()) {
    char *s = static_cast<char *>(std::malloc(name.size() + 1));
    if (!s) {
      for (char *p : cstrs)
        std::free(p);
      return -2;
    }
    std::memcpy(s, name.data(), name.size());
    s[name.size()] = '\0';
    cstrs.push_back(s);
  }

  char **arr =
      static_cast<char **>(std::malloc(sizeof(char *) * cstrs.size()));
  if (!arr) {
    for (char *p : cstrs)
      std::free(p);
    return -3;
  }
  std::memcpy(arr, cstrs.data(), sizeof(char *) * cstrs.size());
  *outNames = arr;
  *count = cstrs.size();
  return 0;
}

// Variant that accepts an existing MlirDialectRegistry C handle.
// The handle typically comes from a Python capsule via
// mlirPythonCapsuleToDialectRegistry (see Interop.h).
int mlir_collect_dialect_names_from_registry(MlirDialectRegistry registry,
                                             char ***outNames, size_t *count) {
  if (!outNames || !count)
    return -1;
  *outNames = nullptr;
  *count = 0;

  mlir::DialectRegistry *reg = mlir::unwrap(registry);
  if (!reg)
    return -4;

  std::vector<const char *> cstrs;
  for (mlir::StringRef name : reg->getDialectNames()) {
    char *s = static_cast<char *>(std::malloc(name.size() + 1));
    if (!s) {
      for (char *p : cstrs)
        std::free(p);
      return -2;
    }
    std::memcpy(s, name.data(), name.size());
    s[name.size()] = '\0';
    cstrs.push_back(s);
  }

  char **arr =
      static_cast<char **>(std::malloc(sizeof(char *) * cstrs.size()));
  if (!arr) {
    for (char *p : cstrs)
      std::free(p);
    return -3;
  }
  std::memcpy(arr, cstrs.data(), sizeof(char *) * cstrs.size());
  *outNames = arr;
  *count = cstrs.size();
  return 0;
}

void mlir_free_dialect_names(char **names, size_t count) {
  if (!names)
    return;
  for (size_t i = 0; i < count; ++i)
    std::free(names[i]);
  std::free(names);
}

} // extern "C"

