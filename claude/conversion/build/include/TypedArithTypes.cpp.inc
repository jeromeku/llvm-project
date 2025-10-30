/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::mlir::typed::FixedPointType,
::mlir::typed::ComplexType

#endif  // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::typed::FixedPointType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::typed::FixedPointType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::typed::ComplexType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::typed::ComplexType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::llvm::LogicalResult>(def)    .Case<::mlir::typed::FixedPointType>([&](auto t) {
      printer << ::mlir::typed::FixedPointType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::typed::ComplexType>([&](auto t) {
      printer << ::mlir::typed::ComplexType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace typed {
namespace detail {
struct FixedPointTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<unsigned, unsigned>;
  FixedPointTypeStorage(unsigned width, unsigned scale) : width(std::move(width)), scale(std::move(scale)) {}

  KeyTy getAsKey() const {
    return KeyTy(width, scale);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (width == std::get<0>(tblgenKey)) && (scale == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static FixedPointTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto width = std::move(std::get<0>(tblgenKey));
    auto scale = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<FixedPointTypeStorage>()) FixedPointTypeStorage(std::move(width), std::move(scale));
  }

  unsigned width;
  unsigned scale;
};
} // namespace detail
FixedPointType FixedPointType::get(::mlir::MLIRContext *context, unsigned width, unsigned scale) {
  return Base::get(context, std::move(width), std::move(scale));
}

FixedPointType FixedPointType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, unsigned width, unsigned scale) {
  return Base::getChecked(emitError, context, std::move(width), std::move(scale));
}

FixedPointType FixedPointType::get(unsigned width, unsigned scale) {
  return Base::get(nullptr, width, scale);
}

FixedPointType FixedPointType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned width, unsigned scale) {
  return Base::getChecked(emitError, nullptr, width, scale);
}

::llvm::LogicalResult FixedPointType::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned width, unsigned scale) {
  if (::mlir::failed(verify(emitError, width, scale)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Type FixedPointType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<unsigned> _result_width;
  ::mlir::FailureOr<unsigned> _result_scale;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'width'
  _result_width = ::mlir::FieldParser<unsigned>::parse(odsParser);
  if (::mlir::failed(_result_width)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse FixedPointType parameter 'width' which is to be a `unsigned`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'scale'
  _result_scale = ::mlir::FieldParser<unsigned>::parse(odsParser);
  if (::mlir::failed(_result_scale)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse FixedPointType parameter 'scale' which is to be a `unsigned`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_width));
  assert(::mlir::succeeded(_result_scale));
  return odsParser.getChecked<FixedPointType>(odsLoc, odsParser.getContext(),
      unsigned((*_result_width)),
      unsigned((*_result_scale)));
}

void FixedPointType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getWidth());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getScale());
  odsPrinter << ">";
}

unsigned FixedPointType::getWidth() const {
  return getImpl()->width;
}

unsigned FixedPointType::getScale() const {
  return getImpl()->scale;
}

} // namespace typed
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::typed::FixedPointType)
namespace mlir {
namespace typed {
namespace detail {
struct ComplexTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<Type>;
  ComplexTypeStorage(Type elementType) : elementType(std::move(elementType)) {}

  KeyTy getAsKey() const {
    return KeyTy(elementType);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (elementType == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ComplexTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto elementType = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ComplexTypeStorage>()) ComplexTypeStorage(std::move(elementType));
  }

  Type elementType;
};
} // namespace detail
ComplexType ComplexType::get(::mlir::MLIRContext *context, Type elementType) {
  return Base::get(context, std::move(elementType));
}

ComplexType ComplexType::get(Type elementType) {
  return Base::get(elementType.getContext(), elementType);
}

::mlir::Type ComplexType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<Type> _result_elementType;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'elementType'
  _result_elementType = ::mlir::FieldParser<Type>::parse(odsParser);
  if (::mlir::failed(_result_elementType)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse ComplexType parameter 'elementType' which is to be a `Type`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_elementType));
  return ComplexType::get(odsParser.getContext(),
      Type((*_result_elementType)));
}

void ComplexType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getElementType());
  odsPrinter << ">";
}

Type ComplexType::getElementType() const {
  return getImpl()->elementType;
}

} // namespace typed
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::typed::ComplexType)
namespace mlir {
namespace typed {

/// Parse a type registered to this dialect.
::mlir::Type TypedArithDialect::parseType(::mlir::DialectAsmParser &parser) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef mnemonic;
  ::mlir::Type genType;
  auto parseResult = generatedTypeParser(parser, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;
  
  parser.emitError(typeLoc) << "unknown  type `"
      << mnemonic << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print a type registered to this dialect.
void TypedArithDialect::printType(::mlir::Type type,
                    ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedTypePrinter(type, printer)))
    return;
  
}
} // namespace typed
} // namespace mlir

#endif  // GET_TYPEDEF_CLASSES

