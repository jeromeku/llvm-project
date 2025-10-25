# 🎓 MLIR Dialect Conversion Tutorial

> **A comprehensive, hands-on tutorial for mastering MLIR's dialect conversion infrastructure**

## 🚀 Quick Start (Choose Your Path)

### 👶 Just Want to Get Started?
→ Read [QUICKSTART.md](QUICKSTART.md) (5 minutes)

### 📖 Want the Full Tutorial?
→ Read [README.md](README.md) (comprehensive guide)

### 🧠 Want to Understand Internals?
→ Read [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) (deep dive)

### 🗺️ Want to Navigate Everything?
→ Read [INDEX.md](INDEX.md) (complete index)

### 📦 Want to Know What's Here?
→ Read [TUTORIAL_COMPLETE.md](TUTORIAL_COMPLETE.md) (overview)

---

## What This Tutorial Covers

✅ **Basic Conversion Patterns** - How to transform operations between dialects
✅ **Type Conversion** - Converting custom types with automatic materialization
✅ **Partial Conversion** - Progressive lowering through multiple abstraction levels
✅ **Dynamic Legality** - Conditional operation conversion
✅ **MLIR Internals** - How the conversion framework actually works

## Three Complete Examples

1. **Simple Arithmetic** ([SimpleArithLowering.cpp](lib/SimpleArithLowering.cpp))
   - Basic operation-to-operation lowering
   - Pattern structure and rewriter usage
   - ConversionTarget setup

2. **Type Conversion** ([TypedArithLowering.cpp](lib/TypedArithLowering.cpp))
   - Custom types (FixedPoint with scale)
   - TypeConverter with materialization
   - Signature conversion

3. **Partial Conversion** ([PartialLowering.cpp](lib/PartialLowering.cpp))
   - Dynamic legality (size-based)
   - Mixed dialect IR
   - Multi-stage lowering

## Try It Now!

```bash
# Build all examples
./build.sh

# Run all examples with explanations
./test.sh

# Or try individual examples:
./build/bin/simple-opt test/simple-arith.mlir -convert-simple-to-arith
./build/bin/typed-opt test/typed-arith.mlir -convert-typed-to-std
./build/bin/partial-opt test/partial-conversion.mlir -partial-lower-high
```

## What Makes This Tutorial Special?

- 📝 **2,500+ lines of heavily annotated code**
- 📚 **20,000 words of documentation** covering beginner to expert level
- 🔗 **50+ links to MLIR source code** for deep understanding
- 🎯 **Runnable examples** you can modify and experiment with
- 🔧 **Complete build system** that auto-detects your MLIR installation
- 🐛 **Debug guides** with concrete troubleshooting commands

## File Structure

```
claude/conversion/
├── START_HERE.md           ← You are here!
├── QUICKSTART.md           ← 5-minute getting started
├── README.md               ← Main tutorial (~10,000 words)
├── IMPLEMENTATION_GUIDE.md ← Internals deep dive (~8,000 words)
├── INDEX.md                ← Complete navigation
├── TUTORIAL_COMPLETE.md    ← Package overview
│
├── include/                ← Dialect definitions (TableGen)
├── lib/                    ← Conversion implementations
├── tools/                  ← Test tools (simple-opt, typed-opt, partial-opt)
├── test/                   ← Test cases (.mlir files)
│
├── build.sh               ← Automated build script
└── test.sh                ← Automated test runner
```

## Next Steps

1. **Read the guide** for your level (see choices at top)
2. **Build the examples**: `./build.sh`
3. **Run the tests**: `./test.sh`
4. **Study the code**: Start with [lib/SimpleArithLowering.cpp](lib/SimpleArithLowering.cpp)
5. **Experiment**: Modify examples, add operations
6. **Go deeper**: Read [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)

## Need Help?

- Check the [QUICKSTART.md](QUICKSTART.md) for common issues
- Read the extensive comments in example files
- Use the debug commands in the guides
- Ask on [MLIR Discourse](https://discourse.llvm.org/c/mlir/31)

---

**Ready? Pick a guide above and start learning!** 🚀

> All links in this tutorial work in VSCode - just Cmd/Ctrl+Click them!
