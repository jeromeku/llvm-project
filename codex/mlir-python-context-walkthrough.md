# MLIR Python IR: Context “with” Dispatch Walkthrough

This guide explains how core MLIR Python IR objects are implemented and shows the exact path taken when you run:

```
from mlir.ir import Context

with Context() as ctx:
    ...
```

It traces Python overlay code → compiled bindings → C API calls → thread‑local defaulting, with clickable source links.

---

## Layering: Where `Context` Comes From

- Compiled bindings define `ir._BaseContext` (nanobind C++ layer):
  - Constructor and context manager methods are exported here:
    - `__init__`: [`mlir/lib/Bindings/Python/IRCore.cpp:2914-2920`](mlir/lib/Bindings/Python/IRCore.cpp#L2914-L2920)
    - `__enter__` / `__exit__`: [`mlir/lib/Bindings/Python/IRCore.cpp:2928-2930`](mlir/lib/Bindings/Python/IRCore.cpp#L2928-L2930)
  - Static `current` accessor (gets thread‑local default):
    - [`mlir/lib/Bindings/Python/IRCore.cpp:2931-2939`](mlir/lib/Bindings/Python/IRCore.cpp#L2931-L2939)

- Python overlay replaces `ir.Context` with a subclass of `ir._BaseContext` that wires site‑specific init:
  - Subclass and rebinding: [`build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs/__init__.py:149-197`](build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs/__init__.py#L149-L197)
  - What the overlay does in `Context.__init__`:
    - Appends a dialect registry, runs post‑init hooks, sets multithreading/thread pool, and loads dialects (either all or selected): [`…:154-190`](build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs/__init__.py#L154-L190)

Summary: `mlir.ir.Context` you import is a Python subclass that inherits `__enter__/__exit__` from the C++ `_BaseContext` class.

Visual
```
User code
  │  from mlir.ir import Context
  ▼
Python overlay: mlir/_mlir_libs/__init__.py
  • class Context(ir._BaseContext)                [build/.../_mlir_libs/__init__.py:149-197]
  • __init__: registry, hooks, threading, dialects
       │ super().__init__() → creates MlirContext  [IRCore.cpp:2915-2919]
       ▼
C++ bindings (nanobind): ir._BaseContext
  • __enter__/__exit__ → TLS push/pop            [IRCore.cpp:2928-2930, 711-718]
  • current (thread-local)                       [IRCore.cpp:2931-2939]
```

---

## Step‑By‑Step: `with Context() as ctx:`

1) Construction
- Python calls the overlay `Context.__init__`, which first calls the base constructor:
  - Base constructor creates the underlying `MlirContext` (multithreading disabled initially):
    - [`mlir/lib/Bindings/Python/IRCore.cpp:2915-2919`](mlir/lib/Bindings/Python/IRCore.cpp#L2915-L2919)
- Then the overlay `Context.__init__` runs:
  - Append registry + post‑init hooks: [`_mlir_libs/__init__.py:154-156`](build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs/__init__.py#L154-L156)
  - Configure threading or thread pool: [`…:164-168`](build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs/__init__.py#L164-L168)
  - Load dialects (all vs selected): [`…:169-190`](build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs/__init__.py#L169-L190)
  - Register LLVM translations if available: [`…:191-195`](build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs/__init__.py#L191-L195)

2) Entering the `with` block
- `Context.__enter__` is inherited from `_BaseContext` and calls `PyMlirContext::contextEnter(...)`:
  - Binding site: [`mlir/lib/Bindings/Python/IRCore.cpp:2928`](mlir/lib/Bindings/Python/IRCore.cpp#L2928)
  - Implementation: pushes the context on a thread‑local stack and returns it:
    - [`mlir/lib/Bindings/Python/IRCore.cpp:711-713`](mlir/lib/Bindings/Python/IRCore.cpp#L711-L713)
    - Push details (copies insertion point / location defaults if same context):
      - [`mlir/lib/Bindings/Python/IRCore.cpp:812-830`](mlir/lib/Bindings/Python/IRCore.cpp#L812-L830)
  - Data structure for the thread‑local stack: `PyThreadContextEntry` (declaration):
  - [`mlir/lib/Bindings/Python/IRModule.h:109-147`](mlir/lib/Bindings/Python/IRModule.h#L109-L147)

Visual: `with Context()` flow
```
with Context() as ctx:
    # __init__ (overlay) → super().__init__()              [IRCore.cpp:2915-2919]
    # __enter__ (C++):
    #   PyMlirContext::contextEnter(ctx)                   [IRCore.cpp:711-713]
    #   → PyThreadContextEntry.pushContext(ctx)            [IRCore.cpp:812-830]
    #   (ctx now on TLS stack; inherits prior loc/ip if same context)
    ...
    # Defaulting resolver:
    #   DefaultingPyMlirContext.resolve() → top-of-stack   [IRCore.cpp:785-794]
    #   DefaultingPyLocation.resolve() → current loc       [IRCore.cpp:1064-1072]

# __exit__ (C++):
#   PyMlirContext::contextExit(...) → popContext           [IRCore.cpp:715-718, 874-882]
```

3) Inside the `with` block
- MLIR API functions declared with a defaultable `context` parameter (type `DefaultingPyMlirContext`) will resolve to the current thread‑local context if `None`:
  - Resolver: [`mlir/lib/Bindings/Python/IRCore.cpp:785-794`](mlir/lib/Bindings/Python/IRCore.cpp#L785-L794)
  - Type declaration: [`mlir/lib/Bindings/Python/IRModule.h:256-262`](mlir/lib/Bindings/Python/IRModule.h#L256-L262)
- Similarly for default locations and insertion points:
  - `DefaultingPyLocation::resolve`: [`mlir/lib/Bindings/Python/IRCore.cpp:1064-1072`](mlir/lib/Bindings/Python/IRCore.cpp#L1064-L1072)
  - `Location.current` property (thread‑local): [`mlir/lib/Bindings/Python/IRCore.cpp:3093-3100`](mlir/lib/Bindings/Python/IRCore.cpp#L3093-L3100)

Examples of defaulting in APIs (no explicit context passed → uses current):
- `Location.unknown(context=None)`: [`mlir/lib/Bindings/Python/IRCore.cpp:3101-3108`](mlir/lib/Bindings/Python/IRCore.cpp#L3101-L3108)
- Many other factory methods accept `DefaultingPyMlirContext context` and thus use the top‑of‑stack context under a `with Context():` block (see multiple usages across `IRCore.cpp`).

4) Exiting the `with` block
- `Context.__exit__` (inherited) calls `PyMlirContext::contextExit(...)`:
  - Binding site: [`mlir/lib/Bindings/Python/IRCore.cpp:2929-2930`](mlir/lib/Bindings/Python/IRCore.cpp#L2929-L2930)
  - Implementation: pops and verifies balance of the context frame:
    - [`mlir/lib/Bindings/Python/IRCore.cpp:715-718`](mlir/lib/Bindings/Python/IRCore.cpp#L715-L718)
    - Pop details and balance checks: [`mlir/lib/Bindings/Python/IRCore.cpp:874-882`](mlir/lib/Bindings/Python/IRCore.cpp#L874-L882)

---

## Useful Accessors During `with`

- The current context bound to the thread:
  - `Context.current`: [`mlir/lib/Bindings/Python/IRCore.cpp:2931-2939`](mlir/lib/Bindings/Python/IRCore.cpp#L2931-L2939)

- The current default `Location` bound to the thread (if any):
  - `Location.current`: [`mlir/lib/Bindings/Python/IRCore.cpp:3093-3100`](mlir/lib/Bindings/Python/IRCore.cpp#L3093-L3100)
  - Enter/exit handlers for `with Location(...):` also use the same stack:
    - Enter: [`mlir/lib/Bindings/Python/IRCore.cpp:1054-1056`](mlir/lib/Bindings/Python/IRCore.cpp#L1054-L1056)
    - Exit:  [`mlir/lib/Bindings/Python/IRCore.cpp:1058-1062`](mlir/lib/Bindings/Python/IRCore.cpp#L1058-L1062)

---

## Putting It Together (Annotated)

```
# 1) Context is Python subclass of C++ _BaseContext
from mlir.ir import Context  # overlay: [_mlir_libs/__init__.py:149-197]

with Context() as ctx:       # __enter__: pushes TLS frame [IRCore.cpp:711-713]
    # DefaultingPyMlirContext resolves to ctx [IRCore.cpp:785-794]
    loc = Location.unknown() # no context arg: uses current [IRCore.cpp:3101-3108]
    # ... any API with DefaultingPyMlirContext gets ctx implicitly

# __exit__: pops TLS frame and checks balance [IRCore.cpp:715-718, 874-882]
```

This is why omitting explicit `context=` in many API calls works inside a `with Context():` block: the defaulting wrappers read the top of the thread‑local context stack.

Visual: Thread‑local stack frames
```
Top of TLS stack  ┌────────────────────────────────────────────┐
                  │ FrameKind=Context                          │
                  │ context = ctx0                             │  ← pushContext(ctx0)
                  │ insertionPoint = (inherited or None)       │
                  │ location = (inherited or None)             │
                  └────────────────────────────────────────────┘
                  ┌────────────────────────────────────────────┐
                  │ FrameKind=Location                         │
                  │ context = ctx0                             │  ← pushLocation(loc)
                  │ insertionPoint = (copied from below)       │
                  │ location = loc                             │
                  └────────────────────────────────────────────┘
                  ┌────────────────────────────────────────────┐
                  │ FrameKind=Context                          │
                  │ context = ctx0                             │  ← nested with Context():
                  │ insertionPoint = (copied from below)       │     copies loc/ip if same ctx
                  │ location = (copied from below)             │
                  └────────────────────────────────────────────┘
Bottom
```

---

## Appendix: Trace a Concrete API Call (with visual)

Example: `Location.unknown()` inside a `with Context()`

- Binding of `Location.unknown(context=None)` uses a defaulting context param:
  - [`mlir/lib/Bindings/Python/IRCore.cpp:3101-3108`](mlir/lib/Bindings/Python/IRCore.cpp#L3101-L3108)
- Defaulting resolves from TLS if `None` is passed:
  - [`mlir/lib/Bindings/Python/IRCore.cpp:785-794`](mlir/lib/Bindings/Python/IRCore.cpp#L785-L794)

Visual
```
User: with Context() as ctx:
         loc = Location.unknown()  # no context arg

Call:  Location.unknown(DefaultingPyMlirContext context=None)
          └─ DefaultingPyMlirContext.resolve() → ctx      [IRCore.cpp:785-794]
              (reads top of TLS stack)
          └─ create MlirLocationUnknown in ctx            [IRCore.cpp:3101-3108]
          └─ wrap as PyLocation(contextRef=ctx)
```

Example: `Operation.create(…, loc=None)` default location resolution

- If no `loc` is provided by user code, bindings will resolve a default location:
  - Many call sites fill default locations via `DefaultingPyLocation::resolve()`; a representative use:
    - [`mlir/lib/Bindings/Python/IRCore.cpp:250`](mlir/lib/Bindings/Python/IRCore.cpp#L250)
- Operation creation requires a `PyLocation &location` and then threads it into `MlirOperationState`:
  - Entry: [`mlir/lib/Bindings/Python/IRCore.cpp:1407-1413`](mlir/lib/Bindings/Python/IRCore.cpp#L1407-L1413)
  - State init (uses location): [`mlir/lib/Bindings/Python/IRCore.cpp:1477-1479`](mlir/lib/Bindings/Python/IRCore.cpp#L1477-L1479)

Visual
```
User: with Context():
         op = Operation.create("arith.addi", ..., loc=None)

Flow:
  parse args → ensure/resolve location (DefaultingPyLocation if needed)
      └─ DefaultingPyLocation.resolve() → current Location  [IRCore.cpp:1064-1072]
  build MlirOperationState(name, location)                  [IRCore.cpp:1477-1479]
  add operands/results/attributes/successors
  create + wrap PyOperation
  optionally insert at current InsertionPoint (if provided)
```

