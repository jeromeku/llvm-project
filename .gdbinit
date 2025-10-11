# ============================================================================
# GDB Init File for C++ Template Debugging (portable, pretty-printer aware)
# ============================================================================

# Enable pretty printing and readable C++
set print pretty on
set print object on
set print static-members on
set print vtbl on
set print demangle on
set demangle-style gnu-v3

# Arrays / readability
set print array on
set print array-indexes on
set print elements 200
set print repeats 10

# Output behavior
set pagination off
set confirm off

# History
set history save on
set history size 10000
set history filename ~/.gdb_history

# Disable ASLR for reproducible debugging (Linux only)
set disable-randomization on

# Thread / fork behavior
set print thread-events on
set follow-fork-mode child
# All-stop mode: when any thread hits a breakpoint, all threads stop.
set non-stop off

# Backtraces
set backtrace past-main on
set backtrace past-entry on
set backtrace limit 100

# Symbol / solib loading
# Limit auto-load to trusted dirs (avoid `set auto-load safe-path /`)
set auto-load safe-path ~/.gdb:~/.config/gdb
set auto-solib-add on
# Optional: let GDB unwind through signal frames (useful in some crash scenarios)
set unwindonsignal on

# ============================================================================
# Visuals
# ============================================================================
set prompt \033[1;32m(gdb) \033[0m

# ============================================================================
# Python: Pretty Printers + Pretty-Printer-Aware Commands
# ============================================================================

python
import gdb
import sys, os

def _load_libstdcxx_printers():
    """
    Try to load libstdc++ pretty-printers if not auto-loaded by the distro.
    This is defensive; many distros already auto-load from .../gdb/auto-load/.
    """
    candidates = [
        '/usr/share/gcc/python',
        '/usr/share/gdb/auto-load',
        '/usr/share/libstdc++/python',              # some distros
        '/usr/lib64/gcc/*/*/python',                # glob patterns won't work here, but retained as docs
        os.path.expanduser('~/.gdb/libstdcxx'),
    ]
    for p in candidates:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    try:
        # libstdcxx printers
        from libstdcxx.v6.printers import register_libstdcxx_printers
        # Register on current objfile (None also works globally in newer gdb)
        try:
            register_libstdcxx_printers(gdb.current_objfile())
        except Exception:
            register_libstdcxx_printers(None)
        gdb.write("Loaded libstdc++ pretty printers\n")
    except Exception as e:
        gdb.write(f"Note: libstdc++ pretty printers not loaded automatically ({e}).\n")

# Register printers (best-effort)
try:
    _load_libstdcxx_printers()
except Exception as _e:
    gdb.write(f"Note: skipping printer setup: {_e}\n")

def _iter_children(v):
    """Yield (name, child) using default pretty-printer if available; else None."""
    viz = gdb.default_visualizer(v)
    if viz is None:
        return None
    try:
        return viz.children()
    except Exception:
        return None

class PVector(gdb.Command):
    """Print std::vector contents using pretty-printers or safe fallbacks.
Usage: pvector EXPR [LIMIT]
Examples:
  pvector myvec
  pvector myns::myvec 16
"""
    def __init__(self):
        super(PVector, self).__init__("pvector", gdb.COMMAND_DATA)

    def invoke(self, arg, from_tty):
        argv = gdb.string_to_argv(arg)
        if not argv:
            gdb.write(self.__doc__)
            return
        expr = argv[0]
        limit = None
        if len(argv) >= 2:
            try:
                limit = int(argv[1])
            except ValueError:
                gdb.write("LIMIT must be an integer\n")
                return

        v = gdb.parse_and_eval(expr)

        # First try pretty-printer (portable across libstdc++ versions)
        children = _iter_children(v)
        if children is not None:
            count = 0
            gdb.write(f"Vector {expr} (via pretty-printer):\n")
            for name, child in children:
                gdb.write(f"  {name} = {child}\n")
                count += 1
                if limit is not None and count >= limit:
                    break
            gdb.write(f"  (shown {count}{'' if limit is None else f' / <= {limit}'} elements)\n")
            return

        # Fallback: try calling size() and operator[]
        try:
            size = int(gdb.parse_and_eval(f"({expr}).size()"))
        except Exception:
            gdb.write("No pretty-printer and cannot call size(); compile with -g and enable libstdc++ printers.\n")
            return

        shown = size if limit is None else min(size, limit)
        gdb.write(f"Vector {expr} (no printer): size={size}\n")
        for i in range(shown):
            try:
                item = gdb.parse_and_eval(f"({expr})[{i}]")
                gdb.write(f"  [{i}] = {item}\n")
            except Exception as e:
                gdb.write(f"  [{i}] = <error: {e}>\n")

class PMap(gdb.Command):
    """Print std::map-like contents using pretty-printers or show size as fallback.
Usage: pmap EXPR [LIMIT]
"""
    def __init__(self):
        super(PMap, self).__init__("pmap", gdb.COMMAND_DATA)

    def invoke(self, arg, from_tty):
        argv = gdb.string_to_argv(arg)
        if not argv:
            gdb.write(self.__doc__)
            return
        expr = argv[0]
        limit = None
        if len(argv) >= 2:
            try:
                limit = int(argv[1])
            except ValueError:
                gdb.write("LIMIT must be an integer\n")
                return

        m = gdb.parse_and_eval(expr)
        children = _iter_children(m)
        if children is not None:
            count = 0
            gdb.write(f"Map {expr} (via pretty-printer):\n")
            for name, pair in children:
                # libstdc++ map printer yields a std::pair-like value
                try:
                    first = pair['first']
                    second = pair['second']
                    gdb.write(f"  [{first}] = {second}\n")
                except Exception:
                    # If structure differs, fall back to generic print
                    gdb.write(f"  {name} = {pair}\n")
                count += 1
                if limit is not None and count >= limit:
                    break
            gdb.write(f"  (shown {count}{'' if limit is None else f' / <= {limit}'} entries)\n")
            return

        # Fallback: just print size()
        try:
            size = int(gdb.parse_and_eval(f"({expr}).size()"))
            gdb.write(f"Map {expr}: size={size}\n")
            gdb.write("Tip: enable pretty-printers to see entries: see libstdc++ v6 printers.\n")
        except Exception:
            gdb.write("No pretty-printer and cannot call size(); compile with -g and enable libstdc++ printers.\n")

# Register commands
PVector()
PMap()
end

# ============================================================================
# Startup Message
# ============================================================================
printf "\n"
printf "================================================\n"
printf "  C++ Template Debugging Environment Loaded\n"
printf "================================================\n"
printf "Custom commands: pvector (pv), pmap (pm), ptype_full (pt),\n"
printf "                 bexcept, pthis, btfull\n"
printf "Pretty printing: enabled (best-effort) for libstdc++\n"
printf "================================================\n"
printf "\n"
