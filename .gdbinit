# ============================================================================
# GDB Init File for C++ Template Debugging
# ============================================================================

# Enable pretty printing for STL containers
set print pretty on
set print object on
set print static-members on
set print vtbl on
set print demangle on
set demangle-style gnu-v3

# Better template display
set print array on
set print array-indexes on

# Pagination and output settings
set pagination off
set confirm off

# History settings
set history save on
set history size 10000
set history filename ~/.gdb_history

# Disable address randomization for reproducible debugging
set disable-randomization on

# Print thread info when stopping
set print thread-events on

# Follow fork behavior (useful for multi-process debugging)
set follow-fork-mode child

# Stop all threads when any thread hits a breakpoint
set scheduler-locking off
set non-stop off

# Enhanced backtrace display
set backtrace past-main on
set backtrace past-entry on
set backtrace limit 100

# Symbol loading
set auto-load safe-path /
set auto-solib-add on

# Memory display preferences
set print elements 200
set print repeats 10

# ============================================================================
# Custom Commands for C++ Template Debugging
# ============================================================================

# Print std::vector contents
define pvector
    if $argc == 0
        help pvector
    else
        set $size = $arg0._M_impl._M_finish - $arg0._M_impl._M_start
        set $capacity = $arg0._M_impl._M_end_of_storage - $arg0._M_impl._M_start
        set $i = 0
        printf "Vector size: %d, capacity: %d\n", $size, $capacity
        while $i < $size
            printf "[%d] = ", $i
            p *($arg0._M_impl._M_start + $i)
            set $i++
        end
    end
end
document pvector
Print the contents of a std::vector
Usage: pvector VECTOR_NAME
end

# Print std::map contents
define pmap
    if $argc == 0
        help pmap
    else
        set $tree = $arg0._M_t
        set $size = $tree._M_impl._M_node_count
        printf "Map size: %d\n", $size
        if $size > 0
            # This is simplified - actual implementation varies by GCC version
            printf "Use 'p $arg0' with pretty-printers for full map display\n"
        end
    end
end
document pmap
Print the size of a std::map
Usage: pmap MAP_NAME
end

# Print template type info
define ptype_full
    if $argc == 0
        help ptype_full
    else
        ptype /o $arg0
    end
end
document ptype_full
Print full type information including template parameters and offsets
Usage: ptype_full VARIABLE_OR_TYPE
end

# Quick breakpoint on exception throw
define bexcept
    catch throw
    commands
        backtrace
        info locals
        continue
    end
end
document bexcept
Set a breakpoint that triggers on C++ exceptions and prints context
Usage: bexcept
end

# Print this pointer in detail
define pthis
    if $argc == 0
        print *this
        ptype /o *this
    else
        print $arg0
        ptype /o $arg0
    end
end
document pthis
Print 'this' pointer with full type information
Usage: pthis [OBJECT]
end

# Better stack trace with local variables
define btfull
    set $frame = 0
    while 1
        frame $frame
        info frame
        info args
        info locals
        set $frame = $frame + 1
    end
end
document btfull
Print full backtrace with arguments and locals for each frame
Usage: btfull
end

# ============================================================================
# Convenience Settings
# ============================================================================

# Shorter aliases
alias bt = backtrace
alias c = continue
alias s = step
alias n = next
alias f = frame
alias p = print
alias q = quit

# STL-specific shortcuts
alias pv = pvector
alias pm = pmap
alias pt = ptype_full

# ============================================================================
# Visual Improvements
# ============================================================================

# Color prompt (if terminal supports it)
set prompt \033[1;32m(gdb) \033[0m

# ============================================================================
# Performance settings for large template-heavy codebases
# ============================================================================

# Cache symbol lookups
set unwindonsignal on

# Don't ask to load symbols from system libraries
set auto-load python-scripts on

# ============================================================================
# Python Pretty Printers (requires GDB with Python support)
# ============================================================================

python
import sys
import os

# Try to load GCC's libstdc++ pretty printers
try:
    # Common locations for libstdc++ printers
    gcc_printer_paths = [
        '/usr/share/gcc/python',
        '/usr/share/gdb/auto-load',
    ]
    
    for path in gcc_printer_paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
    
    from libstdcxx.v6.printers import register_libstdcxx_printers
    register_libstdcxx_printers(None)
    print("Loaded libstdc++ pretty printers")
except ImportError:
    print("Warning: Could not load libstdc++ pretty printers")
    print("Install them with: sudo apt-get install libstdc++6-dbg (Ubuntu/Debian)")
except Exception as e:
    print(f"Error loading pretty printers: {e}")

end

# ============================================================================
# Startup Message
# ============================================================================

printf "\n"
printf "================================================\n"
printf "  C++ Template Debugging Environment Loaded\n"
printf "================================================\n"
printf "Custom commands: pvector, pmap, ptype_full,\n"
printf "                 bexcept, pthis, btfull\n"
printf "Pretty printing: enabled for STL containers\n"
printf "================================================\n"
printf "\n"