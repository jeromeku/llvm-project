import ctypes
import subprocess
import os

def examine_library(lib_path):
    print(f"Examining library: {lib_path}")
    
    # Check if file exists
    if not os.path.exists(lib_path):
        print("Library file not found!")
        return
    
    # Use nm to list symbols
    try:
        result = subprocess.run(['nm', '-D', '--defined-only', lib_path], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            symbols = []
            for line in result.stdout.split('\n'):
                if ' T ' in line:  # Function symbols
                    symbol_name = line.split()[-1]
                    symbols.append(symbol_name)
            
            print(f"Found {len(symbols)} function symbols:")
            for symbol in symbols:  # Show first 10
                print(f"  {symbol}")
            
            # if len(symbols) > 10:
            #     print(f"  ... and {len(symbols) - 10} more")
                
        else:
            print("nm failed:", result.stderr)
            
    except FileNotFoundError:
        print("nm command not available")
    
    # Try to load with ctypes
    try:
        lib = ctypes.CDLL(lib_path)
        print("✓ Library loaded successfully with ctypes")
        return lib
    except Exception as e:
        print(f"✗ Failed to load library: {e}")
        return None

# Usage
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("lib")
    args = parser.parse_args()
    lib = examine_library(args.lib)