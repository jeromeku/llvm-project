from mlir import ir
from mlir.dialects import gpu
PADDING = " " 
IDENT1 = PADDING * 2
IDENT2 = IDENT1 * 2
VERBOSE = False

def lookup_target(target: int):
    if target == gpu.CompilationTarget.Assembly:
        return "ptx"
    elif target == gpu.CompilationTarget.Binary:
        return "cubin"
    elif target == gpu.CompilationTarget.Fatbin:
        return "fatbin"
    else:
        raise ValueError(f"Unrecognized target: {target}")
    
def maybe_print(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

def bytes_to_str(b: bytes):
    b = b.rstrip(b'\x00')
    try:
        s = b.decode('utf-8')
    except UnicodeDecodeError:
        s = b.decode('latin-1')
    return s.replace('\r\n', '\n')

if __name__ == "__main__":
    with ir.Context():
        mod = ir.Module.parseFile('module_after.mlir')
        main_op = mod.operation
        main_block = main_op.regions[0].blocks[0]
        for i, region in enumerate(main_op.regions):
            maybe_print(f"Region {i}: num blocks {len(region.blocks)}")
            for j, block in enumerate(region.blocks):
                maybe_print(f"{IDENT1}Block {j}: num ops {len(block.operations)}")
                for k, op in enumerate(block.operations):
                    maybe_print(f"{IDENT2}Op {k}: {op.name}")
                    if isinstance(op, gpu.BinaryOp):
                        objects: list[gpu.ObjectAttr] = op.objects
                        assert len(objects) == 1
                        gpu_obj = gpu.ObjectAttr(objects[0])
                        target = lookup_target(gpu_obj.format)
                        print(f"Found gpu object, target = {gpu_obj.target} with compilation target {target}")
                        kernel_bytes: bytes = gpu_obj.object
                        kernel_str = bytes_to_str(kernel_bytes)
                        print(kernel_str)
                        break