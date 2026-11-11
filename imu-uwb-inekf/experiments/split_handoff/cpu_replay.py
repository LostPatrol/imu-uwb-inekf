"""Run independent offline replay cases on available cores, preferring physical P cores."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import ctypes
from ctypes import wintypes
import multiprocessing as mp
import os
import struct


def preferred_processors():
    """Read Windows CPU sets; rank one thread per fast core before sharing any core."""
    fallback_count = getattr(os, 'process_cpu_count', os.cpu_count)() or 1
    if os.name != 'nt':
        return [(0, i) for i in sorted(os.sched_getaffinity(0))] if hasattr(os, 'sched_getaffinity') else [(0, i) for i in range(fallback_count)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    get_sets = kernel.GetSystemCpuSetInformation
    get_sets.argtypes = [wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.HANDLE, wintypes.DWORD]
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    affinity = ctypes.c_size_t()
    system_mask = ctypes.c_size_t()
    kernel.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    single_group = kernel.GetActiveProcessorGroupCount() == 1
    has_mask = single_group and kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), ctypes.byref(affinity), ctypes.byref(system_mask))
    size = wintypes.DWORD()
    get_sets(None, 0, ctypes.byref(size), kernel.GetCurrentProcess(), 0)
    buf = ctypes.create_string_buffer(size.value)
    if not size.value or not get_sets(buf, size.value, ctypes.byref(size), kernel.GetCurrentProcess(), 0):
        return [(0, i) for i in range(fallback_count) if not has_mask or affinity.value & (1 << i)]
    per_core = defaultdict(list)
    cursor = 0
    while cursor + 8 <= size.value:
        length, kind = struct.unpack_from('<II', buf, cursor)
        if length < 8 or cursor + length > size.value:
            break
        if kind == 0 and length >= 20:
            _, group, logical, core, _, _, efficiency, flags = struct.unpack_from('<IH6B', buf, cursor + 8)
            # Bit 1 is allocated; bit 2 identifies allocation to this very process.
            if (not has_mask or affinity.value & (1 << logical)) and (not flags & 2 or flags & 4):
                per_core[-efficiency, group, core].append((group, logical))
        cursor += length
    cores = [sorted(per_core[key]) for key in sorted(per_core)]
    return [core[k] for k in range(max(map(len, cores), default=0)) for core in cores if k < len(core)] or [(0, 0)]


def _prefer_worker(counter, processors, fill_machine):
    """Prefer fast cores; spread a full batch across distinct processors on Windows."""
    with counter.get_lock():
        group, logical = processors[counter.value % len(processors)]
        counter.value += 1
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentThread.restype = wintypes.HANDLE
        kernel.SetThreadIdealProcessorEx.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.LPVOID]
        # Request normal-performance QoS for this worker only; leave system power settings alone.
        power = (wintypes.DWORD * 3)(1, 1, 0)
        kernel.SetProcessInformation(wintypes.HANDLE(-1), 4, ctypes.byref(power), ctypes.sizeof(power))
        # PROCESSOR_NUMBER: USHORT group, BYTE processor number, BYTE reserved.
        number = ctypes.create_string_buffer(struct.pack('<HBB', group, logical, 0))
        kernel.SetThreadIdealProcessorEx(kernel.GetCurrentThread(), number, None)
        if fill_machine and kernel.GetActiveProcessorGroupCount() == 1:
            # The one numerical thread per process can otherwise bunch on efficient cores.
            kernel.SetThreadAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
            kernel.SetThreadAffinityMask.restype = ctypes.c_size_t
            kernel.SetThreadAffinityMask(kernel.GetCurrentThread(), 1 << logical)


def replay_pool(case_count, workers=None):
    """Create a pool sized to independent cases and available CPUs, with one BLAS thread each."""
    processors = preferred_processors()
    count = max(1, min(case_count, len(processors), workers or len(processors)))
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[var] = '1'
    ctx = mp.get_context('spawn')
    return ProcessPoolExecutor(max_workers=count, mp_context=ctx, initializer=_prefer_worker,
                               initargs=(ctx.Value('i', 0), processors, count == len(processors)))


if __name__ == '__main__':
    print('Preferred (processor group, logical processor):', preferred_processors())
