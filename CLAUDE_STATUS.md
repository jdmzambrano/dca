# Claude Update: HLS Notebook Status

## Root cause identified (2026-04-29)
The synthesis failure `ERROR: [HLS 200-1715] Encountered problem during source synthesis` was traced to:

```
INFO-FLOW: Error in clang_39: error deleting
  ".autopilot/db/a.g.ld.0.bc.clang.reflow.err.log": permission denied
```

Vitis HLS runs the LTO clang step as a **background process** (`poll_ms 5000`).
On Windows that process keeps `a.g.ld.0.bc.clang.reflow.err.log` open as a
memory-mapped file. On the next synthesis run, Vitis tries to delete the old
log before creating a new one — and Windows refuses because it is still mapped.
This immediately kills the LTO step and fires HLS 200-1715.

The `shutil.rmtree(prj_dir, ignore_errors=True)` in `run_hls_build.py` was
silently failing for the same reason, leaving the locked directory in place.

## Fixes applied to run_hls_build.py
1. **Kill lingering processes before cleanup**: `taskkill /F /IM clang.exe` (and opt.exe, llvm-link.exe) + 2-second sleep so Windows releases handles.
2. **Proper rmtree error handler**: replaced `ignore_errors=True` with an `onerror` handler that `chmod`s and retries; prints a warning if deletion still fails.
3. **Broadened build_prj.tcl patch**: now handles all three variant forms of `config_array_partition -maximum_size` that hls4ml can generate.

## Current state
- `run_hls_build.py` is updated and ready to run.
- The `myproject_prj` directory still exists from failed runs; the updated script will kill the processes and clean it on next run.
- All other parts (training, weight export, hls4ml conversion) remain working.

## To reproduce a clean synthesis run
1. Close Vitis HLS GUI if open.
2. From the project directory: `python run_hls_build.py`
3. Expected output: "Cleaned OK." → "Starting synthesis..." → synthesis runs ~5-15 min.

## Secondary issue (non-fatal)
`config_array_partition -maximum_size 4096` still appears in logs from stored
solution settings. This is an ERROR: 200-101 but does NOT stop synthesis —
it is a deprecated option that Vitis HLS 2024.2 ignores. The broadened patch
in build_prj.tcl removes it from the TCL, and deleting myproject_prj removes
the cached solution settings that were replaying it.
