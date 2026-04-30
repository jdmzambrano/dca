import os
import sys
import shutil
import stat
import subprocess
import traceback
import time

import numpy as np

try:
    import torch
    import torch.nn as nn
    import hls4ml
except Exception as e:
    print('Missing Python dependency:', e)
    sys.exit(1)

WORKDIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORKDIR)

# Rebuild model_hls and load saved weights if available
model_hls = nn.Sequential(
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)

# Try to load saved weights
for name, fname in [('W1.npy', 'W1.npy'), ('b1.npy', 'b1.npy'), ('W2.npy', 'W2.npy'), ('b2.npy', 'b2.npy')]:
    if not os.path.isfile(fname):
        print(f'Warning: expected weight file {fname} not found in {WORKDIR}')

try:
    W1 = np.load('W1.npy')
    b1 = np.load('b1.npy')
    W2 = np.load('W2.npy')
    b2 = np.load('b2.npy')
    model_hls[0].weight.data = torch.from_numpy(W1).float().clone()
    model_hls[0].bias.data = torch.from_numpy(b1).float().clone()
    model_hls[2].weight.data = torch.from_numpy(W2).float().clone()
    model_hls[2].bias.data = torch.from_numpy(b2).float().clone()
    print('Loaded weights into model_hls')
except Exception as e:
    print('Could not load saved weights, aborting conversion:', e)
    traceback.print_exc()
    sys.exit(1)

model_hls.eval()

# Prepare sample IO
if os.path.isfile('x_sample.npy') and os.path.isfile('y_sample.npy'):
    x_sample = np.load('x_sample.npy')
    y_sample = np.load('y_sample.npy')
else:
    # create dummy sample
    x_sample = np.zeros((1, 784), dtype=np.float32)
    y_sample = np.zeros((1, 10), dtype=np.float32)
    print('Created dummy x_sample/y_sample')

# Build hls4ml config with increased reuse factor
config = hls4ml.utils.config_from_pytorch_model(
    model_hls,
    input_shape=(784,),
    granularity='name',
    default_precision='fixed<8,4>',
    default_reuse_factor=784,
)

# Force resource-oriented dense implementations to avoid aggressive full unrolling.
for layer_name, layer_cfg in config.get('LayerName', {}).items():
    if 'Dense' in str(layer_cfg.get('Class', '')) or layer_name in ('_0', '_2'):
        layer_cfg['Strategy'] = 'Resource'
        if layer_name == '_0':
            layer_cfg['ReuseFactor'] = 784
        elif layer_name == '_2':
            layer_cfg['ReuseFactor'] = 64

hls_output_dir = r"C:\Xilinx\Projects\hls4ml_mnist"
os.makedirs(hls_output_dir, exist_ok=True)

print('Converting model to hls4ml project...')
hls_model = hls4ml.converters.convert_from_pytorch_model(
    model_hls,
    hls_config=config,
    output_dir=hls_output_dir,
    backend='VivadoAccelerator',
    board='pynq-z2',
    input_data_tb='x_sample.npy',
    output_data_tb='y_sample.npy',
)
print('Conversion complete, project at', hls_output_dir)

# Patch myproject.cpp: remove #pragma HLS DATAFLOW.
# Vitis HLS 2024.2 strict-dataflow mode rejects static weight arrays (w2, b2, w4, b4)
# that are global includes — they are neither function args nor locally declared.
# With Resource strategy + high reuse factor the layers are sequential anyway,
# so DATAFLOW provides no benefit and causes HLS 200-1715.
myproject_cpp = os.path.join(hls_output_dir, 'firmware', 'myproject.cpp')
if os.path.isfile(myproject_cpp):
    with open(myproject_cpp, 'r', encoding='utf-8') as f:
        src = f.read()
    if '#pragma HLS DATAFLOW' in src:
        src = src.replace('    #pragma HLS DATAFLOW\n', '')
        with open(myproject_cpp, 'w', encoding='utf-8') as f:
            f.write(src)
        print('Patched myproject.cpp: removed #pragma HLS DATAFLOW')
    else:
        print('myproject.cpp: no DATAFLOW pragma found (already clean)')

# Optionally run C-sim if g++ available
can_compile = False
if can_compile:
    print('g++ detected — running C-sim (hls_model.compile())')
    try:
        if sys.platform.startswith('win'):
            _orig_rename = os.rename

            def _safe_rename(src, dst):
                if os.path.exists(dst):
                    os.remove(dst)
                return _orig_rename(src, dst)

            os.rename = _safe_rename

        # Remove wrapper conflict that can cause Windows rename errors
        wrapper = os.path.join(hls_output_dir, 'myproject_test_wrapper.cpp')
        dest_tb = os.path.join(hls_output_dir, 'myproject_test.cpp')
        try:
            if os.path.exists(dest_tb):
                os.remove(dest_tb)
            if os.path.exists(wrapper):
                os.remove(wrapper)
        except Exception:
            pass
        # On Windows, run build via bash if available
        if sys.platform.startswith('win'):
            bash = shutil.which('bash')
            if bash:
                # monkey-patch compile to call bash build_lib.sh
                from types import MethodType
                def _compile_with_bash(self, model):
                    ret_val = subprocess.run([bash, 'build_lib.sh'], cwd=model.config.get_output_dir(), text=True, capture_output=True)
                    print(ret_val.stdout)
                    if ret_val.returncode != 0:
                        print(ret_val.stderr)
                        raise RuntimeError('C-sim compilation failed')
                    return os.path.join(model.config.get_output_dir(), 'firmware', f"{model.config.get_project_name()}-{model.config.get_config_value('Stamp')}.so")
                hls_model.config.backend.compile = MethodType(_compile_with_bash, hls_model.config.backend)
        hls_model.compile()
        print('C-sim complete')
    except Exception as e:
        print('C-sim failed:', e)
else:
    print('Skipping C-sim: g++ not found')

# Prepare Vitis HLS invocation
vitis_hls_bat = r"C:\Xilinx\Vitis_HLS\2024.2\bin\vitis_hls.bat"
if not os.path.isfile(vitis_hls_bat):
    print('vitis_hls.bat not found at', vitis_hls_bat)
    print('Ensure Vitis HLS is installed and path is correct. Aborting synth step.')
    sys.exit(1)

# Kill any background Vitis HLS clang/opt processes still holding file locks
print('Killing lingering Vitis HLS backend processes...')
for proc in ('clang.exe', 'opt.exe', 'llvm-link.exe'):
    os.system(f'taskkill /F /IM {proc} 2>nul')
time.sleep(2)  # give Windows time to release file handles

def _force_remove(func, path, exc_info):
    """onerror handler: strip read-only and retry once."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        print(f'  Warning: could not remove {path}: {e}')

# Ensure a clean project directory to avoid stale Windows file locks
prj_dir = os.path.join(hls_output_dir, 'myproject_prj')
if os.path.isdir(prj_dir):
    print('Removing old project directory...')
    shutil.rmtree(prj_dir, onerror=_force_remove)
    if os.path.isdir(prj_dir):
        print('WARNING: project dir still exists — synthesis may still fail due to locked files.')
    else:
        print('  Cleaned OK.')

# Ensure build_prj.tcl exists, and patch deprecated option
build_tcl = os.path.join(hls_output_dir, 'build_prj.tcl')
if os.path.isfile(build_tcl):
    with open(build_tcl, 'r', encoding='utf-8') as f:
        t = f.read()
    patched = False
    # Remove any variant of the deprecated -maximum_size call
    for old in (
        'catch {config_array_partition -maximum_size $maximum_size}',
        'config_array_partition -maximum_size $maximum_size',
        'config_array_partition -maximum_size 4096',
    ):
        if old in t:
            t = t.replace(old, '# removed -maximum_size for Vitis 2024.2')
            patched = True
    if patched:
        with open(build_tcl, 'w', encoding='utf-8') as f:
            f.write(t)
        print('Patched build_prj.tcl to remove deprecated -maximum_size')
    else:
        print('build_prj.tcl already clean (no -maximum_size found)')
else:
    print('build_prj.tcl not found, writing project files')
    hls_model.write()

# Monkey-patch build to call vitis_hls.bat and capture output
from types import MethodType
from hls4ml.report import parse_vivado_report

def _build_with_vitis(self, model, reset=False, csim=False, synth=True, cosim=False, validation=False, export=True, vsynth=False, fifo_opt=False):
    curr_dir = os.getcwd()
    output_dir = model.config.get_output_dir()
    os.chdir(output_dir)
    cmd = (
        f'"{vitis_hls_bat}" -f build_prj.tcl "reset={reset} '
        f'csim={csim} '
        f'synth={synth} '
        f'cosim={cosim} '
        f'validation={validation} '
        f'export={export} '
        f'vsynth={vsynth} '
        f'fifo_opt={fifo_opt}"'
    )
    print('Running:', cmd)
    ret = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print('=== VITIS STDOUT ===')
    print(ret.stdout[:10000])
    if ret.stderr:
        print('=== VITIS STDERR ===')
        print(ret.stderr[:10000])
    print('vitis_hls return code:', ret.returncode)
    os.chdir(curr_dir)
    if ret.returncode == 0:
        return parse_vivado_report(output_dir)
    else:
        raise RuntimeError('Vitis HLS failed — check output above')

hls_model.config.backend.build = MethodType(_build_with_vitis, hls_model.config.backend)

print('Starting synthesis (this may take many minutes)...')
try:
    report = hls_model.build(csim=False, synth=True, cosim=False, export=True)
    print('Synthesis completed — parse report:')
    print(report)
except Exception as e:
    print('Synthesis failed:', e)
    traceback.print_exc()
    sys.exit(1)

print('Done')
