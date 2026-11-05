"""Verify the existing course environment. Never install or change packages.

Default: metadata and portable SDK-source checks, with no simulator imports.
--cuda: real CUDA, Quadrants compilation and zero-copy checks, without a scene.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def requirements(path):
    result = {}
    for raw in Path(path).read_text(encoding='utf8').splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)', line)
        if not match or match[1] in result:
            raise ValueError('Expected unique exact reference versions in requirements.txt')
        result[match[1]] = match[2]
    if not result:
        raise ValueError('Missing reference package versions')
    return result


def version_checks(expected, lookup=metadata.version):
    rows = []
    for name, version in expected.items():
        try:
            actual = lookup(name)
        except metadata.PackageNotFoundError:
            actual = None
        rows.append(dict(package=name, expected=version, actual=actual, passed=actual == version))
    return rows


def source_checks(pins, roots):
    rows = []
    for relative, expected in pins.items():
        path = Path(relative)
        if (path.is_absolute() or '..' in path.parts or len(path.parts) < 2
                or path.parts[0] not in roots):
            raise ValueError('SDK source paths must be relative to a declared distribution')
        source = roots[path.parts[0]].joinpath(*path.parts[1:])
        actual = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
        rows.append(dict(source=relative, expected=expected, actual=actual, passed=actual == expected))
    return rows


def metadata_check(root=ROOT):
    reference = read(root/'environment_reference.json')
    expected = requirements(root/'requirements.txt')
    versions = version_checks(expected)
    roots = {}
    for package, distribution in (('genesis', 'genesis-world'), ('quadrants', 'quadrants')):
        try:
            roots[package] = Path(metadata.distribution(distribution).locate_file(package)).resolve()
        except metadata.PackageNotFoundError:
            roots[package] = root/'__missing_distribution__'/package
    sdk = read(root/'environment_sdk_pins.json')
    if sdk.get('schema_version') != 'portable-environment-sdk-pins-v1' or not sdk.get('source_sha256'):
        raise ValueError('Missing portable SDK source contract')
    sources = source_checks(sdk['source_sha256'], roots)
    platform_checks = dict(
        python_major_minor=list(sys.version_info[:2]) == reference['python_major_minor'],
        Windows=platform.system() == reference['supported_OS'],
        process_64bit=struct.calcsize('P')*8 == reference['machine_bits'])
    passed = all(platform_checks.values()) and all(r['passed'] for r in versions+sources)
    return dict(schema_version='applied-programming-environment-check-v1',
                status='ENVIRONMENT_METADATA_PASS' if passed else 'ENVIRONMENT_METADATA_MISMATCH',
                passed=passed, python=platform.python_version(), OS=platform.system(),
                machine=platform.machine(), platform_checks=platform_checks,
                environment_name_hint=Path(sys.prefix).name,
                package_versions=versions, SDK_source_checks=sources,
                CUDA_probe_requested=False, CUDA_probe_performed=False,
                automatic_install_or_environment_change=False,
                simulation_steps=0, full_transport_verified=False,
                mismatch_action='Use the existing applied-programming-cuda environment. This checker reports mismatches and does not install, downgrade or modify it.')


def nvidia_driver_information():
    executable = shutil.which('nvidia-smi')
    if executable is None:
        return dict(available=False)
    command = [executable, '--query-gpu=index,name,uuid,driver_version,memory.total',
               '--format=csv,noheader,nounits']
    result = subprocess.run(command, capture_output=True, text=True, timeout=10,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    return dict(available=True, returncode=result.returncode,
                fields=['index', 'name', 'UUID', 'driver_version', 'total_MiB'],
                rows=[line.strip() for line in result.stdout.splitlines() if line.strip()],
                note='Device information only. Owned-process memory is checked by the review supervisor.')


def cuda_check():
    # The optional native check runs only when explicitly requested.
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('PyTorch cannot use CUDA. No CPU fallback is accepted.')
    capability = tuple(torch.cuda.get_device_capability())
    if capability < (7, 5):
        raise RuntimeError('This test requires a CUDA device with compute capability at least7.5.')
    if torch.version.cuda != '12.6':
        raise RuntimeError('The reference PyTorch build is CUDA12.6.')
    device_name = torch.cuda.get_device_name()
    architectures = torch.cuda.get_arch_list()
    value = torch.tensor([1., 2., 3.], dtype=torch.float32, device='cuda')
    if float((value*value).sum().item()) != 14.:
        raise RuntimeError('Actual PyTorch CUDA arithmetic failed')
    import quadrants as qd
    import genesis as gs
    from genesis.utils import misc
    for module, distribution, package in ((torch, 'torch', 'torch'),
            (qd, 'quadrants', 'quadrants'), (gs, 'genesis-world', 'genesis')):
        expected = Path(metadata.distribution(distribution).locate_file(package)).resolve()/'__init__.py'
        if Path(module.__file__).resolve() != expected:
            raise RuntimeError('Imported module differs from its checked distribution: '+package)
    initialized = False
    try:
        gs.init(backend=gs.cuda, precision='32', seed=0, logging_level='warning')
        initialized = True
        if (gs.backend != gs.cuda or gs.device.type != 'cuda'
                or qd.lang.impl.current_cfg().arch != qd.cuda
                or not gs.use_ndarray or not gs.use_zerocopy):
            raise RuntimeError('Genesis/Quadrants did not select the required CUDA ndarray/zero-copy backend')
        values = qd.ndarray(dtype=qd.f64, shape=4)
        flags = qd.ndarray(dtype=qd.i32, shape=4)

        # Eager annotations are required for the installed Quadrants ABI.
        @qd.kernel
        def probe(output: qd.types.ndarray(), finite_flags: qd.types.ndarray()):
            for i in range(4):
                x = qd.f64(i+1)*qd.f64(.0025)
                output[i] = x
                finite_flags[i] = qd.cast((qd.bit_cast(x, qd.i64) & qd.i64(0x7ff0000000000000))
                                        != qd.i64(0x7ff0000000000000), qd.i32)

        probe(values, flags)
        qd.sync()
        view = misc.qd_to_torch(values, copy=False)
        if view.device.type != 'cuda' or view.dtype != torch.float64 or tuple(view.shape) != (4,):
            raise RuntimeError('Required CUDA zero-copy array ABI differs')
        actual = view.detach().cpu().numpy().copy()
        expected = np.arange(1., 5., dtype=np.float64)*.0025
        if not np.array_equal(actual, expected) or not np.array_equal(flags.to_numpy(), np.ones(4, np.int32)):
            raise RuntimeError('Actual Quadrants F64/I64 CUDA kernel failed')
        return dict(passed=True, backend='CUDA', device_name=device_name,
                    compute_capability=list(capability), torch_compiled_architectures=architectures,
                    wheel_reports_sm75='sm_75' in architectures,
                    target_GTX1660_family='GTX1660' in ''.join(device_name.upper().split()),
                    torch_CUDA_runtime=torch.version.cuda, torch_CUDA_arithmetic_passed=True,
                    actual_Quadrants_F64_I64_kernel_passed=True, CUDA_zero_copy_passed=True,
                    Genesis_precision_bits=32, simulation_scenes_created=0, simulation_steps=0,
                    full_transport_verified=False, driver=nvidia_driver_information())
    finally:
        if initialized:
            gs.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda', action='store_true', help='Run actual CUDA/compiler checks without creating a scene')
    parser.add_argument('--output', type=Path, help='Optional fresh JSON receipt path')
    parser.add_argument('--json', action='store_true', help='Print the complete JSON receipt')
    args = parser.parse_args(argv)
    report = metadata_check()
    report['CUDA_probe_requested'] = args.cuda
    if args.cuda and report['passed']:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                report['CUDA_probe'] = cuda_check()
            report['CUDA_probe_performed'] = True
            report['status'] = 'ENVIRONMENT_CUDA_SMOKE_PASS'
        except Exception as exc:
            report.update(passed=False, status='ENVIRONMENT_CUDA_SMOKE_FAILED', CUDA_probe_performed=True,
                          CUDA_probe=dict(passed=False, error_type=type(exc).__name__, message=str(exc)))
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf8') as stream:
            stream.write(text+'\n')
    if args.json:
        print(text)
    else:
        print(report['status'])
        print('Python '+report['python']+' | '+report['OS']+' | '+report['environment_name_hint'])
        print(str(sum(row['passed'] for row in report['package_versions']))+'/'+str(len(report['package_versions']))
              +' package versions, '+str(sum(row['passed'] for row in report['SDK_source_checks']))+'/'+str(len(report['SDK_source_checks']))+' SDK sources match.')
        for row in report['package_versions']:
            if not row['passed']:
                print('Mismatch: '+row['package']+' expected '+row['expected']+', found '+str(row['actual']))
        for row in report['SDK_source_checks']:
            if not row['passed']:
                print('SDK source mismatch: '+row['source'])
        if args.cuda:
            probe_result=report.get('CUDA_probe',{})
            if probe_result.get('passed'):
                print(probe_result['device_name']+' | CUDA compute capability '+'.'.join(map(str,probe_result['compute_capability'])))
                print('CUDA arithmetic, Quadrants compiler and zero-copy checks passed. No simulation was run.')
            elif report['CUDA_probe_performed']:
                print(probe_result.get('error_type','Error')+': '+probe_result.get('message','CUDA check failed'))
        if not report['passed']:
            print(report['mismatch_action'])
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
