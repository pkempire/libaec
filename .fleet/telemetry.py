#!/usr/bin/env python3
"""Frozen CLI correctness and timing workload. Extra streams are synthetic.

No benchmark input or oracle may change during source optimization. Large files
are temporary and deleted before runner synchronization. Timings include complete
graec processes (startup, allocation, codec and file I/O), not just CPU kernels.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import struct
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / '.fleet/build'
FLAGS = ['-n16', '-j64', '-r256', '-m']

def run(args, **kwargs):
    subprocess.run([str(a) for a in args], check=True, **kwargs)

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return {'bytes': path.stat().st_size, 'sha256': h.hexdigest()}

def stream(kind, seed, samples):
    # Specified integer PRNG, independent of Python's random implementation.
    out = bytearray(2 * samples)
    x = seed
    value = 31000
    for i in range(samples):
        x ^= (x << 13) & 0xffffffff
        x ^= x >> 17
        x ^= (x << 5) & 0xffffffff
        if kind == 'entropy':
            v = x & 65535
        elif kind == 'constant':
            v = (0, 65535, 32768, 17321)[(i // 16384) % 4]
        else:
            value = max(0, min(65535, value + (x % 17) - 8))
            v = value
            if kind == 'spikes' and i % 97 == 0:
                v = x & 65535
        struct.pack_into('>H', out, i * 2, v)
    return out

def prepare(tmp, graec, heldout=False, full=False):
    cases = []
    if not heldout:
        typical = tmp / 'typical.dat'
        run([graec, '-d', *FLAGS, ROOT / 'data/typical.rz', typical])
        data = typical.read_bytes()
        typical.unlink()
        path = tmp / 'upstream.dat'
        with path.open('wb') as f:
            for _ in range(500 if full else 1):
                f.write(data)
        cases.append(('upstream500' if full else 'upstream', path))
    for kind in ['smooth', 'spikes', 'constant', 'entropy']:
        path = tmp / (kind + '.dat')
        data = stream(kind, 0x915ac37b if heldout else 0x13579bdf,
                      65536 if heldout else 524288)
        with path.open('wb') as f:
            for _ in range(16 if full else 1):
                f.write(data)
        cases.append((('heldout-' if heldout else ('full-' if full else '')) + kind, path))
    if heldout:
        # Block / RSI boundaries, extrema and direction changes.
        for n in [64, 128, 16320, 16384, 16448]:
            path = tmp / ('boundary-%d.dat' % n)
            path.write_bytes(b''.join(struct.pack('>H', [0, 65535, 1, 32767, 32768, 65534][i % 6]) for i in range(n)))
            cases.append(('boundary-%d' % n, path))
    return cases

def codec_checks(graec, tmp, cases, oracle, create=False, timed=False, repeats=1, direction='both'):
    checks = {}
    rows = []
    for name, raw in cases:
        enc, dec = tmp / 'encoded.rz', tmp / 'decoded.dat'
        raw_hash = digest(raw)
        for rep in range(repeats):
            t0 = time.perf_counter_ns()
            run([graec, *FLAGS, raw, enc])
            t1 = time.perf_counter_ns()
            run([graec, '-d', *FLAGS, enc, dec])
            t2 = time.perf_counter_ns()
            result = {'raw': raw_hash, 'encoded': digest(enc), 'decoded': digest(dec)}
            assert result['raw'] == result['decoded'], (name, 'round trip')
            if not create:
                assert result == oracle[name], (name, result, oracle[name])
            checks[name] = result
            rows.append({'case': name, 'repeat': rep,
                         'encode_wall_s': (t1-t0)/1e9, 'decode_wall_s': (t2-t1)/1e9})
        enc.unlink()
        dec.unlink()
    return checks, rows

def configure():
    # Reconfigure after runner relocation; never change Release compiler flags.
    shutil.rmtree(BUILD, ignore_errors=True)
    run(['cmake', '-S', ROOT, '-B', BUILD, '-DCMAKE_BUILD_TYPE=Release',
         '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '-DCMAKE_C_COMPILER=' + os.environ.get('CC', 'cc')])
    run(['cmake', '--build', BUILD, '-j4'])

def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['build', 'test', 'oracle', 'check', 'workload', 'profile'])
    p.add_argument('--graec', type=Path)
    p.add_argument('--report', type=Path)
    args = p.parse_args()
    graec = (args.graec or BUILD / 'src/graec').resolve()
    os.environ['LD_LIBRARY_PATH'] = str(graec.parent)
    if args.mode == 'build':
        configure()
        return
    if args.mode == 'test':
        configure()
        try:
            run(['ctest', '--test-dir', BUILD, '--output-on-failure'])
            run(['cmake', '--build', BUILD, '--target', 'bench'])
        finally:
            for name in ['bench.dat', 'bench.rz', 'dec.dat', 'typical.dat']:
                (BUILD / 'src' / name).unlink(missing_ok=True)
            for name in ['test.dat', 'test.rz']:
                (BUILD / 'tests' / name).unlink(missing_ok=True)
    oracle_path = ROOT / '.fleet/oracle.json'
    create = args.mode == 'oracle'
    oracle = {} if create else json.loads(oracle_path.read_text())
    checks, rows = {}, []
    with tempfile.TemporaryDirectory(dir=BUILD.parent) as td:
        tmp = Path(td)
        full = args.mode in ['oracle', 'workload', 'profile']
        cases = prepare(tmp, graec, full=full)
        if args.mode == 'profile':
            # Actual complete upstream encode/decode, perf symbols from Release.
            for name, raw in cases[:1]:
                for direction in ['encode', 'decode']:
                    enc = tmp / 'encoded.rz'
                    argv = [graec, *FLAGS, raw, enc] if direction == 'encode' else [graec, '-d', *FLAGS, enc, tmp / 'decoded.dat']
                    output = ROOT / '.evo/evidence' / ('perf-' + direction)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    run(['perf', 'record', '-e', 'cycles:u', '-F', '999', '-o', str(output) + '.data', '--', *argv])
                    with open(str(output) + '.txt', 'w') as f:
                        run(['perf', 'report', '-i', str(output) + '.data', '--stdio', '--no-children', '--sort', 'dso,symbol'], stdout=f)
                    Path(str(output) + '.data').unlink()
            return
        c, r = codec_checks(graec, tmp, cases, oracle, create=create,
                            repeats=2 if args.mode == 'workload' else 1)
        checks.update(c); rows.extend(r)
        if args.mode in ['oracle', 'test', 'check']:
            if create:
                c, r = codec_checks(graec, tmp, prepare(tmp, graec), oracle, create=True)
                checks.update(c); rows.extend(r)
            c, r = codec_checks(graec, tmp, prepare(tmp, graec, heldout=True), oracle, create=create)
            checks.update(c); rows.extend(r)
    if create:
        oracle_path.write_text(json.dumps(checks, indent=2) + '\n')
    report = {'scope': 'complete graec processes including file IO; sequential, warm filesystem',
              'settings': FLAGS, 'rows': rows,
              'children_peak_rss_kib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    # Stable stdout; native measurement records complete harness wall time too.
    print('byte-exact telemetry checks passed: ' + ','.join(checks))

if __name__ == '__main__':
    main()
