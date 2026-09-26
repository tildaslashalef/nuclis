#!/usr/bin/env python3
"""Where a running `nuclis` process's memory is: the GPU side and the CPU side.

On unified memory the split is by who reads the pages, read from the kernel's
own accounting (`footprint -j` for exact per-category bytes, `vmmap -w` for
each region's share mode and file, `mincore` for the model's page-cache pages):

- model weights: the GGUF is a file mapping the GPU reads through no-copy
  buffers. Its pages are clean page cache, so they sit outside the process
  footprint and its RSS; they are counted here with `mincore`.
- session state: heap regions shared with the GPU (`SM=SHM`), the KV cache and
  recurrent state wrapped as no-copy buffers.
- Metal buffers and driver memory: the `IOAccelerator` and graphics categories.
- CPU: the rest of the footprint (private heap, code, stacks).

`--watch SECONDS` samples until Ctrl-C and ends with the peaks; `--json`
prints one report as JSON. macOS only; run it outside a sandbox.
"""
import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest

GB = 1e9
REGION = re.compile(r'^(?P<kind>.+?)\s+[0-9a-f]+-[0-9a-f]+\s+\[\s*(?P<vsize>\S+)\s+(?P<resident>\S+)\s+(?P<dirty>\S+)\s+(?P<swap>\S+)\]'
                    r'\s+\S+\s+SM=(?P<share>\S+)\s*(?P<detail>.*)$')
UNITS = {'': 1, 'K': 1024, 'M': 1024 ** 2, 'G': 1024 ** 3, 'T': 1024 ** 4}


# ---- pure functions (covered by --self-test) --------------------------------

def parse_size(text):
    """vmmap's `33.8M` / `12K` / `824` in bytes."""
    match = re.fullmatch(r'([0-9.]+)([KMGT]?)', text)
    if not match:
        raise ValueError(f'not a vmmap size: {text!r}')
    return int(float(match.group(1)) * UNITS[match.group(2)])


def parse_regions(vmmap_text):
    """The region lines of `vmmap -w` as dicts: kind, sizes in bytes, share mode, detail."""
    regions = []
    for line in vmmap_text.splitlines():
        match = REGION.match(line)
        if not match:
            continue
        region = match.groupdict()
        for key in ('vsize', 'resident', 'dirty', 'swap'):
            region[key] = parse_size(region[key])
        region['kind'] = region['kind'].strip()
        region['detail'] = region['detail'].strip()
        regions.append(region)
    return regions


def classify(categories, regions, weights):
    """The report's rows in bytes from footprint categories, vmmap regions,
    and the resident model files ({path: (resident, size)})."""
    dirty = lambda name: categories.get(name, {}).get('dirty', 0) + categories.get(name, {}).get('swapped', 0)
    shared_heap = sum(r['dirty'] + r['swap'] for r in regions
                      if r['share'] == 'SHM' and (r['kind'].startswith('MALLOC') or r['kind'].startswith('VM_ALLOCATE')))
    metal = dirty('IOAccelerator') + dirty('IOAccelerator (graphics)')
    driver = dirty('Owned physical footprint (unmapped) (graphics)')
    weights_resident = sum(res for res, _ in weights.values())
    return {
        'weights_resident': weights_resident,
        'weights_size': sum(size for _, size in weights.values()),
        'session_state': shared_heap,
        'metal_buffers': metal,
        'driver': driver,
        'gpu_in_footprint': shared_heap + metal + driver,
    }


def human(n):
    return f'{n / GB:6.2f} GB'


# ---- collection --------------------------------------------------------------

def find_pids():
    out = subprocess.run(['pgrep', '-x', 'nuclis'], capture_output=True, text=True).stdout
    return [int(p) for p in out.split()]


def command_line(pid):
    return subprocess.run(['ps', '-o', 'command=', '-p', str(pid)], capture_output=True, text=True).stdout.strip()


def footprint(pid):
    with tempfile.NamedTemporaryFile(suffix='.json') as f:
        subprocess.run(['footprint', '-p', str(pid), '-f', 'bytes', '-j', f.name], capture_output=True, check=True)
        data = json.load(open(f.name))
    process = data['processes'][0]
    return process['categories'], process['auxiliary']['phys_footprint'], process['auxiliary']['phys_footprint_peak']


def resident_bytes(path):
    """(bytes of the file in the page cache, file size), from mincore over a fresh read-only mapping."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    size = os.path.getsize(path)
    page = os.sysconf('SC_PAGE_SIZE')
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, 1, 1, fd, 0)  # PROT_READ, MAP_SHARED
        if addr in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), f'mmap {path}')
        try:
            pages = (size + page - 1) // page
            vec = ctypes.create_string_buffer(pages)
            if libc.mincore(addr, size, vec) != 0:
                raise OSError(ctypes.get_errno(), f'mincore {path}')
            return sum(b & 1 for b in vec.raw) * page, size
        finally:
            libc.munmap(addr, size)
    finally:
        os.close(fd)


def system():
    text = subprocess.run(['vm_stat'], capture_output=True, text=True).stdout
    page = int(re.search(r'page size of (\d+)', text).group(1))
    field = lambda name: int(re.search(rf'{name}:\s+(\d+)', text).group(1)) * page
    total = int(subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True).stdout)
    return {'total': total, 'wired': field('Pages wired down'), 'free': field('Pages free'),
            'compressed': field('Pages occupied by compressor')}


def report(pid):
    categories, phys, peak = footprint(pid)
    regions = parse_regions(subprocess.run(['vmmap', '-w', str(pid)], capture_output=True, text=True).stdout)
    weights = {}
    for r in regions:
        if r['kind'] == 'mapped file' and r['detail'].endswith('.gguf') and r['detail'] not in weights:
            weights[r['detail']] = resident_bytes(r['detail'])
    rows = classify(categories, regions, weights)
    rows.update(pid=pid, command=command_line(pid), footprint=phys, footprint_peak=peak,
                cpu=phys - rows['gpu_in_footprint'], total=phys + rows['weights_resident'],
                weights={p: {'resident': res, 'size': size} for p, (res, size) in weights.items()},
                system=system(), time=time.strftime('%H:%M:%S'))
    return rows


def render(r):
    lines = [f"nuclis pid {r['pid']} at {r['time']}: {r['command']}", '', 'GPU (Metal, unified memory)']
    for path, w in r['weights'].items():
        lines.append(f"  model weights, page cache     {human(w['resident'])} of {w['size'] / GB:.2f} GB  {os.path.basename(path)}")
    if not r['weights']:
        lines.append('  model weights, page cache     none mapped (still loading, or no GGUF)')
    lines += [f"  session state (KV, recurrent) {human(r['session_state'])}  heap shared with the GPU",
              f"  Metal buffers                 {human(r['metal_buffers'])}  scratch, logits, rope tables",
              f"  driver-owned (wired)          {human(r['driver'])}",
              f"  subtotal                      {human(r['weights_resident'] + r['gpu_in_footprint'])}",
              'CPU',
              f"  heap, code, stacks            {human(r['cpu'])}",
              '',
              f"total                           {human(r['total'])}",
              f"  physical footprint            {human(r['footprint'])}  (peak {r['footprint_peak'] / GB:.2f} GB; Activity Monitor's Memory)",
              f"  + resident weights            {human(r['weights_resident'])}  clean file pages, outside the footprint",
              '']
    s = r['system']
    lines.append(f"system: {s['total'] / 1024 ** 3:.0f} GiB, wired {s['wired'] / GB:.2f}, free {s['free'] / GB:.2f}, "
                 f"compressor {s['compressed'] / GB:.2f} GB (the GPU wires the weights while a command buffer runs)")
    return '\n'.join(lines)


def watch(pid, seconds):
    peaks = {}
    print('time      total     weights   state     metal     cpu       footprint  sys wired')
    try:
        while True:
            try:
                r = report(pid)
            except (subprocess.CalledProcessError, IndexError, FileNotFoundError):
                print(f'{time.strftime("%H:%M:%S")}  pid {pid} is gone')
                break
            keys = ('total', 'weights_resident', 'session_state', 'metal_buffers', 'cpu', 'footprint')
            for k in keys:
                peaks[k] = max(peaks.get(k, 0), r[k])
            peaks['wired'] = max(peaks.get('wired', 0), r['system']['wired'])
            print(r['time'], ' '.join(human(r[k]) for k in keys), human(r['system']['wired']), flush=True)
            time.sleep(seconds)
    except KeyboardInterrupt:
        pass
    if peaks:
        print('peak     ', ' '.join(human(peaks[k]) for k in ('total', 'weights_resident', 'session_state', 'metal_buffers', 'cpu', 'footprint')),
              human(peaks['wired']))


# ---- tests -------------------------------------------------------------------

class Tests(unittest.TestCase):
    VMMAP = (
        'mapped file                 300000000-6d55b8000    [ 15.3G  23.6M     0K     0K] r--/r-x SM=SHM          /m/Q.gguf\n'
        'MALLOC_LARGE                6d8c00000-6e0c00000    [128.0M 128.0M 128.0M     0K] rw-/rwx SM=SHM          DefaultMallocZone_0x1\n'
        'MALLOC_LARGE                722800000-72a800000    [128.0M 128.0M 128.0M     0K] rw-/rwx SM=PRV          DefaultMallocZone_0x1\n'
        'MALLOC_LARGE                       1.4G     1.4G     1.4G       0K       0K       0K       0K       15\n')

    def test_sizes(self):
        self.assertEqual(parse_size('824'), 824)
        self.assertEqual(parse_size('12K'), 12 * 1024)
        self.assertEqual(parse_size('1.5G'), 3 * 1024 ** 3 // 2)
        with self.assertRaises(ValueError):
            parse_size('12Q')

    def test_regions_skip_the_summary_and_keep_the_file(self):
        regions = parse_regions(self.VMMAP)
        self.assertEqual([r['share'] for r in regions], ['SHM', 'SHM', 'PRV'])
        self.assertEqual(regions[0]['detail'], '/m/Q.gguf')

    def test_only_shared_heap_is_session_state(self):
        categories = {'IOAccelerator (graphics)': {'dirty': 100, 'swapped': 0},
                      'Owned physical footprint (unmapped) (graphics)': {'dirty': 7, 'swapped': 0}}
        rows = classify(categories, parse_regions(self.VMMAP), {'/m/Q.gguf': (10, 20)})
        self.assertEqual(rows['session_state'], 128 * 1024 ** 2)
        self.assertEqual(rows['metal_buffers'], 100)
        self.assertEqual(rows['gpu_in_footprint'], 128 * 1024 ** 2 + 107)
        self.assertEqual((rows['weights_resident'], rows['weights_size']), (10, 20))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--pid', type=int, help='the process (default: every running `nuclis`)')
    parser.add_argument('--watch', type=float, metavar='SECONDS', help='sample every SECONDS until Ctrl-C, then print the peaks')
    parser.add_argument('--json', action='store_true', help='print the report as JSON')
    parser.add_argument('--self-test', action='store_true', help='run the unit tests')
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[sys.argv[0]], verbosity=1)
    pids = [args.pid] if args.pid else find_pids()
    if not pids:
        sys.exit('no nuclis process is running')
    if args.watch:
        watch(pids[0], args.watch)
        return
    reports = [report(pid) for pid in pids]
    if args.json:
        print(json.dumps(reports if len(reports) > 1 else reports[0], indent=2))
    else:
        print('\n\n'.join(render(r) for r in reports))


if __name__ == '__main__':
    main()
