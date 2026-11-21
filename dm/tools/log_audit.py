#!/usr/bin/env python3
"""Unified nxnative runtime log auditor and A/B comparator.

Parses iOS/macOS `log.txt` files to summarize frame pacing, GPU/CPU timings,
energy efficiency (W, mJ/frame, P-core share, wakeups/s), thermal transitions,
memory footprint / Jetsam headroom, TBDR attachment traffic, GPU texture
storage (gsb_store), and classified `[longframe]` spikes.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

RE_TIMING = re.compile(
    r'\[nvn\] timing up to frame (\d+): frame time p50 ([\d.]+) / p95 ([\d.]+) / max ([\d.]+) ms.*'
    r'NVN submit\+present p50 ([\d.]+) / p95 ([\d.]+) / max ([\d.]+) ms.*'
    r'GPU p50 ([\d.]+) / p95 ([\d.]+) / max ([\d.]+) ms'
)
RE_ONSCREEN = re.compile(
    r'\[nvn\] on-screen up to frame (\d+): (\d+) frames, dropped (\d+) \(discarded (\d+)\), '
    r'p99 ([\d.]+) ms; footprint (\d+) MiB;.*thermal (\d+)'
)
RE_POWER = re.compile(
    r'\[nvn\] power up to frame (\d+): CPU energy ([\d.]+) W, ([\d.]+) mJ per frame \(([\d.]+) fps\); '
    r'CPU (\d+)% \(100 = one core\), (\d+)% of it on performance cores; wakeups (\d+)/s'
)
RE_MEM = re.compile(
    r'\[nvn-mtl\] memory up to frame (\d+): resident (\d+) MB, GPU allocated (\d+) MB,.*'
    r'residency set (\d+),.*headroom to the system memory limit (\d+) MB'
)
RE_RESIDENCY = re.compile(
    r'\[nvn\] residency set up to frame (\d+):.*attachments per frame load (\d+) MB / store (\d+) MB, '
    r'load-all store-all passes (\d+)'
)
RE_GSB = re.compile(
    r'\[nvn\] GPU texture storage up to frame (\d+): \d+ regions, [\d.]+ MB written.*'
    r'\(([\d.]+) regions, ([\d.]+) MB per frame\)'
)
RE_TRANS = re.compile(
    r'\[nvn\] translation up to frame (\d+): (\d+) instructions per draw, '
    r'submit per frame [\d.]+ M instructions / CPU ([\d.]+) ms, (\d+) draws per frame'
)
RE_CLOCK = re.compile(
    r'\[nvn\] translation detail up to frame (\d+):.*mean clock ([\d.]+) GHz'
)
RE_LONGFRAME = re.compile(
    r'\[longframe\] frame (\d+) \(([\d.]+) s\) whole frame ([\d.]+) ms, '
    r'submit\+present ([\d.]+) ms, GPU ([\d.]+) ms; '
    r'new pipelines (\d+) \+ shader libraries (\d+), ([\d.]+) ms in all; '
    r'texture uploads (\d+) first \+ (\d+) re-uploads, ([\d.]+) MB / ([\d.]+) ms; '
    r'draws with a hybrid pipeline (\d+)'
)
RE_THERMAL_CHANGE = re.compile(r'front end: thermal state → (\d+) \((\d+) s after launch\)')
RE_THERMAL_INIT = re.compile(r'front end: thermal state (\d+),')
RE_FOCUS = re.compile(r'oe focus state → (\d+)')
RE_ENV = re.compile(r'boot: env (\S+=\S+)')
RE_GUEST_MEM = re.compile(r'guest memory:.*room to the memory limit ([\d.]+) GB')


@dataclass
class Window:
    frame: int
    ft_p50: float | None = None
    ft_p95: float | None = None
    ft_max: float | None = None
    sub_p50: float | None = None
    sub_max: float | None = None
    gpu_p50: float | None = None
    gpu_p95: float | None = None
    gpu_max: float | None = None
    dropped: int | None = None
    discarded: int | None = None
    p99_ms: float | None = None
    footprint_mib: int | None = None
    thermal: int | None = None
    cpu_w: float | None = None
    mj_per_frame: float | None = None
    fps: float | None = None
    cpu_pct: int | None = None
    pcore_pct: int | None = None
    wakeups: int | None = None
    gpu_alloc_mb: int | None = None
    res_allocs: int | None = None
    headroom_mb: int | None = None
    load_mb: int | None = None
    store_mb: int | None = None
    load_all_passes: int | None = None
    gsb_mb_per_frame: float | None = None
    instr_per_draw: int | None = None
    draws_per_frame: int | None = None
    clock_ghz: float | None = None


@dataclass
class LongFrame:
    frame: int
    time_s: float
    whole_ms: float
    submit_ms: float
    gpu_ms: float
    new_pso: int
    new_lib: int
    make_ms: float
    up_first: int
    up_re: int
    up_mb: float
    up_ms: float
    hybrid_draws: int
    focus_lost: bool = False

    @property
    def cause(self) -> str:
        if self.focus_lost:
            return 'system/focus'
        if self.new_pso > 0 or self.new_lib > 0 or self.hybrid_draws > 0 or self.make_ms >= 3.0:
            return 'pipeline/shader'
        if self.up_ms >= 2.0 or self.up_mb >= 5.0:
            return 'texture-upload'
        if self.gpu_ms >= 20.0 and self.gpu_ms >= self.submit_ms * 1.5:
            return 'gpu-spike'
        if self.submit_ms >= 12.0:
            return 'cpu-submit'
        return 'other/wait'


@dataclass
class SessionAudit:
    path: str
    envs: list[str] = field(default_factory=list)
    mem_limit_gb: float | None = None
    init_thermal: int | None = None
    thermal_transitions: list[tuple[int, int]] = field(default_factory=list)
    windows: dict[int, Window] = field(default_factory=dict)
    longframes: list[LongFrame] = field(default_factory=list)

    def _win(self, frame: int) -> Window:
        w = self.windows.get(frame)
        if w is None:
            w = Window(frame=frame)
            self.windows[frame] = w
        return w

    def ordered_windows(self, min_frame: int = 0) -> list[Window]:
        return [w for f, w in sorted(self.windows.items()) if f >= min_frame]


def parse_log(path: Path) -> SessionAudit:
    audit = SessionAudit(path=str(path))
    focus_state = 1
    with path.open(encoding='utf-8', errors='replace') as fh:
        for line in fh:
            if m := RE_ENV.search(line):
                audit.envs.append(m.group(1))
            elif m := RE_GUEST_MEM.search(line):
                audit.mem_limit_gb = float(m.group(1))
            elif m := RE_THERMAL_INIT.search(line):
                audit.init_thermal = int(m.group(1))
            elif m := RE_THERMAL_CHANGE.search(line):
                audit.thermal_transitions.append((int(m.group(1)), int(m.group(2))))
            elif m := RE_FOCUS.search(line):
                focus_state = int(m.group(1))
            elif m := RE_TIMING.search(line):
                w = audit._win(int(m.group(1)))
                w.ft_p50, w.ft_p95, w.ft_max = float(m.group(2)), float(m.group(3)), float(m.group(4))
                w.sub_p50, w.sub_max = float(m.group(5)), float(m.group(7))
                w.gpu_p50, w.gpu_p95, w.gpu_max = float(m.group(8)), float(m.group(9)), float(m.group(10))
            elif m := RE_ONSCREEN.search(line):
                w = audit._win(int(m.group(1)))
                w.dropped, w.discarded = int(m.group(3)), int(m.group(4))
                w.p99_ms = float(m.group(5))
                w.footprint_mib = int(m.group(6))
                w.thermal = int(m.group(7))
            elif m := RE_POWER.search(line):
                w = audit._win(int(m.group(1)))
                w.cpu_w = float(m.group(2))
                w.mj_per_frame = float(m.group(3))
                w.fps = float(m.group(4))
                w.cpu_pct = int(m.group(5))
                w.pcore_pct = int(m.group(6))
                w.wakeups = int(m.group(7))
            elif m := RE_MEM.search(line):
                w = audit._win(int(m.group(1)))
                w.gpu_alloc_mb = int(m.group(3))
                w.res_allocs = int(m.group(4))
                w.headroom_mb = int(m.group(5))
            elif m := RE_RESIDENCY.search(line):
                w = audit._win(int(m.group(1)))
                w.load_mb = int(m.group(2))
                w.store_mb = int(m.group(3))
                w.load_all_passes = int(m.group(4))
            elif m := RE_GSB.search(line):
                w = audit._win(int(m.group(1)))
                w.gsb_mb_per_frame = float(m.group(3))
            elif m := RE_TRANS.search(line):
                w = audit._win(int(m.group(1)))
                w.instr_per_draw = int(m.group(2))
                w.draws_per_frame = int(m.group(4))
            elif m := RE_CLOCK.search(line):
                w = audit._win(int(m.group(1)))
                w.clock_ghz = float(m.group(2))
            elif m := RE_LONGFRAME.search(line):
                audit.longframes.append(
                    LongFrame(
                        frame=int(m.group(1)),
                        time_s=float(m.group(2)),
                        whole_ms=float(m.group(3)),
                        submit_ms=float(m.group(4)),
                        gpu_ms=float(m.group(5)),
                        new_pso=int(m.group(6)),
                        new_lib=int(m.group(7)),
                        make_ms=float(m.group(8)),
                        up_first=int(m.group(9)),
                        up_re=int(m.group(10)),
                        up_mb=float(m.group(11)),
                        up_ms=float(m.group(12)),
                        hybrid_draws=int(m.group(13)),
                        focus_lost=(focus_state != 1),
                    )
                )
    return audit


def _med(vals: Sequence[float | int]) -> float | None:
    return statistics.median(vals) if vals else None


def summarize(audit: SessionAudit, min_frame: int = 1800) -> dict[str, float | int | None]:
    wins = audit.ordered_windows(min_frame=min_frame)
    if not wins:
        wins = audit.ordered_windows(min_frame=0)
    fps = [w.fps for w in wins if w.fps is not None]
    drops = [w.dropped for w in wins if w.dropped is not None]
    gpu = [w.gpu_p50 for w in wins if w.gpu_p50 is not None]
    gpu_max = [w.gpu_max for w in wins if w.gpu_max is not None]
    sub = [w.sub_p50 for w in wins if w.sub_p50 is not None]
    cpu_w = [w.cpu_w for w in wins if w.cpu_w is not None]
    mj = [w.mj_per_frame for w in wins if w.mj_per_frame is not None]
    pcore = [w.pcore_pct for w in wins if w.pcore_pct is not None]
    wake = [w.wakeups for w in wins if w.wakeups is not None]
    fp = [w.footprint_mib for w in wins if w.footprint_mib is not None]
    gpu_alloc = [w.gpu_alloc_mb for w in wins if w.gpu_alloc_mb is not None]
    hr = [w.headroom_mb for w in wins if w.headroom_mb is not None]
    load_mb = [w.load_mb for w in wins if w.load_mb is not None]
    store_mb = [w.store_mb for w in wins if w.store_mb is not None]
    la = [w.load_all_passes for w in wins if w.load_all_passes is not None]
    gsb = [w.gsb_mb_per_frame for w in wins if w.gsb_mb_per_frame is not None]
    ipd = [w.instr_per_draw for w in wins if w.instr_per_draw is not None]
    clk = [w.clock_ghz for w in wins if w.clock_ghz is not None]
    return {
        'windows': len(wins),
        'last_frame': wins[-1].frame if wins else 0,
        'fps_med': _med(fps),
        'fps_min': min(fps) if fps else None,
        'dropped_total': sum(drops),
        'gpu_p50_med': _med(gpu),
        'gpu_max_peak': max(gpu_max) if gpu_max else None,
        'sub_p50_med': _med(sub),
        'cpu_w_med': _med(cpu_w),
        'mj_med': _med(mj),
        'pcore_med': _med(pcore),
        'pcore_min': min(pcore) if pcore else None,
        'wakeups_med': _med(wake),
        'wakeups_max': max(wake) if wake else None,
        'footprint_peak_mib': max(fp) if fp else None,
        'gpu_alloc_peak_mb': max(gpu_alloc) if gpu_alloc else None,
        'headroom_min_mb': min(hr) if hr else None,
        'tbdr_load_med_mb': _med(load_mb),
        'tbdr_store_med_mb': _med(store_mb),
        'load_all_passes_med': _med(la),
        'gsb_med_mb': _med(gsb),
        'gsb_max_mb': max(gsb) if gsb else None,
        'instr_per_draw_med': _med(ipd),
        'clock_min_ghz': min(clk) if clk else None,
        'clock_med_ghz': _med(clk),
    }


def _fmt(v: float | int | None, suffix: str = '', prec: int = 2) -> str:
    if v is None:
        return '-'
    if isinstance(v, int):
        return f'{v}{suffix}'
    return f'{v:.{prec}f}{suffix}'


def print_single(audit: SessionAudit, min_frame: int, top_longframes: int) -> None:
    s = summarize(audit, min_frame=min_frame)
    print(f'=== nxnative Log Audit: {audit.path} ===')
    if audit.envs:
        print(f'Env overrides: {", ".join(audit.envs)}')
    th_str = f'init={audit.init_thermal}'
    if audit.thermal_transitions:
        th_str += ', transitions=' + ', '.join(f'->{st}@{sec}s' for st, sec in audit.thermal_transitions)
    print(f'Thermal: {th_str} | Memory limit: {_fmt(audit.mem_limit_gb, " GB")} | Windows (>={min_frame}): {s["windows"]} (up to F{s["last_frame"]})')
    print(f'FPS:          med={_fmt(s["fps_med"], " fps", 1)} | min={_fmt(s["fps_min"], " fps", 1)} | dropped={s["dropped_total"]}')
    print(f'Timings:      GPU p50 med={_fmt(s["gpu_p50_med"], " ms")} (peak={_fmt(s["gpu_max_peak"], " ms")}) | Submit p50 med={_fmt(s["sub_p50_med"], " ms")}')
    print(
        f'Power/CPU:    {_fmt(s["cpu_w_med"], " W")} | {_fmt(s["mj_med"], " mJ/frame", 1)} | '
        f'P-cores med={_fmt(s["pcore_med"], "%", 0)} (min={_fmt(s["pcore_min"], "%", 0)}) | '
        f'clock med={_fmt(s["clock_med_ghz"], " GHz")} (min={_fmt(s["clock_min_ghz"], " GHz")})'
    )
    print(f'Wakeups/CPU:  med={_fmt(s["wakeups_med"], "/s", 0)} (max={_fmt(s["wakeups_max"], "/s", 0)}) | instr/draw med={_fmt(s["instr_per_draw_med"], "", 0)}')
    print(
        f'Memory:       peak footprint={_fmt(s["footprint_peak_mib"], " MiB")} | '
        f'peak GPU alloc={_fmt(s["gpu_alloc_peak_mb"], " MB")} | min headroom={_fmt(s["headroom_min_mb"], " MB")}'
    )
    print(
        f'TBDR/Banda:   load={_fmt(s["tbdr_load_med_mb"], " MB/f", 0)} / store={_fmt(s["tbdr_store_med_mb"], " MB/f", 0)} | '
        f'load-all/store-all passes={_fmt(s["load_all_passes_med"], "", 0)} | '
        f'gsb_store med={_fmt(s["gsb_med_mb"], " MB/f")} (max={_fmt(s["gsb_max_mb"], " MB/f")})'
    )

    gameplay_lf = [lf for lf in audit.longframes if lf.frame >= min_frame]
    if gameplay_lf:
        by_cause: dict[str, int] = {}
        for lf in gameplay_lf:
            by_cause[lf.cause] = by_cause.get(lf.cause, 0) + 1
        cause_str = ', '.join(f'{k}={v}' for k, v in sorted(by_cause.items(), key=lambda kv: -kv[1]))
        print(f'Longframes (>={min_frame}): {len(gameplay_lf)} total ({cause_str})')
        worst = sorted(gameplay_lf, key=lambda x: max(x.whole_ms, x.gpu_ms), reverse=True)[:top_longframes]
        for lf in worst:
            print(
                f'  F{lf.frame:<6d} ({lf.time_s:6.2f}s) [{lf.cause:<15s}] '
                f'whole={lf.whole_ms:5.1f}ms sub={lf.submit_ms:4.1f}ms gpu={lf.gpu_ms:5.1f}ms '
                f'pso={lf.new_pso} hy={lf.hybrid_draws} up={lf.up_mb:.1f}MB'
            )


def print_compare(a: SessionAudit, b: SessionAudit, min_frame: int) -> None:
    sa = summarize(a, min_frame=min_frame)
    sb = summarize(b, min_frame=min_frame)
    print(f'=== A/B Comparison (frames >= {min_frame}) ===')
    print(f'A: {a.path}')
    print(f'B: {b.path}')
    rows = [
        ('FPS median', 'fps_med', 'fps', 1),
        ('FPS min', 'fps_min', 'fps', 1),
        ('Dropped frames', 'dropped_total', '', 0),
        ('GPU p50 median', 'gpu_p50_med', 'ms', 2),
        ('GPU max peak', 'gpu_max_peak', 'ms', 2),
        ('Submit p50 median', 'sub_p50_med', 'ms', 2),
        ('CPU power median', 'cpu_w_med', 'W', 2),
        ('Energy / frame', 'mj_med', 'mJ', 1),
        ('P-core share med', 'pcore_med', '%', 0),
        ('P-core share min', 'pcore_min', '%', 0),
        ('CPU clock min', 'clock_min_ghz', 'GHz', 2),
        ('Wakeups median', 'wakeups_med', '/s', 0),
        ('Wakeups peak', 'wakeups_max', '/s', 0),
        ('Peak footprint', 'footprint_peak_mib', 'MiB', 0),
        ('Peak GPU alloc', 'gpu_alloc_peak_mb', 'MB', 0),
        ('Min headroom', 'headroom_min_mb', 'MB', 0),
        ('TBDR load med', 'tbdr_load_med_mb', 'MB/f', 0),
        ('TBDR store med', 'tbdr_store_med_mb', 'MB/f', 0),
        ('gsb_store max', 'gsb_max_mb', 'MB/f', 2),
        ('Instr / draw med', 'instr_per_draw_med', '', 0),
    ]
    print(f'{"Metric":<20s} | {"Log A":>12s} | {"Log B":>12s} | {"Delta (B-A)":>12s}')
    print('-' * 65)
    for label, key, unit, prec in rows:
        va, vb = sa[key], sb[key]
        delta = '-'
        if va is not None and vb is not None:
            diff = vb - va
            sign = '+' if diff > 0 else ''
            delta = f'{sign}{diff:.{prec}f} {unit}'.strip()
        print(f'{label:<20s} | {_fmt(va, " " + unit if unit else "", prec):>12s} | {_fmt(vb, " " + unit if unit else "", prec):>12s} | {delta:>12s}')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Audit and A/B-compare nxnative log.txt telemetry.')
    ap.add_argument('logs', nargs='+', type=Path, help='One log.txt for audit, or two log.txt files with --compare.')
    ap.add_argument('--compare', action='store_true', help='Side-by-side A/B comparison of two logs.')
    ap.add_argument('--min-frame', type=int, default=1800, help='Ignore startup/loading frames below this number (default: 1800).')
    ap.add_argument('--top', type=int, default=8, help='Number of worst gameplay longframes to list (default: 8).')
    args = ap.parse_args(argv)
    if args.compare:
        if len(args.logs) != 2:
            ap.error('--compare requires exactly 2 log files')
        print_compare(parse_log(args.logs[0]), parse_log(args.logs[1]), min_frame=args.min_frame)
    else:
        for p in args.logs:
            print_single(parse_log(p), min_frame=args.min_frame, top_longframes=args.top)
    return 0


if __name__ == '__main__':
    sys.exit(main())
