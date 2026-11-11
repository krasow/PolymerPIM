#!/usr/bin/env python3
"""How much rank-to-rank traffic a Legion profile hid behind DPU kernels.

Usage: legion_overlap.py prof_0.gz [--last MS]   (or a `legion_prof legacy` dir)

A rank cannot copy while its own DPUs run, so overlap means a copy between
ranks A and B while a kernel runs on some third rank.
"""
import argparse, csv, os, re, subprocess, sys, tempfile


def load(d):
    names = {r['tsv']: r['text'] for r in
             csv.DictReader(open(f'{d}/legion_prof_processor.tsv'), delimiter='\t')}
    kernels, copies = [], []
    for path, name in names.items():
        rank = [int(x) for x in re.findall(r'(?:dpu|Device Proc )(\d+)', name)]
        is_kernel = 'Device Proc' in name
        if not is_kernel and not (len(rank) == 2 and ' to ' in name):
            continue
        for r in csv.DictReader(open(f'{d}/{path}'), delimiter='\t'):
            if 'waiting' in r['title']:
                continue
            iv = (float(r['start']) / 1e3, float(r['end']) / 1e3, frozenset(rank))
            (kernels if is_kernel else copies).append(iv)
    return kernels, copies


def union(ivs):
    out = []
    for s, e in sorted(ivs):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def covered(s, e, merged):
    return sum(max(0.0, min(e, b) - max(s, a)) for a, b in merged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('profile')
    ap.add_argument('--last', type=float, help='only the last MS milliseconds')
    ap.add_argument('--legion-prof', default=os.path.join(
        os.path.dirname(__file__), '..', 'opt', 'legion_prof', 'bin', 'legion_prof'))
    a = ap.parse_args()

    d = a.profile
    if not os.path.isdir(d):
        d = tempfile.mkdtemp()
        subprocess.run([a.legion_prof, 'legacy', '-f', '-o', d, a.profile],
                       check=True, stdout=subprocess.DEVNULL)
    kernels, copies = load(d)
    if not copies:
        sys.exit('no rank-to-rank copies in this profile')

    lo = max(e for s, e, _ in kernels + copies) - a.last if a.last else 0.0
    clip = lambda ivs: [(max(s, lo), e, r) for s, e, r in ivs if e > lo]
    kernels, copies = clip(kernels), clip(copies)

    ranks = sorted({r for _, _, rs in kernels for r in rs})
    by_rank = {r: union((s, e) for s, e, rs in kernels if r in rs) for r in ranks}
    k_all = union((s, e) for s, e, _ in kernels)
    c_all = union((s, e) for s, e, _ in copies)
    span = max(e for s, e, _ in kernels + copies) - min(s for s, _, _ in kernels + copies)

    # Per copy: time some rank other than its endpoints was computing.
    hidden = total = 0.0
    eager = 0
    for s, e, ends in copies:
        others = union(iv for r in ranks if r not in ends for iv in by_rank[r])
        total += e - s
        hidden += covered(s, e, others)
        eager += covered(s, s + 1e-3, others) > 0

    K = sum(e - s for s, e in k_all)
    C = sum(e - s for s, e in c_all)
    both = sum(covered(s, e, c_all) for s, e in k_all)
    print(f'span {span:.1f} ms over {len(ranks)} ranks, {len(copies)} rank-to-rank copies')
    print(f'  some kernel running   {K:8.1f} ms  ({K / span:.0%})')
    print(f'  some copy running     {C:8.1f} ms  ({C / span:.0%})')
    print(f'  both at once          {both:8.1f} ms  -> {both / C:.0%} of copy time overlapped')
    print(f'  copy time hidden behind a third rank\'s kernel: {hidden / total:.0%}')
    print(f'  copies that started while another rank computed: {eager}/{len(copies)}')
    print(f'  serial lower bound K+C {K + C:.1f} ms vs actual busy {K + C - both:.1f} ms')


if __name__ == '__main__':
    main()
