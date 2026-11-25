"""LaTeX tables and figures for the revised paper from results2<tag>.json,
analysis2<tag>.json and rate2_C*.mat in results/. Run from sim/ after
aggregate2.m and analyze2.py. The tables of stage gains, the loss split and
the combined pipelines are written between the %<TAB:name> and %</TAB:name>
markers of ../paper/tinyML Edge Caching.tex, and Figs. 2 to 4 to ../paper/."""
import json, re, sys, collections, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import stats
PAPER = '../paper/tinyML Edge Caching.tex'   # tables are written into this file; figures go to ../paper/
A = json.load(open('results/results2.json')); R = json.load(open('results/analysis2.json'))
CL = ['C1', 'C2', 'C3', 'C4']; LAB = ['Class 1', 'Class 2', 'Class 3', 'Class 4']
COL = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.linewidth': 0.6, 'pdf.fonttype': 42, 'legend.fontsize': 7})
SH = lambda n: {'lin': 'linear', 'mlp_s': 'MLP-S', 'mlp_l': 'MLP-L'}[n.split('_', 1)[1]]
def seeds(c): s = A[c]['seeds']; return s if isinstance(s, list) else [s]
def mode(lst):
    c = collections.Counter(lst).most_common(); return c[0][0], c[0][1]
def ci(v):
    v = np.asarray([x for x in v if x is not None and not np.isnan(x)], float)
    if len(v) < 2: return (float(v.mean()) if len(v) else np.nan), np.nan
    return float(v.mean()), float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
def pm(m, h, d=2, sign=True):
    if np.isnan(h): return f"${m:+.{d}f}$" if sign else f"${m:.{d}f}$"
    return f"${m:+.{d}f}\\pm{h:.{d}f}$" if sign else f"${m:.{d}f}\\pm{h:.{d}f}$"
tabs = {}
# ---- stage gains: alone (31) and behind m2H (35); one line per cell ----
AB = {'lin': 'Lin', 'mlp_s': 'S', 'mlp_l': 'L'}
def cell(sel, gains):
    n = len(sel); nsel = sum(b != 'b' for b in sel)
    if 2*nsel < n: return 'Baseline' + (f' ({nsel}/{n})' if nsel else '')
    used = set(b for b in sel if b != 'b'); lab = AB[used.pop().split('_', 1)[1]] if len(used) == 1 else 'Varies'
    mh, hh = ci(gains); return f"{lab}{'' if nsel == n else f' ({nsel}/{n})'}: {pm(mh, hh)}"
rows = []
for j in '1234':
    cH, cG, anyG = [], [], False
    for c in CL:
        ss = seeds(c); sel = [s['best'][int(j) - 1] for s in ss]
        v = [s['single'][b]['dH']*100 for s, b in zip(ss, sel) if b != 'b' and 'dH' in s['single'][b]]
        g = [s['single'][b]['G'] for s, b in zip(ss, sel) if b != 'b' and 'G' in s['single'][b]]
        cH.append(cell(sel, v)); c2 = cell(sel, g)
        if c2.startswith('Baseline'): cG.append('--')
        else: anyG = True; cG.append(pm(*ci(g), 1))
    rows.append(f"S{j}, $\\Delta H_{j}$ & " + ' & '.join(cH) + r' \\')
    if anyG: rows.append(f"S{j}, $G_{j}$ (J) & " + ' & '.join(cG) + r' \\')
rows.append(r'\midrule')
rows.append(r'\multicolumn{5}{@{}l@{}}{\emph{Behind the S2 model $m^H_2(p)$, selected by \eqref{eq:behind}}} \\')
for j in '134':
    cells = []
    for c in CL:
        ss = seeds(c); sel = [s['bestA'][int(j) - 1] for s in ss]
        v = [s['behind'][b]['dHte']*100 for s, b in zip(ss, sel) if b != 'b' and 'dHte' in s['behind'][b]]
        cells.append(cell(sel, v))
    rows.append(f"S{j}, $\\Delta H^A_{j}$ & " + ' & '.join(cells) + r' \\')
tabs['stage_gains'] = '\n'.join(rows)
# ---- loss split ----
rows = []
for i, c in enumerate(CL):
    v = R[c]['split']; v = [x*100 for x in v]
    rows.append(f"Class {i+1} & {v[0]:.2f} & {v[1]:.2f} & {v[2]:.2f} & {v[3]:+.2f} & {v[4]:+.2f} \\\\")
tabs['loss_split'] = '\n'.join(rows)
# ---- pipelines ----
def models(lst):
    out = []
    for j in range(4):
        m, k = mode([x.split(',')[j] for x in lst])
        if m != 'b': out.append(f"S{j+1}-{ {'lin': 'Lin', 'mlp_s': 'S', 'mlp_l': 'L'}[m.split('_', 1)[1]] }" + ('' if k == len(lst) else '$^\\ast$'))
    return ', '.join(out) if out else 'Baseline'
rows = []
for i, c in enumerate(CL):
    r = R[c]
    rows.append(f"{i+1} & {models(r['comb'])} & {pm(r['comb_dH'][0]*100, r['comb_dH'][1]*100)} & {pm(r['comb_G'][0], r['comb_G'][1], 1)} & {models(r['combH'])} & {pm(r['combH_dH'][0]*100, r['combH_dH'][1]*100)} & ${r['combH_dD'][0]*1e3:+.1f}$ & ${r['combH_dBrel'][0]*100:+.0f}\\%$ & ${r['combH_dEsys'][0]:+.0f}$ \\\\")
tabs['combined'] = '\n'.join(rows)
# ---- write into the paper ----
s = open(PAPER).read()
for k, v in tabs.items():
    pat = re.compile(r'(%<TAB:' + k + r'>\n).*?(%</TAB:' + k + r'>)', re.S)
    if pat.search(s): s = pat.sub(lambda m: m.group(1) + v + '\n' + m.group(2), s)
    else: print('marker missing:', k)
open(PAPER, 'w').write(s)
# ---- figures ----
W = 3.4
# energy per inference (S1, S2, S4; S3 uses the S2 architectures)
NM = ['s1_lin','s1_mlp_s','s1_mlp_l','s2_lin','s2_mlp_s','s2_mlp_l','s4_lin','s4_mlp_s','s4_mlp_l']
fig, ax = plt.subplots(figsize=(W, 2.0)); x = np.arange(len(NM)); bw = 0.2
for i, c in enumerate(CL):
    ax.bar(x + (i - 1.5)*bw, [A[c]['inf'][n]['E']*1e6 for n in NM], bw, color=COL[i], label=LAB[i])
ax.set_yscale('log'); ax.set_ylabel(r'Energy per inference ($\mu$J)'); ax.set_xticks(x)
ax.set_xticklabels([n[:2].upper() + ' ' + SH(n) for n in NM], rotation=55, ha='right', fontsize=6.5)
ax.legend(ncol=4, loc='lower center', frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.0), columnspacing=0.8, handlelength=1.2)
ax.grid(axis='y', lw=0.3, alpha=0.5); fig.tight_layout(pad=0.3); fig.savefig('../paper/fig_energy_inf.pdf')
# windowed hit ratio, Class 3, mean over seeds with 95% bands; full 1-h windows only
ss = seeds('C3'); nw = 14; tc = np.arange(nw) + 0.5
def band(key):
    M = np.array([s[key]['Hl'][:nw] for s in ss]); m = M.mean(0)
    h = stats.t.ppf(0.975, len(ss) - 1) * M.std(0, ddof=1) / np.sqrt(len(ss)); return m, h
fig, ax = plt.subplots(figsize=(W, 2.0))
for t0 in (2.4, 8.4):
    ax.axvline(t0, color='k', ls=':', lw=0.7)
ax.text(2.5, 0.272, 'reshuffle', fontsize=6.3); ax.text(8.5, 0.272, 'reshuffle', fontsize=6.3)
hb = np.array(ss[0]['base']['Hl'][:nw])
ax.plot(tc, hb, '-o', color='0.45', lw=1, ms=2.5, label=r'Baseline $\pi^0$')
for key, col, lab in (('comb_res', COL[0], r'Energy-neutral $\pi^\star_p$'), ('combH_res', COL[2], r'Hit-ratio-oriented $\pi^H_p$')):
    m, h = band(key)
    ax.fill_between(tc, m - h, m + h, color=col, alpha=0.2, lw=0)
    ax.plot(tc, m, '-o', color=col, lw=1, ms=2.5, label=lab)
ax.set_xlim(0, nw); ax.set_ylim(0.15, 0.285)
ax.set_xlabel('Time in the test part (h)'); ax.set_ylabel(r'$H_\ell$')
ax.legend(frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=6.3, handlelength=1.5, columnspacing=0.8)
ax.grid(lw=0.3, alpha=0.5)
fig.tight_layout(pad=0.3); fig.savefig('../paper/fig_window.pdf')
# break-even backhaul energy of the selected S2 model vs rate
fig, ax = plt.subplots(figsize=(W, 2.0)); be = {}
for i, c in enumerate(CL):
    o = loadmat(f'results/rate2_{c}.mat', squeeze_me=True, struct_as_record=False)['out']
    lam, eb = [], []
    for rr in np.atleast_1d(o.runs):
        b, m = rr[0], rr[2]
        lam.append(b.__dict__['lambda']); d = b.bytesB - m.bytesB; eb.append((m.EEN - b.EEN)/d if d > 0 else np.nan)
    sel = mode([x['best'][1] for x in seeds(c)])[0]; be[c] = dict(lam=lam, eb=eb, sel=sel)
    ax.plot(lam, np.array(eb)*1e9, '-o', ms=3, lw=1, color=COL[i], label=f'{LAB[i]} ({SH(sel)})')
ax.axhline(1e3, color='k', ls='--', lw=0.6); ax.text(0.04, 1.2e3, r'nominal $e_\mathrm{b}$', fontsize=6.5)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'Request rate $\lambda$ (requests/s)'); ax.set_ylabel(r'Break-even $e_\mathrm{b}$ (nJ/byte)')
ax.legend(frameon=False, fontsize=6.3, loc='lower right', ncol=2); ax.grid(lw=0.3, alpha=0.5, which='both'); ax.set_ylim(0.1, 5e3)
fig.tight_layout(pad=0.3); fig.savefig('../paper/fig_breakeven.pdf')
json.dump(be, open('results/breakeven2.json', 'w'), indent=1, default=float)
print('tables and figures written')
