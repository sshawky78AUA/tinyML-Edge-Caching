"""Tables and numbers for the paper from results2<tag>.json (aggregate2.m)."""
import json, sys, numpy as np
from scipy import stats
tag = sys.argv[1] if len(sys.argv) > 1 else ''
A = json.load(open(f'results2{tag}.json'))
CL = ['C1', 'C2', 'C3', 'C4']
PR = lambda n: 'Baseline' if n == 'b' else n.split('_', 1)[0].upper() + ' ' + {'lin': 'linear', 'mlp_s': 'MLP-S', 'mlp_l': 'MLP-L'}[n.split('_', 1)[1]]
def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]; n = len(v)
    if n < 2: return float(v.mean()) if n else np.nan, np.nan
    return float(v.mean()), float(stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n))
def S(c):
    s = A[c]['seeds']; return s if isinstance(s, list) else [s]
out = {}
for c in CL:
    ss = S(c); n = len(ss); r = {'n': n}
    r['baseH'] = ci([s['base']['H'] for s in ss])[0]; r['baseD'] = ci([s['base']['D'] for s in ss])[0]
    r['baseEsys'] = ci([s['base']['Esys'] for s in ss])[0]; r['baseEEN'] = ci([s['base']['EEN'] for s in ss])[0]
    # single stages (validation for all, test for selected)
    for j in '1234':
        for m in ('lin', 'mlp_s', 'mlp_l'):
            k = f's{j}_{m}'; v = [s['single'].get(k, {}) for s in ss]
            if not v[0] or not v[0].get('feas'): continue
            r[k] = dict(dHva=ci([x['dHva'] for x in v]), Gva=ci([x['Gva'] for x in v]),
                        dHte=ci([x.get('dH', np.nan) for x in v]), Gte=ci([x.get('G', np.nan) for x in v]),
                        dEEN=ci([x.get('dEEN', np.nan) for x in v]))
        sel = [s['best'][int(j) - 1] for s in ss]; r[f'sel{j}'] = sel
        vals = [s['single'][b]['dH'] for s, b in zip(ss, sel) if b != 'b' and 'dH' in s['single'][b]]
        gs = [s['single'][b]['G'] for s, b in zip(ss, sel) if b != 'b' and 'G' in s['single'][b]]
        r[f'selgain{j}'] = (ci(vals), ci(gs), len(vals))
    # split
    sp = [s['split'][s['best'][1]] for s in ss if s['best'][1] != 'b' and s['best'][1] in s['split']]
    r['split'] = np.mean(np.array(sp), 0).tolist() if sp else None
    # energy-neutral pipeline
    r['comb'] = [','.join(s['comb']) for s in ss]
    cr = [s['comb_res'] for s in ss]
    r['comb_dH'] = ci([x['dH'] for x in cr]); r['comb_G'] = ci([x['G'] for x in cr]); r['comb_rho'] = ci([x['rho'] for x in cr])
    r['comb_I'] = ci([s['I'] for s in ss]); r['comb_dD'] = ci([x['dD'] for x in cr]); r['comb_dB'] = ci([x['dBrel'] for x in cr])
    r['comb_dHB'] = ci([x['dHB'] for x in cr])
    # hit-ratio-oriented pipeline
    r['combH'] = [','.join(s['combH']) for s in ss]; r['m2H'] = [s['m2H'] for s in ss]
    hr = [s['combH_res'] for s in ss]
    for f in ('dH', 'dHB', 'dD', 'dBrel', 'dEsys', 'dEEN', 'dElink', 'rho', 'G'):
        r['combH_' + f] = ci([x[f] for x in hr])
    r['A_dH'] = ci([s['A_res']['dH'] for s in ss])
    r['behind'] = {}
    for k in ('s1_lin', 's1_mlp_s', 's1_mlp_l', 's3_lin', 's3_mlp_s', 's3_mlp_l', 's4_lin', 's4_mlp_s', 's4_mlp_l'):
        v = [s['behind'][k] for s in ss if k in s.get('behind', {}) and 'dHva' in s['behind'][k]]
        if v: r['behind'][k] = dict(dHva=ci([x['dHva'] for x in v]), dHte=ci([x.get('dHte', np.nan) for x in v]), nte=sum('dHte' in x for x in v))
    r['bestA'] = [','.join(s['bestA']) for s in ss]
    out[c] = r
json.dump(out, open(f'analysis2{tag}.json', 'w'), indent=1, default=float)
for c in CL:
    r = out[c]; print(f"== {c} n={r['n']} baseH={r['baseH']:.4f}")
    for j in '1234':
        print(f"  S{j} sel {r['sel'+j]}  test dH {r['selgain'+j][0]} G {r['selgain'+j][1]}")
        for m in ('lin', 'mlp_s', 'mlp_l'):
            k = f's{j}_{m}'
            if k in r: print(f"     {k}: dHva {r[k]['dHva'][0]*100:+.2f}±{r[k]['dHva'][1]*100:.2f}  Gva {r[k]['Gva'][0]:+.1f}  dEEN {r[k]['dEEN'][0]:.2f}")
    print('  split', r['split'])
    print(f"  comb {set(r['comb'])} dH {r['comb_dH'][0]*100:+.2f}±{r['comb_dH'][1]*100:.2f} G {r['comb_G'][0]:+.1f}±{r['comb_G'][1]:.1f} I {r['comb_I'][0]*100:+.2f}±{r['comb_I'][1]*100:.2f}")
    print(f"  combH {set(r['combH'])} dH {r['combH_dH'][0]*100:+.2f}±{r['combH_dH'][1]*100:.2f} dD {r['combH_dD'][0]*1e3:+.2f}ms dB {r['combH_dBrel'][0]*100:+.1f}% dEsys {r['combH_dEsys'][0]:+.1f} dEEN {r['combH_dEEN'][0]:+.2f} rho {r['combH_rho'][0]:.2g}")
    print('  behind', {k: (round(v['dHva'][0]*100, 2), round(v['dHte'][0]*100, 2) if v['nte'] else None) for k, v in r['behind'].items()})
