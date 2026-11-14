"""Cycle counts of the int8 kernels on Cortex-M3 (Class 1) and Cortex-M4 (Class 2).

Each test is compiled with arm-none-eabi-gcc -O2 and executed in QEMU
(mps2-an385 for Cortex-M3, mps2-an386 for Cortex-M4) with one instruction per
translation block, so that the execution log lists every executed instruction.
The instructions between roi_begin() and roi_end() are mapped to their
mnemonics with objdump and costed with the instruction timings of the
Cortex-M3 and Cortex-M4 technical reference manuals (pipeline refill P = 2 on
taken branches; neighbouring single loads/stores pipelined). For the
STM32F103 at 72 MHz, two flash wait states are charged on every taken branch,
since sequential fetches are covered by the prefetch buffer. The empty-ROI test
is subtracted. The result is a modelled cycle count, not a measurement.
"""
import subprocess, re, json, sys

CORES = {
    'm3': dict(cpu='cortex-m3', flags=['-mfloat-abi=soft'], machine='mps2-an385', flash_ws=2),
    'm4': dict(cpu='cortex-m4', flags=['-mfpu=fpv4-sp-d16', '-mfloat-abi=hard'], machine='mps2-an386', flash_ws=0),
}
TESTS = {1:'s1_lin',2:'s1_mlp_s',3:'s1_mlp_l',4:'s2_lin',5:'s2_mlp_s',6:'s2_mlp_l',
         7:'s3_lin',8:'s3_mlp_s',9:'s3_mlp_l',10:'s4_lin',11:'s4_mlp_s',12:'s4_mlp_l',
         13:'ewma_item',14:'lru_scan1000',15:'lfu_scan1000',16:'empty',17:'s3_cpu_part'}

def disasm(elf):
    out = subprocess.run(['arm-none-eabi-objdump','-d',elf],capture_output=True,text=True).stdout
    ins, syms = {}, {}
    for line in out.splitlines():
        m = re.match(r'^([0-9a-f]+) <(\w+)>:', line)
        if m: syms[m.group(2)] = int(m.group(1),16); continue
        m = re.match(r'^\s+([0-9a-f]+):\s+((?:[0-9a-f]{4}\s?)+)\s+(\S+)\s*(.*)$', line)
        if m:
            pc = int(m.group(1),16); size = 2*len(m.group(2).split())
            ins[pc] = (m.group(3).split('.')[0], m.group(4), size)
    return ins, syms

def cost(mn, ops, taken, prev_ls, core):
    m4 = core == 'm4'; P = 2
    if mn.startswith(('ldm','stm','push','pop')):
        n = len(re.findall(r'r\d+|lr|pc|sl|fp|ip', ops.split('{')[1] if '{' in ops else ops))
        return 1 + n + (P if 'pc' in ops else 0), False
    if mn in ('ldrd','strd'): return 3, False
    if mn.startswith(('ldr','str')): return (1 if prev_ls else 2), True
    if mn in ('b','bl','bx','blx') or mn.startswith('b') and len(mn) <= 4 and mn not in ('bic','bics','bfi','bfc'):
        return (1 + P) if taken else 1, False
    if mn in ('cbz','cbnz'): return (1 + P) if taken else 1, False
    if mn in ('mla','mls'): return (1 if m4 else 2), False
    if mn in ('smull','umull'): return (1 if m4 else 5), False
    if mn in ('smlal','umlal'): return (1 if m4 else 6), False
    if mn in ('sdiv','udiv'): return 7, False
    return 1, False

def run(test, core):
    c = CORES[core]; elf = f'/tmp/k_{core}_{test}.elf'; log = f'/tmp/k_{core}_{test}.log'
    subprocess.run(['arm-none-eabi-gcc','-O2',f'-mcpu={c["cpu"]}','-mthumb',*c['flags'],'-nostdlib',
                    '-ffreestanding',f'-DTEST={test}','-T','mps2.ld','kern.c','-o',elf],check=True,capture_output=True)
    subprocess.run(['qemu-system-arm','-machine',c['machine'],'-nographic','-semihosting','-kernel',elf,
                    '-accel','tcg,one-insn-per-tb=on','-d','exec,nochain','-D',log],check=True,timeout=600)
    ins, syms = disasm(elf)
    pcs = []
    for line in open(log):
        m = re.search(r'\[[0-9a-f]+/([0-9a-f]+)/', line)
        if m: pcs.append(int(m.group(1),16))
    a, b = syms['roi_begin'], syms['roi_end']
    i0 = pcs.index(a); i1 = pcs.index(b, i0 + 1)
    roi = pcs[i0:i1]
    cyc = ninst = 0; prev_ls = False
    for j, pc in enumerate(roi):
        mn, ops, size = ins[pc]
        nxt = roi[j+1] if j+1 < len(roi) else b
        taken = nxt != pc + size
        cy, ls = cost(mn, ops, taken, prev_ls, core)
        if taken: cy += c['flash_ws']
        cyc += cy; ninst += 1; prev_ls = ls
    return cyc, ninst

res = {}
import os
if os.path.exists('cycles.json'): res = json.load(open('cycles.json'))
for core in CORES:
    for t, name in TESTS.items():
        if f'{core}_{name}' in res: continue
        res[f'{core}_{name}'] = run(t, core)
        print(core, name, res[f'{core}_{name}'], flush=True)
json.dump(res, open('cycles.json','w'), indent=1)
