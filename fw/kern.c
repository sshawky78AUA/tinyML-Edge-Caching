/* int8 inference kernels and baseline cache operations used to obtain the
 * per-decision cycle counts of the Cortex-M classes. The arithmetic follows
 * the TensorFlow Lite int8 reference scheme: int8 x int8 products, int32
 * accumulation with bias, fixed-point requantization, ReLU by clamping.
 * On cores with the DSP extension (__ARM_FEATURE_DSP) the inner product uses
 * SXTB16/SMLAD on pairs of operands, as in CMSIS-NN.
 *
 * Build one binary per test with -DTEST=<n>; the region of interest lies
 * between the calls to roi_begin() and roi_end().                       */
#include <stdint.h>
#if defined(__ARM_FEATURE_DSP)
#include <arm_acle.h>
#endif

void __attribute__((noinline)) roi_begin(void) { __asm volatile("nop"); }
void __attribute__((noinline)) roi_end(void)   { __asm volatile("nop"); }

static inline int32_t requant(int32_t acc, int32_t mult, int shift, int32_t zp) {
  int64_t p = (int64_t)acc * mult;
  int32_t r = (int32_t)((p + ((int64_t)1 << 30)) >> 31);
  r = (shift > 0) ? (r << shift) : (r >> (-shift));
  r += zp;
  if (r < -128) r = -128; if (r > 127) r = 127;
  return r;
}

/* y = act(W x + b), W stored row-major [out][in] */
void __attribute__((noinline)) dense(const int8_t *x, int nin, const int8_t *W,
                                     const int32_t *b, int nout, int8_t *y, int relu) {
  for (int o = 0; o < nout; o++) {
    const int8_t *w = W + o * nin;
    int32_t acc = b[o];
    int i = 0;
#if defined(__ARM_FEATURE_DSP)
    for (; i + 3 < nin; i += 4) {
      uint32_t xv = *(const uint32_t *)(x + i), wv = *(const uint32_t *)(w + i);
      uint32_t x0 = __sxtb16(xv), x1 = __sxtb16(((xv >> 8) | (xv << 24)));
      uint32_t w0 = __sxtb16(wv), w1 = __sxtb16(((wv >> 8) | (wv << 24)));
      acc = __smlad(x0, w0, acc);
      acc = __smlad(x1, w1, acc);
    }
#endif
    for (; i < nin; i++) acc += (int32_t)x[i] * w[i];
    int32_t r = requant(acc, 1395864371, -7, 0);
    if (relu && r < 0) r = 0;
    y[o] = (int8_t)r;
  }
}

static int8_t sigmoid_lut[256];

#define MAXW 64*64
static int8_t W1[MAXW] __attribute__((aligned(4))), W2[MAXW] __attribute__((aligned(4))), W3[48*513+64*513] __attribute__((aligned(4)));
static int32_t B[1024];
static int8_t X[64] __attribute__((aligned(4))), H1[128] __attribute__((aligned(4))), H2[128] __attribute__((aligned(4))), OUT[520] __attribute__((aligned(4)));
static int8_t EMB[515*16];
static int32_t TS[4096]; static int32_t FREQ[4096];
volatile int32_t sink;

static void fill(void) {
  uint32_t s = 12345;
  for (int i = 0; i < MAXW; i++) { s = s*1103515245u + 12345u; W1[i] = (int8_t)(s >> 24); W2[i] = (int8_t)(s >> 16); }
  for (unsigned i = 0; i < sizeof(W3); i++) { s = s*1103515245u + 12345u; W3[i] = (int8_t)(s >> 24); }
  for (int i = 0; i < 1024; i++) B[i] = i;
  for (int i = 0; i < 64; i++) X[i] = (int8_t)(i*7);
  for (int i = 0; i < 520; i++) { s = s*1103515245u + 12345u; OUT[i] = (int8_t)(s >> 24); }
  for (int i = 0; i < 515*16; i++) EMB[i] = (int8_t)i;
  for (int i = 0; i < 4096; i++) { s = s*1103515245u + 12345u; TS[i] = (int32_t)(s >> 1); FREQ[i] = (int32_t)(s & 15); }
  for (int i = 0; i < 256; i++) sigmoid_lut[i] = (int8_t)(i - 128);
}

/* one binary-output MLP inference: nin -> h (x nh layers) -> 1 */
static void mlp(int nin, int h, int nh) {
  if (nh == 0) { dense(X, nin, W1, B, 1, OUT, 0); }
  else {
    dense(X, nin, W1, B, h, H1, 1);
    if (nh > 1) { dense(H1, h, W2, B, h, H2, 1); dense(H2, h, W3, B, 1, OUT, 0); }
    else dense(H1, h, W3, B, 1, OUT, 0);
  }
  sink = sigmoid_lut[(uint8_t)OUT[0]];
}

/* S3: gather Q3=3 embeddings of size d, optional hidden layer, 513 logits, top-2 */
static void s3(int d, int h) {
  int8_t *z = H1; const int tok[3] = {5, 77, 400};
  for (int q = 0; q < 3; q++) for (int j = 0; j < d; j++) z[q*d + j] = EMB[tok[q]*d + j];
  if (h) { dense(z, 3*d, W1, B, h, H2, 1); dense(H2, h, W3, B, 513, OUT, 0); }
  else dense(z, 3*d, W3, B, 513, OUT, 0);
  int b1 = 0, b2 = 1;
  for (int i = 0; i < 512; i++) {
    if (OUT[i] > OUT[b1]) { b2 = b1; b1 = i; } else if (i != b1 && OUT[i] > OUT[b2]) b2 = i;
  }
  sink = b1 + b2;
}

static void s3_cpu_part(void) {
  int8_t *z = H1; const int tok[3] = {5, 77, 400};
  for (int q = 0; q < 3; q++) for (int j = 0; j < 8; j++) z[q*8 + j] = EMB[tok[q]*8 + j];
  int b1 = 0, b2 = 1;
  for (int i = 0; i < 512; i++) {
    if (OUT[i] > OUT[b1]) { b2 = b1; b1 = i; } else if (i != b1 && OUT[i] > OUT[b2]) b2 = i;
  }
  sink = b1 + b2 + z[3];
}

/* baselines: EWMA score of one item over nw=6 windows (Q8 fixed point) */
static void ewma_item(void) {
  static const int32_t c[6] = {3,1,0,2,5,1}; int32_t s = 0, w = 256;
  for (int j = 0; j < 6; j++) { s += c[j]*w; w >>= 1; }
  sink = s;
}
/* LRU / LFU victim scans over n cached items */
static void lru_scan(int n) { int b = 0; for (int i = 1; i < n; i++) if (TS[i] < TS[b]) b = i; sink = b; }
static void lfu_scan(int n) {
  int b = 0;
  for (int i = 1; i < n; i++) if (FREQ[i] < FREQ[b] || (FREQ[i] == FREQ[b] && TS[i] < TS[b])) b = i;
  sink = b;
}

int main(void) {
  fill();
  roi_begin();
#if   TEST == 1
  mlp(7, 0, 0);            /* s1_lin   */
#elif TEST == 2
  mlp(7, 16, 2);           /* s1_mlp_s */
#elif TEST == 3
  mlp(7, 64, 2);           /* s1_mlp_l */
#elif TEST == 4
  mlp(6, 0, 0);            /* s2_lin   */
#elif TEST == 5
  mlp(6, 16, 2);           /* s2_mlp_s */
#elif TEST == 6
  mlp(6, 64, 2);           /* s2_mlp_l */
#elif TEST == 7
  s3(8, 0);                /* s3_lin   */
#elif TEST == 8
  s3(8, 32);               /* s3_mlp_s */
#elif TEST == 9
  s3(16, 64);              /* s3_mlp_l */
#elif TEST == 10
  mlp(9, 0, 0);            /* s4_lin   */
#elif TEST == 11
  mlp(9, 16, 2);           /* s4_mlp_s */
#elif TEST == 12
  mlp(9, 64, 2);           /* s4_mlp_l */
#elif TEST == 13
  ewma_item();             /* S1 baseline, per item */
#elif TEST == 14
  lru_scan(1000);          /* LRU scan of 1000 items */
#elif TEST == 15
  lfu_scan(1000);          /* LFU scan of 1000 items */
#elif TEST == 16
  ;                        /* empty ROI: marker overhead */
#elif TEST == 17
  s3_cpu_part();           /* S3 gather + top-2 only (CPU part on Class 3) */
#endif
  roi_end();
  /* semihosting exit */
  register int r0 __asm("r0") = 0x18; register int r1 __asm("r1") = 0x20026;
  __asm volatile("bkpt 0xab" :: "r"(r0), "r"(r1));
  for (;;);
}

/* minimal vector table and reset handler */
extern uint32_t _estack, _sidata, _sdata, _edata, _sbss, _ebss;
void Reset_Handler(void) {
  uint32_t *s = &_sidata, *d = &_sdata; while (d < &_edata) *d++ = *s++;
  for (d = &_sbss; d < &_ebss;) *d++ = 0;
#if defined(__ARM_FP)
  *(volatile uint32_t *)0xE000ED88 |= (0xF << 20);   /* enable FPU */
#endif
  main();
}
__attribute__((section(".isr_vector"))) const void *vectors[] = { &_estack, Reset_Handler };
