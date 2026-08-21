# raw-editor — Phase 5/6 algorithm reference

Research notes gathered ahead of implementation. Source for demosaic:
Getreuer, *Malvar-He-Cutler Linear Image Demosaicking*, IPOL 2011
(https://www.ipol.im/pub/art/2011/g_mhcd/), which reproduces Malvar/He/Cutler
ICASSP 2004.

## Phase 6 — Malvar-He-Cutler demosaic (replaces bilinear)

MHC is bilinear interpolation plus a **cross-channel Laplacian correction**: the
missing channel borrows the high-frequency detail of the channel that *was*
sampled at that site. That is the whole idea, and it is why a purely linear 5x5
method beats naive bilinear so visibly (PSNR 29.66 vs 29.47 on Kodak, and much
better on edges — bilinear's zipper artifacts largely vanish).

Correction weights, rounded to dyadic rationals so the filters are exact in
integer arithmetic:

    alpha = 1/2   (G at R/B sites, 5-point Laplacian of R or B)
    beta  = 5/8   (R/B at G sites, 9-point Laplacian of G)
    gamma = 3/4   (R at B and B at R sites, 5-point Laplacian of the other)

### The eight 5x5 filters (coefficients scaled by 8 — divide the sum by 8)

Offsets are (dx, dy) from the centre pixel. "Same-axis" means the axis along
which the *needed* colour is actually sampled.

**G at R sites, and G at B sites** (identical shape):

    centre        : 4
    (+-1, 0), (0, +-1) : 2      <- bilinear average of the 4 axial greens
    (+-2, 0), (0, +-2) : -1     <- -alpha * Laplacian of R (or B)

**R at G sites** (and, by symmetry, B at G sites): centre 5; the two
same-axis neighbours (where the R samples actually are) 4; same-axis +-2: -1;
all four diagonals: -1; the perpendicular +-2: +1/2.

Written for the case where the R neighbours are horizontal:

    ( 0,-2): 1/2
    (-1,-1): -1              (+1,-1): -1
    (-2, 0): -1   (-1,0): 4   (0,0): 5   (+1,0): 4   (+2,0): -1
    (-1,+1): -1              (+1,+1): -1
    ( 0,+2): 1/2

When the R neighbours are vertical, this filter is simply **transposed**.

Integer form (divide by 16 instead, all coefficients doubled) — from the paper:

    R = ( F(i,j-2) + F(i,j+2)
        - 2*( F(i-1,j-1) + F(i+1,j-1) + F(i-2,j) + F(i+2,j)
            + F(i-1,j+1) + F(i+1,j+1) )
        + 8*( F(i-1,j) + F(i+1,j) )
        + 10*F(i,j) ) / 16

**R at B sites** (and B at R sites — diagonal neighbours):

    centre               : 6
    the 4 diagonals      : 2     <- bilinear average of the 4 diagonal samples
    (+-2,0), (0,+-2)     : -3/2  <- -gamma * Laplacian of the other channel

### Implementation warning

The paper's Figure 3 labels the four G-site filters by "red rows" / "blue rows",
and those labels are easy to mis-transcribe (the drawn orientation and the text's
index convention disagree about which index runs horizontally). **Do not copy the
orientations from the figure.** Derive each filter's orientation from the
unambiguous physical fact: at a green site, the needed R samples lie along
whichever axis actually carries R in that row. Pick the orientation at runtime
from `cfa.color_at(row, col+1)`.

Verify with a synthetic round trip rather than by eye: take a known RGB image,
mosaic it to the GF1's GBRG pattern, demosaic, and assert PSNR is meaningfully
above the bilinear path's PSNR on the same input. That test catches a transposed
filter immediately, which visual inspection will not.

Note the GF1 is **GBRG**, not the RGGB the literature assumes — the phase offset
must come from `cfa.color_at()`, never a hardcoded parity check.

## Phase 5 — sharpening and noise reduction

Both are neighbourhood operations, so they cannot live in the existing
single-pass fragment shader; they need their own passes over the whole image
before the adjustment chain samples it. Do them as compute passes writing an
intermediate `rgba16float` texture.

**Sharpening (unsharp mask).** Separable Gaussian: one horizontal pass, one
vertical, then `out = in + amount * (in - blur)`. Two 1-D passes cost 2*r taps
instead of r^2, so this is cheap even at 12MP. Two quality points that matter:
apply it to **luminance only** (sharpening chroma produces coloured edge
fringes), and include a threshold so flat, noisy areas are left alone —
without it, sharpening at ISO 400+ mostly amplifies grain. Radius ~1-2px for
capture sharpening.

**Noise reduction.** A true bilateral filter is O(r^2) per pixel and not
separable, since the range weights depend on the centre pixel. Options in
increasing order of effort:

1. *Chroma-only blur* — convert to a luma/chroma space, blur chroma hard with a
   large separable Gaussian, leave luma alone. Cheap, and it removes the blotchy
   colour noise that dominates the GF1's high-ISO look. Best effort-to-payoff
   ratio; do this first.
2. *Separable-approximation bilateral on luma* — run the bilateral horizontally
   then vertically. Not mathematically correct, but it's the standard photo-NR
   shortcut and the artifacts are hard to see.
3. *Guided filter* — edge-preserving, runs in O(1) per pixel via box filters
   (summed-area or separable), and generally beats bilateral on gradients where
   bilateral produces staircasing. Bart Wronski's write-up is the clearest
   practical reference: https://bartwronski.com/2019/09/22/local-linear-models-guided-filter/

Recommended split for v1: chroma blur (strength slider) + luma bilateral
(strength slider), matching how Lightroom exposes "Color" and "Luminance" noise
reduction separately.

**Ordering.** NR before sharpening, both before the tone/colour chain, and all of
it on scene-referred linear data.

**Performance note.** These passes run on the whole image every time their own
sliders change. If that turns out to be too slow on the GTX 960 at 12MP, the
standard fix is to run detail passes on a screen-resolution proxy while dragging
and only do the full-resolution pass on export — not to make the filters cheaper.
