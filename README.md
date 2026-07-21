# oxidctf

oxidctf is a Rust port of
[HolyWu/VapourSynth-DCTFilter](https://github.com/HolyWu/VapourSynth-DCTFilter) /
[Mr-Z-2697/VapourSynth-DCTFilter](https://github.com/Mr-Z-2697/VapourSynth-DCTFilter),
a [VapourSynth](https://www.vapoursynth.com/) plugin that suppresses selected
frequencies of a clip. For each n x n block it performs a Discrete Cosine
Transform (DCT), scales down the selected frequency values, and then reverses
the process with an Inverse Discrete Cosine Transform (IDCT).

## Install

```
pip install vapoursynth-oxidctf
```

## API

```python
oxidctf.DCTFilter(
  clip: vs.VideoNode,
  factors: Sequence[float],
  nsize: int = 8,
  planes: Sequence[int] = [0, 1, 2],
) -> vs.VideoNode
```

- `clip` — Clip to process. Any format with either an integer sample type of
  8-16 bit depth or a float sample type of 32 bit depth is supported.
- `factors` — Either `nsize` or `nsize * nsize` floating point numbers, all of
  which must be in the range `0.0 <= x <= 1.0`.

  With `nsize` values, these correspond to scaling factors for the rows and
  columns of the n x n DCT blocks. The leftmost number corresponds to the top
  row, left column. This is the DC component of the transform and should always
  be left as `1.0`. The row and column numbers are multiplied together to get
  the scale factor for each of the values in a block.

  With `nsize * nsize` values, they are used directly as a row-major
  per-coefficient scaling matrix, which allows patterns that the outer product
  cannot express.
- `nsize` — Size of the DCT blocks.
- `planes` — Which planes to process. Any unfiltered planes are copied from the
  input clip.

## How this all works

Any image is, in essence, a grid of numbers. A grayscale picture is one such
grid. A color picture is a few grids stacked together, one per plane. Each
number records how "bright" a single pixel is. oxidctf chops each plane into
small square tiles (by default `8 x 8`, so 64 numbers apiece) and works on one
tile at a time.

But why tiles, and why a cosine transform? The reason is that a block of pixels
can be described in two equivalent ways. The perhaps obvious way is the one I
initially described: list the 64 brightness values. The less obvious way is to
instead ask how "wavy" the block is. A perfectly flat block (every pixel the
same shade) has no waviness at all. A block that alternates light and dark like
a checkerboard is extremely wavy. In practice, most blocks lie somewhere
between, and they can be built by adding together a fixed set of wave patterns,
each contributing a little or a lot.

The Discrete Cosine Transform (DCT) is a way to switch from the first
way I described to the second. It rewrites the block as a weighted sum of cosine
ripples of increasing frequency. The slowest ripple (no ripple at all, a
constant) is called the DC component; its weight is just the average brightness
of the block. The faster ripples are the AC components; their weights say how
much fine detail, and of what fineness, the block contains. Notably, the switch
is lossless, which means the transformation is reversible. The Inverse DCT
(IDCT) can take the list of weights and reconstruct the original 64 pixels
exactly.

Once we have a block expressed as a set of frequency weights, suppressing a
frequency becomes quite easy. To soften fine detail, we can shrink the weights
of the fast ripples toward zero and then transform back. This is what the
`factors` argument controls: one multiplier per frequency, each between `0.0`
(erase the frequency entirely) and `1.0` (leave the frequency untouched). The DC
weight is normally left at `1.0`, since scaling the average brightness would
darken the image. So the whole operation, for one block, is a round trip:

1. Transform to frequencies.
2. Scale the frequencies.
3. Transform back.

### Math time

Now is the point where I have to pull out math notation that I definitely did
not forget from school. Let's put the cosine recipe for an `n x n` block into a
single matrix $C$, the DCT matrix. Its entry in row $k$, column $j$ is:

$$C_{k,j} = a_k \cos\!\left(\frac{\pi\,(2j+1)\,k}{2n}\right),\qquad a_0 = \sqrt{\tfrac{1}{n}},\quad a_k = \sqrt{\tfrac{2}{n}}\ \ (k > 0).$$

Row $k$ of $C$ is the $k$-th cosine ripple, sampled at the $n$ pixel positions.
The leading constants $a_k$ are chosen so that $C$ is orthonormal, which is a
nice property that means its inverse is merely its transpose:
$C^{-1} = C^{\mathsf{T}}$. This fact will be invaluable later on.

A block, though, is two-dimensional. The transform is applied first along the
rows and then along the columns, an operation that, for a block $B$, comes out
as sandwiching $B$ between the matrix and its transpose:

$$F = C\,B\,C^{\mathsf{T}}.$$

Here $F$ holds the 64 frequency weights. We scale them by multiplying each
weight, position for position, by its factor (let's call the grid of factors
$M$) and then invert the transform to return to pixels:

$$B' = C^{\mathsf{T}}\,(F \odot M)\,C,$$

where $\odot$ denotes entry-by-entry multiplication. When the caller supplies
only `nsize` factors rather than a full `nsize * nsize` grid, $M$ is built as the
outer product of that list with itself: the factor for row $y$, column $x$ is
the row factor times the column factor.

### Naive first approach

Read literally, the formula above prescribes, for **every single block** of the
image, four matrix multiplications and one scaling pass:

- $CB$ — Transform the columns.
- $(CB)\,C^{\mathsf{T}}$ — Transform the rows, yielding the frequencies $F$.
- $F \odot M$ — Scale each frequency by its factor.
- $C^{\mathsf{T}}(\dots)$ and $(\dots)\,C$ — The two multiplications of the inverse.

If there's anything I've learned from writing VapourSynth plugins, it's that
every nanosecond of overhead counts. A 1080p plane holds more than 30,000
`8 x 8` blocks, and the typical video will have tens of thousands of such
planes. Each of the four multiplications above costs on the order of $n^3$
arithmetic operations per block, and every one reads and writes a fresh little
intermediate matrix. This algorithm worked fine when I implemented it, but it
wasn't exactly fast.

### Optimizing

The key thing to notice is that the whole round trip (forward transform, scale,
inverse transform) is linear. Composing linear steps yields another linear step,
so the entire per-block operation can be expressed as a single fixed matrix.
Since that matrix depends only on $C$ and the factors, never on the pixels, it
can be computed once, when the filter is created, and then reused for every
block of every frame.

Accomplishing this requires flattening a block's 64 numbers into a single column
of length $n^2$. Under that flattening, the two-sided product
$C B C^{\mathsf{T}}$ becomes an ordinary one-sided product by the
[Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product)
$D = C \otimes C$, an $n^2 \times n^2$ matrix that applies $C$ to rows and
columns at once. Scaling by the factors is multiplication by a diagonal matrix
$\operatorname{diag}(f)$ whose entries are the flattened grid $M$, and the
inverse transform is $D^{\mathsf{T}}$. Chaining the three gives the single
operator

$$A = D^{\mathsf{T}}\,\operatorname{diag}(f)\,D,\qquad D = C \otimes C,$$

so that filtering a flattened block $x$ is now the lone product $A x$. oxidctf
builds $A$ during filter creation and stores it. Further note that $A$ is
symmetric, because $\operatorname{diag}(f)$ is: $A^{\mathsf{T}} = D^{\mathsf{T}}\operatorname{diag}(f)\,D = A$. Symmetry means it makes no difference whether
blocks are multiplied on the left or the right, which frees the implementation
to lay the data out in whichever direction is fastest. What I went with was
storing the blocks as rows, computed as $xA$.

Collapsing four multiplications into one is only half the story, however. The
other half is that oxidctf does not process blocks one at a time. It gathers an
entire row of blocks and stacks their flattened forms as the rows of one tall
matrix $x$ of shape `blocks x n²`. A single matrix product $xA$ then filters the
whole row at once.

There are also a couple of smaller decisions at play here. Input pixels may be
8- or 16-bit integers, or 32-bit floats, but all arithmetic is done in 32-bit
float, with conversion back to the source type deferred to the final write. And
because the block grid must tile the plane evenly, oxidctf pads each plane by
mirroring pixels along its right and bottom edges until the dimensions are
multiples of the block size, then crops the padding away at the end, so we don't
need to special-case the edges.
