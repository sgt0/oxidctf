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
