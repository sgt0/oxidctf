use std::ffi::{CStr, CString, c_void};

use const_str::cstr;
use ndarray::Array2;
use oxiblas_ndarray::blas::gemm_ndarray;
use vapours::{enums::ColorRange, frame::VapoursVideoFrame, generic::HoldsVideoFormat};
use vapoursynth4_rs::{
  SampleType, VideoInfo,
  core::CoreRef,
  declare_plugin,
  ffi::VSFrame,
  frame::{Frame, FrameContext, VideoFrame},
  key,
  map::{AppendMode, MapRef, Value},
  node::{
    ActivationReason, Dependencies, Filter, FilterDependency, Node, RequestPattern, VideoNode,
  },
  utils::is_constant_video_format,
};

/// A pixel type that can be converted to and from `f32`.
trait Pixel: Copy {
  fn to_f32(self) -> f32;
  fn from_f32(value: f32, peak: f32) -> Self;
}

impl Pixel for u8 {
  #[inline]
  fn to_f32(self) -> f32 {
    f32::from(self)
  }

  #[inline]
  fn from_f32(value: f32, peak: f32) -> Self {
    (value + 0.5).clamp(0.0, peak) as Self
  }
}

impl Pixel for u16 {
  #[inline]
  fn to_f32(self) -> f32 {
    f32::from(self)
  }

  #[inline]
  fn from_f32(value: f32, peak: f32) -> Self {
    (value + 0.5).clamp(0.0, peak) as Self
  }
}

impl Pixel for f32 {
  #[inline]
  fn to_f32(self) -> f32 {
    self
  }

  #[inline]
  fn from_f32(value: f32, _peak: f32) -> Self {
    value
  }
}

struct DctFilter {
  /// Input video node.
  node: VideoNode,

  /// Block size.
  nsize: usize,

  /// Precomputed combined DCT operator matrix.
  /// A = D^T * diag(f) * D, where D = C (x) C.
  a: Array2<f32>,

  /// Indicates whether or not the plane at index `i` should be processed.
  process_planes: [bool; 3],

  /// Maximum pixel value for integer formats.
  peak: f32,
}

impl DctFilter {
  fn filter_plane<T: Pixel>(&self, src: &VideoFrame, dst: &mut VideoFrame, plane: i32) {
    let n = self.nsize;
    let nn = n * n;
    let width = src.frame_width(plane) as usize;
    let height = src.frame_height(plane) as usize;
    let stride = src.stride(plane) as usize / size_of::<T>();
    let srcp = src.as_slice::<T>(plane);
    let dst_stride = dst.stride(plane) as usize / size_of::<T>();
    let dstp = dst.as_mut_slice::<T>(plane);

    let blocks = width / n;
    let mut x = Array2::<f32>::zeros((blocks, nn));
    let mut out = Array2::<f32>::zeros((blocks, nn));

    for y in (0..height).step_by(n) {
      {
        let xs = x.as_slice_mut().expect("x should be contiguous");
        for yy in 0..n {
          let row = &srcp[stride * (y + yy)..stride * (y + yy) + width];
          for b in 0..blocks {
            let dst_off = b * nn + yy * n;
            for xx in 0..n {
              xs[dst_off + xx] = row[b * n + xx].to_f32();
            }
          }
        }
      }

      // `a` is symmetric, so `x * a` == `x * a^T`.
      gemm_ndarray(1.0, &x, &self.a, 0.0, &mut out);

      let os = out.as_slice().expect("out should be contiguous");
      for yy in 0..n {
        let row = &mut dstp[dst_stride * (y + yy)..dst_stride * (y + yy) + width];
        for b in 0..blocks {
          let src_off = b * nn + yy * n;
          for xx in 0..n {
            row[b * n + xx] = T::from_f32(os[src_off + xx], self.peak);
          }
        }
      }
    }
  }
}

/// Pad `node` by mirroring `pad_width`/`pad_height` pixels on the right/bottom.
/// Returns the padded node and its updated `VideoInfo`.
fn pad_node(
  core: &CoreRef<'_>,
  node: VideoNode,
  pad_width: i32,
  pad_height: i32,
) -> Result<(VideoNode, VideoInfo), CString> {
  let vi = node.info().clone();

  let Some(resize) = core.get_plugin_by_id(cstr!("com.vapoursynth.resize")) else {
    return Err(cstr!("oxidctf.DCTFilter: the resize plugin is missing.").to_owned());
  };

  let mut args = core.create_map();
  args
    .set(key!(c"clip"), Value::VideoNode(node), AppendMode::Replace)
    .and_then(|()| {
      args.set(
        key!(c"width"),
        Value::Int(i64::from(vi.width + pad_width)),
        AppendMode::Replace,
      )
    })
    .and_then(|()| {
      args.set(
        key!(c"height"),
        Value::Int(i64::from(vi.height + pad_height)),
        AppendMode::Replace,
      )
    })
    .and_then(|()| {
      args.set(
        key!(c"src_width"),
        Value::Float(f64::from(vi.width + pad_width)),
        AppendMode::Replace,
      )
    })
    .and_then(|()| {
      args.set(
        key!(c"src_height"),
        Value::Float(f64::from(vi.height + pad_height)),
        AppendMode::Replace,
      )
    })
    .expect("should set resize.Point arguments");

  let ret = resize.invoke(cstr!("Point"), &args);
  if let Some(e) = ret.get_error() {
    return Err(e.to_owned());
  }

  let node = ret
    .get_video_node(key!(c"clip"), 0)
    .map_err(|_| cstr!("oxidctf.DCTFilter: failed to get resize.Point output.").to_owned())?;
  let vi = node.info().clone();
  Ok((node, vi))
}

/// Crop `pad_width`/`pad_height` padding pixels off the right/bottom.
fn crop_node(
  core: &CoreRef<'_>,
  node: VideoNode,
  pad_width: i32,
  pad_height: i32,
) -> Result<VideoNode, CString> {
  let Some(std_plugin) = core.get_plugin_by_id(cstr!("com.vapoursynth.std")) else {
    return Err(cstr!("oxidctf.DCTFilter: the std plugin is missing.").to_owned());
  };

  let mut args = core.create_map();
  args
    .set(key!(c"clip"), Value::VideoNode(node), AppendMode::Replace)
    .and_then(|()| {
      args.set(
        key!(c"right"),
        Value::Int(i64::from(pad_width)),
        AppendMode::Replace,
      )
    })
    .and_then(|()| {
      args.set(
        key!(c"bottom"),
        Value::Int(i64::from(pad_height)),
        AppendMode::Replace,
      )
    })
    .expect("should set std.Crop arguments");

  let ret = std_plugin.invoke(cstr!("Crop"), &args);
  if let Some(e) = ret.get_error() {
    return Err(e.to_owned());
  }

  ret
    .get_video_node(key!(c"clip"), 0)
    .map_err(|_| cstr!("oxidctf.DCTFilter: failed to get std.Crop output.").to_owned())
}

impl Filter for DctFilter {
  type Error = CString;
  type FrameType = VideoFrame;
  type FilterData = ();

  fn create(
    input: MapRef<'_>,
    mut output: MapRef<'_>,
    _data: Option<Box<Self::FilterData>>,
    mut core: CoreRef<'_>,
  ) -> Result<(), Self::Error> {
    let Ok(node) = input.get_video_node(key!(c"clip"), 0) else {
      return Err(cstr!("oxidctf.DCTFilter: failed to get clip.").to_owned());
    };

    let vi = node.info().clone();

    if !is_constant_video_format(&vi)
      || (vi.format.sample_type == SampleType::Integer && vi.format.bits_per_sample > 16)
      || (vi.format.sample_type == SampleType::Float && vi.format.bits_per_sample != 32)
    {
      return Err(
        cstr!("oxidctf.DCTFilter: only constant format 8-16 bit integer and 32 bit float input supported.")
          .to_owned(),
      );
    }

    let nsize = match input.get_int(key!(c"nsize"), 0) {
      Ok(n) => {
        if !(1..=64).contains(&n) {
          return Err(
            cstr!("oxidctf.DCTFilter: `nsize` must be between 1 and 64 (inclusive).").to_owned(),
          );
        }
        n as usize
      }
      Err(_) => 8,
    };

    let factors = input
      .get_float_array(key!(c"factors"))
      .map_err(|_| cstr!("oxidctf.DCTFilter: failed to get factors.").to_owned())?;

    if factors.len() != nsize && factors.len() != nsize * nsize {
      return Err(
        CString::new(format!(
          "oxidctf.DCTFilter: number of elements in factors must be {nsize} (if specifying row/column factors) or {} (if specifying a full coefficient matrix).",
          nsize * nsize
        ))
        .expect("should create CString from String"),
      );
    }

    if factors.iter().any(|&f| !(0.0..=1.0).contains(&f)) {
      return Err(
        cstr!("oxidctf.DCTFilter: factor must be between 0.0 and 1.0 (inclusive)").to_owned(),
      );
    }

    let num_planes = vi.format.num_planes as usize;
    let num_plane_args = input.num_elements(key!(c"planes")).unwrap_or(0);
    let mut process_planes = [num_plane_args <= 0; 3];

    for i in 0..num_plane_args {
      let plane = input
        .get_int_saturated(key!(c"planes"), i)
        .expect("should get plane index");

      if plane < 0 || plane as usize >= num_planes {
        return Err(cstr!("oxidctf.DCTFilter: plane index out of range").to_owned());
      }

      if process_planes[plane as usize] {
        return Err(cstr!("oxidctf.DCTFilter: plane specified twice").to_owned());
      }

      process_planes[plane as usize] = true;
    }

    // Orthonormal DCT-II matrix.
    let size = nsize as f64;
    let dct = Array2::from_shape_fn((nsize, nsize), |(k, j)| {
      let a = if k == 0 {
        (1.0 / size).sqrt()
      } else {
        (2.0 / size).sqrt()
      };
      a * (std::f64::consts::PI * ((2 * j + 1) * k) as f64 / (2.0 * size)).cos()
    });

    // `factors` is either per-row/column factors or a full per-coefficient
    // matrix.
    let factors_2d = if factors.len() == nsize {
      Array2::from_shape_fn((nsize, nsize), |(y, x)| factors[y] * factors[x])
    } else {
      Array2::from_shape_fn((nsize, nsize), |(y, x)| factors[nsize * y + x])
    };

    // Fold forward DCT, coefficient scaling, and inverse DCT into a single
    // operator on row-major flattened blocks.
    //
    // A = D^T * diag(f) * D, where D = C (x) C.
    //
    // A is symmetric because diag(f) is.
    let nn = nsize * nsize;
    let d = Array2::from_shape_fn((nn, nn), |(r, c)| {
      let (u, v) = (r / nsize, r % nsize);
      let (i, j) = (c / nsize, c % nsize);
      dct[[u, i]] * dct[[v, j]]
    });
    let f_col = Array2::from_shape_fn((nn, 1), |(r, _)| factors_2d[[r / nsize, r % nsize]]);
    let fd = &d * &f_col;
    let a = d.t().dot(&fd).mapv(|v| v as f32);

    let peak = vi.peak_value(None, Some(ColorRange::Full));

    // Pad the clip so every plane's dimensions are multiples of the block size.
    let mod_w = (nsize << vi.format.sub_sampling_w) as i32;
    let mod_h = (nsize << vi.format.sub_sampling_h) as i32;
    let pad_width = (mod_w - vi.width % mod_w) % mod_w;
    let pad_height = (mod_h - vi.height % mod_h) % mod_h;

    let (node, vi) = if pad_width > 0 || pad_height > 0 {
      pad_node(&core, node, pad_width, pad_height)?
    } else {
      (node, vi)
    };

    let filter = Self {
      node,
      nsize,
      a,
      process_planes,
      peak,
    };

    let deps = [FilterDependency {
      source: filter.node.as_ptr(),
      request_pattern: RequestPattern::StrictSpatial,
    }];

    core.create_video_filter(
      output,
      cstr!("DCTFilter"),
      &vi,
      Box::new(filter),
      Dependencies::new(&deps).unwrap(),
    );

    // Crop the padding off to restore the original dimensions.
    if pad_width > 0 || pad_height > 0 {
      let node = output
        .get_video_node(key!(c"clip"), 0)
        .map_err(|_| cstr!("oxidctf.DCTFilter: failed to get filter output.").to_owned())?;
      output.clear();
      let node = crop_node(&core, node, pad_width, pad_height)?;
      output
        .set(key!(c"clip"), Value::VideoNode(node), AppendMode::Replace)
        .expect("should set output clip");
    }

    Ok(())
  }

  fn get_frame(
    &self,
    n: i32,
    activation_reason: ActivationReason,
    _frame_data: *mut *mut c_void,
    mut ctx: FrameContext,
    core: CoreRef<'_>,
  ) -> Result<Option<VideoFrame>, Self::Error> {
    match activation_reason {
      ActivationReason::Initial => {
        ctx.request_frame_filter(n, &self.node);
      }
      ActivationReason::AllFramesReady => {
        let src = self.node.get_frame_filter(n, &mut ctx);
        let info = self.node.info();
        let num_planes = info.format.num_planes;

        // Copy the unprocessed planes from the source frame.
        let plane_src: Vec<*const VSFrame> = (0..3)
          .map(|i| {
            if self.process_planes[i] {
              std::ptr::null()
            } else {
              src.as_ptr()
            }
          })
          .collect();
        let planes = [0, 1, 2];

        let mut dst = core.new_video_frame2(
          &info.format,
          info.width,
          info.height,
          &plane_src,
          &planes,
          Some(&src),
        );

        for plane in (0..num_planes).filter(|&plane| self.process_planes[plane as usize]) {
          match (info.format.sample_type, info.format.bytes_per_sample) {
            (SampleType::Integer, 1) => self.filter_plane::<u8>(&src, &mut dst, plane),
            (SampleType::Integer, 2) => self.filter_plane::<u16>(&src, &mut dst, plane),
            _ => self.filter_plane::<f32>(&src, &mut dst, plane),
          }
        }

        return Ok(Some(dst));
      }
      ActivationReason::Error => {}
    }

    Ok(None)
  }

  const NAME: &'static CStr = cstr!("DCTFilter");
  const ARGS: &'static CStr = cstr!("clip:vnode;factors:float[];nsize:int:opt;planes:int[]:opt;");
  const RETURN_TYPE: &'static CStr = cstr!("clip:vnode;");
}

declare_plugin!(
  c"sgt.oxidctf",
  c"oxidctf",
  c"DCT/IDCT Frequency Suppressor",
  (0, 1),
  VAPOURSYNTH_API_VERSION,
  0,
  (DctFilter, None)
);
