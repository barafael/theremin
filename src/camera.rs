//! V4L2 MJPEG capture on a background thread.

use crate::config::CameraConfig;
use crate::types::Frame;
use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, Sender, TrySendError, bounded};
use std::thread;
use v4l::buffer::Type;
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::Capture;
use v4l::{Format, FourCC};
use web_time::Instant;
use zune_jpeg::JpegDecoder;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;

pub struct Camera {
    pub frames: Receiver<Frame>,
    pub width: usize,
    pub height: usize,
}

impl Camera {
    pub fn open(cfg: &CameraConfig) -> Result<Self> {
        let dev =
            Device::with_path(&cfg.device).with_context(|| format!("opening {}", cfg.device))?;
        let fmt = Format::new(cfg.width, cfg.height, FourCC::new(b"MJPG"));
        let fmt = dev.set_format(&fmt).context("setting camera format")?;
        if fmt.fourcc != FourCC::new(b"MJPG") {
            anyhow::bail!("camera does not support MJPG, got {}", fmt.fourcc);
        }
        let params = dev.set_params(&v4l::video::capture::Parameters::with_fps(cfg.fps))?;
        apply_controls(&dev, cfg);
        log::info!(
            "camera {} {}x{} @ {}/{} fps",
            cfg.device,
            fmt.width,
            fmt.height,
            params.interval.denominator,
            params.interval.numerator
        );
        let (tx, rx) = bounded::<Frame>(1);
        let width = fmt.width as usize;
        let height = fmt.height as usize;
        let mirror = cfg.mirror;
        thread::Builder::new()
            .name("camera".into())
            .spawn(move || {
                if let Err(e) = capture_loop(dev, width, height, mirror, tx) {
                    log::error!("camera thread stopped: {e:#}");
                }
            })
            .context("spawning camera thread")?;
        Ok(Self {
            frames: rx,
            width,
            height,
        })
    }
}

/// Set UVC controls by name; unknown controls are logged and skipped.
fn apply_controls(dev: &Device, cfg: &CameraConfig) {
    use v4l::control::{Control, Value};
    let descs = match dev.query_controls() {
        Ok(d) => d,
        Err(e) => {
            log::warn!("cannot query camera controls: {e}");
            return;
        }
    };
    // v4l2-ctl style names: lowercase, non-alphanumerics collapsed to '_'.
    let normalize = |n: &str| -> String {
        let mut out = String::new();
        for c in n.chars() {
            if c.is_ascii_alphanumeric() {
                out.push(c.to_ascii_lowercase());
            } else if !out.ends_with('_') && !out.is_empty() {
                out.push('_');
            }
        }
        out.trim_end_matches('_').to_string()
    };
    let id_of = |name: &str| {
        descs
            .iter()
            .find(|d| normalize(&d.name) == normalize(name))
            .map(|d| d.id)
    };
    let mut wanted: Vec<(String, i64)> = Vec::new();
    wanted.push((
        "exposure_dynamic_framerate".into(),
        cfg.dynamic_framerate as i64,
    ));
    if let Some(exp) = cfg.exposure {
        wanted.push(("auto_exposure".into(), 1));
        wanted.push(("exposure_time_absolute".into(), exp as i64));
    }
    for (k, v) in &cfg.controls {
        wanted.push((k.clone(), *v));
    }
    for (name, value) in wanted {
        let Some(id) = id_of(&name) else {
            log::warn!("camera has no control named {name}");
            continue;
        };
        let typ = descs.iter().find(|d| d.id == id).map(|d| d.typ);
        let val = match typ {
            Some(v4l::control::Type::Boolean) => Value::Boolean(value != 0),
            _ => Value::Integer(value),
        };
        match dev.set_control(Control { id, value: val }) {
            Ok(()) => log::info!("camera control {name} = {value}"),
            Err(e) => log::warn!("setting camera control {name} = {value} failed: {e}"),
        }
    }
}

fn capture_loop(
    dev: Device,
    width: usize,
    height: usize,
    mirror: bool,
    tx: Sender<Frame>,
) -> Result<()> {
    let mut stream =
        MmapStream::with_buffers(&dev, Type::VideoCapture, 4).context("creating mmap stream")?;
    let mut seq = 0u64;
    loop {
        let (buf, _meta) = stream.next().context("capturing frame")?;
        let captured_at = Instant::now();
        let opts = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
        let mut dec = JpegDecoder::new_with_options(buf, opts);
        let rgb = match dec.decode() {
            Ok(px) => px,
            Err(e) => {
                log::warn!("jpeg decode failed: {e}");
                continue;
            }
        };
        let (w, h) = dec.dimensions().unwrap_or((width, height));
        if w != width || h != height || rgb.len() != w * h * 3 {
            log::warn!("unexpected frame size {w}x{h}");
            continue;
        }
        let rgb = if mirror { mirror_rgb(&rgb, w, h) } else { rgb };
        seq += 1;
        let frame = Frame {
            width: w,
            height: h,
            rgb,
            captured_at,
            sequence: seq,
        };
        match tx.try_send(frame) {
            Ok(()) => {}
            // Consumer is slower than the camera: drop the stale frame, keep the fresh one.
            Err(TrySendError::Full(frame)) => {
                let _ = tx.try_send(frame);
            }
            Err(TrySendError::Disconnected(_)) => return Ok(()),
        }
    }
}

fn mirror_rgb(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; src.len()];
    for y in 0..h {
        let row = &src[y * w * 3..(y + 1) * w * 3];
        let orow = &mut out[y * w * 3..(y + 1) * w * 3];
        for x in 0..w {
            let s = x * 3;
            let d = (w - 1 - x) * 3;
            orow[d..d + 3].copy_from_slice(&row[s..s + 3]);
        }
    }
    out
}
