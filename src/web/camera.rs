//! Camera frames in a browser: `getUserMedia` into a hidden `<video>`, drawn (mirrored) onto a
//! canvas and read back as RGB. With `?image=<url>` a still picture stands in for the camera,
//! which is handy to try the page without one.

use super::js_err;
use crate::config::CameraConfig;
use crate::types::Frame;
use anyhow::{Context, Result, anyhow};
use js_sys::{Object, Reflect};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    CanvasRenderingContext2d, HtmlCanvasElement, HtmlImageElement, HtmlVideoElement, MediaStream,
    MediaStreamConstraints,
};
use web_time::Instant;

enum Source {
    Camera {
        video: HtmlVideoElement,
        last_time: f64,
    },
    Still(HtmlImageElement),
}

pub struct Grabber {
    source: Source,
    canvas: HtmlCanvasElement,
    ctx: CanvasRenderingContext2d,
    mirror: bool,
    sequence: u64,
}

impl Grabber {
    /// The camera, or the still picture named by `?image=`.
    pub async fn open(cfg: &CameraConfig, still: Option<String>) -> Result<Self> {
        let document = web_sys::window()
            .and_then(|w| w.document())
            .context("no document")?;
        let canvas: HtmlCanvasElement = document
            .create_element("canvas")
            .map_err(js_err)?
            .unchecked_into();
        let options = Object::new();
        Reflect::set(&options, &"willReadFrequently".into(), &JsValue::TRUE).map_err(js_err)?;
        let ctx: CanvasRenderingContext2d = canvas
            .get_context_with_context_options("2d", &options)
            .map_err(js_err)?
            .context("no 2d canvas context")?
            .unchecked_into();

        let (source, mirror) = match still {
            Some(url) => {
                let img = HtmlImageElement::new().map_err(js_err)?;
                img.set_cross_origin(Some("anonymous"));
                img.set_src(&url);
                JsFuture::from(img.decode())
                    .await
                    .map_err(js_err)
                    .with_context(|| format!("loading {url}"))?;
                // A photo is not a mirror image.
                (Source::Still(img), false)
            }
            None => (
                Source::Camera {
                    video: open_camera(cfg).await?,
                    last_time: -1.0,
                },
                cfg.mirror,
            ),
        };
        Ok(Self {
            source,
            canvas,
            ctx,
            mirror,
            sequence: 0,
        })
    }

    /// Whether the picture is mirrored (needed to tell left and right hands apart).
    pub fn mirrored(&self) -> bool {
        self.mirror
    }

    /// The newest frame, or None if the camera has not produced a new one since the last call.
    pub fn grab(&mut self) -> Result<Option<Frame>> {
        let (w, h) = match &mut self.source {
            Source::Camera { video, last_time } => {
                let t = video.current_time();
                if video.ready_state() < 2 || video.video_width() == 0 || t == *last_time {
                    return Ok(None);
                }
                *last_time = t;
                (video.video_width(), video.video_height())
            }
            Source::Still(img) => (img.natural_width(), img.natural_height()),
        };
        if self.canvas.width() != w || self.canvas.height() != h {
            self.canvas.set_width(w);
            self.canvas.set_height(h);
        }
        let (fw, fh) = (w as f64, h as f64);
        if self.mirror {
            self.ctx
                .set_transform(-1.0, 0.0, 0.0, 1.0, fw, 0.0)
                .map_err(js_err)?;
        } else {
            self.ctx
                .set_transform(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
                .map_err(js_err)?;
        }
        match &self.source {
            Source::Camera { video, .. } => {
                self.ctx.draw_image_with_html_video_element(video, 0.0, 0.0)
            }
            Source::Still(img) => self.ctx.draw_image_with_html_image_element(img, 0.0, 0.0),
        }
        .map_err(js_err)?;
        let rgba = self
            .ctx
            .get_image_data(0.0, 0.0, fw, fh)
            .map_err(js_err)?
            .data();
        let rgb: Vec<u8> = rgba
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|&[r, g, b, _]| [r, g, b])
            .collect();
        self.sequence += 1;
        Ok(Some(Frame {
            width: w as usize,
            height: h as usize,
            rgb,
            captured_at: Instant::now(),
            sequence: self.sequence,
        }))
    }
}

async fn open_camera(cfg: &CameraConfig) -> Result<HtmlVideoElement> {
    let window = web_sys::window().context("no window")?;
    let devices = window.navigator().media_devices().map_err(|_| {
        anyhow!("this browser gives no camera access here (camera needs https or localhost)")
    })?;
    let video_constraints = Object::new();
    let ideal = |v: f64| -> Result<JsValue> {
        let o = Object::new();
        Reflect::set(&o, &"ideal".into(), &v.into()).map_err(js_err)?;
        Ok(o.into())
    };
    Reflect::set(
        &video_constraints,
        &"width".into(),
        &ideal(cfg.width as f64)?,
    )
    .map_err(js_err)?;
    Reflect::set(
        &video_constraints,
        &"height".into(),
        &ideal(cfg.height as f64)?,
    )
    .map_err(js_err)?;
    Reflect::set(
        &video_constraints,
        &"frameRate".into(),
        &ideal(cfg.fps as f64)?,
    )
    .map_err(js_err)?;
    Reflect::set(&video_constraints, &"facingMode".into(), &"user".into()).map_err(js_err)?;
    let constraints = MediaStreamConstraints::new();
    constraints.set_audio(&JsValue::FALSE);
    constraints.set_video(&video_constraints);
    let promise = devices
        .get_user_media_with_constraints(&constraints)
        .map_err(js_err)?;
    let stream: MediaStream = JsFuture::from(promise)
        .await
        .map_err(js_err)
        .context("camera access was refused or no camera is available")?
        .unchecked_into();

    let document = window.document().context("no document")?;
    let video: HtmlVideoElement = document
        .create_element("video")
        .map_err(js_err)?
        .unchecked_into();
    video.set_muted(true);
    video.set_attribute("playsinline", "").map_err(js_err)?;
    // In the page but invisible: some browsers stop decoding video that isn't.
    video
        .set_attribute(
            "style",
            "position:fixed;left:0;top:0;width:2px;height:2px;opacity:0;pointer-events:none",
        )
        .map_err(js_err)?;
    document
        .body()
        .context("no body")?
        .append_child(&video)
        .map_err(js_err)?;
    video.set_src_object(Some(&stream));
    JsFuture::from(video.play().map_err(js_err)?)
        .await
        .map_err(js_err)
        .context("starting the camera video")?;
    log::info!("camera {}x{}", video.video_width(), video.video_height());
    Ok(video)
}
