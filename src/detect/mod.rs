//! Hand tracking: palm detection + landmark regression with MediaPipe-style ROI tracking.

mod image_ops;
mod landmark;
mod model;
mod palm;

use crate::config::DetectConfig;
use crate::types::{Frame, Hand, Handedness, Vec3, lm};
use anyhow::Result;
use image_ops::Roi;
use web_time::Instant;

pub struct Tracker {
    palm: palm::PalmDetector,
    landmark: landmark::LandmarkModel,
    cfg: DetectConfig,
    tracks: Vec<Track>,
    frame_count: u32,
    next_id: u32,
    pub last_inference_ms: f32,
    mirror: bool,
}

struct Track {
    id: u32,
    roi: Roi,
    hand: Hand,
}

/// Where onnxruntime-web comes from in the browser: the official npm build, matching the ONNX
/// Runtime API version that `ort-web` speaks (its `+1.30`).
#[cfg(target_arch = "wasm32")]
const ORT_WEB_DIST: &str = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

impl Tracker {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(models_dir: &std::path::Path, cfg: &DetectConfig, mirror: bool) -> Result<Self> {
        use anyhow::Context;
        let env = ort::init()
            .with_name("theremin")
            .build()
            .context("initializing onnxruntime")?;
        let palm = model::Model::load_file(&env, &models_dir.join(palm::FILE), cfg.threads)?;
        let landmark =
            model::Model::load_file(&env, &models_dir.join(landmark::FILE), cfg.threads)?;
        Ok(Self::with_models(palm, landmark, cfg, mirror))
    }

    /// Load onnxruntime-web and the models from `models_url` (a directory URL ending in `/`).
    #[cfg(target_arch = "wasm32")]
    pub async fn new_web(models_url: &str, cfg: &DetectConfig, mirror: bool) -> Result<Self> {
        use anyhow::Context;
        let dist = ort_web::Dist::new(ORT_WEB_DIST).with_binary_name("ort-wasm-simd-threaded.wasm");
        let api = ort_web::api(dist)
            .await
            .context("loading onnxruntime-web")?;
        ort::set_api(api);
        // No usage beacon to pyke.io from visitors' browsers.
        let env = ort::init()
            .with_name("theremin")
            .with_telemetry(false)
            .build()
            .context("initializing onnxruntime")?;
        log::info!("onnxruntime-web loaded");
        let palm = model::Model::load_url(&env, &format!("{models_url}{}", palm::FILE)).await?;
        log::info!("palm model ready");
        let landmark =
            model::Model::load_url(&env, &format!("{models_url}{}", landmark::FILE)).await?;
        log::info!("landmark model ready");
        Ok(Self::with_models(palm, landmark, cfg, mirror))
    }

    fn with_models(
        palm: model::Model,
        landmark: model::Model,
        cfg: &DetectConfig,
        mirror: bool,
    ) -> Self {
        Self {
            palm: palm::PalmDetector::new(palm),
            landmark: landmark::LandmarkModel::new(landmark),
            cfg: cfg.clone(),
            tracks: Vec::new(),
            frame_count: 0,
            next_id: 1,
            last_inference_ms: 0.0,
            mirror,
        }
    }

    /// [`process`](Self::process) for the desktop, where inference completes without waiting.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn process_blocking(&mut self, frame: &Frame) -> Result<Vec<Hand>> {
        let mut fut = std::pin::pin!(self.process(frame));
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        loop {
            if let std::task::Poll::Ready(r) = fut.as_mut().poll(&mut cx) {
                return r;
            }
            // The native runtime never suspends; this is only reached if that changes.
            std::thread::yield_now();
        }
    }

    pub async fn process(&mut self, frame: &Frame) -> Result<Vec<Hand>> {
        let t0 = Instant::now();
        self.frame_count += 1;
        let need_detect = self.tracks.len() < self.cfg.max_hands
            && (self.tracks.is_empty()
                || self.frame_count.is_multiple_of(self.cfg.redetect_interval));

        if need_detect {
            let t = Instant::now();
            let dets = self
                .palm
                .detect(frame, self.cfg.palm_score_threshold)
                .await?;
            log::trace!(
                "palm detection {:.1} ms, {} candidates",
                t.elapsed().as_secs_f32() * 1000.0,
                dets.len()
            );
            for d in dets {
                if self.tracks.len() >= self.cfg.max_hands {
                    break;
                }
                let roi = d.roi();
                let overlaps = self
                    .tracks
                    .iter()
                    .any(|t| t.roi.center.dist2d(roi.center) < 0.5 * (t.roi.side + roi.side) * 0.5);
                if overlaps {
                    continue;
                }
                let id = self.next_id;
                self.next_id += 1;
                self.tracks.push(Track {
                    id,
                    roi,
                    hand: Hand {
                        landmarks: [Vec3::default(); 21],
                        handedness: Handedness::Right,
                        handedness_score: 0.0,
                        presence: 0.0,
                        track_id: id,
                        aspect: frame.width as f32 / frame.height as f32,
                    },
                });
            }
        }

        let mut i = 0;
        while i < self.tracks.len() {
            let track = &mut self.tracks[i];
            let result = self.landmark.run(frame, &track.roi).await?;
            if result.presence < self.cfg.presence_threshold {
                log::debug!("track {} lost (presence {:.2})", track.id, result.presence);
                self.tracks.remove(i);
                continue;
            }
            // Handedness from the model assumes a mirrored (selfie) image.
            let mut handedness = if result.right {
                Handedness::Right
            } else {
                Handedness::Left
            };
            if !self.mirror {
                handedness = match handedness {
                    Handedness::Left => Handedness::Right,
                    Handedness::Right => Handedness::Left,
                };
            }
            track.hand = Hand {
                landmarks: result.landmarks_px.map(|p| {
                    Vec3::new(
                        p.x / frame.width as f32,
                        p.y / frame.height as f32,
                        p.z / frame.width as f32,
                    )
                }),
                handedness,
                handedness_score: 1.0,
                presence: result.presence,
                track_id: track.id,
                aspect: frame.width as f32 / frame.height as f32,
            };
            track.roi = Roi::from_landmarks(
                &result.landmarks_px,
                frame.width as f32,
                frame.height as f32,
            );
            i += 1;
        }
        // Drop duplicate tracks that converged onto the same hand.
        let mut j = 1;
        while j < self.tracks.len() {
            let a = self.tracks[j].hand.landmarks[lm::WRIST];
            let dup = self.tracks[..j]
                .iter()
                .any(|t| t.hand.landmarks[lm::WRIST].dist2d(a) < 0.05);
            if dup {
                self.tracks.remove(j);
            } else {
                j += 1;
            }
        }
        self.last_inference_ms = t0.elapsed().as_secs_f32() * 1000.0;
        Ok(self.tracks.iter().map(|t| t.hand.clone()).collect())
    }
}
