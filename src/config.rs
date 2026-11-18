//! User-tunable settings. On the desktop they load from `~/.config/theremin/config.toml` if
//! present; on the web the defaults apply, and the keyboard changes them while playing.

use crate::music::Scale;
use crate::synth::Timbre;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub camera: CameraConfig,
    pub detect: DetectConfig,
    pub play: PlayConfig,
    pub sound: SoundConfig,
    pub ui: UiConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct CameraConfig {
    pub device: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    /// Mirror the frame horizontally so the picture behaves like a mirror.
    pub mirror: bool,
    /// Let the camera lower its frame rate in low light (UVC `exposure_dynamic_framerate`).
    /// Off keeps the frame rate (and so the control rate of the instrument) steady.
    pub dynamic_framerate: bool,
    /// Manual exposure time (UVC `exposure_time_absolute` units, often 100 µs). None = auto.
    pub exposure: Option<i32>,
    /// Extra V4L2 controls to set by name, e.g. { brightness = 10, gain = 50 }.
    pub controls: std::collections::BTreeMap<String, i64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct DetectConfig {
    pub models_dir: Option<PathBuf>,
    pub palm_score_threshold: f32,
    pub presence_threshold: f32,
    pub max_hands: usize,
    /// Run the palm detector at least every N frames while fewer than `max_hands` are tracked.
    pub redetect_interval: u32,
    pub threads: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    Left,
    Right,
}

/// How hand positions become pitch and volume. Positions are fractions of the camera frame
/// (as seen in the mirrored picture: 0 = left edge, 0 = top edge).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct PlayConfig {
    /// 2: one hand plays pitch (left/right), the other volume (up/down), like a real theremin.
    /// 1: a single hand does both.
    pub hands: u8,
    /// Which hand plays pitch in two-hand mode. Pitch rises toward this side of the picture.
    pub pitch_hand: Side,
    /// Lowest note (MIDI number, 48 = C3) at the far end of the pitch zone.
    pub low_note: i32,
    /// Range of the pitch zone in octaves.
    pub octaves: f32,
    /// Horizontal band (left, right) of the frame that spans the pitch range.
    pub pitch_zone: [f32; 2],
    /// Vertical band (top = full volume, bottom = silent) for the volume hand.
    pub volume_zone: [f32; 2],
    /// Exponent applied to the volume position (higher = more resolution at low volume).
    pub volume_curve: f32,
    /// Root of the scale used for snapping and for the note grid, e.g. "C", "F#", "Bb".
    pub key: String,
    pub scale: Scale,
    /// Pull toward the nearest scale note: 0 = free (a real theremin), 1 = fully quantized.
    pub snap: f32,
    /// Closing the volume hand into a fist cuts the sound.
    pub fist_mute: bool,
    /// Mean finger straightness (0..1) below which the hand counts as a fist, and above which
    /// it counts as open again (hysteresis).
    pub fist_enter: f32,
    pub fist_exit: f32,
    /// One Euro filter on hand positions: lower min_cutoff = steadier notes at rest,
    /// higher beta = less lag on fast moves.
    pub min_cutoff: f32,
    pub beta: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SoundConfig {
    pub timbre: Timbre,
    /// Master output gain, 0..1.
    pub gain: f32,
    /// Portamento time constant between control updates.
    pub glide_ms: f32,
    pub attack_ms: f32,
    pub release_ms: f32,
    /// Added vibrato depth in semitones (0 = only what your hand does) and rate in Hz.
    pub vibrato_depth: f32,
    pub vibrato_rate: f32,
    /// Reverb wet level, 0..1.
    pub reverb: f32,
    /// Requested audio buffer size in frames (lower = less latency). Falls back to the device
    /// default if refused.
    pub buffer_frames: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct UiConfig {
    /// Show the camera picture behind the instrument (toggle with V).
    pub show_video: bool,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            device: "/dev/video0".into(),
            width: 640,
            height: 480,
            fps: 60,
            mirror: true,
            dynamic_framerate: false,
            exposure: None,
            controls: Default::default(),
        }
    }
}
impl Default for DetectConfig {
    fn default() -> Self {
        Self {
            models_dir: None,
            palm_score_threshold: 0.5,
            presence_threshold: 0.5,
            max_hands: 2,
            redetect_interval: 10,
            threads: 4,
        }
    }
}
impl Default for PlayConfig {
    fn default() -> Self {
        Self {
            hands: 2,
            pitch_hand: Side::Right,
            low_note: 48,
            octaves: 3.0,
            pitch_zone: [0.30, 0.95],
            volume_zone: [0.20, 0.85],
            volume_curve: 1.6,
            key: "C".into(),
            scale: Scale::Chromatic,
            snap: 0.0,
            fist_mute: true,
            fist_enter: 0.55,
            fist_exit: 0.68,
            min_cutoff: 1.5,
            beta: 3.0,
        }
    }
}
impl Default for SoundConfig {
    fn default() -> Self {
        Self {
            timbre: Timbre::Classic,
            gain: 0.5,
            glide_ms: 25.0,
            attack_ms: 15.0,
            release_ms: 90.0,
            vibrato_depth: 0.0,
            vibrato_rate: 5.5,
            reverb: 0.25,
            buffer_frames: Some(256),
        }
    }
}
impl Default for UiConfig {
    fn default() -> Self {
        Self { show_video: true }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn config_dir() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("theremin")
}

#[cfg(not(target_arch = "wasm32"))]
impl Config {
    pub fn load(path: Option<&PathBuf>) -> anyhow::Result<Self> {
        use anyhow::Context;
        let path = path
            .cloned()
            .unwrap_or_else(|| config_dir().join("config.toml"));
        if path.exists() {
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;
            toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
        } else {
            Ok(Self::default())
        }
    }

    pub fn models_dir(&self) -> PathBuf {
        if let Some(d) = &self.detect.models_dir {
            return d.clone();
        }
        if let Ok(d) = std::env::var("THEREMIN_MODELS") {
            return PathBuf::from(d);
        }
        let mut candidates = vec![config_dir().join("models")];
        if let Some(dir) = std::env::current_exe()
            .ok()
            .as_deref()
            .and_then(|e| e.parent())
        {
            candidates.push(dir.join("models"));
            candidates.push(dir.join("../../models"));
        }
        candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models"));
        candidates
            .into_iter()
            .find(|p| p.join("palm_detection_full_inf_post_192x192.onnx").exists())
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models"))
    }
}
