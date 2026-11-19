//! The playing app: the session (settings changed from the keyboard), what the window shows for
//! a frame, and on the desktop the tracking thread. The web version drives the same pieces from
//! the browser's event loop (`crate::web`).

use crate::config::Config;
use crate::music::{self, midi_to_hz};
use crate::synth::SynthParams;
use crate::theremin::{Reading, Theremin};
use crate::types::{Frame, Hand};
use crate::ui::{NoteMark, ThereminWindow};
use crate::wireframe;
use slint::{Color, Image, ModelRc, Rgb8Pixel, SharedPixelBuffer, VecModel};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use web_time::Instant;

#[cfg(not(target_arch = "wasm32"))]
pub use desktop::play;

#[cfg(not(target_arch = "wasm32"))]
mod desktop {
    use super::*;
    use crate::synth::Audio;
    use crate::{camera, detect};
    use anyhow::{Context, Result};
    use crossbeam_channel::{Receiver, RecvTimeoutError};
    use slint::{ComponentHandle, SharedString};
    use std::sync::atomic::AtomicBool;
    use web_time::Duration;

    pub fn play(cfg: Config) -> Result<()> {
        let params = Arc::new(SynthParams::new(&cfg.sound));
        let audio = Audio::start(&cfg.sound, params.clone()).context("starting audio output")?;
        let ui = ThereminWindow::new()?;
        ui.set_show_video(cfg.ui.show_video);
        let (tx, rx) = crossbeam_channel::unbounded::<String>();
        ui.on_command(move |c| {
            let _ = tx.send(c.to_string());
        });

        let stop = Arc::new(AtomicBool::new(false));
        let weak = ui.as_weak();
        let (stop2, params2) = (stop.clone(), params.clone());
        let audio_info = format!("{} Hz", audio.sample_rate);
        let worker = std::thread::Builder::new()
            .name("tracking".into())
            .spawn(move || {
                if let Err(e) = worker(cfg, params2, rx, weak.clone(), stop2, audio_info) {
                    log::error!("{e:#}");
                    let msg = format!("error: {e:#}");
                    let _ = weak.upgrade_in_event_loop(move |ui| {
                        ui.set_hint(SharedString::from(msg));
                        ui.set_info("stopped".into());
                    });
                }
            })?;
        ui.run()?;
        stop.store(true, Ordering::SeqCst);
        params.amp.set(0.0);
        let _ = worker.join();
        // Let the release finish instead of cutting the tone off with a click.
        std::thread::sleep(Duration::from_millis(150));
        drop(audio);
        Ok(())
    }

    fn worker(
        cfg: Config,
        params: Arc<SynthParams>,
        commands: Receiver<String>,
        ui: slint::Weak<ThereminWindow>,
        stop: Arc<AtomicBool>,
        audio_info: String,
    ) -> Result<()> {
        let mut tracker = detect::Tracker::new(&cfg.models_dir(), &cfg.detect, cfg.camera.mirror)?;
        let camera = camera::Camera::open(&cfg.camera)?;
        let mut session = Session::new(&cfg, params.clone());
        let mut fps = Fps::new();
        while !stop.load(Ordering::SeqCst) {
            for c in commands.try_iter() {
                session.command(&c);
            }
            let frame = match camera.frames.recv_timeout(Duration::from_millis(100)) {
                Ok(f) => f,
                Err(RecvTimeoutError::Timeout) => {
                    params.amp.set(0.0);
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => anyhow::bail!("camera stopped"),
            };
            let hands = tracker.process_blocking(&frame)?;
            let reading = session.theremin.update(&hands, Instant::now());
            params.freq.set(midi_to_hz(reading.midi));
            params.amp.set(reading.amp);
            params.brightness.set(reading.brightness);

            let view = View::new(
                &mut session,
                &frame,
                hands,
                &reading,
                fps.tick(),
                tracker.last_inference_ms,
                &audio_info,
            );
            let _ = ui.upgrade_in_event_loop(move |ui| view.apply(&ui));
        }
        Ok(())
    }
}

/// Settings the player changes with the keyboard while playing.
pub struct Session {
    pub theremin: Theremin,
    pub params: Arc<SynthParams>,
    marks_dirty: bool,
}

const SNAP_STEPS: [f32; 3] = [0.0, 0.5, 1.0];
const REVERB_STEPS: [f32; 4] = [0.0, 0.25, 0.45, 0.7];

fn next_step(steps: &[f32], v: f32) -> f32 {
    steps
        .iter()
        .copied()
        .find(|&s| s > v + 1e-3)
        .unwrap_or(steps[0])
}

impl Session {
    pub fn new(cfg: &Config, params: Arc<SynthParams>) -> Self {
        Self {
            theremin: Theremin::new(cfg.play.clone()),
            params,
            marks_dirty: true,
        }
    }

    /// Apply a key command from the window.
    pub fn command(&mut self, c: &str) {
        let cfg = &mut self.theremin.cfg;
        match c {
            "s" | "S" => cfg.scale = cfg.scale.next(),
            "k" | "K" => self.theremin.root = (self.theremin.root + 1).rem_euclid(12),
            "j" | "J" => self.theremin.root = (self.theremin.root - 1).rem_euclid(12),
            "q" | "Q" => cfg.snap = next_step(&SNAP_STEPS, cfg.snap),
            "h" | "H" => cfg.hands = if cfg.hands >= 2 { 1 } else { 2 },
            "up" => cfg.low_note = (cfg.low_note + 12).min(96),
            "down" => cfg.low_note = (cfg.low_note - 12).max(24),
            "t" | "T" => self.params.set_timbre(self.params.timbre().next()),
            "r" | "R" => self
                .params
                .reverb
                .set(next_step(&REVERB_STEPS, self.params.reverb.get())),
            "m" | "M" => {
                let m = self.params.muted.load(Ordering::Relaxed);
                self.params.muted.store(!m, Ordering::Relaxed);
            }
            _ => return,
        }
        self.marks_dirty = true;
    }

    fn settings_line(&self) -> String {
        let t = &self.theremin;
        let snap = match t.cfg.snap {
            s if s <= 0.0 => "free pitch".to_string(),
            s if s >= 1.0 => "snap to scale".to_string(),
            s => format!("snap {:.0}%", s * 100.0),
        };
        format!(
            "{} {}  ·  {}  ·  {}  ·  {} hand{}  ·  reverb {:.0}%  ·  {}–{}",
            music::pitch_class_name(t.root),
            t.scale().name(),
            snap,
            self.params.timbre().name(),
            t.cfg.hands.clamp(1, 2),
            if t.cfg.hands >= 2 { "s" } else { "" },
            self.params.reverb.get() * 100.0,
            music::note_name(t.cfg.low_note),
            music::note_name(t.high_note()),
        )
    }

    fn marks(&self) -> Vec<NoteMark> {
        let t = &self.theremin;
        let scale = t.scale();
        let chromatic = scale.intervals().len() == 12;
        (t.cfg.low_note..=t.high_note())
            .map(|m| {
                let in_scale = scale.contains(t.root, m);
                let is_root = (m - t.root).rem_euclid(12) == 0;
                let kind = if is_root {
                    2
                } else if in_scale {
                    1
                } else {
                    0
                };
                // Label every scale note, except in chromatic mode where only naturals fit.
                let natural = music::pitch_class_name(m).len() == 1;
                let label = if is_root {
                    music::note_name(m)
                } else if in_scale && (!chromatic || natural) {
                    music::pitch_class_name(m).to_string()
                } else {
                    String::new()
                };
                NoteMark {
                    x: t.midi_to_x(m as f32),
                    label: label.into(),
                    kind,
                }
            })
            .collect()
    }
}

/// Everything the window shows for one frame, built on the tracking thread.
pub struct View {
    frame: SharedPixelBuffer<Rgb8Pixel>,
    aspect: f32,
    hands: Vec<Hand>,
    colors: Vec<Color>,
    marks: Option<Vec<NoteMark>>,
    zone: (f32, f32),
    volume_zone: [f32; 2],
    reading: Reading,
    pitch_x: f32,
    muted: bool,
    tone_enabled: bool,
    note: String,
    cents: String,
    freq: String,
    hint: String,
    settings: String,
    info: String,
}

impl View {
    /// Snapshot what to show; takes the note grid only when it changed since the last view.
    pub fn new(
        s: &mut Session,
        frame: &Frame,
        hands: Vec<Hand>,
        r: &Reading,
        fps: f32,
        infer_ms: f32,
        audio: &str,
    ) -> Self {
        let mut buf = SharedPixelBuffer::<Rgb8Pixel>::new(frame.width as u32, frame.height as u32);
        buf.make_mut_bytes().copy_from_slice(&frame.rgb);
        let pitch_color = Color::from_rgb_u8(90, 200, 255);
        let volume_color = Color::from_rgb_u8(255, 190, 60);
        let colors = (0..hands.len())
            .map(|i| {
                if Some(i) == r.pitch_hand {
                    pitch_color
                } else if Some(i) == r.volume_hand {
                    volume_color
                } else {
                    Color::from_rgb_u8(150, 150, 160)
                }
            })
            .collect();
        let nearest = r.midi.round() as i32;
        let cents = ((r.midi - nearest as f32) * 100.0).round() as i32;
        let two = s.theremin.cfg.hands >= 2;
        let hint = match (r.pitch_hand.is_some(), r.volume_hand.is_some()) {
            (false, false) if two => {
                "Raise both hands. Right hand: left/right for pitch, up/down for tone. Left hand: raise for volume."
            }
            (false, false) => "Raise one hand: left/right for pitch, up/down for volume.",
            (true, false) => "Now raise your other hand to set the volume.",
            (false, true) => "Now raise your pitch hand.",
            _ => "",
        };
        let marks = std::mem::take(&mut s.marks_dirty).then(|| s.marks());
        Self {
            frame: buf,
            aspect: frame.width as f32 / frame.height as f32,
            hands,
            colors,
            marks,
            zone: s.theremin.pitch_zone_x(),
            volume_zone: s.theremin.cfg.volume_zone,
            pitch_x: s.theremin.midi_to_x(r.midi),
            tone_enabled: s.theremin.tone_hand_active(),
            muted: s.params.muted.load(Ordering::Relaxed),
            note: music::note_name(nearest),
            cents: format!("{cents:+} ¢"),
            freq: format!("{:.1} Hz", midi_to_hz(r.midi)),
            hint: hint.into(),
            settings: s.settings_line(),
            info: format!("{fps:.0} fps · {infer_ms:.1} ms · {audio}"),
            reading: r.clone(),
        }
    }

    pub fn apply(self, ui: &ThereminWindow) {
        let (paths, points) = wireframe::build(&self.hands, &self.colors, self.aspect, 4.0);
        let r = &self.reading;
        ui.set_frame(Image::from_rgb8(self.frame));
        ui.set_frame_aspect(self.aspect);
        ui.set_paths(paths);
        ui.set_points(points);
        if let Some(marks) = self.marks {
            ui.set_marks(ModelRc::new(VecModel::from(marks)));
        }
        ui.set_zone_left(self.zone.0);
        ui.set_zone_right(self.zone.1);
        ui.set_vol_top(self.volume_zone[0]);
        ui.set_vol_bottom(self.volume_zone[1]);
        ui.set_pitch_active(r.pitch_pos.is_some());
        ui.set_pitch_x(self.pitch_x);
        if let Some(p) = r.pitch_pos {
            ui.set_hand_x(p.x);
            ui.set_hand_y(p.y);
        }
        ui.set_volume_active(r.volume_pos.is_some());
        if let Some(p) = r.volume_pos {
            ui.set_volume_hand_y(p.y);
        }
        ui.set_volume(r.volume_level);
        ui.set_amp(if self.muted { 0.0 } else { r.amp });
        ui.set_fist(r.fist);
        ui.set_tone_enabled(self.tone_enabled);
        ui.set_tone_active(r.tone_level.is_some());
        ui.set_tone(r.tone);
        ui.set_muted(self.muted);
        ui.set_note(self.note.into());
        ui.set_cents(self.cents.into());
        ui.set_freq(self.freq.into());
        ui.set_hint(self.hint.into());
        ui.set_settings(self.settings.into());
        ui.set_info(self.info.into());
    }
}

pub struct Fps {
    last: Instant,
    frames: u32,
    total: u64,
    value: f32,
}

impl Default for Fps {
    fn default() -> Self {
        Self::new()
    }
}

impl Fps {
    pub fn new() -> Self {
        Self {
            last: Instant::now(),
            frames: 0,
            total: 0,
            value: 0.0,
        }
    }
    /// Frames ticked so far.
    pub fn frames_total(&self) -> u64 {
        self.total
    }

    pub fn tick(&mut self) -> f32 {
        self.frames += 1;
        self.total += 1;
        let dt = self.last.elapsed().as_secs_f32();
        if dt >= 0.5 {
            self.value = self.frames as f32 / dt;
            self.frames = 0;
            self.last = Instant::now();
        }
        self.value
    }
}
