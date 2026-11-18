//! The theremin voice.
//!
//! The control loop (hand tracking, ~60 Hz) writes target pitch and volume into [`SynthParams`];
//! the sound glides toward them so the steps between camera frames are inaudible. The voice is a
//! sine run through an asymmetric soft clipper (a theremin's heterodyne oscillator is a slightly
//! distorted sine), a gentle low-pass, and a small stereo reverb.
//!
//! On the desktop that is [`Voice`], rendered sample by sample in a cpal callback. In a browser
//! the same chain is a Web Audio graph (`crate::web::audio`) built from the same [`Shape`]s.

use crate::config::SoundConfig;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Timbre {
    /// Nearly pure sine.
    Ether,
    /// Warm, slightly reedy: the sound of a vintage tube theremin.
    #[default]
    Classic,
    /// Strong even and odd harmonics, darker filter.
    Cello,
    /// Mostly odd harmonics, clarinet-like.
    Reed,
}

/// The waveshaper and filter settings behind a [`Timbre`].
pub struct Shape {
    /// Gain into the tanh clipper: higher = more harmonics.
    pub drive: f32,
    /// DC offset into the clipper: adds even harmonics.
    pub asym: f32,
    /// Low-pass cutoff in Hz.
    pub cutoff: f32,
}

impl Timbre {
    pub const ALL: [Timbre; 4] = [Timbre::Ether, Timbre::Classic, Timbre::Cello, Timbre::Reed];

    pub fn name(self) -> &'static str {
        match self {
            Timbre::Ether => "ether",
            Timbre::Classic => "classic",
            Timbre::Cello => "cello",
            Timbre::Reed => "reed",
        }
    }

    pub fn next(self) -> Self {
        Self::ALL[(self as usize + 1) % Self::ALL.len()]
    }

    pub fn from_index(i: u32) -> Self {
        Self::ALL.get(i as usize).copied().unwrap_or_default()
    }

    pub fn shape(self) -> Shape {
        match self {
            Timbre::Ether => Shape {
                drive: 0.5,
                asym: 0.0,
                cutoff: 9000.0,
            },
            Timbre::Classic => Shape {
                drive: 1.8,
                asym: 0.25,
                cutoff: 5000.0,
            },
            Timbre::Cello => Shape {
                drive: 3.5,
                asym: 0.4,
                cutoff: 2200.0,
            },
            Timbre::Reed => Shape {
                drive: 7.0,
                asym: 0.05,
                cutoff: 3200.0,
            },
        }
    }
}

/// An f32 shared without locks (bit-cast into an AtomicU32).
pub struct AtomicF32(AtomicU32);

impl AtomicF32 {
    pub fn new(v: f32) -> Self {
        Self(AtomicU32::new(v.to_bits()))
    }
    pub fn get(&self) -> f32 {
        f32::from_bits(self.0.load(Relaxed))
    }
    pub fn set(&self, v: f32) {
        self.0.store(v.to_bits(), Relaxed)
    }
}

/// Targets written by the control thread, read once per audio buffer.
pub struct SynthParams {
    /// Oscillator frequency in Hz.
    pub freq: AtomicF32,
    /// Amplitude 0..1 before master gain.
    pub amp: AtomicF32,
    pub reverb: AtomicF32,
    pub muted: AtomicBool,
    timbre: AtomicU32,
}

impl Shape {
    /// The clipper's transfer function, normalized so a full-scale sine peaks near ±1 with no DC
    /// at rest: `(tanh(d·(x + a)) − tanh(d·a)) / (tanh(d·(1 + a)) − tanh(d·a))`.
    pub fn transfer(&self, x: f32) -> f32 {
        let off = (self.drive * self.asym).tanh();
        let norm = ((self.drive * (1.0 + self.asym)).tanh() - off).max(1e-3);
        ((self.drive * (x + self.asym)).tanh() - off) / norm
    }
}

impl SynthParams {
    pub fn new(cfg: &SoundConfig) -> Self {
        Self {
            freq: AtomicF32::new(220.0),
            amp: AtomicF32::new(0.0),
            reverb: AtomicF32::new(cfg.reverb),
            muted: AtomicBool::new(false),
            timbre: AtomicU32::new(cfg.timbre as u32),
        }
    }
    pub fn timbre(&self) -> Timbre {
        Timbre::from_index(self.timbre.load(Relaxed))
    }
    pub fn set_timbre(&self, t: Timbre) {
        self.timbre.store(t as u32, Relaxed)
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use desktop::{Audio, Voice};

#[cfg(not(target_arch = "wasm32"))]
mod desktop {
    use super::*;
    use anyhow::{Context, Result, anyhow};
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use cpal::{BufferSize, FromSample, SampleFormat, SizedSample, StreamConfig};
    use std::f32::consts::TAU;
    use std::sync::Arc;

    /// One-pole smoothing coefficient for a time constant in seconds.
    fn coef(tau: f32, sr: f32) -> f32 {
        if tau <= 0.0 {
            1.0
        } else {
            1.0 - (-1.0 / (tau * sr)).exp()
        }
    }

    pub struct Voice {
        sr: f32,
        log_freq: f32,
        amp: f32,
        phase: f32,
        vib_phase: f32,
        drive: f32,
        asym: f32,
        cutoff: f32,
        wet: f32,
        lp1: f32,
        lp2: f32,
        dc_x: f32,
        dc_y: f32,
        k_glide: f32,
        k_attack: f32,
        k_release: f32,
        vib_depth: f32,
        vib_rate: f32,
        gain: f32,
        reverb: Reverb,
    }

    impl Voice {
        pub fn new(cfg: &SoundConfig, sr: f32) -> Self {
            let s = cfg.timbre.shape();
            Self {
                sr,
                log_freq: 220f32.log2(),
                amp: 0.0,
                phase: 0.0,
                vib_phase: 0.0,
                drive: s.drive,
                asym: s.asym,
                cutoff: s.cutoff,
                wet: cfg.reverb,
                lp1: 0.0,
                lp2: 0.0,
                dc_x: 0.0,
                dc_y: 0.0,
                k_glide: coef(cfg.glide_ms / 1000.0, sr),
                k_attack: coef(cfg.attack_ms / 1000.0, sr),
                k_release: coef(cfg.release_ms / 1000.0, sr),
                vib_depth: cfg.vibrato_depth / 12.0,
                vib_rate: cfg.vibrato_rate,
                gain: cfg.gain.clamp(0.0, 1.0),
                reverb: Reverb::new(sr),
            }
        }

        /// Fill `out` with interleaved frames of `channels` samples.
        pub fn render<T: SizedSample + FromSample<f32>>(
            &mut self,
            out: &mut [T],
            channels: usize,
            p: &SynthParams,
        ) {
            let target_lf = p.freq.get().clamp(20.0, 8000.0).log2();
            let target_amp = if p.muted.load(Relaxed) {
                0.0
            } else {
                p.amp.get().clamp(0.0, 1.0)
            };
            // Timbre and reverb move once per buffer (a few ms), which is smooth enough.
            let s = p.timbre().shape();
            let k = 0.15;
            self.drive += (s.drive - self.drive) * k;
            self.asym += (s.asym - self.asym) * k;
            self.cutoff += (s.cutoff - self.cutoff) * k;
            self.wet += (p.reverb.get().clamp(0.0, 1.0) - self.wet) * k;
            let shape = Shape {
                drive: self.drive,
                asym: self.asym,
                cutoff: self.cutoff,
            };
            let a_lp = 1.0 - (-TAU * self.cutoff / self.sr).exp();
            let dry = 1.0 - 0.5 * self.wet;
            // Starting from silence, jump straight to the new pitch instead of sweeping up to it.
            if self.amp < 1e-4 && target_amp > 0.0 {
                self.log_freq = target_lf;
            }

            for frame in out.chunks_mut(channels) {
                self.log_freq += (target_lf - self.log_freq) * self.k_glide;
                let k_amp = if target_amp > self.amp {
                    self.k_attack
                } else {
                    self.k_release
                };
                self.amp += (target_amp - self.amp) * k_amp;

                let vib = if self.vib_depth > 0.0 {
                    self.vib_phase = (self.vib_phase + self.vib_rate / self.sr).fract();
                    (TAU * self.vib_phase).sin() * self.vib_depth
                } else {
                    0.0
                };
                let f = (self.log_freq + vib).exp2();
                self.phase = (self.phase + f / self.sr).fract();
                let s = (TAU * self.phase).sin();
                let shaped = shape.transfer(s);
                // DC blocker: the asymmetric clipper leaves an offset that depends on the drive.
                let y = shaped - self.dc_x + 0.995 * self.dc_y;
                self.dc_x = shaped;
                self.dc_y = y;
                self.lp1 += (y - self.lp1) * a_lp;
                self.lp2 += (self.lp1 - self.lp2) * a_lp;
                let x = self.lp2 * self.amp * self.gain;

                let (wl, wr) = self.reverb.process(x);
                let l = soft_clip(x * dry + wl * self.wet);
                let r = soft_clip(x * dry + wr * self.wet);
                match frame {
                    [] => {}
                    [m] => *m = T::from_sample(0.5 * (l + r)),
                    [fl, fr, rest @ ..] => {
                        *fl = T::from_sample(l);
                        *fr = T::from_sample(r);
                        for c in rest {
                            *c = T::from_sample(0.0);
                        }
                    }
                }
            }
        }
    }

    fn soft_clip(x: f32) -> f32 {
        if x.abs() < 0.8 {
            x
        } else {
            x.signum() * (0.8 + 0.2 * ((x.abs() - 0.8) / 0.2).tanh())
        }
    }

    struct Comb {
        buf: Vec<f32>,
        idx: usize,
        filt: f32,
    }

    impl Comb {
        fn new(len: usize) -> Self {
            Self {
                buf: vec![0.0; len.max(1)],
                idx: 0,
                filt: 0.0,
            }
        }
        fn process(&mut self, x: f32) -> f32 {
            const FEEDBACK: f32 = 0.84;
            const DAMP: f32 = 0.25;
            let y = self.buf[self.idx];
            self.filt = y * (1.0 - DAMP) + self.filt * DAMP;
            self.buf[self.idx] = x + self.filt * FEEDBACK;
            self.idx = (self.idx + 1) % self.buf.len();
            y
        }
    }

    struct Allpass {
        buf: Vec<f32>,
        idx: usize,
    }

    impl Allpass {
        fn new(len: usize) -> Self {
            Self {
                buf: vec![0.0; len.max(1)],
                idx: 0,
            }
        }
        fn process(&mut self, x: f32) -> f32 {
            let b = self.buf[self.idx];
            self.buf[self.idx] = x + b * 0.5;
            self.idx = (self.idx + 1) % self.buf.len();
            b - x
        }
    }

    /// A small Freeverb-style room: four parallel combs into two allpass chains (one per channel).
    struct Reverb {
        combs: [Comb; 4],
        left: [Allpass; 2],
        right: [Allpass; 2],
    }

    impl Reverb {
        fn new(sr: f32) -> Self {
            let n = |len: usize| (len as f32 * sr / 44100.0) as usize;
            Self {
                combs: [
                    Comb::new(n(1116)),
                    Comb::new(n(1277)),
                    Comb::new(n(1422)),
                    Comb::new(n(1617)),
                ],
                left: [Allpass::new(n(556)), Allpass::new(n(441))],
                right: [Allpass::new(n(579)), Allpass::new(n(464))],
            }
        }
        fn process(&mut self, x: f32) -> (f32, f32) {
            // The tiny offset keeps the feedback loops out of denormal territory during silence.
            let input = x * 0.12 + 1e-18;
            let s: f32 = self.combs.iter_mut().map(|c| c.process(input)).sum();
            let l = self.left.iter_mut().fold(s, |acc, ap| ap.process(acc));
            let r = self.right.iter_mut().fold(s, |acc, ap| ap.process(acc));
            (l, r)
        }
    }

    /// A running audio output stream. Dropping it stops the sound.
    pub struct Audio {
        _stream: cpal::Stream,
        pub device: String,
        pub sample_rate: u32,
        pub buffer: BufferSize,
    }

    impl Audio {
        pub fn start(cfg: &SoundConfig, params: Arc<SynthParams>) -> Result<Self> {
            let host = cpal::default_host();
            let device = host
                .default_output_device()
                .context("no default audio output device")?;
            let name = device
                .description()
                .map(|d| d.to_string())
                .unwrap_or_else(|_| "default".into());
            let supported = device
                .default_output_config()
                .context("querying the audio output format")?;
            let format = supported.sample_format();
            let base: StreamConfig = supported.into();

            let mut attempts = Vec::new();
            if let Some(n) = cfg.buffer_frames {
                attempts.push(StreamConfig {
                    buffer_size: BufferSize::Fixed(n),
                    ..base
                });
            }
            attempts.push(StreamConfig {
                buffer_size: BufferSize::Default,
                ..base
            });
            let mut last_err = None;
            for config in attempts {
                match build_stream(&device, config, format, cfg, params.clone()) {
                    Ok(stream) => {
                        stream.play().context("starting audio stream")?;
                        log::info!(
                            "audio: {name}, {} Hz, {} ch, {format}, buffer {:?}",
                            config.sample_rate,
                            config.channels,
                            config.buffer_size
                        );
                        return Ok(Self {
                            _stream: stream,
                            device: name,
                            sample_rate: config.sample_rate,
                            buffer: config.buffer_size,
                        });
                    }
                    Err(e) => {
                        log::debug!("audio config {config:?} refused: {e:#}");
                        last_err = Some(e);
                    }
                }
            }
            Err(last_err.unwrap_or_else(|| anyhow!("no audio configuration to try")))
        }
    }

    fn build_stream(
        device: &cpal::Device,
        config: StreamConfig,
        format: SampleFormat,
        cfg: &SoundConfig,
        params: Arc<SynthParams>,
    ) -> Result<cpal::Stream> {
        match format {
            SampleFormat::F32 => build_typed::<f32>(device, config, cfg, params),
            SampleFormat::F64 => build_typed::<f64>(device, config, cfg, params),
            SampleFormat::I16 => build_typed::<i16>(device, config, cfg, params),
            SampleFormat::I32 => build_typed::<i32>(device, config, cfg, params),
            SampleFormat::U16 => build_typed::<u16>(device, config, cfg, params),
            SampleFormat::U8 => build_typed::<u8>(device, config, cfg, params),
            other => Err(anyhow!("unsupported sample format {other}")),
        }
    }

    fn build_typed<T: SizedSample + FromSample<f32>>(
        device: &cpal::Device,
        config: StreamConfig,
        cfg: &SoundConfig,
        params: Arc<SynthParams>,
    ) -> Result<cpal::Stream> {
        let channels = config.channels as usize;
        let mut voice = Voice::new(cfg, config.sample_rate as f32);
        let stream = device.build_output_stream(
            config,
            move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
                voice.render(data, channels, &params)
            },
            |e| log::warn!("audio stream: {e}"),
            None,
        )?;
        Ok(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(cfg: &SoundConfig, p: &SynthParams, frames: usize) -> Vec<f32> {
        let mut v = Voice::new(cfg, 48000.0);
        let mut out = vec![0f32; frames * 2];
        v.render(&mut out, 2, p);
        out
    }

    #[test]
    fn silent_until_played_and_bounded() {
        let cfg = SoundConfig {
            reverb: 0.0,
            ..Default::default()
        };
        let p = SynthParams::new(&cfg);
        assert!(render(&cfg, &p, 4800).iter().all(|s| s.abs() < 1e-6));
        p.amp.set(1.0);
        p.freq.set(440.0);
        for t in Timbre::ALL {
            p.set_timbre(t);
            let out = render(&cfg, &p, 48000);
            let peak = out.iter().fold(0f32, |m, s| m.max(s.abs()));
            assert!(peak > 0.1 && peak <= 1.0, "{t:?} peak {peak}");
            assert!(out.iter().all(|s| s.is_finite()));
        }
    }

    #[test]
    fn pitch_follows_target() {
        let cfg = SoundConfig {
            reverb: 0.0,
            timbre: Timbre::Ether,
            ..Default::default()
        };
        let p = SynthParams::new(&cfg);
        p.amp.set(1.0);
        p.freq.set(440.0);
        let out = render(&cfg, &p, 48000);
        // Count rising zero crossings over the last half second, left channel.
        let left: Vec<f32> = out.chunks(2).map(|f| f[0]).skip(24000).collect();
        let crossings = left
            .windows(2)
            .filter(|w| w[0] < 0.0 && w[1] >= 0.0)
            .count();
        assert!(
            (crossings as i32 - 220).abs() <= 2,
            "{crossings} crossings in 0.5 s"
        );
    }
}
