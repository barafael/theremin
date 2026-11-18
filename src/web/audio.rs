//! The theremin voice as a Web Audio graph, so it runs on the browser's audio thread and keeps
//! playing smoothly while the page's main thread is busy with hand tracking:
//!
//! ```text
//! sine osc ──▶ waveshaper ──▶ DC block ──▶ low-pass ──▶ envelope ──▶ master ─┬─▶ dry ──▶ out
//!   ▲ detune                 (Shape::transfer)                              └─▶ reverb ─▶ wet ──▶ out
//!   └─ vibrato LFO
//! ```
//!
//! Pitch glide and the envelope are `setTargetAtTime`, the same one-pole smoothing the desktop
//! voice does per sample.

use super::js_err;
use crate::config::SoundConfig;
use crate::synth::{SynthParams, Timbre};
use anyhow::Result;
use std::cell::Cell;
use std::sync::atomic::Ordering;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::Closure;
use web_sys::{
    AudioContext, AudioContextOptions, AudioContextState, BiquadFilterNode, BiquadFilterType,
    GainNode, OscillatorNode, OscillatorType, OverSampleType, WaveShaperNode,
};

pub struct WebSynth {
    ctx: AudioContext,
    osc: OscillatorNode,
    shaper: WaveShaperNode,
    lowpass: BiquadFilterNode,
    envelope: GainNode,
    dry: GainNode,
    wet: GainNode,
    glide: f64,
    attack: f64,
    release: f64,
    amp_target: Cell<f32>,
    timbre: Cell<Option<Timbre>>,
    reverb: Cell<f32>,
}

impl WebSynth {
    pub fn new(cfg: &SoundConfig) -> Result<Self> {
        let options = AudioContextOptions::new();
        options.set_latency_hint(&"interactive".into());
        let ctx = AudioContext::new_with_context_options(&options).map_err(js_err)?;

        let osc = ctx.create_oscillator().map_err(js_err)?;
        osc.set_type(OscillatorType::Sine);
        osc.frequency().set_value(220.0);

        // Vibrato: an LFO in cents onto the oscillator's detune.
        let lfo = ctx.create_oscillator().map_err(js_err)?;
        lfo.frequency().set_value(cfg.vibrato_rate);
        let lfo_depth = ctx.create_gain().map_err(js_err)?;
        lfo_depth.gain().set_value(cfg.vibrato_depth * 100.0);
        lfo.connect_with_audio_node(&lfo_depth).map_err(js_err)?;
        lfo_depth
            .connect_with_audio_param(&osc.detune())
            .map_err(js_err)?;

        let shaper = ctx.create_wave_shaper().map_err(js_err)?;
        shaper.set_oversample(OverSampleType::N4x);
        // The asymmetric clipper leaves a DC offset; block it like the desktop voice does.
        let dc_block = ctx.create_biquad_filter().map_err(js_err)?;
        dc_block.set_type(BiquadFilterType::Highpass);
        dc_block.frequency().set_value(15.0);
        let lowpass = ctx.create_biquad_filter().map_err(js_err)?;
        lowpass.set_type(BiquadFilterType::Lowpass);
        // Q 0.5: two coincident real poles, the desktop voice's two one-pole stages.
        lowpass.q().set_value(0.5);
        let envelope = ctx.create_gain().map_err(js_err)?;
        envelope.gain().set_value(0.0);
        let master = ctx.create_gain().map_err(js_err)?;
        master.gain().set_value(cfg.gain.clamp(0.0, 1.0));

        let reverb = ctx.create_convolver().map_err(js_err)?;
        reverb.set_buffer(Some(&room_impulse(&ctx).map_err(js_err)?));
        let dry = ctx.create_gain().map_err(js_err)?;
        let wet = ctx.create_gain().map_err(js_err)?;

        osc.connect_with_audio_node(&shaper).map_err(js_err)?;
        shaper.connect_with_audio_node(&dc_block).map_err(js_err)?;
        dc_block.connect_with_audio_node(&lowpass).map_err(js_err)?;
        lowpass.connect_with_audio_node(&envelope).map_err(js_err)?;
        envelope.connect_with_audio_node(&master).map_err(js_err)?;
        master.connect_with_audio_node(&dry).map_err(js_err)?;
        master.connect_with_audio_node(&reverb).map_err(js_err)?;
        reverb.connect_with_audio_node(&wet).map_err(js_err)?;
        dry.connect_with_audio_node(&ctx.destination())
            .map_err(js_err)?;
        wet.connect_with_audio_node(&ctx.destination())
            .map_err(js_err)?;
        osc.start().map_err(js_err)?;
        lfo.start().map_err(js_err)?;

        let synth = Self {
            ctx,
            osc,
            shaper,
            lowpass,
            envelope,
            dry,
            wet,
            glide: cfg.glide_ms as f64 / 1000.0,
            attack: cfg.attack_ms as f64 / 1000.0,
            release: cfg.release_ms as f64 / 1000.0,
            amp_target: Cell::new(0.0),
            timbre: Cell::new(None),
            reverb: Cell::new(-1.0),
        };
        synth.set_timbre(cfg.timbre);
        synth.set_reverb(cfg.reverb);
        Ok(synth)
    }

    /// Browsers start audio suspended until the page gets a click or a key press.
    pub fn running(&self) -> bool {
        self.ctx.state() == AudioContextState::Running
    }

    pub fn resume(&self) {
        if !self.running() {
            let _ = self.ctx.resume();
        }
    }

    pub fn sample_rate(&self) -> f32 {
        self.ctx.sample_rate()
    }

    /// Resume the audio on the page's first click, tap or key press.
    pub fn resume_on_gesture(self: &std::rc::Rc<Self>) {
        let Some(window) = web_sys::window() else {
            return;
        };
        for event in ["pointerdown", "keydown"] {
            let synth = self.clone();
            let cb = Closure::<dyn FnMut()>::new(move || synth.resume());
            let _ = window.add_event_listener_with_callback_and_bool(
                event,
                cb.as_ref().unchecked_ref(),
                true,
            );
            // Lives as long as the page.
            cb.forget();
        }
    }

    /// Move toward the latest targets from the control loop.
    pub fn update(&self, p: &SynthParams) {
        let now = self.ctx.current_time();
        let target = if p.muted.load(Ordering::Relaxed) {
            0.0
        } else {
            p.amp.get().clamp(0.0, 1.0)
        };
        let freq = self.osc.frequency();
        let hz = p.freq.get().clamp(20.0, 8000.0);
        if self.amp_target.get() <= 1e-4 && target > 0.0 {
            // Starting from silence: jump to the pitch instead of sweeping up to it.
            let _ = freq.cancel_scheduled_values(now);
            let _ = freq.set_value_at_time(hz, now);
        } else {
            let _ = freq.set_target_at_time(hz, now, self.glide);
        }
        let tau = if target > self.amp_target.get() {
            self.attack
        } else {
            self.release
        };
        let _ = self.envelope.gain().set_target_at_time(target, now, tau);
        self.amp_target.set(target);
        self.set_timbre(p.timbre());
        self.set_reverb(p.reverb.get());
    }

    fn set_timbre(&self, t: Timbre) {
        if self.timbre.get() == Some(t) {
            return;
        }
        self.timbre.set(Some(t));
        let shape = t.shape();
        const N: usize = 2049;
        let curve: Vec<f32> = (0..N)
            .map(|i| shape.transfer(i as f32 / (N - 1) as f32 * 2.0 - 1.0))
            .collect();
        self.shaper
            .set_curve_opt_f32_array(Some(&js_sys::Float32Array::from(&curve[..])));
        let _ = self.lowpass.frequency().set_target_at_time(
            shape.cutoff,
            self.ctx.current_time(),
            0.03,
        );
    }

    fn set_reverb(&self, wet: f32) {
        if (self.reverb.get() - wet).abs() < 1e-4 {
            return;
        }
        self.reverb.set(wet);
        let now = self.ctx.current_time();
        let _ = self
            .dry
            .gain()
            .set_target_at_time(1.0 - 0.5 * wet, now, 0.05);
        let _ = self.wet.gain().set_target_at_time(wet * 0.7, now, 0.05);
    }
}

/// A stereo room: two seconds of decaying noise, a little different per channel.
fn room_impulse(ctx: &AudioContext) -> Result<web_sys::AudioBuffer, wasm_bindgen::JsValue> {
    let sr = ctx.sample_rate();
    let len = (sr * 2.0) as usize;
    let buffer = ctx.create_buffer(2, len as u32, sr)?;
    let mut seed = 0x9E37_79B9u32;
    for ch in 0..2 {
        let mut data: Vec<f32> = (0..len)
            .map(|i| {
                // xorshift noise
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                let noise = seed as f32 / u32::MAX as f32 * 2.0 - 1.0;
                let t = i as f32 / sr;
                noise * (-t * 3.2).exp()
            })
            .collect();
        buffer.copy_to_channel(&mut data, ch)?;
    }
    Ok(buffer)
}
