//! Render the instrument window for a still photo, offscreen, to a PNG. Useful to check the UI
//! without a camera or a display.
//!
//!     cargo run --release --example snapshot -- photo.jpg out.png [key commands…]
//!
//! Key commands are the window's shortcuts, e.g. `s q` = next scale, snap 50%.
#![cfg(not(target_arch = "wasm32"))]

use anyhow::{Context, Result};
use slint::platform::software_renderer::{MinimalSoftwareWindow, RepaintBufferType};
use slint::platform::{Platform, WindowAdapter};
use slint::{ComponentHandle, PhysicalSize, Rgb8Pixel};
use std::rc::Rc;
use std::sync::Arc;
use theremin::app::{Session, View};
use theremin::config::Config;
use theremin::detect;
use theremin::synth::SynthParams;
use theremin::types::Frame;
use theremin::ui::ThereminWindow;
use web_time::Instant;

struct Offscreen(Rc<MinimalSoftwareWindow>);

impl Platform for Offscreen {
    fn create_window_adapter(&self) -> Result<Rc<dyn WindowAdapter>, slint::PlatformError> {
        Ok(self.0.clone())
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [input, output, commands @ ..] = args.as_slice() else {
        anyhow::bail!("usage: snapshot <photo> <out.png> [key commands…]");
    };
    let (w, h) = (1180u32, 860u32);
    let window = MinimalSoftwareWindow::new(RepaintBufferType::NewBuffer);
    window.set_size(PhysicalSize::new(w, h));
    slint::platform::set_platform(Box::new(Offscreen(window.clone())))
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let cfg = Config::default();
    let img = image::open(input)
        .with_context(|| format!("opening {input}"))?
        .to_rgb8();
    let frame = Frame {
        width: img.width() as usize,
        height: img.height() as usize,
        rgb: img.into_raw(),
        captured_at: Instant::now(),
        sequence: 1,
    };
    let mut tracker = detect::Tracker::new(&cfg.models_dir(), &cfg.detect, false)?;
    let hands = tracker.process_blocking(&frame)?;
    let mut session = Session::new(&cfg, Arc::new(SynthParams::new(&cfg.sound)));
    for c in commands {
        session.command(c);
    }
    let reading = session.theremin.update(&hands, Instant::now());
    println!(
        "{} hand(s); midi {:.2}, volume {:.2}, amp {:.2}",
        hands.len(),
        reading.midi,
        reading.volume,
        reading.amp
    );

    let ui = ThereminWindow::new()?;
    ui.show()?;
    View::new(
        &mut session,
        &frame,
        hands,
        &reading,
        60.0,
        tracker.last_inference_ms,
        "48000 Hz",
    )
    .apply(&ui);
    slint::platform::update_timers_and_animations();
    let mut buf = vec![Rgb8Pixel::default(); (w * h) as usize];
    window.draw_if_needed(|renderer| {
        renderer.render(&mut buf, w as usize);
    });
    let bytes: Vec<u8> = buf.iter().flat_map(|p| [p.r, p.g, p.b]).collect();
    image::RgbImage::from_raw(w, h, bytes)
        .context("image buffer")?
        .save(output)?;
    println!("wrote {output}");
    Ok(())
}
