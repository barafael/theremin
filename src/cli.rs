//! The desktop command line.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;
use theremin::config::Config;
use theremin::music::{self, midi_to_hz};
use theremin::synth::{Audio, SynthParams};
use theremin::types::{Frame, lm};
use theremin::{app, camera, detect, hand};
use web_time::{Duration, Instant};

#[derive(Parser)]
#[command(
    name = "theremin",
    about = "A theremin you play with your hands in front of a webcam"
)]
struct Cli {
    /// Path to config.toml (default: ~/.config/theremin/config.toml)
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Play (the default).
    Play {
        /// Play with a single hand: left/right = pitch, up/down = volume.
        #[arg(long)]
        one_hand: bool,
    },
    /// Sweep through the pitch range, dark to bright, for a few seconds to test the sound output.
    Tone,
    /// Check models, camera, and audio output.
    Check,
    /// Run the detector on an image file and write an annotated PNG next to it.
    Image { path: PathBuf },
    /// Print the default configuration.
    DefaultConfig,
}

pub fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info,ort=warn"))
        .init();
    let cli = Cli::parse();
    let mut cfg = Config::load(cli.config.as_ref())?;
    match cli.cmd.unwrap_or(Cmd::Play { one_hand: false }) {
        Cmd::Play { one_hand } => {
            if one_hand {
                cfg.play.hands = 1;
            }
            app::play(cfg)
        }
        Cmd::Tone => tone(cfg),
        Cmd::Check => check(cfg),
        Cmd::Image { path } => image_debug(cfg, &path),
        Cmd::DefaultConfig => {
            print!("{}", toml::to_string_pretty(&Config::default())?);
            Ok(())
        }
    }
}

fn tone(cfg: Config) -> Result<()> {
    let params = Arc::new(SynthParams::new(&cfg.sound));
    let audio = Audio::start(&cfg.sound, params.clone())?;
    let lo = cfg.play.low_note as f32;
    let hi = lo + cfg.play.octaves * 12.0;
    println!(
        "audio: {} ({} Hz, buffer {:?}); sweeping {} → {}, dark → bright, with {} timbre",
        audio.device,
        audio.sample_rate,
        audio.buffer,
        music::note_name(lo as i32),
        music::note_name(hi as i32),
        params.timbre().name()
    );
    let dur = 4.0;
    let start = Instant::now();
    loop {
        let t = start.elapsed().as_secs_f32() / dur;
        if t >= 1.0 {
            break;
        }
        params.freq.set(midi_to_hz(lo + t * (hi - lo)));
        params.brightness.set(t);
        params
            .amp
            .set((std::f32::consts::PI * t).sin().sqrt() * 0.8);
        std::thread::sleep(Duration::from_millis(10));
    }
    params.amp.set(0.0);
    std::thread::sleep(Duration::from_millis(400));
    Ok(())
}

fn check(cfg: Config) -> Result<()> {
    println!("models dir: {}", cfg.models_dir().display());
    match detect::Tracker::new(&cfg.models_dir(), &cfg.detect, cfg.camera.mirror) {
        Ok(_) => println!("models: ok"),
        Err(e) => println!("models: FAILED: {e:#}"),
    }
    match camera::Camera::open(&cfg.camera) {
        Ok(cam) => match cam.frames.recv_timeout(Duration::from_secs(3)) {
            Ok(f) => println!("camera: ok ({}x{})", f.width, f.height),
            Err(e) => println!("camera: opened but no frames ({e})"),
        },
        Err(e) => println!("camera: FAILED: {e:#}"),
    }
    let params = Arc::new(SynthParams::new(&cfg.sound));
    match Audio::start(&cfg.sound, params) {
        Ok(a) => println!(
            "audio: ok ({}, {} Hz, buffer {:?})",
            a.device, a.sample_rate, a.buffer
        ),
        Err(e) => println!("audio: FAILED: {e:#}"),
    }
    Ok(())
}

fn image_debug(cfg: Config, path: &PathBuf) -> Result<()> {
    let img = image::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let frame = Frame {
        width: w,
        height: h,
        rgb: img.into_raw(),
        captured_at: Instant::now(),
        sequence: 1,
    };
    let mut tracker = detect::Tracker::new(&cfg.models_dir(), &cfg.detect, false)?;
    let hands = tracker.process_blocking(&frame)?;
    println!(
        "{} hand(s) in {:.1} ms",
        hands.len(),
        tracker.last_inference_ms
    );
    let mut out = image::RgbImage::from_raw(w as u32, h as u32, frame.rgb.clone()).unwrap();
    for hand in &hands {
        let f = hand::HandFeatures::compute(hand);
        println!(
            "track {} {:?} presence {:.2} palm ({:.2}, {:.2}) openness {:.2} straightness {:?}",
            hand.track_id,
            hand.handedness,
            hand.presence,
            f.palm_center.x,
            f.palm_center.y,
            f.openness(),
            f.straightness.map(|s| (s * 100.0).round() / 100.0)
        );
        let px = |i: usize| {
            (
                (hand.landmarks[i].x * w as f32) as i32,
                (hand.landmarks[i].y * h as f32) as i32,
            )
        };
        for &(a, b) in lm::CONNECTIONS {
            draw_line(&mut out, px(a), px(b), [255, 255, 255]);
        }
        for i in 0..21 {
            let (x, y) = px(i);
            draw_dot(
                &mut out,
                x,
                y,
                4,
                if i == 0 {
                    [255, 200, 0]
                } else {
                    [90, 200, 255]
                },
            );
        }
    }
    let out_path = path.with_extension("landmarks.png");
    out.save(&out_path)?;
    println!("wrote {}", out_path.display());
    Ok(())
}

fn draw_dot(img: &mut image::RgbImage, x: i32, y: i32, r: i32, c: [u8; 3]) {
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                let (px, py) = (x + dx, y + dy);
                if px >= 0 && py >= 0 && (px as u32) < img.width() && (py as u32) < img.height() {
                    img.put_pixel(px as u32, py as u32, image::Rgb(c));
                }
            }
        }
    }
}

fn draw_line(img: &mut image::RgbImage, a: (i32, i32), b: (i32, i32), c: [u8; 3]) {
    let n = (a.0 - b.0).abs().max((a.1 - b.1).abs()).max(1);
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let x = (a.0 as f32 + (b.0 - a.0) as f32 * t) as i32;
        let y = (a.1 as f32 + (b.1 - a.1) as f32 * t) as i32;
        if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
            img.put_pixel(x as u32, y as u32, image::Rgb(c));
        }
    }
}
