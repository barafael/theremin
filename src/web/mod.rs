//! The browser version: the same instrument and window as on the desktop, fed by `getUserMedia`,
//! onnxruntime-web, and a Web Audio voice, all driven from the page's event loop.
//!
//! Page parameters: `?image=<url>` plays from a still picture instead of the camera,
//! `?hands=1` starts in one-hand mode.

mod audio;
mod camera;

use crate::app::{Fps, Session, View};
use crate::config::Config;
use crate::detect::Tracker;
use crate::music::midi_to_hz;
use crate::synth::SynthParams;
use crate::ui::ThereminWindow;
use anyhow::{Result, anyhow};
use audio::WebSynth;
use slint::{ComponentHandle, SharedString};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use wasm_bindgen::{JsCast, JsValue};
use web_time::Instant;

/// Where the page serves the models (Trunk copies them next to `index.html`).
const MODELS_URL: &str = "models/";

pub fn start() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);
    if let Err(e) = start_inner() {
        log::error!("{e:#}");
        page_failed(&format!("{e:#}"));
    }
}

fn start_inner() -> Result<()> {
    let mut cfg = Config::default();
    if query("hands").as_deref() == Some("1") {
        cfg.play.hands = 1;
    }
    let ui = ThereminWindow::new()?;
    ui.set_show_video(cfg.ui.show_video);
    ui.set_info("loading…".into());

    let params = Arc::new(SynthParams::new(&cfg.sound));
    let synth = Rc::new(WebSynth::new(&cfg.sound)?);
    synth.resume_on_gesture();
    let session = Rc::new(RefCell::new(Session::new(&cfg, params)));
    {
        let (session, synth) = (session.clone(), synth.clone());
        ui.on_command(move |c| {
            let mut s = session.borrow_mut();
            s.command(&c);
            synth.update(&s.params);
        });
    }

    let handle = ui.clone_strong();
    // In Slint's event loop rather than a bare wasm-bindgen task: property changes made from a
    // task the event loop doesn't know about never wake it to repaint.
    slint::spawn_local(async move {
        if let Err(e) = play(cfg, &handle, session, synth).await {
            log::error!("{e:#}");
            handle.set_hint(SharedString::from(format!("{e:#}")));
            handle.set_info("stopped".into());
        }
    })
    .map_err(|e| anyhow!("{e}"))?;
    page_loaded();
    // On the web this hands control to the browser and does not return.
    ui.run()?;
    Ok(())
}

async fn play(
    cfg: Config,
    ui: &ThereminWindow,
    session: Rc<RefCell<Session>>,
    synth: Rc<WebSynth>,
) -> Result<()> {
    let still = query("image");
    ui.set_hint(
        if still.is_some() {
            "Loading the picture…"
        } else {
            "Allow the camera to play."
        }
        .into(),
    );
    let mut grabber = camera::Grabber::open(&cfg.camera, still).await?;
    ui.set_hint("Loading the hand-tracking models…".into());
    let mut tracker = Tracker::new_web(MODELS_URL, &cfg.detect, grabber.mirrored()).await?;
    log::info!("hand-tracking models loaded");
    // Keys go to the focused element; give the instrument the focus without a click first.
    if let Some(canvas) = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id("canvas"))
    {
        let _ = canvas.unchecked_into::<web_sys::HtmlElement>().focus();
    }
    let params = session.borrow().params.clone();
    let mut fps = Fps::new();
    loop {
        animation_frame().await;
        let Some(frame) = grabber.grab()? else {
            continue;
        };
        let hands = tracker.process(&frame).await?;
        if fps.frames_total().is_power_of_two() || fps.frames_total() == 0 {
            log::info!(
                "frame {}: {} hand(s), {:.1} ms",
                fps.frames_total(),
                hands.len(),
                tracker.last_inference_ms
            );
        }
        let reading = session.borrow_mut().theremin.update(&hands, Instant::now());
        params.freq.set(midi_to_hz(reading.midi));
        params.amp.set(reading.amp);
        params.brightness.set(reading.brightness);
        synth.update(&params);
        let audio_info = format!("{:.0} Hz", synth.sample_rate());
        let view = View::new(
            &mut session.borrow_mut(),
            &frame,
            hands,
            &reading,
            fps.tick(),
            tracker.last_inference_ms,
            &audio_info,
        );
        view.apply(ui);
        if !synth.running() {
            ui.set_hint("Click or press any key to turn on the sound.".into());
        }
    }
}

/// Resolves on the browser's next animation frame.
async fn animation_frame() {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        if let Some(w) = web_sys::window() {
            let _ = w.request_animation_frame(&resolve);
        }
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

fn query(name: &str) -> Option<String> {
    let search = web_sys::window()?.location().search().ok()?;
    web_sys::UrlSearchParams::new_with_str(&search)
        .ok()?
        .get(name)
}

/// The page shows a loading note until the instrument is up, and a failure note if it can't start.
fn page_loaded() {
    if let Some(body) = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.body())
    {
        let _ = body.class_list().remove_1("loading");
    }
}

fn page_failed(message: &str) {
    let Some(document) = web_sys::window().and_then(|w| w.document()) else {
        return;
    };
    if let Some(body) = document.body() {
        let _ = body.class_list().add_1("failed");
    }
    if let Some(el) = document.get_element_by_id("failure") {
        el.set_text_content(Some(message));
    }
}

/// A JavaScript exception as an error, with its message when it has one.
pub(crate) fn js_err(e: JsValue) -> anyhow::Error {
    if let Some(err) = e.dyn_ref::<js_sys::Error>() {
        return anyhow!("{}", String::from(err.message()));
    }
    anyhow!("{}", e.as_string().unwrap_or_else(|| format!("{e:?}")))
}
