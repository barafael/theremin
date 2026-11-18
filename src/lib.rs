//! theremin: a theremin played with bare hands in front of an ordinary webcam, on the desktop
//! or in a browser.

pub mod app;
#[cfg(not(target_arch = "wasm32"))]
pub mod camera;
pub mod config;
pub mod detect;
pub mod filter;
pub mod hand;
pub mod music;
pub mod synth;
pub mod theremin;
pub mod types;
#[cfg(target_arch = "wasm32")]
pub mod web;
pub mod wireframe;

pub mod ui {
    slint::include_modules!();
}
