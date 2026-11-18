//! Builds the Slint wireframe models (paths + points) from tracked hands.

use crate::types::{Hand, lm};
use crate::ui::{HandPath, Landmark};
use slint::{Color, ModelRc, SharedString, VecModel};
use std::fmt::Write;

/// Bone paths and landmark points, one color per hand. Paths use a viewbox of
/// `1000 * aspect` x 1000 (Slint fits path viewboxes without stretching, so the viewbox must have
/// the frame's aspect ratio); points use normalized coordinates. `point_radius` is in logical pixels.
pub fn build(
    hands: &[Hand],
    colors: &[Color],
    aspect: f32,
    point_radius: f32,
) -> (ModelRc<HandPath>, ModelRc<Landmark>) {
    let sx = 1000.0 * aspect;
    let mut paths = Vec::with_capacity(hands.len());
    let mut points = Vec::with_capacity(hands.len() * 21);
    for (hi, hand) in hands.iter().enumerate() {
        let color = colors
            .get(hi)
            .copied()
            .unwrap_or(Color::from_rgb_u8(150, 150, 160));
        let mut cmd = String::with_capacity(21 * 24);
        for &(a, b) in lm::CONNECTIONS {
            let pa = hand.landmarks[a];
            let pb = hand.landmarks[b];
            let _ = write!(
                cmd,
                "M {:.1} {:.1} L {:.1} {:.1} ",
                pa.x * sx,
                pa.y * 1000.0,
                pb.x * sx,
                pb.y * 1000.0
            );
        }
        paths.push(HandPath {
            commands: SharedString::from(cmd),
            color: color.with_alpha(0.75),
        });
        for (i, p) in hand.landmarks.iter().enumerate() {
            let tip = i != 0 && i % 4 == 0;
            let (r, c) = if tip {
                (point_radius * 1.5, color)
            } else {
                (point_radius, Color::from_argb_f32(0.8, 0.92, 0.92, 0.96))
            };
            points.push(Landmark {
                x: p.x,
                y: p.y,
                r,
                color: c,
            });
        }
    }
    (
        ModelRc::new(VecModel::from(paths)),
        ModelRc::new(VecModel::from(points)),
    )
}
