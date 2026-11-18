//! One Euro filter (Casiez et al. 2012) on a 2D signal: smooth at rest, responsive when moving.

use crate::types::Vec2;
use web_time::Instant;

pub struct OneEuro {
    min_cutoff: f32,
    beta: f32,
    d_cutoff: f32,
    prev: Option<(Vec2, Vec2, Instant)>,
}

impl OneEuro {
    pub fn new(min_cutoff: f32, beta: f32) -> Self {
        Self {
            min_cutoff,
            beta,
            d_cutoff: 1.0,
            prev: None,
        }
    }

    pub fn reset(&mut self) {
        self.prev = None;
    }

    fn alpha(cutoff: f32, dt: f32) -> f32 {
        let tau = 1.0 / (2.0 * std::f32::consts::PI * cutoff);
        1.0 / (1.0 + tau / dt)
    }

    pub fn filter(&mut self, x: Vec2, now: Instant) -> Vec2 {
        let Some((px, pdx, pt)) = self.prev else {
            self.prev = Some((x, Vec2::default(), now));
            return x;
        };
        let dt = now.duration_since(pt).as_secs_f32().max(1e-3);
        let dx = x.sub(px).scale(1.0 / dt);
        let ad = Self::alpha(self.d_cutoff, dt);
        let dxf = Vec2::new(pdx.x + ad * (dx.x - pdx.x), pdx.y + ad * (dx.y - pdx.y));
        let cutoff = self.min_cutoff + self.beta * dxf.len();
        let a = Self::alpha(cutoff, dt);
        let xf = Vec2::new(px.x + a * (x.x - px.x), px.y + a * (x.y - px.y));
        self.prev = Some((xf, dxf, now));
        xf
    }
}
