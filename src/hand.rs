//! Geometric features derived from hand landmarks.

use crate::types::{Hand, Vec2, Vec3, lm};

pub struct HandFeatures {
    /// Mean of wrist and the four finger knuckles, normalized frame coords (0..1).
    pub palm_center: Vec3,
    /// Wrist to middle knuckle distance in isotropic frame units; scale reference.
    pub hand_size: f32,
    /// Straightness of each finger, 0..1 (1 = straight), index 0 = thumb.
    pub straightness: [f32; 5],
}

impl HandFeatures {
    pub fn compute(h: &Hand) -> Self {
        // Landmarks are normalized per axis (x by width, y by height, z by width). Distances
        // need isotropic units, so scale x and z by the aspect ratio.
        let l: [Vec3; 21] = h
            .landmarks
            .map(|p| Vec3::new(p.x * h.aspect, p.y, p.z * h.aspect));
        let palm_pts = [
            lm::WRIST,
            lm::INDEX_MCP,
            lm::MIDDLE_MCP,
            lm::RING_MCP,
            lm::PINKY_MCP,
        ];
        let mut c = Vec3::default();
        for &i in &palm_pts {
            c = c.add(h.landmarks[i]);
        }
        let palm_center = c.scale(1.0 / palm_pts.len() as f32);
        let hand_size = l[lm::MIDDLE_MCP].dist2d(l[lm::WRIST]).max(1e-4);
        let chain = |a: usize, b: usize, c: usize, d: usize| -> f32 {
            let path = l[a].dist(l[b]) + l[b].dist(l[c]) + l[c].dist(l[d]);
            if path < 1e-5 {
                0.0
            } else {
                l[a].dist(l[d]) / path
            }
        };
        let straightness = [
            chain(lm::THUMB_CMC, lm::THUMB_MCP, lm::THUMB_IP, lm::THUMB_TIP),
            chain(lm::INDEX_MCP, lm::INDEX_PIP, lm::INDEX_DIP, lm::INDEX_TIP),
            chain(
                lm::MIDDLE_MCP,
                lm::MIDDLE_PIP,
                lm::MIDDLE_DIP,
                lm::MIDDLE_TIP,
            ),
            chain(lm::RING_MCP, lm::RING_PIP, lm::RING_DIP, lm::RING_TIP),
            chain(lm::PINKY_MCP, lm::PINKY_PIP, lm::PINKY_DIP, lm::PINKY_TIP),
        ];
        Self {
            palm_center,
            hand_size,
            straightness,
        }
    }

    pub fn palm(&self) -> Vec2 {
        Vec2::new(self.palm_center.x, self.palm_center.y)
    }

    /// Mean straightness of the four fingers (thumb excluded): ~1 open hand, low for a fist.
    pub fn openness(&self) -> f32 {
        self.straightness[1..].iter().sum::<f32>() / 4.0
    }
}
