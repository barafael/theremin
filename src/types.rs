//! Shared data types passed between the pipeline stages.

use serde::{Deserialize, Serialize};
use web_time::Instant;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    pub fn sub(self, o: Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
    pub fn add(self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
    pub fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    pub fn cross(self, o: Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    pub fn len(self) -> f32 {
        self.dot(self).sqrt()
    }
    pub fn len2d(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
    pub fn normalized(self) -> Self {
        let l = self.len();
        if l > 1e-6 { self.scale(1.0 / l) } else { self }
    }
    pub fn dist(self, o: Self) -> f32 {
        self.sub(o).len()
    }
    pub fn dist2d(self, o: Self) -> f32 {
        self.sub(o).len2d()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
    pub fn sub(self, o: Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y)
    }
    pub fn add(self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y)
    }
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s)
    }
    pub fn len(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
    pub fn dist(self, o: Self) -> f32 {
        self.sub(o).len()
    }
    pub fn clamp01(self) -> Self {
        Self::new(self.x.clamp(0.0, 1.0), self.y.clamp(0.0, 1.0))
    }
}

/// An RGB8 camera frame.
pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
    pub captured_at: Instant,
    pub sequence: u64,
}

impl Frame {
    #[inline]
    pub fn pixel(&self, x: usize, y: usize) -> [u8; 3] {
        let i = (y * self.width + x) * 3;
        [self.rgb[i], self.rgb[i + 1], self.rgb[i + 2]]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Handedness {
    Left,
    Right,
}

/// MediaPipe landmark indices.
pub mod lm {
    pub const WRIST: usize = 0;
    pub const THUMB_CMC: usize = 1;
    pub const THUMB_MCP: usize = 2;
    pub const THUMB_IP: usize = 3;
    pub const THUMB_TIP: usize = 4;
    pub const INDEX_MCP: usize = 5;
    pub const INDEX_PIP: usize = 6;
    pub const INDEX_DIP: usize = 7;
    pub const INDEX_TIP: usize = 8;
    pub const MIDDLE_MCP: usize = 9;
    pub const MIDDLE_PIP: usize = 10;
    pub const MIDDLE_DIP: usize = 11;
    pub const MIDDLE_TIP: usize = 12;
    pub const RING_MCP: usize = 13;
    pub const RING_PIP: usize = 14;
    pub const RING_DIP: usize = 15;
    pub const RING_TIP: usize = 16;
    pub const PINKY_MCP: usize = 17;
    pub const PINKY_PIP: usize = 18;
    pub const PINKY_DIP: usize = 19;
    pub const PINKY_TIP: usize = 20;

    /// Bone connections for drawing the skeleton.
    pub const CONNECTIONS: &[(usize, usize)] = &[
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (0, 17),
    ];
}

/// A detected hand. Landmark x/y are in *normalized frame coordinates* (0..1, origin top-left,
/// not mirrored), z is relative depth in the same units (negative = closer to camera).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Hand {
    pub landmarks: [Vec3; 21],
    pub handedness: Handedness,
    pub handedness_score: f32,
    pub presence: f32,
    /// Stable id across frames while the hand is tracked.
    pub track_id: u32,
    /// Frame width / height, needed to do geometry in isotropic units.
    pub aspect: f32,
}
