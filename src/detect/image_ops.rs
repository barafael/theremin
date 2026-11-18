//! Letterboxing and rotated-crop sampling into NCHW float tensors.

use crate::types::{Frame, Vec3};

/// Square, rotated region of interest in frame pixel coordinates.
#[derive(Clone, Copy, Debug)]
pub struct Roi {
    pub center: Vec3,
    pub side: f32,
    /// Rotation such that crop "up" maps to (sin a, -cos a) in the image.
    pub angle: f32,
}

impl Roi {
    /// Rotation angle for which the crop's up direction points from `from` to `to`.
    pub fn angle_from(from: Vec3, to: Vec3) -> f32 {
        let d = to.sub(from);
        d.x.atan2(-d.y)
    }

    /// Map crop-normalized coordinates (0..1 across the crop) to frame pixels.
    #[inline]
    pub fn to_frame(&self, u: f32, v: f32) -> (f32, f32) {
        let ox = (u - 0.5) * self.side;
        let oy = (v - 0.5) * self.side;
        let (s, c) = self.angle.sin_cos();
        (
            self.center.x + c * ox - s * oy,
            self.center.y + s * ox + c * oy,
        )
    }

    /// Next-frame ROI from the current landmarks (frame pixels).
    pub fn from_landmarks(pts: &[Vec3; 21], frame_w: f32, frame_h: f32) -> Self {
        let angle = Self::angle_from(pts[0], pts[9]);
        let (s, c) = angle.sin_cos();
        // Bounding box in the rotated (hand-aligned) frame.
        let (mut minx, mut miny, mut maxx, mut maxy) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for p in pts {
            let rx = c * p.x + s * p.y;
            let ry = -s * p.x + c * p.y;
            minx = minx.min(rx);
            maxx = maxx.max(rx);
            miny = miny.min(ry);
            maxy = maxy.max(ry);
        }
        let cx = (minx + maxx) * 0.5;
        let cy = (miny + maxy) * 0.5;
        let center = Vec3::new(c * cx - s * cy, s * cx + c * cy, 0.0);
        let extent = (maxx - minx).max(maxy - miny);
        let side = (extent * 1.6).max(0.08 * frame_w.min(frame_h));
        Self {
            center,
            side,
            angle,
        }
    }
}

/// Letterbox the frame into a square `size` tensor, NCHW RGB 0..1.
/// Returns the tensor plus (scale, pad_x, pad_y) mapping frame pixels -> canvas pixels.
pub fn letterbox(frame: &Frame, size: usize) -> (Vec<f32>, f32, f32, f32) {
    let scale = size as f32 / frame.width.max(frame.height) as f32;
    let cw = (frame.width as f32 * scale).round() as usize;
    let ch = (frame.height as f32 * scale).round() as usize;
    let pad_x = ((size - cw) / 2) as f32;
    let pad_y = ((size - ch) / 2) as f32;
    let mut out = vec![0.0f32; 3 * size * size];
    let plane = size * size;
    for y in 0..ch {
        let sy = ((y as f32 + 0.5) / scale - 0.5).clamp(0.0, (frame.height - 1) as f32);
        for x in 0..cw {
            let sx = ((x as f32 + 0.5) / scale - 0.5).clamp(0.0, (frame.width - 1) as f32);
            let px = sample_bilinear(frame, sx, sy);
            let oi = (y + pad_y as usize) * size + (x + pad_x as usize);
            out[oi] = px[0];
            out[plane + oi] = px[1];
            out[2 * plane + oi] = px[2];
        }
    }
    (out, scale, pad_x, pad_y)
}

/// Sample a rotated square crop into an NCHW RGB 0..1 tensor of `size`x`size`.
pub fn crop_roi(frame: &Frame, roi: &Roi, size: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; 3 * size * size];
    let plane = size * size;
    let w = frame.width as f32;
    let h = frame.height as f32;
    for y in 0..size {
        let v = (y as f32 + 0.5) / size as f32;
        for x in 0..size {
            let u = (x as f32 + 0.5) / size as f32;
            let (fx, fy) = roi.to_frame(u, v);
            if fx < 0.0 || fy < 0.0 || fx > w - 1.0 || fy > h - 1.0 {
                continue;
            }
            let px = sample_bilinear(frame, fx, fy);
            let oi = y * size + x;
            out[oi] = px[0];
            out[plane + oi] = px[1];
            out[2 * plane + oi] = px[2];
        }
    }
    out
}

#[inline]
fn sample_bilinear(frame: &Frame, x: f32, y: f32) -> [f32; 3] {
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(frame.width - 1);
    let y1 = (y0 + 1).min(frame.height - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let p00 = frame.pixel(x0, y0);
    let p10 = frame.pixel(x1, y0);
    let p01 = frame.pixel(x0, y1);
    let p11 = frame.pixel(x1, y1);
    let mut out = [0.0; 3];
    for c in 0..3 {
        let top = p00[c] as f32 * (1.0 - fx) + p10[c] as f32 * fx;
        let bot = p01[c] as f32 * (1.0 - fx) + p11[c] as f32 * fx;
        out[c] = (top * (1.0 - fy) + bot * fy) / 255.0;
    }
    out
}
