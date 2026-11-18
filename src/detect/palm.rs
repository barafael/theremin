//! Palm detector (PINTO's export of MediaPipe palm_detection_full with decode+NMS baked in).

use super::image_ops::{Roi, letterbox};
use super::model::Model;
use crate::types::{Frame, Vec3};
use anyhow::Result;
use ort::value::Tensor;

pub const FILE: &str = "palm_detection_full_inf_post_192x192.onnx";
const SIZE: usize = 192;

pub struct PalmDetector {
    model: Model,
}

/// Detection in frame pixel coordinates.
#[derive(Clone, Debug)]
pub struct PalmDetection {
    pub score: f32,
    pub center: Vec3,
    pub width: f32,
    pub wrist: Vec3,
    pub middle_mcp: Vec3,
}

impl PalmDetection {
    /// Hand ROI the way MediaPipe builds it: 2.6x the palm box, shifted half a box toward the fingers.
    pub fn roi(&self) -> Roi {
        let angle = Roi::angle_from(self.wrist, self.middle_mcp);
        let dir = self.middle_mcp.sub(self.wrist).normalized();
        let center = self.center.add(dir.scale(0.5 * self.width));
        Roi {
            center: Vec3::new(center.x, center.y, 0.0),
            side: self.width * 2.6,
            angle,
        }
    }
}

impl PalmDetector {
    pub fn new(model: Model) -> Self {
        Self { model }
    }

    pub async fn detect(&mut self, frame: &Frame, threshold: f32) -> Result<Vec<PalmDetection>> {
        let (tensor, scale, pad_x, pad_y) = letterbox(frame, SIZE);
        let input = Tensor::from_array(([1usize, 3, SIZE, SIZE], tensor))?;
        let outputs = self.model.run(input).await?;
        let first = outputs
            .first()
            .ok_or_else(|| anyhow::anyhow!("palm model has no outputs"))?;
        let (shape, data) = (&first.shape, &first.data);
        let rows = shape[0] as usize;
        let cols = shape[1] as usize;
        anyhow::ensure!(cols == 8, "unexpected palm output shape {shape:?}");
        let to_frame = |x: f32, y: f32| {
            Vec3::new(
                (x * SIZE as f32 - pad_x) / scale,
                (y * SIZE as f32 - pad_y) / scale,
                0.0,
            )
        };
        let mut out = Vec::new();
        for r in 0..rows {
            let d = &data[r * cols..(r + 1) * cols];
            if d[0] < threshold {
                continue;
            }
            out.push(PalmDetection {
                score: d[0],
                center: to_frame(d[1], d[2]),
                width: d[3] * SIZE as f32 / scale,
                wrist: to_frame(d[4], d[5]),
                middle_mcp: to_frame(d[6], d[7]),
            });
        }
        out.sort_by(|a, b| b.score.total_cmp(&a.score));
        Ok(out)
    }
}
