//! Hand landmark model (PINTO's export of MediaPipe hand_landmark_sparse).

use super::image_ops::{Roi, crop_roi};
use super::model::{Model, output};
use crate::types::{Frame, Vec3};
use anyhow::Result;
use ort::value::Tensor;

pub const FILE: &str = "hand_landmark_sparse_Nx3x224x224.onnx";
const SIZE: usize = 224;

pub struct LandmarkModel {
    model: Model,
}

pub struct LandmarkResult {
    /// Landmarks in frame pixel coordinates; z in frame pixel scale, relative to the wrist.
    pub landmarks_px: [Vec3; 21],
    pub presence: f32,
    pub right: bool,
}

impl LandmarkModel {
    pub fn new(model: Model) -> Self {
        Self { model }
    }

    pub async fn run(&mut self, frame: &Frame, roi: &Roi) -> Result<LandmarkResult> {
        let tensor = crop_roi(frame, roi, SIZE);
        let input = Tensor::from_array(([1usize, 3, SIZE, SIZE], tensor))?;
        let outputs = self.model.run(input).await?;
        let xyz = &output(&outputs, "xyz_x21")?.data;
        let score = &output(&outputs, "hand_score")?.data;
        let handed = &output(&outputs, "lefthand_0_or_righthand_1")?.data;
        anyhow::ensure!(
            !score.is_empty() && !handed.is_empty(),
            "empty landmark outputs"
        );
        anyhow::ensure!(
            xyz.len() >= 63,
            "unexpected landmark output length {}",
            xyz.len()
        );
        let mut landmarks_px = [Vec3::default(); 21];
        for (i, p) in landmarks_px.iter_mut().enumerate() {
            let u = xyz[i * 3] / SIZE as f32;
            let v = xyz[i * 3 + 1] / SIZE as f32;
            let (fx, fy) = roi.to_frame(u, v);
            *p = Vec3::new(fx, fy, xyz[i * 3 + 2] * roi.side / SIZE as f32);
        }
        Ok(LandmarkResult {
            landmarks_px,
            presence: score[0],
            right: handed[0] > 0.5,
        })
    }
}
