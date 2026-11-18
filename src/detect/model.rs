//! One ONNX model behind the same async interface on both targets: the native ONNX Runtime on
//! the desktop (where running completes immediately) and onnxruntime-web through `ort-web` in a
//! browser (where running is a promise).

use anyhow::Result;
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;

pub struct Model {
    session: Session,
    #[cfg(target_arch = "wasm32")]
    run_options: ort::session::RunOptions,
}

/// One f32 output, copied out of the runtime.
pub struct Output {
    pub name: String,
    pub shape: Vec<i64>,
    pub data: Vec<f32>,
}

impl Model {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_file(env: &Environment, path: &std::path::Path, threads: usize) -> Result<Self> {
        use anyhow::{Context, anyhow};
        let session = Session::builder(env)
            .map_err(|e| anyhow!("{e}"))?
            .with_intra_threads(threads)
            .map_err(|e| anyhow!("{e}"))?
            .commit_from_file(path)
            .with_context(|| format!("loading {}", path.display()))?;
        Ok(Self { session })
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn load_url(env: &Environment, url: &str) -> Result<Self> {
        use anyhow::{Context, anyhow};
        let session = Session::builder(env)
            .map_err(|e| anyhow!("{e}"))?
            .commit_from_url(url)
            .await
            .with_context(|| format!("loading {url}"))?;
        Ok(Self {
            session,
            run_options: ort::session::RunOptions::new()?,
        })
    }

    /// Run with a single f32 input and return every output.
    pub async fn run(&mut self, input: Tensor<f32>) -> Result<Vec<Output>> {
        #[cfg(not(target_arch = "wasm32"))]
        let outputs = self.session.run(ort::inputs![input])?;
        #[cfg(target_arch = "wasm32")]
        let outputs = {
            let mut outputs = self
                .session
                .run_async(ort::inputs![input], &self.run_options)
                .await?;
            ort_web::sync_outputs(&mut outputs).await?;
            outputs
        };
        outputs
            .iter()
            .map(|(name, value)| {
                let (shape, data) = value.try_extract_tensor::<f32>()?;
                Ok(Output {
                    name: name.to_string(),
                    shape: shape.to_vec(),
                    data: data.to_vec(),
                })
            })
            .collect()
    }
}

/// Find an output by name.
pub fn output<'a>(outputs: &'a [Output], name: &str) -> Result<&'a Output> {
    outputs
        .iter()
        .find(|o| o.name == name)
        .ok_or_else(|| anyhow::anyhow!("model has no output {name:?}"))
}
