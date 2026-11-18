#[cfg(not(target_arch = "wasm32"))]
mod cli;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    cli::main()
}

/// In a browser: Trunk loads this binary and wasm-bindgen runs `main` on load.
#[cfg(target_arch = "wasm32")]
fn main() {
    theremin::web::start();
}
