# theremin

A theremin you play with your bare hands in front of an ordinary webcam, on the desktop or in a
browser. Rust throughout: MediaPipe hand models through ONNX Runtime, a small synth, and a Slint
window that shows the note grid over your camera picture.

**Play it in your browser: <https://barafael.github.io/theremin/>** (needs a webcam; nothing
leaves the page).

## Playing

```sh
cargo build --release
./target/release/theremin            # same as `theremin play`
./target/release/theremin play --one-hand
```

Like the real instrument, one hand plays pitch and the other plays volume:

| hand | movement | effect |
|---|---|---|
| right (pitch) | left ↔ right | pitch, continuously, rising to the right, across `play.octaves` (C3–C6 by default) |
| left (volume) | down ↕ up | volume: low = silent, high = loud |
| left (volume) | close into a fist | instant silence (for staccato and phrasing), open to play again |

Roles follow position, not the model's left/right guess: with both hands up, the one further to
the right plays pitch. Set `play.pitch_hand = "left"` to play the other way round. With
`--one-hand` (or **H** while playing) a single hand does both: left/right = pitch, up/down =
volume.

The window shows the camera picture with the note grid over the pitch zone (scale notes
brighter, the key's root in blue), a glowing line at the sounding pitch, the volume rail on the
left, and the note name with its frequency and how many cents you are off.

### Keys

| key | |
|---|---|
| **S** | next scale: chromatic, major, minor, pentatonic, blues, dorian, harmonic minor |
| **K** / **J** | key up / down a semitone |
| **Q** | snap: free pitch → 50% pull toward scale notes → fully quantized |
| **T** | timbre: ether (near-sine), classic (warm tube theremin), cello, reed |
| **H** | one hand / two hands |
| **R** | reverb amount |
| **↑** / **↓** | shift the range an octave |
| **M** | mute |
| **V** | camera picture on/off |

Free pitch is how a real theremin works and where the vibrato and glissandi live. Snap makes it
much easier to hit notes while you learn.

## In a browser

```sh
trunk serve        # http://127.0.0.1:8080
```

The same instrument and window, compiled to WebAssembly and drawn on a `<canvas>`:

| | desktop | browser |
|---|---|---|
| camera | V4L2 (`camera.rs`) | `getUserMedia` (`web/camera.rs`) |
| inference | ONNX Runtime, native | onnxruntime-web 1.30 via [`ort-web`](https://ort.pyke.io/backends/web), same `ort` code (`detect/model.rs`) |
| sound | cpal callback rendering `Voice` | Web Audio graph with the same waveshaper curve (`web/audio.rs`), on the browser's audio thread |
| loop | tracking thread | `slint::spawn_local` on the page's event loop (`web/mod.rs`) |

- Browsers keep audio off until the page gets a click or key press; the window says so.
- `?image=models/test_hand.jpg` plays from a still picture instead of the camera, `?hands=1`
  starts in one-hand mode.
- Camera access needs `https://` or `localhost`.
- The page loads onnxruntime-web from jsdelivr; ort-web's usage telemetry is switched off. The
  models and the camera picture stay on your machine.
- Speed: about 35–45 ms per frame with two hands (release build, 4 threads). `trunk serve` builds
  release by default (`Trunk.toml`), since a dev build is ~3× slower in the browser.
- Threads need cross-origin isolation: `trunk serve` sends `Cross-Origin-Opener-Policy:
  same-origin` and `Cross-Origin-Embedder-Policy: credentialless`. A host that can't set headers
  (GitHub Pages) still works, single-threaded and slower.

To publish: `trunk build --release --cargo-profile web` writes a static site into `dist/`
(about 3.3 MB of wasm plus 15 MB of models). Pushing to `main` deploys it to GitHub Pages
(`.github/workflows/pages.yml`). Pages can't send the isolation headers, so the hosted version
tracks single-threaded: about 80 ms per frame instead of 40.

## Other commands

| command | what it does |
|---|---|
| `theremin tone` | 4-second sweep through the range: tests the sound output |
| `theremin check` | tests models, camera, and audio output |
| `theremin image <photo>` | runs the hand detector on a photo, writes `<photo>.landmarks.png` |
| `theremin default-config` | prints the default `config.toml` |
| `cargo run --release --example snapshot -- <photo> <out.png> [keys…]` | renders the window offscreen for a still photo |

## Configuration

`~/.config/theremin/config.toml` (start with `theremin default-config > …`). The useful knobs:

- `play.pitch_zone` / `play.volume_zone`: which part of the camera picture spans the pitch range
  (x, left→right) and the volume range (y, top = loud → bottom = silent).
- `play.low_note`, `play.octaves`: the range (MIDI numbers, 48 = C3).
- `play.min_cutoff`, `play.beta`: One Euro smoothing of hand positions. Lower `min_cutoff` gives
  steadier notes at rest; higher `beta` gives less lag on fast moves. A heavier filter also eats
  hand vibrato, so `sound.vibrato_depth` (semitones) can add synthetic vibrato.
- `play.fist_enter` / `fist_exit`: finger straightness thresholds for the fist mute.
- `sound.glide_ms`, `attack_ms`, `release_ms`: how the synth smooths between camera frames.
- `sound.buffer_frames`: audio buffer (256 frames ≈ 5 ms at 48 kHz); the device default is used
  if the request is refused.

## How it works

1. **camera.rs**: MJPEG frames from V4L2 at 60 fps, decoded and mirrored on a capture thread.
2. **detect/**: palm detection, then per-hand landmark regression with ROI tracking (stable
   track ids across frames).
3. **theremin.rs**: assigns pitch/volume roles to tracks, filters palm positions (One Euro), and
   maps them to fractional MIDI pitch (optionally snapped to a scale) and amplitude.
4. **synth.rs**: on the desktop, the audio callback glides toward the latest targets every sample, so the
   ~60 Hz control rate is inaudible. The voice is a sine through an asymmetric soft clipper
   (a theremin's heterodyne oscillator is a slightly distorted sine), a DC blocker, a two-pole
   low-pass, and a small Freeverb-style stereo reverb.
5. **app.rs** + **ui/app.slint**: the window. **web/**: the browser front end (see above).

## Setup

- **Camera access.** The seat ACL grants `/dev/video0` to the active session; otherwise add
  yourself to `video` (`sudo usermod -aG video $USER`, re-login). The camera's
  `exposure_dynamic_framerate` is switched off so low light does not drop the control rate to
  10 fps; set `camera.exposure` for a fixed exposure if frames get too dark.
- **Audio.** Uses the default ALSA output (PipeWire/PulseAudio provide one).
- **Models** live in `models/`: Google's MediaPipe palm detection and hand landmark models
  (Apache-2.0), as converted to ONNX by [PINTO0309](https://github.com/PINTO0309/PINTO_model_zoo).
  The native ONNX Runtime binary is downloaded at build time. `models/test_hand.jpg` is by
  Evan-Amos, CC BY-SA 3.0 (see `models/test_hand.txt`).

## License

MIT or Apache-2.0, at your option.
- **Toolchain.** `rust-toolchain.toml` pins stable with the `wasm32-unknown-unknown` target. The
  nightly from 2026-10-31 fails to compile `zbus` (pulled in by Slint's accessibility support).
- **Web.** [Trunk](https://trunkrs.dev) (`cargo install trunk`).
