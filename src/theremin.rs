//! Turns tracked hands into theremin controls: pitch from the horizontal position of the pitch
//! hand, volume from the height of the volume hand, a fist to cut the sound.

use crate::config::{PlayConfig, Side};
use crate::filter::OneEuro;
use crate::hand::HandFeatures;
use crate::music::{self, Scale};
use crate::types::{Hand, Handedness, Vec2};
use std::collections::HashMap;
use web_time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Pitch,
    Volume,
}

/// One control update.
#[derive(Clone, Debug, Default)]
pub struct Reading {
    /// Index into the `hands` slice for each role (in one-hand mode both point to the same hand).
    pub pitch_hand: Option<usize>,
    pub volume_hand: Option<usize>,
    /// Filtered palm positions, normalized frame coordinates.
    pub pitch_pos: Option<Vec2>,
    pub volume_pos: Option<Vec2>,
    /// Pitch from the hand position before and after scale snapping (fractional MIDI).
    pub raw_midi: f32,
    pub midi: f32,
    /// Height of the volume hand within the volume zone, 0 (bottom) .. 1 (top).
    pub volume_level: f32,
    /// Volume from the hand height after the response curve, 0..1, before the fist mute.
    pub volume: f32,
    pub fist: bool,
    /// Final amplitude target for the synth.
    pub amp: f32,
}

pub struct Theremin {
    pub cfg: PlayConfig,
    pub root: i32,
    roles: HashMap<u32, Role>,
    pitch_filter: OneEuro,
    volume_filter: OneEuro,
    fist: bool,
    last_midi: f32,
}

impl Theremin {
    pub fn new(cfg: PlayConfig) -> Self {
        let root = music::parse_key(&cfg.key).unwrap_or_else(|| {
            log::warn!("unknown key {:?}, using C", cfg.key);
            0
        });
        Self {
            pitch_filter: OneEuro::new(cfg.min_cutoff, cfg.beta),
            volume_filter: OneEuro::new(cfg.min_cutoff, cfg.beta),
            last_midi: cfg.low_note as f32,
            cfg,
            root,
            roles: HashMap::new(),
            fist: false,
        }
    }

    pub fn high_note(&self) -> i32 {
        self.cfg.low_note + (self.cfg.octaves * 12.0).round() as i32
    }

    /// Position along the pitch axis toward the pitch side: 0 = lowest note end, 1 = highest.
    fn pitch_t(&self, x: f32) -> f32 {
        let [a, b] = self.cfg.pitch_zone;
        let x = if self.cfg.pitch_hand == Side::Left {
            1.0 - x
        } else {
            x
        };
        ((x - a) / (b - a).max(1e-3)).clamp(0.0, 1.0)
    }

    /// Where note `m` sits on the frame's x axis, for drawing the note grid.
    pub fn midi_to_x(&self, m: f32) -> f32 {
        let [a, b] = self.cfg.pitch_zone;
        let span = (self.high_note() - self.cfg.low_note).max(1) as f32;
        let x = a + (m - self.cfg.low_note as f32) / span * (b - a);
        if self.cfg.pitch_hand == Side::Left {
            1.0 - x
        } else {
            x
        }
    }

    /// The pitch-zone edges in frame x, left to right.
    pub fn pitch_zone_x(&self) -> (f32, f32) {
        let (a, b) = (
            self.midi_to_x(self.cfg.low_note as f32),
            self.midi_to_x(self.high_note() as f32),
        );
        (a.min(b), a.max(b))
    }

    pub fn scale(&self) -> Scale {
        self.cfg.scale
    }

    pub fn update(&mut self, hands: &[Hand], now: Instant) -> Reading {
        self.assign_roles(hands);
        let find = |role: Role| {
            hands
                .iter()
                .position(|h| self.roles.get(&h.track_id) == Some(&role))
        };
        let (pitch_hand, volume_hand) = if self.cfg.hands >= 2 {
            (find(Role::Pitch), find(Role::Volume))
        } else {
            let h = find(Role::Pitch).or(if hands.is_empty() { None } else { Some(0) });
            (h, h)
        };

        let mut r = Reading {
            pitch_hand,
            volume_hand,
            raw_midi: self.last_midi,
            midi: self.last_midi,
            ..Default::default()
        };

        match pitch_hand {
            Some(i) => {
                let p = self
                    .pitch_filter
                    .filter(HandFeatures::compute(&hands[i]).palm(), now);
                let span = (self.high_note() - self.cfg.low_note) as f32;
                r.raw_midi = self.cfg.low_note as f32 + self.pitch_t(p.x) * span;
                r.midi = self.cfg.scale.snap(self.root, r.raw_midi, self.cfg.snap);
                r.pitch_pos = Some(p);
                self.last_midi = r.midi;
            }
            None => self.pitch_filter.reset(),
        }

        match volume_hand {
            Some(i) => {
                let f = HandFeatures::compute(&hands[i]);
                let p = if volume_hand == pitch_hand {
                    r.pitch_pos.unwrap_or_else(|| f.palm())
                } else {
                    self.volume_filter.filter(f.palm(), now)
                };
                let [top, bottom] = self.cfg.volume_zone;
                let v = ((bottom - p.y) / (bottom - top).max(1e-3)).clamp(0.0, 1.0);
                r.volume_level = v;
                r.volume = v.powf(self.cfg.volume_curve.max(0.1));
                r.volume_pos = Some(p);
                let open = f.openness();
                if self.fist {
                    self.fist = open < self.cfg.fist_exit;
                } else {
                    self.fist = open < self.cfg.fist_enter;
                }
                r.fist = self.fist && self.cfg.fist_mute;
            }
            None => {
                self.volume_filter.reset();
                self.fist = false;
            }
        }

        r.amp = if pitch_hand.is_some() && volume_hand.is_some() && !r.fist {
            r.volume
        } else {
            0.0
        };
        r
    }

    /// Keep a stable role per tracked hand. With two hands visible the one further toward the
    /// pitch side plays pitch; a lone new hand gets its role from the model's handedness.
    fn assign_roles(&mut self, hands: &[Hand]) {
        self.roles
            .retain(|id, _| hands.iter().any(|h| h.track_id == *id));
        let pitch_handedness = match self.cfg.pitch_hand {
            Side::Right => Handedness::Right,
            Side::Left => Handedness::Left,
        };
        let toward_pitch = |h: &Hand| {
            let x = HandFeatures::compute(h).palm_center.x;
            if self.cfg.pitch_hand == Side::Right {
                x
            } else {
                -x
            }
        };
        match hands {
            [] => {}
            [h] => {
                self.roles
                    .entry(h.track_id)
                    .or_insert(if h.handedness == pitch_handedness {
                        Role::Pitch
                    } else {
                        Role::Volume
                    });
            }
            [a, b, ..] => {
                let (da, db) = (toward_pitch(a), toward_pitch(b));
                let consistent = match (self.roles.get(&a.track_id), self.roles.get(&b.track_id)) {
                    (Some(ra), Some(rb)) => ra != rb,
                    _ => false,
                };
                // Re-assign by position unless the hands are close together and roles are fine.
                if !consistent || (da - db).abs() > 0.08 {
                    let (pitch, volume) = if da >= db { (a, b) } else { (b, a) };
                    self.roles.insert(pitch.track_id, Role::Pitch);
                    self.roles.insert(volume.track_id, Role::Volume);
                }
                for extra in &hands[2..] {
                    self.roles.remove(&extra.track_id);
                }
            }
        }
        if self.cfg.hands < 2 {
            // One-hand mode: the hand that last played pitch keeps playing.
            if let Some(h) = hands
                .iter()
                .find(|h| self.roles.get(&h.track_id) == Some(&Role::Pitch))
            {
                let id = h.track_id;
                self.roles.retain(|k, _| *k == id);
            } else if let Some(h) = hands.first() {
                self.roles.clear();
                self.roles.insert(h.track_id, Role::Pitch);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Vec3;

    /// An open hand with its palm centred at (x, y).
    fn hand(id: u32, x: f32, y: f32, handedness: Handedness) -> Hand {
        let mut l = [Vec3::new(x, y, 0.0); 21];
        l[crate::types::lm::WRIST] = Vec3::new(x, y + 0.08, 0.0);
        for (f, &mcp) in [5usize, 9, 13, 17].iter().enumerate() {
            let dx = (f as f32 - 1.5) * 0.02;
            for j in 0..4 {
                l[mcp + j] = Vec3::new(x + dx, y - 0.02 - 0.03 * j as f32, 0.0);
            }
        }
        for j in 1..5 {
            l[j] = Vec3::new(x - 0.03 - 0.01 * j as f32, y + 0.04 - 0.02 * j as f32, 0.0);
        }
        Hand {
            landmarks: l,
            handedness,
            handedness_score: 1.0,
            presence: 1.0,
            track_id: id,
            aspect: 4.0 / 3.0,
        }
    }

    #[test]
    fn two_hands_pitch_and_volume() {
        let mut t = Theremin::new(PlayConfig::default());
        let now = Instant::now();
        // Volume hand high on the left, pitch hand at the right end of the zone.
        let hands = [
            hand(1, 0.15, 0.25, Handedness::Left),
            hand(2, 0.95, 0.5, Handedness::Right),
        ];
        let r = t.update(&hands, now);
        assert_eq!(r.pitch_hand, Some(1));
        assert_eq!(r.volume_hand, Some(0));
        assert!(r.midi > 80.0, "midi {}", r.midi);
        assert!(r.amp > 0.8, "amp {}", r.amp);
        // Without the volume hand there is no sound.
        let r = t.update(&hands[1..], now);
        assert_eq!(r.amp, 0.0);
    }

    #[test]
    fn roles_follow_position_not_handedness() {
        let mut t = Theremin::new(PlayConfig::default());
        let now = Instant::now();
        // Both reported as right hands; the right-most one plays pitch.
        let hands = [
            hand(7, 0.8, 0.5, Handedness::Right),
            hand(8, 0.2, 0.5, Handedness::Right),
        ];
        let r = t.update(&hands, now);
        assert_eq!(r.pitch_hand, Some(0));
        assert_eq!(r.volume_hand, Some(1));
    }

    #[test]
    fn one_hand_mode_and_grid() {
        let cfg = PlayConfig {
            hands: 1,
            scale: Scale::Major,
            snap: 1.0,
            ..Default::default()
        };
        let mut t = Theremin::new(cfg);
        let r = t.update(&[hand(3, 0.6, 0.3, Handedness::Left)], Instant::now());
        assert_eq!(r.pitch_hand, Some(0));
        assert_eq!(r.volume_hand, Some(0));
        assert!(r.amp > 0.0);
        assert_eq!(r.midi, r.midi.round());
        assert!(Scale::Major.contains(0, r.midi as i32));
        let (a, b) = t.pitch_zone_x();
        assert!((a - 0.30).abs() < 1e-4 && (b - 0.95).abs() < 1e-4);
        assert!((t.midi_to_x(t.cfg.low_note as f32) - 0.30).abs() < 1e-4);
    }

    #[test]
    fn left_handed_pitch_rises_to_the_left() {
        let cfg = PlayConfig {
            pitch_hand: Side::Left,
            ..Default::default()
        };
        let t = Theremin::new(cfg);
        assert!(t.midi_to_x(t.high_note() as f32) < t.midi_to_x(t.cfg.low_note as f32));
        assert!(t.pitch_t(0.05) > 0.99);
    }
}
