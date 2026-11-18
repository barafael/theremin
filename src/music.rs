//! Notes, scales, and pitch quantization. Pitches are fractional MIDI note numbers (69 = A4).

use serde::{Deserialize, Serialize};

// In a browser Slint draws with its built-in font, which has no ♯ (U+266F).
#[cfg(not(target_arch = "wasm32"))]
const NAMES: [&str; 12] = [
    "C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B",
];
#[cfg(target_arch = "wasm32")]
const NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

pub fn midi_to_hz(m: f32) -> f32 {
    440.0 * ((m - 69.0) / 12.0).exp2()
}

/// Pitch class name, e.g. "F♯".
pub fn pitch_class_name(pc: i32) -> &'static str {
    NAMES[pc.rem_euclid(12) as usize]
}

/// Note name with octave, e.g. "A4".
pub fn note_name(m: i32) -> String {
    format!("{}{}", pitch_class_name(m), m.div_euclid(12) - 1)
}

/// Parse a key name like "C", "f#", "Bb", "E♭" into a pitch class.
pub fn parse_key(s: &str) -> Option<i32> {
    let mut chars = s.trim().chars();
    let base: i32 = match chars.next()?.to_ascii_uppercase() {
        'C' => 0,
        'D' => 2,
        'E' => 4,
        'F' => 5,
        'G' => 7,
        'A' => 9,
        'B' => 11,
        _ => return None,
    };
    let mut pc = base;
    for c in chars {
        match c {
            '#' | '♯' => pc += 1,
            'b' | '♭' => pc -= 1,
            _ => return None,
        }
    }
    Some(pc.rem_euclid(12))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scale {
    #[default]
    Chromatic,
    Major,
    Minor,
    Pentatonic,
    Blues,
    Dorian,
    Harmonic,
}

impl Scale {
    pub const ALL: [Scale; 7] = [
        Scale::Chromatic,
        Scale::Major,
        Scale::Minor,
        Scale::Pentatonic,
        Scale::Blues,
        Scale::Dorian,
        Scale::Harmonic,
    ];

    /// Semitones above the root.
    pub fn intervals(self) -> &'static [i32] {
        match self {
            Scale::Chromatic => &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            Scale::Major => &[0, 2, 4, 5, 7, 9, 11],
            Scale::Minor => &[0, 2, 3, 5, 7, 8, 10],
            Scale::Pentatonic => &[0, 2, 4, 7, 9],
            Scale::Blues => &[0, 3, 5, 6, 7, 10],
            Scale::Dorian => &[0, 2, 3, 5, 7, 9, 10],
            Scale::Harmonic => &[0, 2, 3, 5, 7, 8, 11],
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Scale::Chromatic => "chromatic",
            Scale::Major => "major",
            Scale::Minor => "minor",
            Scale::Pentatonic => "pentatonic",
            Scale::Blues => "blues",
            Scale::Dorian => "dorian",
            Scale::Harmonic => "harmonic minor",
        }
    }

    pub fn next(self) -> Self {
        let i = Self::ALL.iter().position(|&s| s == self).unwrap_or(0);
        Self::ALL[(i + 1) % Self::ALL.len()]
    }

    /// Whether MIDI note `m` belongs to this scale with the given root pitch class.
    pub fn contains(self, root: i32, m: i32) -> bool {
        self.intervals().contains(&(m - root).rem_euclid(12))
    }

    /// The scale note closest to the fractional pitch `m`.
    pub fn nearest(self, root: i32, m: f32) -> i32 {
        let center = m.round() as i32;
        (center - 6..=center + 6)
            .filter(|&n| self.contains(root, n))
            .min_by(|a, b| (*a as f32 - m).abs().total_cmp(&(*b as f32 - m).abs()))
            .unwrap_or(center)
    }

    /// Pull `m` toward the nearest scale note. `strength` 0 leaves it free, 1 quantizes fully.
    pub fn snap(self, root: i32, m: f32, strength: f32) -> f32 {
        if strength <= 0.0 {
            return m;
        }
        let n = self.nearest(root, m) as f32;
        n + (m - n) * (1.0 - strength.min(1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_frequencies() {
        assert_eq!(note_name(69), "A4");
        assert_eq!(note_name(60), "C4");
        assert!((midi_to_hz(69.0) - 440.0).abs() < 1e-3);
        assert!((midi_to_hz(81.0) - 880.0).abs() < 1e-2);
        assert_eq!(parse_key("F#"), Some(6));
        assert_eq!(parse_key("Bb"), Some(10));
        assert_eq!(parse_key("cb"), Some(11));
        assert_eq!(parse_key("H"), None);
    }

    #[test]
    fn snapping() {
        // C major: 61.4 is 1.4 above C (60) and 0.6 below D (62).
        assert_eq!(Scale::Major.nearest(0, 61.4), 62);
        assert_eq!(Scale::Major.nearest(0, 60.4), 60);
        assert_eq!(Scale::Pentatonic.nearest(0, 65.0), 64); // F is not in C pentatonic; E is closer than G
        assert_eq!(Scale::Major.snap(0, 61.4, 1.0), 62.0);
        assert!((Scale::Major.snap(0, 61.4, 0.5) - 61.7).abs() < 1e-4);
        assert_eq!(Scale::Major.snap(0, 61.4, 0.0), 61.4);
        // G major contains F♯.
        assert!(Scale::Major.contains(7, 66));
        assert!(!Scale::Major.contains(7, 65));
    }
}
