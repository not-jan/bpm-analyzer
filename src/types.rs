//! Core types for BPM detection results.

/// A single beat timing event.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeatTiming {
    /// Time in seconds since analysis started
    pub time_seconds: f64,
    /// Strength of the beat (0.0 to 1.0)
    pub strength: f32,
}

impl BeatTiming {
    /// Creates a new beat timing.
    pub fn new(time_seconds: f64, strength: f32) -> Self {
        Self {
            time_seconds,
            strength,
        }
    }
}

/// A BPM detection candidate with its confidence value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BpmCandidate {
    /// The detected tempo in beats per minute
    pub bpm: f32,
    /// Confidence value from autocorrelation (higher = more confident)
    pub confidence: f32,
}

impl BpmCandidate {
    /// Creates a new BPM candidate.
    pub fn new(bpm: f32, confidence: f32) -> Self {
        Self { bpm, confidence }
    }

    /// Returns true if this candidate's BPM is within the given range.
    pub fn in_range(&self, min: f32, max: f32) -> bool {
        self.bpm >= min && self.bpm <= max
    }
}

impl From<(f32, f32)> for BpmCandidate {
    fn from((bpm, confidence): (f32, f32)) -> Self {
        Self::new(bpm, confidence)
    }
}

/// Result from BPM detection containing up to 5 candidates sorted by confidence.
#[derive(Debug, Clone)]
pub struct BpmDetection {
    candidates: Vec<BpmCandidate>,
    /// Recent beat timings (up to last 8 beats)
    beat_timings: Vec<BeatTiming>,
}

impl BpmDetection {
    /// Creates a new detection result from an array of candidates.
    pub fn from_array(arr: [(f32, f32); 5]) -> Self {
        let candidates = arr
            .into_iter()
            .filter(|(bpm, _)| *bpm > 0.0) // Filter out padding zeros
            .map(BpmCandidate::from)
            .collect();
        Self {
            candidates,
            beat_timings: Vec::new(),
        }
    }

    /// Creates a new detection result with beat timings.
    pub fn with_beats(arr: [(f32, f32); 5], beat_timings: Vec<BeatTiming>) -> Self {
        let candidates = arr
            .into_iter()
            .filter(|(bpm, _)| *bpm > 0.0)
            .map(BpmCandidate::from)
            .collect();
        Self {
            candidates,
            beat_timings,
        }
    }

    /// Returns the top BPM candidate (highest confidence).
    pub fn top_candidate(&self) -> Option<&BpmCandidate> {
        self.candidates.first()
    }

    /// Returns all candidates.
    pub fn candidates(&self) -> &[BpmCandidate] {
        &self.candidates
    }

    /// Returns the most likely BPM value.
    pub fn bpm(&self) -> Option<f32> {
        self.top_candidate().map(|c| c.bpm)
    }

    /// Returns candidates within a specific BPM range.
    pub fn in_range(&self, min: f32, max: f32) -> Vec<BpmCandidate> {
        self.candidates
            .iter()
            .filter(|c| c.in_range(min, max))
            .copied()
            .collect()
    }

    /// Returns the detected beat timings.
    pub fn beat_timings(&self) -> &[BeatTiming] {
        &self.beat_timings
    }

    /// Returns the most recent beat timing, if available.
    pub fn last_beat(&self) -> Option<&BeatTiming> {
        self.beat_timings.last()
    }

    /// Returns the time interval between the last two beats, if available.
    pub fn last_beat_interval(&self) -> Option<f64> {
        if self.beat_timings.len() >= 2 {
            let len = self.beat_timings.len();
            Some(self.beat_timings[len - 1].time_seconds - self.beat_timings[len - 2].time_seconds)
        } else {
            None
        }
    }
}
