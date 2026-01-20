//! Configuration types for the BPM analyzer.

use crate::error::{Error, Result};

/// Size of the cross-thread audio queue
const QUEUE_SIZE: usize = 4096;
/// Number of levels for discrete wavelet transform decomposition
pub(crate) const DWT_LEVELS: usize = 4;
/// Size of the audio buffer for capture
const AUDIO_BUFFER_SIZE: u32 = 256;
/// Window size for DWT analysis (must be power of 2)
const DWT_WINDOW_SIZE: usize = 65536;
/// Target sampling rate for analysis (Hz)
pub(crate) const TARGET_SAMPLING_RATE: f64 = 22050.0;

/// Default minimum BPM for detection range
const MIN_BPM: f32 = 40.0;
/// Default maximum BPM for detection range
const MAX_BPM: f32 = 240.0;

/// Configuration for the BPM analyzer.
///
/// Use the builder pattern to customize analyzer parameters:
///
/// # Example
///
/// ```
/// use bpm_analyzer::AnalyzerConfig;
///
/// let config = AnalyzerConfig::builder()
///     .min_bpm(60.0)
///     .max_bpm(180.0)
///     .window_size(32768)
///     .build();
/// ```
#[derive(Clone, Debug, Copy, bon::Builder)]
pub struct AnalyzerConfig {
    /// Minimum BPM to detect (default: 40.0)
    #[builder(default = MIN_BPM)]
    min_bpm: f32,
    /// Maximum BPM to detect (default: 240.0)
    #[builder(default = MAX_BPM)]
    max_bpm: f32,
    /// Size of the analysis window in samples (default: 65536)
    #[builder(default = DWT_WINDOW_SIZE)]
    window_size: usize,
    /// Size of the audio queue between threads (default: 4096)
    #[builder(default = QUEUE_SIZE)]
    queue_size: usize,
    /// Size of the audio capture buffer (default: 256)
    #[builder(default = AUDIO_BUFFER_SIZE)]
    buffer_size: u32,
}

impl AnalyzerConfig {
    /// Creates a configuration preset optimized for electronic music (100-160 BPM).
    pub fn electronic() -> Self {
        Self::builder().min_bpm(100.0).max_bpm(160.0).build()
    }

    /// Creates a configuration preset optimized for hip-hop (80-110 BPM).
    pub fn hip_hop() -> Self {
        Self::builder().min_bpm(80.0).max_bpm(110.0).build()
    }

    /// Creates a configuration preset optimized for classical music (40-100 BPM).
    pub fn classical() -> Self {
        Self::builder().min_bpm(40.0).max_bpm(100.0).build()
    }

    /// Creates a configuration preset optimized for rock/pop (110-140 BPM).
    pub fn rock_pop() -> Self {
        Self::builder().min_bpm(110.0).max_bpm(140.0).build()
    }

    /// Returns the minimum BPM setting.
    pub fn min_bpm(&self) -> f32 {
        self.min_bpm
    }

    /// Returns the maximum BPM setting.
    pub fn max_bpm(&self) -> f32 {
        self.max_bpm
    }

    /// Returns the window size setting.
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Returns the queue size setting.
    pub fn queue_size(&self) -> usize {
        self.queue_size
    }

    /// Returns the buffer size setting.
    pub fn buffer_size(&self) -> u32 {
        self.buffer_size
    }

    /// Validates the configuration and returns an error if invalid.
    pub fn validate(&self) -> Result<()> {
        if self.min_bpm <= 0.0 {
            return Err(Error::InvalidConfig("min_bpm must be positive".to_string()));
        }
        if self.max_bpm <= self.min_bpm {
            return Err(Error::InvalidConfig(
                "max_bpm must be greater than min_bpm".to_string(),
            ));
        }
        if self.window_size == 0 || !self.window_size.is_power_of_two() {
            return Err(Error::InvalidConfig(
                "window_size must be a power of 2".to_string(),
            ));
        }
        if self.queue_size == 0 {
            return Err(Error::InvalidConfig(
                "queue_size must be positive".to_string(),
            ));
        }
        if self.buffer_size == 0 {
            return Err(Error::InvalidConfig(
                "buffer_size must be positive".to_string(),
            ));
        }
        Ok(())
    }
}
