//! Error types for the BPM analyzer.

use cpal::SampleFormat;

/// Errors that can occur during BPM analysis
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("CPAL error: {0}")]
    DevicesError(#[from] cpal::DevicesError),
    #[error("Unsupported sample format: {0}")]
    UnsupportedSampleFormat(SampleFormat),
    #[error("Failed to get device name: {0}")]
    DeviceNameError(#[from] cpal::DeviceNameError),
    #[error("No input device found")]
    NoDeviceFound,
    #[error("Device not found: {0}")]
    DeviceNotFound(String),
    #[error("Failed to get stream configuration: {0}")]
    StreamConfigError(#[from] cpal::DefaultStreamConfigError),
    #[error("Resampling error: {0}")]
    ResampleError(resampler::ResampleError),
    #[error("Osclet error: {0}")]
    Osclet(#[from] osclet::OscletError),
    #[error("Unsupported sample rate: {0} Hz")]
    UnsupportedSampleRate(u32),
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Result type alias for BPM analyzer operations
pub type Result<T, E = Error> = std::result::Result<T, E>;
