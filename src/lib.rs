//! # BPM Analyzer
//!
//! A real-time BPM (beats per minute) detection library that analyzes audio input
//! using wavelet decomposition and autocorrelation techniques.
//!
//! ## Features
//!
//! - Real-time audio capture from system audio devices
//! - Wavelet-based onset detection using Discrete Wavelet Transform (DWT)
//! - Multi-band envelope analysis
//! - Autocorrelation-based tempo estimation
//! - Configurable BPM range and analysis parameters
//!
//! ## Example
//!
//! ```no_run
//! use bpm_analyzer::{AnalyzerConfig, begin};
//!
//! // Configure the analyzer with default settings
//! let config = AnalyzerConfig::builder()
//!     .min_bpm(60.0)
//!     .max_bpm(180.0)
//!     .build();
//!
//! // Start the analyzer and receive BPM candidates
//! let bpm_receiver = begin(config).expect("Failed to start analyzer");
//!
//! // Process BPM candidates
//! for detection in bpm_receiver.iter() {
//!     if let Some(bpm) = detection.bpm() {
//!         println!("Detected BPM: {:.1}", bpm);
//!     }
//! }
//! ```

// Module declarations
pub mod analyzer;
pub mod config;
pub mod device;
pub mod dsp;
pub mod error;
pub mod types;

// Re-exports for public API
pub use analyzer::{begin, begin_with_device};
pub use config::AnalyzerConfig;
pub use crossbeam_channel::Receiver;
pub use device::{AudioDevice, get_default_device, get_device_by_name, list_audio_devices};
pub use error::{Error, Result};
pub use types::{BeatTiming, BpmCandidate, BpmDetection};
