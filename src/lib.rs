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
//! for peaks in bpm_receiver.iter() {
//!     // Each entry contains (bpm, confidence) pairs
//!     if let Some((bpm, confidence)) = peaks.first() {
//!         println!("Detected BPM: {} (confidence: {})", bpm, confidence);
//!     }
//! }
//! ```

use cpal::{
    BufferSize, FromSample, SampleFormat, SizedSample,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use crossbeam_channel::{Sender, bounded};
use fundsp::prelude32::*;
use osclet::{BorderMode, DaubechiesFamily, Osclet};
use resampler::{Attenuation, Latency, ResamplerFir, SampleRate};
use ringbuffer::{AllocRingBuffer, RingBuffer};

pub mod dsp;

/// Size of the cross-thread audio queue
/// Size of the cross-thread audio queue
const QUEUE_SIZE: usize = 4096;
/// Number of levels for discrete wavelet transform decomposition
const DWT_LEVELS: usize = 4;
/// Size of the audio buffer for capture
const AUDIO_BUFFER_SIZE: u32 = 256;
/// Window size for DWT analysis (must be power of 2)
const DWT_WINDOW_SIZE: usize = 65536;
/// Target sampling rate for analysis (Hz)
const TARGET_SAMPLING_RATE: f64 = 22050.0;

/// Default minimum BPM for detection range
const MIN_BPM: f32 = 40.0;
/// Default maximum BPM for detection range
const MAX_BPM: f32 = 240.0;

/// Internal buffer wrapper for processing audio chunks of varying sizes.
///
/// This enum handles both full-sized buffers (aligned for SIMD processing)
/// and partial buffers (for the last chunk that may be smaller than the max buffer size).
#[allow(clippy::large_enum_variant)]
enum TransientBuffer<'a> {
    /// A full buffer that can be processed using SIMD operations
    Full(BufferRef<'a>),
    /// A partial buffer for the remaining samples
    Partial {
        /// The buffer containing the samples
        buffer: BufferArray<U1>,
        /// The actual number of valid samples in the buffer
        length: usize,
    },
}

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
    #[error("Failed to get stream configuration: {0}")]
    StreamConfigError(#[from] cpal::DefaultStreamConfigError),
    #[error("Resampling error: {0}")]
    ResampleError(resampler::ResampleError),
    #[error("Osclet error: {0}")]
    Osclet(#[from] osclet::OscletError),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

pub use crossbeam_channel::Receiver;

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

/// Starts the BPM analyzer with the given configuration.
///
/// This function initializes the audio input stream and spawns a background thread
/// that performs real-time BPM analysis. It returns a receiver that yields arrays of
/// BPM candidates with their confidence values.
///
/// # Arguments
///
/// * `config` - The analyzer configuration
///
/// # Returns
///
/// A `Receiver` that yields arrays of 5 `(bpm, confidence)` tuples, where:
/// - `bpm`: The detected tempo in beats per minute
/// - `confidence`: The autocorrelation value indicating detection confidence
///
/// The results are sorted by confidence (highest first).
///
/// # Errors
///
/// Returns an error if:
/// - No suitable audio input device is found
/// - The audio stream cannot be configured or started
/// - The device's sample format is unsupported
///
/// # Example
///
/// ```no_run
/// use bpm_analyzer::{AnalyzerConfig, begin};
///
/// let config = AnalyzerConfig::builder().build();
/// let receiver = begin(config)?;
///
/// for peaks in receiver.iter() {
///     println!("Top BPM candidate: {:.1}", peaks[0].0);
/// }
/// # Ok::<(), bpm_analyzer::Error>(())
/// ```
pub fn begin(config: AnalyzerConfig) -> Result<crossbeam_channel::Receiver<[(f32, f32); 5]>> {
    let host = cpal::default_host();

    let loopback_device = host
        .input_devices()?
        .find_map(|device| match device.description() {
            #[cfg(target_os = "macos")]
            Ok(desc) if desc.name().contains("BlackHole") => Some(Ok(device)),
            Err(e) => Some(Err(e)),
            Ok(_) => None,
        })
        .transpose()?
        .or_else(|| host.default_input_device())
        .ok_or(Error::NoDeviceFound)?;

    let device_name = loopback_device.description()?.name().to_string();

    tracing::info!("Using audio device: {}", device_name);

    let supported_config = loopback_device.default_input_config()?;

    let mut stream_config = supported_config.config();

    stream_config.buffer_size = BufferSize::Fixed(config.buffer_size);

    let sample_rate = stream_config.sample_rate as f64;

    tracing::info!(
        "Sampling with {:?} Hz on {} channels",
        stream_config.sample_rate,
        stream_config.channels
    );

    let (audio_sender, audio_receiver) = bounded(config.queue_size);
    let (bmp_sender, bpm_receiver) = bounded(config.queue_size);

    match supported_config.sample_format() {
        SampleFormat::F32 => run::<f32>(&loopback_device, &stream_config, audio_sender)?,
        SampleFormat::I16 => run::<i16>(&loopback_device, &stream_config, audio_sender)?,
        SampleFormat::U16 => run::<u16>(&loopback_device, &stream_config, audio_sender)?,
        other => {
            return Err(Error::UnsupportedSampleFormat(other));
        }
    }

    std::thread::spawn(move || run_analysis(sample_rate, audio_receiver, bmp_sender, config));

    Ok(bpm_receiver)
}

impl<'a> TransientBuffer<'a> {
    /// Processes this buffer through an audio processing node.
    ///
    /// # Returns
    ///
    /// A tuple containing the processed buffer and the number of valid samples.
    fn process<N: AudioUnit>(&'a self, node: &mut N) -> (BufferArray<U1>, usize) {
        match self {
            TransientBuffer::Full(buffer_ref) => {
                let mut buffer = BufferArray::<U1>::new();

                node.process(MAX_BUFFER_SIZE, buffer_ref, &mut buffer.buffer_mut());

                (buffer, MAX_BUFFER_SIZE)
            }
            TransientBuffer::Partial { buffer, length } => {
                let mut output_buffer = BufferArray::<U1>::new();

                node.process(
                    *length,
                    &buffer.buffer_ref(),
                    &mut output_buffer.buffer_mut(),
                );

                (output_buffer, *length)
            }
        }
    }
}

/// Main analysis loop that processes audio samples and detects BPM.
///
/// This function:
/// 1. Resamples audio to 22.05 kHz
/// 2. Accumulates samples in a ring buffer
/// 3. Performs multi-level discrete wavelet transform
/// 4. Extracts onset envelopes from each frequency band
/// 5. Computes autocorrelation to find periodic patterns
/// 6. Identifies BPM candidates based on peak autocorrelation values
///
/// # Arguments
///
/// * `sample_rate` - Original audio sample rate
/// * `audio_receiver` - Channel receiving stereo audio samples
/// * `bpm_sender` - Channel to send BPM detection results
/// * `config` - Analyzer configuration
fn run_analysis(
    sample_rate: f64,
    audio_receiver: Receiver<(f32, f32)>,
    bpm_sender: Sender<[(f32, f32); 5]>,
    config: AnalyzerConfig,
) -> Result<()> {
    let now = std::time::Instant::now();

    let dwt_executor = Osclet::make_daubechies_f32(DaubechiesFamily::Db4, BorderMode::Wrap);

    let mut resampler = ResamplerFir::new(
        1,
        SampleRate::Hz48000,
        SampleRate::Hz22050,
        Latency::Sample64,
        Attenuation::Db90,
    );

    tracing::info!("Resampling buffer: {}", resampler.buffer_size_output());

    let resampling_factor = TARGET_SAMPLING_RATE / sample_rate;
    tracing::info!(
        "Resampling factor: {}, every {}th sample",
        resampling_factor,
        (sample_rate / TARGET_SAMPLING_RATE).round()
    );

    let mut ring_buffer = AllocRingBuffer::<f32>::new(config.window_size);

    let once = std::sync::Once::new();

    let mut filter_chain = dsp::alpha_lpf(0.99f32) >> dsp::fwr::<f32>();

    let mut resampled_output = vec![0.0f32; resampler.buffer_size_output()];

    loop {
        // Read all available audio samples from the channel
        let input = audio_receiver
            .try_iter()
            // Mix to mono
            .map(|(l, r)| (l + r) * 0.5)
            .collect::<Vec<_>>();

        let mut input = &input[..];

        while !input.is_empty() {
            let (consumed, produced) = resampler
                .resample(input, &mut resampled_output)
                .map_err(Error::ResampleError)?;
            ring_buffer.extend(resampled_output[..produced].iter().copied());

            input = &input[consumed..];
        }

        if ring_buffer.is_full() {
            once.call_once(|| {
                let time = now.elapsed();
                tracing::info!(
                    "Initial audio buffer filled with {} samples in {:.2?}",
                    ring_buffer.len(),
                    time
                );
            });

            let signal = ring_buffer.to_vec();

            let dwt = dwt_executor.multi_dwt(&signal, DWT_LEVELS)?;

            let bands = dwt
                .levels
                .into_iter()
                .enumerate()
                .map(|(i, band)| {
                    filter_chain.reset();

                    let band = band
                        .approximations
                        .chunks(MAX_BUFFER_SIZE)
                        .map(|chunk| {
                            if chunk.len() == MAX_BUFFER_SIZE {
                                let buffer = unsafe {
                                    std::slice::from_raw_parts::<'_, F32x>(
                                        chunk.as_ptr() as *const _,
                                        MAX_BUFFER_SIZE / SIMD_LEN,
                                    )
                                };

                                TransientBuffer::Full(BufferRef::new(buffer))
                            } else {
                                let mut buffer = BufferArray::<U1>::new();
                                buffer.channel_f32_mut(0)[..chunk.len()].copy_from_slice(chunk);

                                TransientBuffer::Partial {
                                    buffer,
                                    length: chunk.len(),
                                }
                            }
                        })
                        .map(|buffer| buffer.process(&mut filter_chain))
                        .flat_map(|(mut band, length)| {
                            band.channel_f32(0)
                                .iter()
                                .copied()
                                .take(length)
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();

                    let downsampling_factor = 1 << (3 - i);

                    let mut band = (0..band.len())
                        .step_by(downsampling_factor)
                        .map(|i| band[i])
                        .collect::<Vec<_>>();

                    let mean = band.iter().copied().sum::<f32>() / band.len() as f32;

                    // Normalization in each band (mean removal)
                    band.iter_mut().for_each(|sample| *sample -= mean);

                    band.resize(4096, 0.0);

                    band
                })
                .collect::<Vec<_>>();

            let summed_bands = (0..4096)
                .map(|i| bands.iter().map(|band| band[i]).sum::<f32>())
                .collect::<Vec<_>>();

            let min_lag = ((4096.0 / 3.0) * 60.0 / config.max_bpm) as usize;
            let max_lag = ((4096.0 / 3.0) * 60.0 / config.min_bpm) as usize;

            let ac = autocorrelation(&summed_bands, max_lag);

            let mut peaks = ac
                .into_iter()
                .enumerate()
                .skip(min_lag)
                .take(max_lag - min_lag)
                .collect::<Vec<_>>();
            peaks.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

            let peaks = &peaks[1..6];

            let peaks = peaks
                .iter()
                .copied()
                .map(|(lag, v)| ((60.0 * (4096.0 / 3.0)) / (lag as f32), v))
                .collect::<Vec<_>>();

            if let Ok(()) = bpm_sender.try_send(peaks.try_into().unwrap()) {}
        }
    }
}

/// Starts the audio input stream on the given device with the given configuration,
/// sending audio samples to the provided channel sender.
fn run<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sender: Sender<(f32, f32)>,
) -> Result<()>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let channels = config.channels as usize;
    let err_fn = |err| tracing::error!("an error occurred on stream: {}", err);
    let stream = device.build_input_stream(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| read_data(data, channels, sender.clone()),
        err_fn,
        None,
    );
    if let Ok(stream) = stream
        && let Ok(()) = stream.play()
    {
        std::mem::forget(stream);
    }

    tracing::info!("Input stream built.");

    Ok(())
}

/// Callback function to read audio data from the input device.
/// and sends it to the provided channel sender.
fn read_data<T>(input: &[T], channels: usize, sender: Sender<(f32, f32)>)
where
    T: SizedSample,
    f32: FromSample<T>,
{
    for frame in input.chunks(channels) {
        let mut left = 0.0;
        let mut right = 0.0;
        for (channel, sample) in frame.iter().enumerate() {
            if channel & 1 == 0 {
                left = sample.to_sample::<f32>();
            } else {
                right = sample.to_sample::<f32>();
            }
        }

        if let Ok(()) = sender.try_send((left, right)) {}
    }
}

/// Computes the autocorrelation of a signal for lags [0, max_lag).
/// Returns a Vec<f32> of length max_lag.
///
/// signal: input signal (e.g., summed band envelopes)
/// max_lag: maximum lag in samples to compute
pub fn autocorrelation(signal: &[f32], max_lag: usize) -> Vec<f32> {
    let n = signal.len();

    let max_lag = std::cmp::Ord::min(max_lag, n);

    let mut ac = vec![0.0f32; max_lag];

    for lag in 0..max_lag {
        let mut sum = 0.0f32;

        for i in 0..(n - lag) {
            sum += signal[i] * signal[i + lag];
        }

        ac[lag] = sum / n as f32;
    }

    ac
}
