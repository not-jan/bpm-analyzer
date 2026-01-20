//! Core BPM analysis functionality.

use cpal::{
    BufferSize, FromSample, SampleFormat, SizedSample,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use crossbeam_channel::{Receiver, Sender, bounded};
use fundsp::prelude32::*;
use osclet::{BorderMode, DaubechiesFamily, Osclet};
use resampler::{Attenuation, Latency, ResamplerFir, SampleRate};
use ringbuffer::{AllocRingBuffer, RingBuffer};

use crate::{
    config::{AnalyzerConfig, DWT_LEVELS, TARGET_SAMPLING_RATE},
    dsp,
    error::{Error, Result},
    types::{BeatTiming, BpmDetection},
};

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

/// Starts the BPM analyzer with the given configuration using the default device.
///
/// This automatically selects an audio input device, preferring BlackHole on macOS,
/// then falling back to the system default input device.
///
/// # Arguments
///
/// * `config` - The analyzer configuration
///
/// # Returns
///
/// A `Receiver` that yields `BpmDetection` results containing up to 5 BPM candidates
/// sorted by confidence (highest first).
///
/// # Errors
///
/// Returns an error if:
/// - The configuration is invalid
/// - No suitable audio input device is found
/// - The audio stream cannot be configured or started
/// - The device's sample format is unsupported
///
/// # Example
///
/// ```no_run
/// use bpm_analyzer::{AnalyzerConfig, begin};
///
/// let config = AnalyzerConfig::electronic();
/// let receiver = begin(config)?;
///
/// for detection in receiver.iter() {
///     if let Some(bpm) = detection.bpm() {
///         println!("Detected BPM: {:.1}", bpm);
///     }
/// }
/// # Ok::<(), bpm_analyzer::Error>(())
/// ```
pub fn begin(config: AnalyzerConfig) -> Result<Receiver<BpmDetection>> {
    // Validate configuration
    config.validate()?;
    let host = cpal::default_host();

    let device = host
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

    begin_with_device(config, &device)
}

/// Starts the BPM analyzer with the given configuration and a specific audio device.
///
/// Use this function when you need to specify which audio device to use.
/// To list available devices, use [`list_audio_devices`](crate::list_audio_devices).
/// To get a device by name, use [`get_device_by_name`](crate::get_device_by_name).
///
/// # Arguments
///
/// * `config` - The analyzer configuration
/// * `device` - The audio input device to use
///
/// # Returns
///
/// A `Receiver` that yields `BpmDetection` results containing up to 5 BPM candidates
/// sorted by confidence (highest first).
///
/// # Errors
///
/// Returns an error if:
/// - The configuration is invalid
/// - The audio stream cannot be configured or started
/// - The device's sample format is unsupported
///
/// # Example
///
/// ```no_run
/// use bpm_analyzer::{AnalyzerConfig, begin_with_device, get_device_by_name};
///
/// let config = AnalyzerConfig::electronic();
/// let device = get_device_by_name("BlackHole 2ch")?;
/// let receiver = begin_with_device(config, &device)?;
///
/// for detection in receiver.iter() {
///     if let Some(bpm) = detection.bpm() {
///         println!("Detected BPM: {:.1}", bpm);
///     }
/// }
/// # Ok::<(), bpm_analyzer::Error>(())
/// ```
pub fn begin_with_device(
    config: AnalyzerConfig,
    device: &cpal::Device,
) -> Result<Receiver<BpmDetection>> {
    // Validate configuration
    config.validate()?;

    let device_name = device.description()?.name().to_string();

    tracing::info!("Using audio device: {}", device_name);

    let supported_config = device.default_input_config()?;

    let mut stream_config = supported_config.config();

    stream_config.buffer_size = BufferSize::Fixed(config.buffer_size());

    let sample_rate = stream_config.sample_rate as f64;

    tracing::info!(
        "Sampling with {:?} Hz on {} channels",
        stream_config.sample_rate,
        stream_config.channels
    );

    let (audio_sender, audio_receiver) = bounded(config.queue_size());
    let (bpm_sender, bpm_receiver) = bounded(config.queue_size());

    match supported_config.sample_format() {
        SampleFormat::F32 => run::<f32>(device, &stream_config, audio_sender)?,
        SampleFormat::I16 => run::<i16>(device, &stream_config, audio_sender)?,
        SampleFormat::U16 => run::<u16>(device, &stream_config, audio_sender)?,
        other => {
            return Err(Error::UnsupportedSampleFormat(other));
        }
    }

    std::thread::spawn(move || run_analysis(sample_rate, audio_receiver, bpm_sender, config));

    Ok(bpm_receiver)
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
    bpm_sender: Sender<BpmDetection>,
    config: AnalyzerConfig,
) -> Result<()> {
    let now = std::time::Instant::now();

    let dwt_executor = Osclet::make_daubechies_f32(DaubechiesFamily::Db4, BorderMode::Wrap);

    // Create resampler based on actual sample rate
    let input_sample_rate = match sample_rate as u32 {
        16000 => SampleRate::Hz16000,
        22050 => SampleRate::Hz22050,
        32000 => SampleRate::Hz32000,
        44100 => SampleRate::Hz44100,
        48000 => SampleRate::Hz48000,
        88200 => SampleRate::Hz88200,
        96000 => SampleRate::Hz96000,
        176400 => SampleRate::Hz176400,
        192000 => SampleRate::Hz192000,
        _ => return Err(Error::UnsupportedSampleRate(sample_rate as u32)),
    };

    let mut resampler = ResamplerFir::new(
        1,
        input_sample_rate,
        SampleRate::Hz22050,
        Latency::Sample64,
        Attenuation::Db90,
    );

    tracing::info!("Resampling buffer: {}", resampler.buffer_size_output());

    let resampling_factor = TARGET_SAMPLING_RATE / sample_rate;
    let window_length = config.window_size() as f64 / TARGET_SAMPLING_RATE;

    tracing::info!(
        "Analysis window: {} samples ({:.2} seconds)",
        config.window_size(),
        window_length
    );

    tracing::info!(
        "Resampling factor: {}, every {}th sample",
        resampling_factor,
        (sample_rate / TARGET_SAMPLING_RATE).round()
    );

    let mut ring_buffer = AllocRingBuffer::<f32>::new(config.window_size());

    let once = std::sync::Once::new();

    let mut filter_chain = dsp::alpha_lpf(0.99f32) >> dsp::fwr::<f32>();

    let mut resampled_output = vec![0.0f32; resampler.buffer_size_output()];

    // Pre-allocate buffers to reduce allocations in hot loop
    let mut input_buffer = Vec::with_capacity(4096);
    let mut signal = vec![0.0f32; config.window_size()];
    let mut bands = vec![vec![0.0f32; 4096]; DWT_LEVELS];
    let mut summed_bands = vec![0.0f32; 4096];
    let mut peaks_buffer = Vec::with_capacity(1024);
    
    // Beat detection state
    let mut beat_timings: Vec<BeatTiming> = Vec::with_capacity(8);
    let mut prev_summed_bands = vec![0.0f32; 4096];
    let mut samples_processed = 0usize;
    let mut current_bpm: Option<f32> = None;

    loop {
        // Read all available audio samples from the channel
        input_buffer.clear();
        input_buffer.extend(
            audio_receiver
                .try_iter()
                // Mix to mono
                .map(|(l, r)| (l + r) * 0.5),
        );

        let mut input_slice = &input_buffer[..];

        while !input_slice.is_empty() {
            let (consumed, produced) = resampler
                .resample(input_slice, &mut resampled_output)
                .map_err(Error::ResampleError)?;
            ring_buffer.extend(resampled_output[..produced].iter().copied());
            samples_processed += produced;

            input_slice = &input_slice[consumed..];
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

            // Copy ring buffer into pre-allocated signal buffer
            signal = ring_buffer.to_vec();

            let dwt = dwt_executor.multi_dwt(&signal, DWT_LEVELS)?;

            // Process each band and store in pre-allocated buffers
            for (band_idx, level) in dwt.levels.into_iter().enumerate() {
                filter_chain.reset();

                // Process approximations through filter chain
                let mut processed_samples = Vec::with_capacity(level.approximations.len());

                for chunk in level.approximations.chunks(MAX_BUFFER_SIZE) {
                    let transient_buffer = if chunk.len() == MAX_BUFFER_SIZE {
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
                    };

                    let (mut output, length) = transient_buffer.process(&mut filter_chain);
                    processed_samples.extend_from_slice(&output.channel_f32(0)[..length]);
                }

                let downsampling_factor = 1 << (DWT_LEVELS - 1 - band_idx);

                // Downsample and store in pre-allocated band buffer
                let band_buffer = &mut bands[band_idx];
                band_buffer.fill(0.0);

                let downsampled_len = processed_samples.len() / downsampling_factor;
                let samples_to_copy = std::cmp::min(downsampled_len, 4096);

                for (i, sample_idx) in (0..processed_samples.len())
                    .step_by(downsampling_factor)
                    .take(samples_to_copy)
                    .enumerate()
                {
                    band_buffer[i] = processed_samples[sample_idx];
                }

                // Normalization (mean removal)
                if samples_to_copy > 0 {
                    let mean: f32 =
                        band_buffer[..samples_to_copy].iter().sum::<f32>() / samples_to_copy as f32;
                    band_buffer[..samples_to_copy]
                        .iter_mut()
                        .for_each(|sample| *sample -= mean);
                }
            }

            // Sum bands into pre-allocated buffer with frequency weighting
            // Weight higher frequencies more for better beat detection across the spectrum
            // Band 0 (highest): 2.0x weight for high-frequency transients (hi-hats, cymbals)
            // Band 1 (high):    1.5x weight for mid-high transients (snares)
            // Band 2 (mid):     1.0x weight for mid-range (claps, vocals)
            // Band 3 (low):     0.5x weight for low-end (kicks) - reduced to balance energy
            summed_bands.fill(0.0);
            for i in 0..4096 {
                summed_bands[i] = bands[0][i] * 2.0  // Highest frequencies
                                + bands[1][i] * 1.5  // High frequencies
                                + bands[2][i] * 1.0  // Mid frequencies
                                + bands[3][i] * 0.5; // Low frequencies
            }
            
            // Simple but effective beat detection using energy differences
            // Calculate onset strength for each point and find peaks
            let mut onset_strengths = Vec::with_capacity(4096);
            
            for i in 0..summed_bands.len() {
                // Calculate increase in energy from previous frame
                let onset = (summed_bands[i] - prev_summed_bands[i]).max(0.0);
                onset_strengths.push(onset);
            }
            
            // Use 90th percentile instead of median for stricter threshold
            let mut sorted_onsets = onset_strengths.clone();
            sorted_onsets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let percentile_90 = sorted_onsets[(sorted_onsets.len() * 9) / 10];
            
            // Threshold is 1.5x the 90th percentile - captures strong onsets only
            let threshold = (percentile_90 * 1.5).max(0.05);
            
            // Peak picking: find prominent local maxima
            for i in 10..(onset_strengths.len() - 10) {
                let current = onset_strengths[i];
                
                // Stricter local maximum check over a larger 21-sample window
                // Must be stronger than all neighbors within 10 samples
                let is_local_max = (i.saturating_sub(10)..i).all(|j| current > onset_strengths[j])
                    && ((i + 1)..=std::cmp::min(i + 10, onset_strengths.len() - 1)).all(|j| current >= onset_strengths[j]);
                
                if is_local_max && current > threshold {
                    // Calculate the time of this beat
                    let beat_sample = samples_processed - config.window_size() + (i * (config.window_size() / 4096));
                    let beat_time = beat_sample as f64 / TARGET_SAMPLING_RATE;
                    
                    // Tempo-based validation: only enforce after we have at least 3 beats
                    // This allows the system to bootstrap and adapt to tempo changes
                    let mut tempo_valid = beat_timings.len() < 3; // Always allow first 3 beats
                    
                    if !tempo_valid {
                        if let Some(bpm) = current_bpm {
                            if let Some(last_beat) = beat_timings.last() {
                                let interval = beat_time - last_beat.time_seconds;
                                let expected_interval = 60.0 / bpm as f64;
                                
                                // Allow ±30% deviation from expected interval
                                // Also accept half-time (2x interval) and double-time (0.5x interval)
                                let deviation = (interval / expected_interval).abs();
                                tempo_valid = (0.7..=1.3).contains(&deviation)  // Normal time (±30%)
                                    || (1.7..=2.3).contains(&deviation)         // Half time
                                    || (0.4..=0.6).contains(&deviation);        // Double time
                            } else {
                                tempo_valid = true; // No previous beat to compare
                            }
                        } else {
                            tempo_valid = true; // No BPM reference yet
                        }
                    }
                    
                    // Add beat if it's not too close to the previous one (0.15s min spacing)
                    let should_add = beat_timings.last()
                        .map(|last: &BeatTiming| (beat_time - last.time_seconds) > 0.15)
                        .unwrap_or(true);
                    
                    if should_add && tempo_valid {
                        let normalized_strength = (current / threshold).min(2.0);
                        beat_timings.push(BeatTiming::new(beat_time, normalized_strength));
                        
                        // Keep only the last 8 beats
                        if beat_timings.len() > 8 {
                            beat_timings.remove(0);
                        }
                    }
                }
            }
            
            // Store current bands for next iteration
            prev_summed_bands.copy_from_slice(&summed_bands);

            let min_lag = ((4096.0 / window_length) * 60.0 / config.max_bpm() as f64) as usize;
            let max_lag = ((4096.0 / window_length) * 60.0 / config.min_bpm() as f64) as usize;

            let ac = autocorrelation(&summed_bands, max_lag);

            // Reuse peaks buffer
            peaks_buffer.clear();
            peaks_buffer.extend(
                ac.iter()
                    .enumerate()
                    .skip(min_lag)
                    .take(max_lag - min_lag)
                    .map(|(idx, &val)| (idx, val)),
            );
            peaks_buffer
                .sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

            // Skip the first peak (which is usually the zero-lag) and take up to 5 peaks
            let peak_count = std::cmp::Ord::min(peaks_buffer.len().saturating_sub(1), 5);
            if peak_count > 0 {
                let mut result = [(0.0f32, 0.0f32); 5];
                for (i, &(lag, v)) in peaks_buffer[1..=peak_count].iter().enumerate() {
                    let bpm = (60.0 * (4096.0 / window_length as f32)) / (lag as f32);
                    result[i] = (bpm, v);
                }
                
                // Update current BPM for beat validation (use the highest confidence BPM)
                // If BPM changes significantly (>10%), reset beat timings to allow re-syncing
                if result[0].0 > 0.0 {
                    if let Some(old_bpm) = current_bpm {
                        let bpm_change = ((result[0].0 - old_bpm) / old_bpm).abs();
                        if bpm_change > 0.1 {
                            // Significant BPM change detected (song change or tempo shift)
                            // Keep only the last beat to maintain continuity
                            if !beat_timings.is_empty() {
                                let last = beat_timings.last().cloned().unwrap();
                                beat_timings.clear();
                                beat_timings.push(last);
                            }
                        }
                    }
                    current_bpm = Some(result[0].0);
                }

                let _ = bpm_sender.try_send(BpmDetection::with_beats(result, beat_timings.clone()));
            }
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

/// Callback function to read audio data from the input device
/// and sends it to the provided channel sender.
fn read_data<T>(input: &[T], channels: usize, sender: Sender<(f32, f32)>)
where
    T: SizedSample,
    f32: FromSample<T>,
{
    for frame in input.chunks(channels) {
        let left = if !frame.is_empty() {
            frame[0].to_sample::<f32>()
        } else {
            0.0
        };

        let right = if channels > 1 && frame.len() > 1 {
            frame[1].to_sample::<f32>()
        } else {
            left // For mono, duplicate to both channels
        };

        let _ = sender.try_send((left, right));
    }
}

/// Computes the autocorrelation of a signal for lags [0, max_lag).
///
/// # Arguments
///
/// * `signal` - Input signal (e.g., summed band envelopes)
/// * `max_lag` - Maximum lag in samples to compute
///
/// # Returns
///
/// A Vec<f32> of autocorrelation values for each lag
fn autocorrelation(signal: &[f32], max_lag: usize) -> Vec<f32> {
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
