//! Digital signal processing nodes for audio analysis.
//!
//! This module provides custom audio processing nodes built on top of the fundsp library,
//! specifically designed for onset detection and envelope extraction:
//!
//! - [`Input`]: Audio input node that receives samples from a channel
//! - [`AlphaLpf`]: Single-pole low-pass filter for envelope smoothing
//! - [`FullWaveRectification`]: Full-wave rectifier for onset detection
//!
//! # Example
//!
//! ```rust,ignore
//! use bpm_analyzer::dsp::*;
//!
//! // Create an onset detection chain:
//! // lowpass filter -> full-wave rectification
//! let onset_detector = alpha_lpf(0.99) >> fwr::<f32>();
//! ```

use crossbeam_channel::Receiver;
use fundsp::prelude32::*;

/// Base ID for custom audio nodes
const BASE_ID: u64 = 0x1337;

/// Audio input node that receives samples from a crossbeam channel.
/// 
/// This node pulls stereo audio data from a channel receiver and outputs it
/// as a 2-channel audio stream. It's useful for integrating external audio sources
/// into a fundsp processing graph.
#[derive(Clone)]
pub struct Input<F: Real> {
    receiver: Receiver<(F, F)>,
}

impl<F: Real> AudioNode for Input<F> {
    const ID: u64 = BASE_ID;

    type Inputs = U0;

    type Outputs = U2;

    /// Receives one stereo frame from the channel.
    /// Returns zeros if the channel is disconnected or empty.
    #[inline]
    fn tick(&mut self, _input: &Frame<f32, Self::Inputs>) -> Frame<f32, Self::Outputs> {
        let (left, right) = self
            .receiver
            .recv()
            .inspect_err(|e| {
                tracing::error!(error =? e);
            })
            .unwrap_or_else(|_| (F::zero(), F::zero()));
        [convert(left), convert(right)].into()
    }
}

/// Creates an audio input node from a channel receiver.
/// 
/// # Arguments
/// 
/// * `receiver` - A channel receiver providing stereo `(left, right)` audio samples
/// 
/// # Returns
/// 
/// A 2-channel audio node that outputs the received samples
pub fn input<F: Real>(receiver: Receiver<(F, F)>) -> An<Input<F>> {
    An(Input { receiver })
}

/// Single-pole low-pass filter with configurable smoothing coefficient.
/// 
/// This filter implements a simple recursive smoothing formula:
/// `y[n] = (1 - α) * x[n] + α * x[n-1]`
/// 
/// where α (alpha) controls the smoothing:
/// - α close to 0: minimal smoothing (faster response)
/// - α close to 1: heavy smoothing (slower response)
/// 
/// Commonly used for envelope following and smoothing onset detection signals.
#[derive(Clone)]
pub struct AlphaLpf<F: Real> {
    /// Smoothing coefficient (0.0 to 1.0)
    alpha: F,
    /// Previous input sample
    previous: F,
}

impl<F: Real> AudioNode for AlphaLpf<F> {
    const ID: u64 = BASE_ID + 1;

    type Inputs = U1;

    type Outputs = U1;

    /// Processes one sample through the low-pass filter.
    #[inline]
    fn tick(&mut self, input: &Frame<f32, Self::Inputs>) -> Frame<f32, Self::Outputs> {
        let x = convert(input[0]);
        let value = (F::one() - self.alpha) * x + self.alpha * self.previous;
        self.previous = x;

        [convert(value)].into()
    }

    /// Resets the filter state to zero.
    fn reset(&mut self) {
        self.previous = F::zero();
    }
}

/// Creates a single-pole low-pass filter.
/// 
/// # Arguments
/// 
/// * `alpha` - Smoothing coefficient (0.0 to 1.0). Higher values = more smoothing.
/// 
/// # Example
/// 
/// ```rust,ignore
/// // Create a heavily smoothing filter for envelope extraction
/// let envelope_filter = alpha_lpf(0.99);
/// ```
pub fn alpha_lpf<F: Real>(alpha: F) -> An<AlphaLpf<F>> {
    An(AlphaLpf {
        alpha,
        previous: F::zero(),
    })
}

/// Full-wave rectifier that outputs the absolute value of the input signal.
/// 
/// This operation is essential for onset detection, as it converts bipolar audio
/// signals into unipolar envelopes that can be used to detect amplitude changes.
/// 
/// The rectified signal is typically followed by a low-pass filter to extract
/// the amplitude envelope.
#[derive(Clone)]
pub struct FullWaveRectification<F: Real> {
    _marker: std::marker::PhantomData<F>,
}

impl<F: Real> AudioNode for FullWaveRectification<F> {
    const ID: u64 = BASE_ID + 2;

    type Inputs = U1;

    type Outputs = U1;

    /// Computes the absolute value of the input sample.
    #[inline]
    fn tick(&mut self, input: &Frame<f32, Self::Inputs>) -> Frame<f32, Self::Outputs> {
        [convert(input[0].abs())].into()
    }

    /// SIMD-optimized batch processing of multiple samples.
    fn process(&mut self, size: usize, input: &BufferRef, output: &mut BufferMut) {
        (0..simd_items(size)).for_each(|i| {
            let input = input.channel(0)[i];

            let output = &mut output.channel_mut(0)[i];

            *output = input.abs();
        });
    }
}

/// Creates a full-wave rectifier node.
/// 
/// # Returns
/// 
/// A 1-input, 1-output node that outputs |x| for input x.
/// 
/// # Example
/// 
/// ```rust,ignore
/// // Create an onset detector: rectify, then smooth
/// let onset_detector = fwr::<f32>() >> alpha_lpf(0.95);
/// ```
pub fn fwr<F: Real>() -> An<FullWaveRectification<F>> {
    An(FullWaveRectification {
        _marker: std::marker::PhantomData,
    })
}
