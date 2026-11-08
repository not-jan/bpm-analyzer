//! Example: BPM detection on samples you feed in yourself
//!
//! Instead of capturing from an input device, this feeds a synthetic 130 BPM click track
//! into the analyzer with `begin_with_receiver`. In a real application the frames would
//! come from your decoder or player.

use bpm_analyzer::{AnalyzerConfig, begin_with_receiver, bounded};

const SAMPLE_RATE: u32 = 44_100;
const TARGET_BPM: f32 = 130.0;

/// A short, exponentially decaying noise burst on every beat.
fn click_track(bpm: f32, seconds: f32) -> impl Iterator<Item = f32> {
    let period = (60.0 / bpm * SAMPLE_RATE as f32) as usize;
    let mut seed = 12345u32;
    (0..(SAMPLE_RATE as f32 * seconds) as usize).map(move |i| {
        let t = (i % period) as f32 / SAMPLE_RATE as f32;
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let noise = (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
        0.9 * noise * (-80.0 * t).exp()
    })
}

fn main() -> Result<(), bpm_analyzer::Error> {
    let (sender, frames) = bounded(1 << 16);
    let detections = begin_with_receiver(AnalyzerConfig::electronic(), SAMPLE_RATE, frames)?;

    // Feed the audio from another thread; dropping the sender ends the analysis.
    std::thread::spawn(move || {
        for sample in click_track(TARGET_BPM, 20.0) {
            // blocks while the queue is full, so nothing is dropped
            if sender.send((sample, sample)).is_err() {
                break;
            }
        }
    });

    // The loop ends when the analysis thread is done.
    let mut last = None;
    for detection in detections.iter() {
        if let Some(bpm) = detection.bpm() {
            last = Some(bpm);
        }
    }

    match last {
        Some(bpm) => println!("Fed {TARGET_BPM} BPM, detected {bpm:.1} BPM"),
        None => println!("No BPM detected"),
    }
    Ok(())
}
