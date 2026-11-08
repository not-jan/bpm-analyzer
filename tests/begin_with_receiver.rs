//! Tests for analysing caller-supplied samples (`begin_with_receiver`).

use std::time::{Duration, Instant};

use bpm_analyzer::{AnalyzerConfig, Error, begin_with_receiver, bounded};
use crossbeam_channel::TrySendError;

/// Click track: a 30 ms, exponentially decaying noise burst on every beat.
fn click_track(bpm: f32, rate: u32, seconds: f32) -> Vec<f32> {
    let n = (rate as f32 * seconds) as usize;
    let period = (60.0 / bpm * rate as f32) as usize;
    let mut seed = 12345u32;
    (0..n)
        .map(|i| {
            let t = (i % period) as f32 / rate as f32;
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
            0.9 * noise * (-80.0 * t).exp()
        })
        .collect()
}

#[test]
fn detects_tempo_of_fed_samples() {
    for (bpm, rate) in [(120.0f32, 44_100u32), (130.0, 48_000), (150.0, 44_100)] {
        let (tx, rx) = bounded(1 << 16);
        let detections = begin_with_receiver(AnalyzerConfig::electronic(), rate, rx).unwrap();

        let feeder = std::thread::spawn(move || {
            for s in click_track(bpm, rate, 20.0) {
                if tx.send((s, s)).is_err() {
                    break;
                }
            }
        });
        feeder.join().unwrap();

        // The channel closes when the analysis thread ends (sender gone, queue drained).
        let bpms: Vec<f32> = detections.iter().filter_map(|d| d.bpm()).collect();
        assert!(!bpms.is_empty(), "{bpm} BPM @ {rate} Hz: no detection");
        assert!(
            bpms.iter().rev().take(3).any(|b| (b - bpm).abs() <= 4.0),
            "{bpm} BPM @ {rate} Hz expected, last detections: {:?}",
            &bpms[bpms.len().saturating_sub(5)..]
        );
    }
}

#[test]
fn analysis_thread_ends_when_sender_is_dropped() {
    let (tx, rx) = bounded(1024);
    let detections = begin_with_receiver(AnalyzerConfig::electronic(), 44_100, rx).unwrap();
    drop(tx);

    // `iter()` only returns once the result channel is closed, i.e. the thread ended.
    let start = Instant::now();
    let count = detections.iter().count();
    assert_eq!(count, 0);
    assert!(start.elapsed() < Duration::from_secs(5));
}

#[test]
fn analysis_thread_ends_when_receiver_is_dropped() {
    let (tx, rx) = bounded(1 << 16);
    let detections = begin_with_receiver(AnalyzerConfig::electronic(), 44_100, rx).unwrap();
    drop(detections);

    // The analysis thread only notices that nobody listens once it has its first result.
    // Keep feeding (10 s of audio is far more than a window) until it ends and drops its
    // end of the audio channel: then sending fails with `Disconnected`.
    let start = Instant::now();
    for s in click_track(120.0, 44_100, 10.0).into_iter().cycle() {
        let mut frame = (s, s);
        loop {
            match tx.try_send(frame) {
                Ok(()) => break,
                Err(TrySendError::Full(f)) => {
                    frame = f;
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(TrySendError::Disconnected(_)) => return, // the thread has ended
            }
        }
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "analysis thread keeps running although nobody reads the results"
        );
    }
}

#[test]
fn rejects_unsupported_sample_rate() {
    let (_tx, rx) = bounded(16);
    let err = begin_with_receiver(AnalyzerConfig::electronic(), 12_345, rx).unwrap_err();
    assert!(matches!(err, Error::UnsupportedSampleRate(12_345)));
}
