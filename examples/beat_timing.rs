//! Example: Real-time BPM and beat timing detection
//!
//! This example demonstrates how to use the BPM analyzer to detect
//! both tempo and individual beat timings in real-time.

use bpm_analyzer::{AnalyzerConfig, begin};

fn main() -> Result<(), bpm_analyzer::Error> {
    // Configure the analyzer for electronic music
    let config = AnalyzerConfig::electronic();

    println!("Starting BPM analyzer...");
    println!("Listening for beats and tempo...\n");

    // Start the analyzer
    let receiver = begin(config)?;

    // Process detections
    for detection in receiver.iter() {
        // Print BPM if detected
        if let Some(bpm) = detection.bpm() {
            println!("🎵 Detected BPM: {:.1}", bpm);
        }

        // Print beat timings
        if let Some(last_beat) = detection.last_beat() {
            print!(
                "🥁 Beat at {:.2}s (strength: {:.2})",
                last_beat.time_seconds, last_beat.strength
            );

            // Show beat interval if we have at least 2 beats
            if let Some(interval) = detection.last_beat_interval() {
                let instant_bpm = 60.0 / interval;
                print!(" | Interval: {:.3}s ({:.1} BPM)", interval, instant_bpm);
            }
            println!();
        }

        // Print all recent beat timings
        let beats = detection.beat_timings();
        if beats.len() >= 4 {
            println!("📊 Recent beats: {} detected", beats.len());
            for (i, beat) in beats.iter().rev().take(4).enumerate() {
                println!(
                    "   {} {:.2}s ago (strength: {:.2})",
                    if i == 0 { "•" } else { " " },
                    beats.last().unwrap().time_seconds - beat.time_seconds,
                    beat.strength
                );
            }
        }

        println!();
    }

    Ok(())
}
