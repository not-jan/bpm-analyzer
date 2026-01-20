# BPM Analyzer

A real-time BPM (beats per minute) detection library and application for Rust, using wavelet decomposition and autocorrelation analysis.

Based on [Audio Analysis using the Discrete Wavelet Transform](https://soundlab.cs.princeton.edu/publications/2001_amta_aadwt.pdf) by George Tzanetakis, Georg Essl and Perry Cook.

## How It Works

The BPM analyzer uses a sophisticated multi-stage signal processing pipeline:

1. **Audio Capture**: Captures audio from system input or loopback devices
2. **Resampling**: Downsamples to 22.05 kHz for efficient processing
3. **Wavelet Decomposition**: Applies 4-level Daubechies D4 wavelet transform to separate frequency bands
4. **Onset Detection**: Extracts amplitude envelopes from each band using full-wave rectification and low-pass filtering
5. **Beat Detection**: Identifies individual beats using onset strength analysis
6. **Autocorrelation**: Computes autocorrelation on the summed envelopes to find periodic patterns
7. **Peak Detection**: Identifies BPM candidates based on autocorrelation peaks within the specified range

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
bpm-analyzer = "0.1"
```

## Usage

### As a Library

```rust
use bpm_analyzer::{AnalyzerConfig, begin};

fn main() -> Result<(), bpm_analyzer::Error> {
    // Configure the analyzer
    let config = AnalyzerConfig::builder()
        .min_bpm(60.0)
        .max_bpm(180.0)
        .build();

    // Start the analyzer with default device
    let bpm_receiver = begin(config)?;

    // Process BPM detections
    for detection in bpm_receiver.iter() {
        if let Some(bpm) = detection.bpm() {
            println!("Detected BPM: {:.1}", bpm);
        }
        
        // Access beat timings
        for beat in detection.beat_timings() {
            println!("Beat at {:.2}s (strength: {:.2})", beat.time_seconds, beat.strength);
        }
        
        // Get the interval between last two beats
        if let Some(interval) = detection.last_beat_interval() {
            println!("Last beat interval: {:.3}s", interval);
        }
    }

    Ok(())
}
```

#### Selecting a Specific Audio Device

```rust
use bpm_analyzer::{AnalyzerConfig, begin_with_device, list_audio_devices, get_device_by_name};

// List all available devices
let devices = list_audio_devices()?;
for device in &devices {
    println!("{} {}", device.name, if device.is_default { "(default)" } else { "" });
}

// Use a specific device by name
let config = AnalyzerConfig::electronic();
let device = get_device_by_name("BlackHole 2ch")?;
let receiver = begin_with_device(config, &device)?;

for detection in receiver.iter() {
    if let Some(bpm) = detection.bpm() {
        println!("Detected BPM: {:.1}", bpm);
    }
}
```

### As a Standalone Application

Run the GUI application:

```bash
cargo run --release --features bin
```

With custom BPM range:

```bash
cargo run --release --features bin -- --min-bpm 80 --max-bpm 160
```

### Command-Line Options

- `-m, --min-bpm <MIN_BPM>`: Minimum BPM to detect (default: 40)
- `-M, --max-bpm <MAX_BPM>`: Maximum BPM to detect (default: 240)

## Configuration

### AnalyzerConfig

The analyzer behavior can be customized using `AnalyzerConfig`:

```rust
let config = AnalyzerConfig::builder()
    .min_bpm(40.0)           // Minimum BPM to detect
    .max_bpm(240.0)          // Maximum BPM to detect
    .window_size(65536)      // Analysis window size (must be power of 2)
    .queue_size(4096)        // Audio queue size
    .buffer_size(256)        // Audio capture buffer size
    .build();
```

#### Parameters

- **min_bpm** / **max_bpm**: Define the range of tempos to detect. Narrowing this range can improve accuracy for specific genres.
- **window_size**: Size of the analysis window in samples. Larger windows provide better frequency resolution but slower response. 
- **queue_size**: Size of the inter-thread audio queue. I don't think you need to change this.
- **buffer_size**: Audio capture buffer size. Smaller values reduce latency but may increase CPU usage. I'm also not convinced that this needs to be changed.

## Audio Setup

### macOS

For system audio capture, install [BlackHole](https://github.com/ExistentialAudio/BlackHole):

```bash
brew install blackhole-2ch
```

Then configure your system to route audio through BlackHole using a multi-output device.

## Dependencies

This library builds on several excellent Rust crates:

- [fundsp](https://github.com/SamiPerttu/fundsp) - Audio processing framework
- [osclet](https://crates.io/crates/osclet) - Wavelet transform library
- [cpal](https://github.com/RustAudio/cpal) - Cross-platform audio I/O
- [resampler](https://crates.io/crates/resampler) - Audio resampling

## Limitations

- **Tempo Changes**: The analyzer works best with consistent tempo. Sudden tempo changes may take time to adapt.
- **Polyrhythmic Music**: Complex polyrhythmic patterns may produce multiple strong candidates.
- **Very Slow/Fast Tempos**: Accuracy may decrease outside the 40-240 BPM range.
