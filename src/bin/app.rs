use anyhow::Result;
use bpm_analyzer::{AnalyzerConfig, BpmDetection};
use clap::Parser;
use crossbeam_channel::Receiver;
use egui::{CentralPanel, Color32, CornerRadius, Pos2, SidePanel, Stroke, Vec2};
use egui_plotter::EguiBackend;
use itertools::Itertools;
use plotters::{
    chart::ChartBuilder,
    prelude::{IntoDrawingArea, IntoSegmentedCoord},
    series::Histogram,
    style::{BLUE, Color, WHITE},
};
use ringbuffer::{AllocRingBuffer, RingBuffer};
use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

#[derive(clap::Parser, Debug)]
struct Args {
    #[clap(short = 'm', long, default_value_t = 40.0)]
    min_bpm: f32,
    #[clap(short = 'M', long, default_value_t = 240.0)]
    max_bpm: f32,
}

pub struct BpmApp {
    min_bpm: f32,
    max_bpm: f32,
    data: AllocRingBuffer<(f32, f32)>,
    receiver: Receiver<BpmDetection>,
    last_beat_timestamp: Option<f64>,
    beat_pulse_strength: f32,
    last_detection: Option<BpmDetection>,
}

impl BpmApp {
    pub fn new(min_bpm: f32, max_bpm: f32, receiver: Receiver<BpmDetection>) -> Self {
        Self {
            min_bpm,
            max_bpm,
            data: AllocRingBuffer::new(256),
            receiver,
            last_beat_timestamp: None,
            beat_pulse_strength: 0.0,
            last_detection: None,
        }
    }
}

impl eframe::App for BpmApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();

        // Process incoming detections
        for detection in self.receiver.try_iter() {
            self.data.extend(
                detection.candidates().iter().map(|c| (c.bpm, c.confidence)).collect::<Vec<_>>()
            );
            
            // Check for new beat using audio timestamp
            if let Some(last_beat) = detection.last_beat() {
                if let Some(prev_timestamp) = self.last_beat_timestamp {
                    // Only trigger pulse if this is a new beat (different timestamp)
                    if (last_beat.time_seconds - prev_timestamp).abs() > 0.05 {
                        self.last_beat_timestamp = Some(last_beat.time_seconds);
                        self.beat_pulse_strength = last_beat.strength.min(1.0);
                    }
                } else {
                    self.last_beat_timestamp = Some(last_beat.time_seconds);
                    self.beat_pulse_strength = last_beat.strength.min(1.0);
                }
            }
            
            self.last_detection = Some(detection);
        }

        // Decay beat pulse smoothly (faster decay, with cutoff)
        self.beat_pulse_strength *= 0.85;
        if self.beat_pulse_strength < 0.01 {
            self.beat_pulse_strength = 0.0;
        }

        SidePanel::right("right_panel").min_width(200.0).show(ctx, |ui| {
            let peaks = self
                .data
                .iter()
                .map(|(x, v)| (x.round() as i32, *v))
                .into_group_map();

            let mut peaks = peaks
                .into_iter()
                .map(|(k, v)| (k, v.into_iter().sum::<f32>()))
                .collect::<Vec<_>>();

            peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            ui.heading("BPM Detection");
            ui.separator();

            if let Some((bpm, _)) = peaks.first() {
                ui.label(egui::RichText::new(format!("{bpm} BPM"))
                    .size(48.0)
                    .strong());
            } else {
                ui.label(egui::RichText::new("-- BPM")
                    .size(48.0)
                    .weak());
            }

            ui.add_space(20.0);

            // Beat timing section
            if let Some(detection) = &self.last_detection {
                ui.heading("Beat Timing");
                ui.separator();

                if let Some(last_beat) = detection.last_beat() {
                    ui.label(format!("Last Beat: {:.2}s", last_beat.time_seconds));
                    ui.label(format!("Strength: {:.2}", last_beat.strength));
                    
                    if let Some(interval) = detection.last_beat_interval() {
                        let instant_bpm = 60.0 / interval;
                        ui.label(format!("Interval: {:.3}s", interval));
                        ui.label(egui::RichText::new(format!("Instant BPM: {:.1}", instant_bpm))
                            .color(Color32::from_rgb(100, 150, 255)));
                    }
                } else {
                    ui.label(egui::RichText::new("Waiting for beats...")
                        .weak());
                }

                ui.add_space(10.0);
                
                // Beat pulse indicator
                let pulse_size = 80.0 + self.beat_pulse_strength * 40.0;
                let pulse_alpha = (self.beat_pulse_strength * 255.0) as u8;
                let (rect, _response) = ui.allocate_exact_size(
                    Vec2::new(ui.available_width(), 120.0),
                    egui::Sense::hover(),
                );
                
                let center = rect.center();
                let radius = pulse_size / 2.0;
                
                ui.painter().circle(
                    center,
                    radius,
                    Color32::from_rgba_unmultiplied(100, 150, 255, pulse_alpha),
                    Stroke::new(2.0, Color32::from_rgb(100, 150, 255)),
                );
                
                ui.painter().text(
                    center,
                    egui::Align2::CENTER_CENTER,
                    "♪",
                    egui::FontId::proportional(40.0),
                    Color32::WHITE,
                );

                ui.add_space(10.0);

                // Beat history timeline
                let beat_count = detection.beat_timings().len();
                if beat_count > 1 {
                    ui.label(format!("Recent Beats: {}", beat_count));
                    
                    let timeline_height = 60.0;
                    let (timeline_rect, _) = ui.allocate_exact_size(
                        Vec2::new(ui.available_width(), timeline_height),
                        egui::Sense::hover(),
                    );
                    
                    // Draw timeline background
                    ui.painter().rect_filled(
                        timeline_rect,
                        CornerRadius::same(4),
                        Color32::from_gray(40),
                    );
                    
                    // Get beat timings
                    let beats = detection.beat_timings();
                    if let (Some(first), Some(last)) = (beats.first(), beats.last()) {
                        let time_range = (last.time_seconds - first.time_seconds).max(0.1);
                        
                        // Draw beats
                        for beat in beats {
                            let normalized_time = (beat.time_seconds - first.time_seconds) / time_range;
                            let x = timeline_rect.left() + (normalized_time as f32) * timeline_rect.width();
                            let y = timeline_rect.center().y;
                            
                            let beat_radius = 4.0 + beat.strength * 4.0;
                            ui.painter().circle_filled(
                                Pos2::new(x, y),
                                beat_radius,
                                Color32::from_rgb(100, 200, 255),
                            );
                        }
                    }
                }
            }
        });

        CentralPanel::default().show(ctx, |ui| {
            let root = EguiBackend::new(ui).into_drawing_area();
            root.fill(&WHITE).unwrap();
            let mut chart_builder = ChartBuilder::on(&root);

            let max = self.data.iter().map(|(_, v)| *v).sum::<f32>();

            chart_builder
                .margin(5)
                .set_left_and_bottom_label_area_size(20);

            let mut chart_context = chart_builder
                .build_cartesian_2d(
                    (self.min_bpm as i32..self.max_bpm as i32).into_segmented(),
                    0.0..max,
                )
                .unwrap();

            chart_context.configure_mesh().draw().unwrap();
            chart_context
                .draw_series(
                    Histogram::vertical(&chart_context)
                        .style(BLUE.filled())
                        .margin(0)
                        .data(
                            self.data
                                .iter()
                                .copied()
                                .map(|(a, b)| (a.round() as i32, b)),
                        ),
                )
                .unwrap();
            root.present().unwrap();
        });
    }
}

fn main() -> Result<()> {
    let fmt_layer = fmt::layer().with_target(false);
    let filter_layer = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new("info"))
        .unwrap();

    tracing_subscriber::registry()
        .with(filter_layer)
        .with(fmt_layer)
        .init();

    let args = Args::parse();

    let config = AnalyzerConfig::builder()
        .max_bpm(args.max_bpm)
        .min_bpm(args.min_bpm)
        .build();

    let bpm_receiver = bpm_analyzer::begin(config)?;

    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "BPM Analyzer",
        native_options,
        Box::new(|_cc| {
            Ok(Box::new(BpmApp::new(
                args.min_bpm,
                args.max_bpm,
                bpm_receiver,
            )))
        }),
    )
    .map_err(|e| anyhow::anyhow!("Failed to start app: {e:?}"))?;

    Ok(())
}
