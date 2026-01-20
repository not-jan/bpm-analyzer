use anyhow::Result;
use bpm_analyzer::AnalyzerConfig;
use clap::Parser;
use crossbeam_channel::Receiver;
use egui::{CentralPanel, SidePanel};
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
    receiver: Receiver<[(f32, f32); 5]>,
}

impl BpmApp {
    pub fn new(min_bpm: f32, max_bpm: f32, receiver: Receiver<[(f32, f32); 5]>) -> Self {
        Self {
            min_bpm,
            max_bpm,
            data: AllocRingBuffer::new(256),
            receiver,
        }
    }
}

impl eframe::App for BpmApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();

        self.data
            .extend(self.receiver.try_iter().flat_map(|arr| arr.into_iter()));

        SidePanel::right("right_panel").show(ctx, |ui| {
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

            if let Some((bpm, _)) = peaks.first() {
                ui.label(format!("{bpm} BPM"));
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
