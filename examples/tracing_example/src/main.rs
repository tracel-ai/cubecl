use clap::{Parser, ValueEnum};
use cubecl::prelude::*;
use opentelemetry::trace::TracerProvider;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use std::error::Error;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

static APP_NAME: &str = "dop_timer";

#[derive(Debug, Clone, ValueEnum)]
pub enum TracingMode {
    /// Print to stderr.
    Console,

    /// Export to OTEL via gRPC.
    Otel,
}

/// Timing tool for measuring the performance of collective operations.
///
/// Currently only supports `all_reduce`.
#[derive(Parser, Debug)]
pub struct Args {
    /// Enable/Set tracing mode.
    #[arg(long, value_enum)]
    pub tracing: Option<TracingMode>,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn Error + Send + Sync + 'static>> {
    let args = Args::parse();
    println!("{:?}", args);


    let tracing_provider = match &args.tracing {
        None => None,
        Some(TracingMode::Console) => {
            let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
            let subscriber = tracing_subscriber::fmt()
                .with_env_filter(env_filter)
                .with_span_events(FmtSpan::ENTER | FmtSpan::EXIT);

            let subscriber = subscriber.with_writer(std::io::stderr);

            subscriber.try_init()?;

            None
        }
        Some(TracingMode::Otel) => {
            let exporter = opentelemetry_otlp::SpanExporter::builder()
                .with_tonic()
                .build()?;

            let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
                .with_batch_exporter(exporter)
                .with_sampler(opentelemetry_sdk::trace::Sampler::AlwaysOn)
                .with_resource(Resource::builder().with_service_name(APP_NAME).build())
                .build();

            opentelemetry::global::set_tracer_provider(provider.clone());

            let tracer = provider.tracer(APP_NAME);

            let telemetry = tracing_opentelemetry::layer().with_tracer(tracer);

            let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
            tracing_subscriber::registry()
                .with(env_filter)
                .with(telemetry)
                .try_init()?;

            opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());

            Some(provider)
        }
    };

    let count = [cfg!(feature = "cuda"), cfg!(feature = "wgpu")]
        .iter()
        .filter(|x| **x)
        .count();
    assert_eq!(count, 1, "exactly one backend must be enabled");

    #[cfg(feature = "cuda")]
    launch::<cubecl::cuda::CudaRuntime>(&args, &Default::default())?;

    #[cfg(feature = "wgpu")]
    launch::<cubecl::wgpu::WgpuRuntime>(&args, &Default::default())?;

    if let Some(provider) = tracing_provider {
        provider.shutdown()?;
    }

    Ok(())
}

#[allow(unused)]
#[tracing::instrument(level = "trace", skip(args, device))]
pub fn launch<R: Runtime>(
    args: &Args,
    device: &R::Device,
) -> Result<(), Box<dyn Error + Send + Sync + 'static>> {
    let client = R::client(device);

    let input = [-1., 10., 1., 5.];
    let device_handle = client.create_from_slice(f32::as_bytes(&input));

    let ram_bytes = client.read_one(device_handle.clone());
    let output = f32::from_bytes(&ram_bytes);

    Ok(())
}
