use cubecl::{
    prelude::Client,
    server::Collective,
    std::throughput::{measure_link_curve, transfer_route},
    throughput::{LinkRoute, MemoryCurve},
};

const COLLECTIVES: [Collective; 3] = [
    Collective::AllReduce,
    Collective::AllGather,
    Collective::ReduceScatter,
];

fn main() {
    let clients = multi_device::clients();
    println!("{} devices", clients.len());

    for (from, source) in clients.iter().enumerate() {
        for (to, destination) in clients.iter().enumerate().filter(|(to, _)| *to != from) {
            let transfer = LinkRoute::Transfer(transfer_route(source, destination));
            for route in [transfer, LinkRoute::PeerLoad, LinkRoute::PeerStore] {
                let pair = [source.clone(), destination.clone()];
                print_curve(&format!("{route:?} {from} -> {to}"), &pair, route, 1.0);
            }
        }
    }

    let mut runtimes: Vec<&str> = clients.iter().map(Client::name).collect();
    runtimes.dedup();
    for runtime in runtimes {
        let group: Vec<Client> = clients
            .iter()
            .filter(|client| client.name() == runtime)
            .cloned()
            .collect();
        for members in 2..=group.len() {
            for route in [LinkRoute::RingPeerStore, LinkRoute::RingPeerLoad] {
                let label = format!("{route:?} over {members} {runtime}");
                print_curve(&label, &group[..members], route, 1.0);
            }
            for kind in COLLECTIVES {
                let route = LinkRoute::Collective(kind);
                let label = format!("{kind} over {members} {runtime}");
                print_curve(&label, &group[..members], route, bus_factor(kind, members));
            }
        }
    }
}

fn print_curve(label: &str, members: &[Client], route: LinkRoute, bus_factor: f64) {
    let curve: MemoryCurve = measure_link_curve(members, route);
    if curve.points().is_empty() {
        println!("\n  {label}: unsupported");
        return;
    }

    println!(
        "\n  {label:<24}{:>12}{:>12}{:>10}{:>10}",
        "algbw", "busbw", "time rsd", "max"
    );
    for point in curve.points() {
        let rate = curve.ceiling_at_bytes(point.bytes).unwrap_or(f64::NAN);
        println!(
            "    {:>10}{:>14.2} GB/s{:>7.2} GB/s{:>10}{:>10}",
            bytes_label(point.bytes),
            rate / 1e9,
            rate * bus_factor / 1e9,
            percent_label(point.value.relative_std_dev(), ""),
            percent_label(point.value.relative_max(), "+")
        );
    }
}

fn percent_label(fraction: Option<f64>, sign: &str) -> String {
    match fraction {
        Some(fraction) => format!("{sign}{:.1}%", fraction * 100.0),
        None => String::from("N/A"),
    }
}

/// The share of the bytes each member must send, which nccl-tests' bus bandwidth scales by so a
/// collective compares across member counts and against the link's own rate.
fn bus_factor(kind: Collective, members: usize) -> f64 {
    let n = members as f64;
    match kind {
        Collective::AllReduce => 2.0 * (n - 1.0) / n,
        _ => (n - 1.0) / n,
    }
}

fn bytes_label(bytes: u64) -> String {
    match bytes {
        bytes if bytes >= 1 << 20 => format!("{} MiB", bytes >> 20),
        bytes => format!("{} KiB", bytes >> 10),
    }
}
