use cubecl::{
    benchmark::{Benchmark, MultiDeviceRun},
    client::{ClientGroup, MemberBuffers},
    future::block_on,
    ir::{ElemType, FloatKind},
    server::{Handle, ReduceOperation},
};

/// A decode step's hidden state at batch one.
const HIDDEN: usize = 2048;

struct AllReduce {
    group: ClientGroup,
    dtype: ElemType,
}

impl Benchmark for AllReduce {
    type Input = Vec<Handle>;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let bytes = HIDDEN * self.dtype.size();
        self.group
            .clients()
            .iter()
            .map(|client| client.empty(bytes))
            .collect()
    }

    fn execute(&self, buffers: Self::Input) -> Result<(), String> {
        let buffers = buffers
            .into_iter()
            .map(|buffer| MemberBuffers {
                src: buffer.clone(),
                dst: buffer,
            })
            .collect();
        self.group
            .all_reduce(buffers, self.dtype, ReduceOperation::Sum);
        Ok(())
    }

    fn name(&self) -> String {
        format!("all-reduce of [1, {HIDDEN}] {:?}", self.dtype)
    }

    fn sync(&self) {
        for client in self.group.clients() {
            block_on(client.sync()).expect("the member syncs");
        }
    }
}

fn main() {
    let clients = multi_device::clients();
    for (reading, client) in clients.iter().enumerate() {
        for (owning, peer) in clients
            .iter()
            .enumerate()
            .filter(|(owning, _)| *owning != reading)
        {
            let access = client.peer_access(peer);
            println!("member {reading} has {access:?} access to member {owning}'s memory");
        }
    }
    let group = ClientGroup::new(clients).expect("the devices form a collective group");
    let f16 = ElemType::Float(FloatKind::F16);
    let dtype = match group.reduces(f16) {
        true => f16,
        false => ElemType::Float(FloatKind::F32),
    };
    let bench = AllReduce { group, dtype };
    let run = MultiDeviceRun::new(&bench, bench.group.clients())
        .run()
        .expect("the all-reduce runs");
    println!("{}", bench.name());
    for (member, durations) in run.members.iter().enumerate() {
        println!("  member {member}: {durations}");
    }
    println!(
        "  slowest median {:?}, spread {:.2}x",
        run.median(),
        run.spread()
    );
    println!("  wall: {}", run.wall);
}
