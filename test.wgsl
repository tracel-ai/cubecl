@group(0)
@binding(0)
var<storage, read_write> input_0_global: array<f32>;

@group(0)
@binding(1)
var<storage, read_write> output_0_global: array<atomic<u32>>;

@group(0)
@binding(2)
var<storage, read_write> info: array<u32>;

const WORKGROUP_SIZE_X = 1u;
const WORKGROUP_SIZE_Y = 1u;
const WORKGROUP_SIZE_Z = 1u;

var<workgroup> shared_0: array<f32, 1024>;

@compute
@workgroup_size(1, 1, 1)
fn main(
    @builtin(local_invocation_index) local_idx: u32,
) {let rank: u32 = info[0];
    let _0 = &output_0_global[0];
    storageBarrier();
    workgroupBarrier();
    let _1 = atomicLoad(_0);
    atomicStore(_0, _1);
    let _2 = atomicExchange(_0, 2u);
    let _3 = atomicCompareExchangeWeak(_0, 3u, 4u);
    let _4 = atomicAdd(_0, 1u);
    let _5 = atomicSub(_0, 2u);
    let _11 = atomicMin(_0, 3u);
    let _6 = atomicMax(_0, 4u);
    let _7 = atomicAnd(_0, 5u);
    let _8 = atomicOr(_0, 6u);
    let _9 = atomicXor(_0, 7u);
}