
@group(0)
@binding(0)
var<storage, read_write> input_0_global: array<f32>;

@group(0)
@binding(1)
var<storage, read_write> output_0_global: array<vec4<f32>>;

@group(0)
@binding(2)
var<storage, read_write> info: array<u32>;

@group(0)
@binding(3)
var<storage, read_write> scalars_uint: array<u32, 3>;

var<workgroup> shared_memory_0: array<vec4<f32>, 16>;

const WORKGROUP_SIZE_X = 1u;
const WORKGROUP_SIZE_Y = 1u;
const WORKGROUP_SIZE_Z = 1u;

@compute
@workgroup_size(1, 1, 1)
fn main(
) {let rank: u32 = info[0];
    let rank_2: u32 = rank * 2u;
    var l_0_0: u32;
    var l_0_1: u32;
    var l_0_2: u32;
    var l_0_3: u32;
    var l_0_4: u32;
    var l_0_5: bool;
    var l_0_6: u32;
    var l_0_7: u32;
    var l_0_8: u32;
    var l_0_9: vec4<f32>;
    var l_0_10: f32;
    l_0_0 = rank - 2u;
    l_0_1 = info[(0u * rank_2) + rank + l_0_0 + 1u];
    l_0_0 = rank - 1u;
    l_0_2 = info[(0u * rank_2) + rank + l_0_0 + 1u];
    l_0_0 = scalars_uint[2] * l_0_2;
    l_0_3 = 0u + l_0_0;
    l_0_3 = l_0_3 + 0u;
    l_0_0 = scalars_uint[0] * l_0_2;
    l_0_0 = l_0_0 + scalars_uint[1];
    l_0_0 = l_0_0 + l_0_3;
    l_0_4 = scalars_uint[0] * 8u;
    l_0_4 = l_0_4 + scalars_uint[1];
    l_0_5 = scalars_uint[0] < 8u;
    if l_0_5 {
        l_0_6 = 0u * l_0_2;
        l_0_6 = l_0_0 + l_0_6;
        l_0_6 = l_0_6 / 1u;
        l_0_7 = 0u * 8u;
        l_0_7 = l_0_4 + l_0_7;
        l_0_7 = l_0_7 / 4u;
        l_0_9 = vec4(
            f32(0f),
            f32(0f),
            f32(0f),
            f32(0f),
        );
        l_0_8 = l_0_6 + 0u;
        l_0_10 = input_0_global[l_0_8];
        l_0_9[0u] = f32(l_0_10);
        l_0_8 = l_0_6 + 1u;
        l_0_10 = input_0_global[l_0_8];
        l_0_9[1u] = f32(l_0_10);
        l_0_8 = l_0_6 + 2u;
        l_0_10 = input_0_global[l_0_8];
        l_0_9[2u] = f32(l_0_10);
        l_0_8 = l_0_6 + 3u;
        l_0_10 = input_0_global[l_0_8];
        l_0_9[3u] = f32(l_0_10);
        shared_memory_0[l_0_7] = vec4<f32>(l_0_9);
        l_0_8 = 1u * l_0_2;
        l_0_8 = l_0_0 + l_0_8;
        l_0_8 = l_0_8 / 1u;
        l_0_7 = 1u * 8u;
        l_0_7 = l_0_4 + l_0_7;
        l_0_7 = l_0_7 / 4u;
        l_0_9[0u] = f32(0f);
        l_0_9[1u] = f32(0f);
        l_0_9[2u] = f32(0f);
        l_0_9[3u] = f32(0f);
        l_0_6 = l_0_8 + 0u;
        l_0_10 = input_0_global[l_0_6];
        l_0_9[0u] = f32(l_0_10);
        l_0_6 = l_0_8 + 1u;
        l_0_10 = input_0_global[l_0_6];
        l_0_9[1u] = f32(l_0_10);
        l_0_6 = l_0_8 + 2u;
        l_0_10 = input_0_global[l_0_6];
        l_0_9[2u] = f32(l_0_10);
        l_0_6 = l_0_8 + 3u;
        l_0_10 = input_0_global[l_0_6];
        l_0_9[3u] = f32(l_0_10);
        shared_memory_0[l_0_7] = vec4<f32>(l_0_9);
        l_0_8 = 2u * l_0_2;
        l_0_8 = l_0_0 + l_0_8;
        l_0_8 = l_0_8 / 1u;
        l_0_7 = 2u * 8u;
        l_0_7 = l_0_4 + l_0_7;
        l_0_7 = l_0_7 / 4u;
        l_0_9[0u] = f32(0f);
        l_0_9[1u] = f32(0f);
        l_0_9[2u] = f32(0f);
        l_0_9[3u] = f32(0f);
        l_0_6 = l_0_8 + 0u;
        l_0_10 = input_0_global[l_0_6];
        l_0_9[0u] = f32(l_0_10);
        l_0_6 = l_0_8 + 1u;
        l_0_10 = input_0_global[l_0_6];
        l_0_9[1u] = f32(l_0_10);
        l_0_6 = l_0_8 + 2u;
        l_0_10 = input_0_global[l_0_6];
        l_0_9[2u] = f32(l_0_10);
        l_0_6 = l_0_8 + 3u;
        l_0_10 = input_0_global[l_0_6];
        l_0_9[3u] = f32(l_0_10);
        shared_memory_0[l_0_7] = vec4<f32>(l_0_9);
        l_0_8 = 3u * l_0_2;
        l_0_8 = l_0_0 + l_0_8;
        l_0_8 = l_0_8 / 1u;
        l_0_7 = 3u * 8u;
        l_0_7 = l_0_4 + l_0_7;
        l_0_7 = l_0_7 / 4u;
        l_0_9[0u] = f32(0f);
        l_0_9[1u] = f32(0f);
        l_0_9[2u] = f32(0f);
        l_0_9[3u] = f32(0f);
        l_0_6 = l_0_8 + 0u;
        l_0_10 = input_0_global[l_0_6];
        l_0_9[0u] = f32(l_0_10);
        l_0_6 = l_0_8 + 1u;
        l_0_10 = input_0_global[l_0_6];
        l_0_9[1u] = f32(l_0_10);
        l_0_6 = l_0_8 + 2u;
        l_0_10 = input_0_global[l_0_6];
        l_0_9[2u] = f32(l_0_10);
        l_0_6 = l_0_8 + 3u;
        l_0_10 = input_0_global[l_0_6];
        l_0_9[3u] = f32(l_0_10);
        shared_memory_0[l_0_7] = vec4<f32>(l_0_9);
    }

    for (var l_1_0: u32 = 0u; l_1_0 < 16u; l_1_0++) {
        l_0_9 = shared_memory_0[l_1_0];
        output_0_global[l_1_0] = vec4<f32>(l_0_9);
    }
}
