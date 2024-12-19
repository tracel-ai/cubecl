@group(0)
@binding(0)
var<storage, read_write> input_0_global: array<vec4<f32>>;

@group(0)
@binding(1)
var<storage, read_write> input_1_global: array<vec4<f32>>;

@group(0)
@binding(2)
var<storage, read_write> output_0_global: array<vec4<f32>>;

@group(0)
@binding(3)
var<storage, read_write> info: array<u32>;

const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;
const WORKGROUP_SIZE_Z = 1u;

@compute
@workgroup_size(16, 16, 1)
fn execute_unary_kernel_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
let id = (global_id.z * num_workgroups.x * WORKGROUP_SIZE_X * num_workgroups.y * WORKGROUP_SIZE_Y) + (global_id.y * num_workgroups.x * WORKGROUP_SIZE_X) + global_id.x;
let b_0_0 = info[5u];
let b_0_1 = id < b_0_0;
if b_0_1 {

for (var l_2_0: u32 = 0u; l_2_0 < 256u; l_2_0++) {
let b_2_1 = l_2_0 % 2u;
let b_2_2 = b_2_1 == 0u;
if b_2_2 {
var l_3_6: u32;
l_3_6 = info[0u];
let b_3_0 = select(vec4<f32>(0), input_0_global[id], id < l_3_6);
var l_3_7: u32;
l_3_7 = info[1u];
let b_3_1 = select(vec4<f32>(0), input_1_global[id], id < l_3_7);
let b_3_2 = b_3_0 * b_3_1;
let b_3_3 = cos(b_3_2);
var l_3_8: u32;
l_3_8 = info[2u];
let b_3_4 = select(vec4<f32>(0), output_0_global[id], id < l_3_8);
let b_3_5 = b_3_4 - b_3_3;
var l_3_9: u32;
var l_3_10: bool;
l_3_9 = info[2u];
l_3_10 = id < l_3_9;
if l_3_10 {
output_0_global[id] = b_3_5;
}
} else {
var l_3_6: u32;
l_3_6 = info[0u];
let b_3_0 = select(vec4<f32>(0), input_0_global[id], id < l_3_6);
var l_3_7: u32;
l_3_7 = info[1u];
let b_3_1 = select(vec4<f32>(0), input_1_global[id], id < l_3_7);
let b_3_2 = b_3_0 * b_3_1;
let b_3_3 = cos(b_3_2);
var l_3_8: u32;
l_3_8 = info[2u];
let b_3_4 = select(vec4<f32>(0), output_0_global[id], id < l_3_8);
let b_3_5 = b_3_4 + b_3_3;
var l_3_9: u32;
var l_3_10: bool;
l_3_9 = info[2u];
l_3_10 = id < l_3_9;
if l_3_10 {
output_0_global[id] = b_3_5;
}
}
}
}
}