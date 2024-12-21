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
let l_0_0 = info[5u];
let l_0_1 = id < l_0_0;
if l_0_1 {

for (var l_mut_2_0: u32 = 0u; l_mut_2_0 < 256u; l_mut_2_0++) {
let l_2_1 = l_mut_2_0 % 2u;
let l_2_2 = l_2_1 == 0u;
if l_2_2 {
var l_mut_3_6: u32;
var l_mut_3_7: bool;
var l_mut_3_8: vec4<f32>;
l_mut_3_6 = info[0u];
l_mut_3_7 = id < l_mut_3_6;
l_mut_3_8 = input_0_global[id];
let l_3_0 = vec4(
select(0f, l_mut_3_8[0], l_mut_3_7),
select(0f, l_mut_3_8[1], l_mut_3_7),
select(0f, l_mut_3_8[2], l_mut_3_7),
select(0f, l_mut_3_8[3], l_mut_3_7),
);
var l_mut_3_9: u32;
var l_mut_3_10: bool;
var l_mut_3_11: vec4<f32>;
l_mut_3_9 = info[1u];
l_mut_3_10 = id < l_mut_3_9;
l_mut_3_11 = input_1_global[id];
let l_3_1 = vec4(
select(0f, l_mut_3_11[0], l_mut_3_10),
select(0f, l_mut_3_11[1], l_mut_3_10),
select(0f, l_mut_3_11[2], l_mut_3_10),
select(0f, l_mut_3_11[3], l_mut_3_10),
);
let l_3_2 = l_3_0 * l_3_1;
let l_3_3 = cos(l_3_2);
var l_mut_3_12: u32;
var l_mut_3_13: bool;
var l_mut_3_14: vec4<f32>;
l_mut_3_12 = info[2u];
l_mut_3_13 = id < l_mut_3_12;
l_mut_3_14 = output_0_global[id];
let l_3_4 = vec4(
select(0f, l_mut_3_14[0], l_mut_3_13),
select(0f, l_mut_3_14[1], l_mut_3_13),
select(0f, l_mut_3_14[2], l_mut_3_13),
select(0f, l_mut_3_14[3], l_mut_3_13),
);
let l_3_5 = l_3_4 - l_3_3;
var l_mut_3_15: u32;
var l_mut_3_16: bool;
l_mut_3_15 = info[2u];
l_mut_3_16 = id < l_mut_3_15;
if l_mut_3_16 {
output_0_global[id] = l_3_5;
}
} else {
var l_mut_3_6: u32;
var l_mut_3_7: bool;
var l_mut_3_8: vec4<f32>;
l_mut_3_6 = info[0u];
l_mut_3_7 = id < l_mut_3_6;
l_mut_3_8 = input_0_global[id];
let l_3_0 = vec4(
select(0f, l_mut_3_8[0], l_mut_3_7),
select(0f, l_mut_3_8[1], l_mut_3_7),
select(0f, l_mut_3_8[2], l_mut_3_7),
select(0f, l_mut_3_8[3], l_mut_3_7),
);
var l_mut_3_9: u32;
var l_mut_3_10: bool;
var l_mut_3_11: vec4<f32>;
l_mut_3_9 = info[1u];
l_mut_3_10 = id < l_mut_3_9;
l_mut_3_11 = input_1_global[id];
let l_3_1 = vec4(
select(0f, l_mut_3_11[0], l_mut_3_10),
select(0f, l_mut_3_11[1], l_mut_3_10),
select(0f, l_mut_3_11[2], l_mut_3_10),
select(0f, l_mut_3_11[3], l_mut_3_10),
);
let l_3_2 = l_3_0 * l_3_1;
let l_3_3 = cos(l_3_2);
var l_mut_3_12: u32;
var l_mut_3_13: bool;
var l_mut_3_14: vec4<f32>;
l_mut_3_12 = info[2u];
l_mut_3_13 = id < l_mut_3_12;
l_mut_3_14 = output_0_global[id];
let l_3_4 = vec4(
select(0f, l_mut_3_14[0], l_mut_3_13),
select(0f, l_mut_3_14[1], l_mut_3_13),
select(0f, l_mut_3_14[2], l_mut_3_13),
select(0f, l_mut_3_14[3], l_mut_3_13),
);
let l_3_5 = l_3_4 + l_3_3;
var l_mut_3_15: u32;
var l_mut_3_16: bool;
l_mut_3_15 = info[2u];
l_mut_3_16 = id < l_mut_3_15;
if l_mut_3_16 {
output_0_global[id] = l_3_5;
}
}
}
}
}