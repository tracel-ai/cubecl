@group(0) @binding(0)
var<storage, read_write> buffer_0_global: array<vec4<u32>>;

@group(0) @binding(1)
var<storage, read_write> buffer_1_global: array<vec4<u32>>;

@group(0) @binding(2)
var<storage, read_write> buffer_2_global: array<vec4<u32>>;

struct info_st {
    static_meta: array<u32, 4>,
}

@group(0) @binding(3)
var<storage, read> info: info_st;

const WORKGROUP_SIZE_X = 1u;
const WORKGROUP_SIZE_Y = 1u;
const WORKGROUP_SIZE_Z = 1u;

@compute @workgroup_size(1, 1, 1)
fn test_function_n_4(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>,) {
    let id = (u32(global_id.z) * u32(num_workgroups.x) * u32(WORKGROUP_SIZE_X) * u32(num_workgroups.y) * u32(WORKGROUP_SIZE_Y)) + (u32(global_id.y) * u32(num_workgroups.x) * u32(WORKGROUP_SIZE_X)) + u32(global_id.x);

    let l_2 = info.static_meta[u32(1)];
    let l_7 = id < l_2;
    if l_7 {
        let l_24 = info.static_meta[u32(0)];
        let l_26 = id < l_24;
        let l_27 = !l_26;
        if l_27 { }
        let l_28 = min(id, l_24);
        let l_29 = &buffer_0_global[l_28];
        let l_30 = info.static_meta[u32(1)];
        let l_32 = id < l_30;
        let l_33 = !l_32;
        if l_33 { }
        let l_34 = min(id, l_30);
        let l_35 = &buffer_1_global[l_34];
        let l_16 = *l_29;
        let l_17 = *l_35;
        let l_18 = l_16 <= l_17;
        let l_19 = vec4<u32>(l_18);
        let l_36 = info.static_meta[u32(2)];
        let l_38 = id < l_36;
        let l_39 = !l_38;
        if l_39 { }
        let l_40 = min(id, l_36);
        let l_41 = &buffer_2_global[l_40];
        * l_41 = l_19;
    }
}