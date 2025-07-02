use cubecl::prelude::*;
use cubecl_core as cubecl;

#[cube(launch_unchecked)]
fn plane_dim(output: &mut Array<u32>) {
    output[0] = u32::cast_from(PLANE_DIM);
}

pub(crate) fn get_plane_dim<R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    cube_dim: CubeDim,
) -> u32 {
    unsafe {
        let elem = u32::as_elem_native_unchecked();
        let output = client.empty(elem.size());
        plane_dim::launch_unchecked(
            client,
            CubeCount::Static(1, 1, 1),
            cube_dim,
            ArrayArg::Handle {
                handle: ArrayHandleRef::<R>::from_raw_parts(&output, 1, elem.size()),
                vectorization_factor: 1,
            },
        );

        let value = client.read_one(output.binding());

        u32::from_bytes(&value)[0]
    }
}
