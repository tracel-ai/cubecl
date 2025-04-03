use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::stage::StageConfig;
use crate::matmul::components::{Ident, InputIdent, MatrixLayout};

// Warning: assumes contiguous tiling layout with
// Lhs: row major tiling order and row major matrix layout
// Rhs: col major tiling order and col major matrix layout

#[derive(CubeType, Clone, Copy)]
/// Makes stages actually share a common shared memories.
/// Tiles can be interleaved, in cols for Lhs, rows for Rhs
pub struct VirtualSharedMemory<ES: Numeric> {
    /// Underlying actual shared memory
    /// pub for testing
    pub physical: SharedMemory<Line<ES>>,
    #[cube(comptime)]
    offset: u32,
    #[cube(comptime)]
    jump: u32,
    #[cube(comptime)]
    /// Width of one buffer
    buffer_width: u32,

    #[cube(comptime)]
    num_smem: u32,
    #[cube(comptime)]
    num_buffers: u32,
}

#[cube]
impl<ES: Numeric> VirtualSharedMemory<ES> {
    fn virtual_to_physical_index_lhs(&self, virtual_index: u32) -> u32 {
        let total_width = self.num_buffers * self.buffer_width;
        let virtual_strided = virtual_index / total_width;
        let y = virtual_index % total_width;
        self.virtual_coordinates_to_physical_index_lhs(virtual_strided, y)
    }

    fn virtual_coordinates_to_physical_index_lhs(
        &self,
        virtual_strided: u32,
        virtual_contiguous: u32,
    ) -> u32 {
        let buffer = (virtual_contiguous / self.buffer_width) % self.num_buffers;
        let pos_in_buffer = virtual_contiguous % self.buffer_width;
        let pos_contiguous = self.offset + pos_in_buffer + buffer * self.jump;

        let physical_stride = self.num_buffers * self.num_smem * self.buffer_width;

        virtual_strided * physical_stride + pos_contiguous
    }

    pub fn read_at_index(&self, virtual_index: u32) -> Line<ES> {
        self.physical[self.virtual_to_physical_index_lhs(virtual_index)]
    }

    pub fn read_at_coordinates(&self, virtual_x: u32, virtual_y: u32) -> Line<ES> {
        self.physical[self.virtual_coordinates_to_physical_index_lhs(virtual_x, virtual_y)]
    }

    pub fn write_at_index(&mut self, virtual_index: u32, new_val: Line<ES>) {
        self.physical[self.virtual_to_physical_index_lhs(virtual_index)] = new_val;
    }

    pub fn write_at_coordinates(&mut self, virtual_x: u32, virtual_y: u32, new_val: Line<ES>) {
        self.physical[self.virtual_coordinates_to_physical_index_lhs(virtual_x, virtual_y)] =
            new_val;
    }
}

#[derive_cube_comptime]
pub enum SharedMemoryLayout {
    SideBySide,
    Interleaved,
}

#[cube]
pub fn make_virtual_shared_memories<ES: Numeric, S: StageConfig>(
    #[comptime] sizes: Vec<u32>,
    #[comptime] layout: SharedMemoryLayout,
    #[comptime] ident: Ident,
    #[comptime] config: S,
) -> Sequence<VirtualSharedMemory<ES>> {
    let td = config.tiling_dimensions(ident);
    make_virtual_shared_memories_testable::<ES>(
        sizes,
        layout,
        ident,
        config.matrix_layout(ident),
        config.line_size(ident),
        td.tile_shape_col(),
        td.tile_shape_row(),
    )
}

#[cube]
pub fn make_virtual_shared_memories_testable<ES: Numeric>(
    #[comptime] sizes: Vec<u32>,
    #[comptime] layout: SharedMemoryLayout,
    #[comptime] ident: Ident,
    #[comptime] matrix_layout: MatrixLayout,
    #[comptime] line_size: u32,
    #[comptime] tile_shape_col: u32,
    #[comptime] tile_shape_row: u32,
) -> Sequence<VirtualSharedMemory<ES>> {
    match ident.as_input() {
        InputIdent::Lhs => {
            if let MatrixLayout::ColMajor = matrix_layout {
                panic!("lhs must be col major")
            }
        }
        InputIdent::Rhs => {
            if let MatrixLayout::RowMajor = matrix_layout {
                panic!("rhs must be row major")
            }
        }
    }

    let (num_smem, total_size) = comptime! {
        let num_smem = sizes.len() ;
        let mut total_size: u32 = 0;
        for smem in 0..num_smem {
            total_size += sizes[smem];
        }
        (num_smem as u32, total_size)
    };

    let physical = SharedMemory::<ES>::new_lined(total_size, line_size);

    let mut sequence = Sequence::new();

    let buffer_width = comptime!(match matrix_layout {
        MatrixLayout::RowMajor => tile_shape_col,
        MatrixLayout::ColMajor => tile_shape_row,
    }) / line_size;

    let jump = comptime! {match layout {
        SharedMemoryLayout::SideBySide => 1u32,
        SharedMemoryLayout::Interleaved => num_smem,
    }} * buffer_width;

    let mut i = comptime![0u32];

    let num_buffers = 2u32;

    #[unroll]
    for _ in 0..num_smem {
        let offset = comptime! {match layout {
            SharedMemoryLayout::SideBySide => i * buffer_width * 2,
            SharedMemoryLayout::Interleaved => i * buffer_width,
        }};

        sequence.push(VirtualSharedMemory::<ES> {
            physical,
            offset,
            jump,
            buffer_width,
            num_smem,
            num_buffers,
        });

        comptime![i += 1];
    }

    sequence
}

#[cube(launch)]
fn test_virtual<ES: Numeric>(
    output: &mut Array<Line<ES>>,
    #[comptime] ident: Ident,
    #[comptime] smem_layout: SharedMemoryLayout,
) {
    let line_size = 4u32;
    let tile_shape_row = 16u32;
    let tile_shape_col = 16u32;
    let tile_size = tile_shape_row * tile_shape_col;
    let num_buffers = 2u32;
    let num_tiles_per_buffer = 2u32;
    let v_smem_size = num_buffers * num_tiles_per_buffer * tile_size / line_size;

    let matrix_layout = match ident.as_input() {
        InputIdent::Lhs => MatrixLayout::RowMajor,
        InputIdent::Rhs => MatrixLayout::ColMajor,
    };

    let mut v_smems = make_virtual_shared_memories_testable::<ES>(
        comptime!(vec![v_smem_size, v_smem_size]),
        smem_layout,
        ident,
        matrix_layout,
        4u32,
        16u32,
        16u32,
    );

    let v_smem_0 = v_smems.index_mut(0);
    for i in 0..v_smem_size {
        v_smem_0.write_at_index(i, Line::cast_from(i));
    }

    for i in 0..2 * v_smem_size {
        output[i] = v_smem_0.physical[i];
    }

    let v_smem_1 = v_smems.index_mut(1);
    for i in 0..v_smem_size {
        v_smem_1.write_at_index(i, Line::cast_from(i + 1000));
    }

    for i in 0..2 * v_smem_size {
        output[i] = v_smem_1.physical[i];
    }
}

pub fn virtual_smem_test_side_by_side<R: Runtime>(
    client: ComputeClient<R::Server, R::Channel>,
    ident: Ident,
) {
    let output = client.empty(core::mem::size_of::<u32>() * 512);

    test_virtual::launch::<f32, R>(
        &client,
        CubeCount::new_single(),
        CubeDim::new(1, 1, 1),
        unsafe { ArrayArg::from_raw_parts::<f32>(&output, 512, 4) },
        ident,
        SharedMemoryLayout::SideBySide,
    );

    let actual = client.read_one(output.binding());
    let actual = f32::from_bytes(&actual);

    let expected = &[
        0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0,
        4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0, 1000.0, 1000.0,
        1000.0, 1000.0, 1001.0, 1001.0, 1001.0, 1001.0, 1002.0, 1002.0, 1002.0, 1002.0, 1003.0,
        1003.0, 1003.0, 1003.0, 1004.0, 1004.0, 1004.0, 1004.0, 1005.0, 1005.0, 1005.0, 1005.0,
        1006.0, 1006.0, 1006.0, 1006.0, 1007.0, 1007.0, 1007.0, 1007.0, 8.0, 8.0, 8.0, 8.0, 9.0,
        9.0, 9.0, 9.0, 10.0, 10.0, 10.0, 10.0, 11.0, 11.0, 11.0, 11.0, 12.0, 12.0, 12.0, 12.0,
        13.0, 13.0, 13.0, 13.0, 14.0, 14.0, 14.0, 14.0, 15.0, 15.0, 15.0, 15.0, 1008.0, 1008.0,
        1008.0, 1008.0, 1009.0, 1009.0, 1009.0, 1009.0, 1010.0, 1010.0, 1010.0, 1010.0, 1011.0,
        1011.0, 1011.0, 1011.0, 1012.0, 1012.0, 1012.0, 1012.0, 1013.0, 1013.0, 1013.0, 1013.0,
        1014.0, 1014.0, 1014.0, 1014.0, 1015.0, 1015.0, 1015.0, 1015.0, 16.0, 16.0, 16.0, 16.0,
        17.0, 17.0, 17.0, 17.0, 18.0, 18.0, 18.0, 18.0, 19.0, 19.0, 19.0, 19.0, 20.0, 20.0, 20.0,
        20.0, 21.0, 21.0, 21.0, 21.0, 22.0, 22.0, 22.0, 22.0, 23.0, 23.0, 23.0, 23.0, 1016.0,
        1016.0, 1016.0, 1016.0, 1017.0, 1017.0, 1017.0, 1017.0, 1018.0, 1018.0, 1018.0, 1018.0,
        1019.0, 1019.0, 1019.0, 1019.0, 1020.0, 1020.0, 1020.0, 1020.0, 1021.0, 1021.0, 1021.0,
        1021.0, 1022.0, 1022.0, 1022.0, 1022.0, 1023.0, 1023.0, 1023.0, 1023.0, 24.0, 24.0, 24.0,
        24.0, 25.0, 25.0, 25.0, 25.0, 26.0, 26.0, 26.0, 26.0, 27.0, 27.0, 27.0, 27.0, 28.0, 28.0,
        28.0, 28.0, 29.0, 29.0, 29.0, 29.0, 30.0, 30.0, 30.0, 30.0, 31.0, 31.0, 31.0, 31.0, 1024.0,
        1024.0, 1024.0, 1024.0, 1025.0, 1025.0, 1025.0, 1025.0, 1026.0, 1026.0, 1026.0, 1026.0,
        1027.0, 1027.0, 1027.0, 1027.0, 1028.0, 1028.0, 1028.0, 1028.0, 1029.0, 1029.0, 1029.0,
        1029.0, 1030.0, 1030.0, 1030.0, 1030.0, 1031.0, 1031.0, 1031.0, 1031.0, 32.0, 32.0, 32.0,
        32.0, 33.0, 33.0, 33.0, 33.0, 34.0, 34.0, 34.0, 34.0, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0,
        36.0, 36.0, 37.0, 37.0, 37.0, 37.0, 38.0, 38.0, 38.0, 38.0, 39.0, 39.0, 39.0, 39.0, 1032.0,
        1032.0, 1032.0, 1032.0, 1033.0, 1033.0, 1033.0, 1033.0, 1034.0, 1034.0, 1034.0, 1034.0,
        1035.0, 1035.0, 1035.0, 1035.0, 1036.0, 1036.0, 1036.0, 1036.0, 1037.0, 1037.0, 1037.0,
        1037.0, 1038.0, 1038.0, 1038.0, 1038.0, 1039.0, 1039.0, 1039.0, 1039.0, 40.0, 40.0, 40.0,
        40.0, 41.0, 41.0, 41.0, 41.0, 42.0, 42.0, 42.0, 42.0, 43.0, 43.0, 43.0, 43.0, 44.0, 44.0,
        44.0, 44.0, 45.0, 45.0, 45.0, 45.0, 46.0, 46.0, 46.0, 46.0, 47.0, 47.0, 47.0, 47.0, 1040.0,
        1040.0, 1040.0, 1040.0, 1041.0, 1041.0, 1041.0, 1041.0, 1042.0, 1042.0, 1042.0, 1042.0,
        1043.0, 1043.0, 1043.0, 1043.0, 1044.0, 1044.0, 1044.0, 1044.0, 1045.0, 1045.0, 1045.0,
        1045.0, 1046.0, 1046.0, 1046.0, 1046.0, 1047.0, 1047.0, 1047.0, 1047.0, 48.0, 48.0, 48.0,
        48.0, 49.0, 49.0, 49.0, 49.0, 50.0, 50.0, 50.0, 50.0, 51.0, 51.0, 51.0, 51.0, 52.0, 52.0,
        52.0, 52.0, 53.0, 53.0, 53.0, 53.0, 54.0, 54.0, 54.0, 54.0, 55.0, 55.0, 55.0, 55.0, 1048.0,
        1048.0, 1048.0, 1048.0, 1049.0, 1049.0, 1049.0, 1049.0, 1050.0, 1050.0, 1050.0, 1050.0,
        1051.0, 1051.0, 1051.0, 1051.0, 1052.0, 1052.0, 1052.0, 1052.0, 1053.0, 1053.0, 1053.0,
        1053.0, 1054.0, 1054.0, 1054.0, 1054.0, 1055.0, 1055.0, 1055.0, 1055.0, 56.0, 56.0, 56.0,
        56.0, 57.0, 57.0, 57.0, 57.0, 58.0, 58.0, 58.0, 58.0, 59.0, 59.0, 59.0, 59.0, 60.0, 60.0,
        60.0, 60.0, 61.0, 61.0, 61.0, 61.0, 62.0, 62.0, 62.0, 62.0, 63.0, 63.0, 63.0, 63.0, 1056.0,
        1056.0, 1056.0, 1056.0, 1057.0, 1057.0, 1057.0, 1057.0, 1058.0, 1058.0, 1058.0, 1058.0,
        1059.0, 1059.0, 1059.0, 1059.0, 1060.0, 1060.0, 1060.0, 1060.0, 1061.0, 1061.0, 1061.0,
        1061.0, 1062.0, 1062.0, 1062.0, 1062.0, 1063.0, 1063.0, 1063.0, 1063.0,
    ];

    assert_eq!(actual, expected);
}

pub fn virtual_smem_test_interleaved<R: Runtime>(
    client: ComputeClient<R::Server, R::Channel>,
    ident: Ident,
) {
    let output = client.empty(core::mem::size_of::<u32>() * 512);

    test_virtual::launch::<f32, R>(
        &client,
        CubeCount::new_single(),
        CubeDim::new(1, 1, 1),
        unsafe { ArrayArg::from_raw_parts::<f32>(&output, 512, 4) },
        ident,
        SharedMemoryLayout::Interleaved,
    );

    let actual = client.read_one(output.binding());
    let actual = f32::from_bytes(&actual);

    let expected = &[
        0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 1000.0,
        1000.0, 1000.0, 1000.0, 1001.0, 1001.0, 1001.0, 1001.0, 1002.0, 1002.0, 1002.0, 1002.0,
        1003.0, 1003.0, 1003.0, 1003.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0,
        7.0, 7.0, 7.0, 7.0, 1004.0, 1004.0, 1004.0, 1004.0, 1005.0, 1005.0, 1005.0, 1005.0, 1006.0,
        1006.0, 1006.0, 1006.0, 1007.0, 1007.0, 1007.0, 1007.0, 8.0, 8.0, 8.0, 8.0, 9.0, 9.0, 9.0,
        9.0, 10.0, 10.0, 10.0, 10.0, 11.0, 11.0, 11.0, 11.0, 1008.0, 1008.0, 1008.0, 1008.0,
        1009.0, 1009.0, 1009.0, 1009.0, 1010.0, 1010.0, 1010.0, 1010.0, 1011.0, 1011.0, 1011.0,
        1011.0, 12.0, 12.0, 12.0, 12.0, 13.0, 13.0, 13.0, 13.0, 14.0, 14.0, 14.0, 14.0, 15.0, 15.0,
        15.0, 15.0, 1012.0, 1012.0, 1012.0, 1012.0, 1013.0, 1013.0, 1013.0, 1013.0, 1014.0, 1014.0,
        1014.0, 1014.0, 1015.0, 1015.0, 1015.0, 1015.0, 16.0, 16.0, 16.0, 16.0, 17.0, 17.0, 17.0,
        17.0, 18.0, 18.0, 18.0, 18.0, 19.0, 19.0, 19.0, 19.0, 1016.0, 1016.0, 1016.0, 1016.0,
        1017.0, 1017.0, 1017.0, 1017.0, 1018.0, 1018.0, 1018.0, 1018.0, 1019.0, 1019.0, 1019.0,
        1019.0, 20.0, 20.0, 20.0, 20.0, 21.0, 21.0, 21.0, 21.0, 22.0, 22.0, 22.0, 22.0, 23.0, 23.0,
        23.0, 23.0, 1020.0, 1020.0, 1020.0, 1020.0, 1021.0, 1021.0, 1021.0, 1021.0, 1022.0, 1022.0,
        1022.0, 1022.0, 1023.0, 1023.0, 1023.0, 1023.0, 24.0, 24.0, 24.0, 24.0, 25.0, 25.0, 25.0,
        25.0, 26.0, 26.0, 26.0, 26.0, 27.0, 27.0, 27.0, 27.0, 1024.0, 1024.0, 1024.0, 1024.0,
        1025.0, 1025.0, 1025.0, 1025.0, 1026.0, 1026.0, 1026.0, 1026.0, 1027.0, 1027.0, 1027.0,
        1027.0, 28.0, 28.0, 28.0, 28.0, 29.0, 29.0, 29.0, 29.0, 30.0, 30.0, 30.0, 30.0, 31.0, 31.0,
        31.0, 31.0, 1028.0, 1028.0, 1028.0, 1028.0, 1029.0, 1029.0, 1029.0, 1029.0, 1030.0, 1030.0,
        1030.0, 1030.0, 1031.0, 1031.0, 1031.0, 1031.0, 32.0, 32.0, 32.0, 32.0, 33.0, 33.0, 33.0,
        33.0, 34.0, 34.0, 34.0, 34.0, 35.0, 35.0, 35.0, 35.0, 1032.0, 1032.0, 1032.0, 1032.0,
        1033.0, 1033.0, 1033.0, 1033.0, 1034.0, 1034.0, 1034.0, 1034.0, 1035.0, 1035.0, 1035.0,
        1035.0, 36.0, 36.0, 36.0, 36.0, 37.0, 37.0, 37.0, 37.0, 38.0, 38.0, 38.0, 38.0, 39.0, 39.0,
        39.0, 39.0, 1036.0, 1036.0, 1036.0, 1036.0, 1037.0, 1037.0, 1037.0, 1037.0, 1038.0, 1038.0,
        1038.0, 1038.0, 1039.0, 1039.0, 1039.0, 1039.0, 40.0, 40.0, 40.0, 40.0, 41.0, 41.0, 41.0,
        41.0, 42.0, 42.0, 42.0, 42.0, 43.0, 43.0, 43.0, 43.0, 1040.0, 1040.0, 1040.0, 1040.0,
        1041.0, 1041.0, 1041.0, 1041.0, 1042.0, 1042.0, 1042.0, 1042.0, 1043.0, 1043.0, 1043.0,
        1043.0, 44.0, 44.0, 44.0, 44.0, 45.0, 45.0, 45.0, 45.0, 46.0, 46.0, 46.0, 46.0, 47.0, 47.0,
        47.0, 47.0, 1044.0, 1044.0, 1044.0, 1044.0, 1045.0, 1045.0, 1045.0, 1045.0, 1046.0, 1046.0,
        1046.0, 1046.0, 1047.0, 1047.0, 1047.0, 1047.0, 48.0, 48.0, 48.0, 48.0, 49.0, 49.0, 49.0,
        49.0, 50.0, 50.0, 50.0, 50.0, 51.0, 51.0, 51.0, 51.0, 1048.0, 1048.0, 1048.0, 1048.0,
        1049.0, 1049.0, 1049.0, 1049.0, 1050.0, 1050.0, 1050.0, 1050.0, 1051.0, 1051.0, 1051.0,
        1051.0, 52.0, 52.0, 52.0, 52.0, 53.0, 53.0, 53.0, 53.0, 54.0, 54.0, 54.0, 54.0, 55.0, 55.0,
        55.0, 55.0, 1052.0, 1052.0, 1052.0, 1052.0, 1053.0, 1053.0, 1053.0, 1053.0, 1054.0, 1054.0,
        1054.0, 1054.0, 1055.0, 1055.0, 1055.0, 1055.0, 56.0, 56.0, 56.0, 56.0, 57.0, 57.0, 57.0,
        57.0, 58.0, 58.0, 58.0, 58.0, 59.0, 59.0, 59.0, 59.0, 1056.0, 1056.0, 1056.0, 1056.0,
        1057.0, 1057.0, 1057.0, 1057.0, 1058.0, 1058.0, 1058.0, 1058.0, 1059.0, 1059.0, 1059.0,
        1059.0, 60.0, 60.0, 60.0, 60.0, 61.0, 61.0, 61.0, 61.0, 62.0, 62.0, 62.0, 62.0, 63.0, 63.0,
        63.0, 63.0, 1060.0, 1060.0, 1060.0, 1060.0, 1061.0, 1061.0, 1061.0, 1061.0, 1062.0, 1062.0,
        1062.0, 1062.0, 1063.0, 1063.0, 1063.0, 1063.0,
    ];

    assert_eq!(actual, expected);
}
