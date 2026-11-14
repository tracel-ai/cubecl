//! Both kinds of device memory on the default device (Intel's USM device allocations where it has them, and
//! coarse-grained SVM, which every usable device has): pages written from the host, read and written by a kernel,
//! read back, and freed behind the queue.

use crate::compute::cl::{self, api};
use crate::device::{self, Info};
use crate::memory::{Memory, OpenClStorage};
use cubecl_server::storage::ComputeStorage;

const INCREMENT: &str = "__kernel void increment(__global const uint* a, __global uint* b) {
    size_t i = get_global_id(0);
    b[i] = a[i] + 1;
}";

#[test]
fn both_memories_carry_data_to_kernels_and_back() {
    let device = *device::devices()
        .first()
        .expect("the machine should have a usable OpenCL device");
    let info = Info(device);
    let context =
        cl3::context::create_context(&[device], std::ptr::null(), None, std::ptr::null_mut())
            .expect("a context should be created");
    // SAFETY: a queue on a live context and its device, default properties.
    let queue = unsafe {
        cl3::command_queue::create_command_queue_with_properties(context, device, [0].as_ptr())
    }
    .expect("a queue should be created");
    let program = cl3::program::create_program_with_source(context, &[INCREMENT])
        .expect("the program should be created");
    cl3::program::build_program(program, &[device], c"", None, std::ptr::null_mut())
        .expect("the program should build");
    let kernel =
        cl3::kernel::create_kernel(program, c"increment").expect("the kernel should be created");

    let input: Vec<u32> = (0..256).collect();
    let bytes = input.len() * size_of::<u32>();
    for memory in [Memory::of(device, &info), Memory::Svm] {
        let mut storage = OpenClStorage::new(memory, context, queue, 128, 1 << 20);
        let (a, b) = (storage.alloc(bytes as u64), storage.alloc(bytes as u64));
        let (a, b) = (
            a.expect("a page should be allocated"),
            b.expect("a page should be allocated"),
        );
        let (from, to) = (
            storage.get(&a).expect("the first page should be live"),
            storage.get(&b).expect("the second page should be live"),
        );
        let mut output = vec![0u32; input.len()];
        // SAFETY: the queue, kernel and pages are live, both copies blocking, the run finished before the read.
        unsafe {
            memory
                .copy(queue, true, from.at(0), input.as_ptr().cast(), bytes)
                .expect("the copy to the device should be enqueued");
            memory
                .set_arg(kernel, 0, from.at(0).cast_const())
                .expect("argument 0 should be set");
            memory
                .set_arg(kernel, 1, to.at(0).cast_const())
                .expect("argument 1 should be set");
            cl::launch(queue, kernel, [input.len() / 64, 1, 1], [64, 1, 1], false)
                .expect("the kernel should be enqueued");
            memory
                .copy(
                    queue,
                    true,
                    output.as_mut_ptr().cast(),
                    to.at(0).cast_const(),
                    bytes,
                )
                .expect("the output should be read back");
        }
        let expected: Vec<u32> = (1..=256).collect();
        assert_eq!(output, expected, "{memory:?}");
        storage.dealloc(a.id);
        storage.dealloc(b.id);
        // Dropping the storage finishes the queue and frees the released pages.
    }

    api().clReleaseKernel(kernel);
    api().clReleaseProgram(program);
    api().clReleaseCommandQueue(queue);
    api().clReleaseContext(context);
}
