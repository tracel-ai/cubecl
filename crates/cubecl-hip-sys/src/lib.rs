#![allow(clippy::too_many_arguments)]
#![allow(clippy::useless_transmute)]
#![allow(improper_ctypes)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(unused_variables)]

pub mod hipconfig;
pub use hipconfig::*;

mod bindings;
#[allow(unused)]
pub use bindings::*;

#[cfg(target_os = "linux")]
#[cfg(test)]
mod tests {
    use super::bindings::*;
    use std::{ffi::CString, ptr, time::Instant};

    #[test]
    fn test_launch_kernel_end_to_end() {
        // Kernel that computes y values of a linear equation in slop-intercept form
        let source = CString::new(
            r#"
extern "C" __global__ void kernel(float a, float *x, float *b, float *out, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    out[tid] = x[tid] * a + b[tid];
  }
}
 "#,
        )
        .expect("Should construct kernel string");

        let func_name = CString::new("kernel".to_string()).unwrap();
        // reference: https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.0/user_guide/hip_rtc.html

        // Step 0: Select the GPU device
        unsafe {
            let status = hipSetDevice(0);
            assert_eq!(status, HIP_SUCCESS, "Should set the GPU device");
        }

        let free: usize = 0;
        let total: usize = 0;
        unsafe {
            let status = hipMemGetInfo(
                &free as *const _ as *mut usize,
                &total as *const _ as *mut usize,
            );
            assert_eq!(
                status, HIP_SUCCESS,
                "Should get the available memory of the device"
            );
            println!("Free: {} | Total:{}", free, total);
        }

        // Step 1: Create the program
        let mut program: hiprtcProgram = ptr::null_mut();
        unsafe {
            let status = hiprtcCreateProgram(
                &mut program,    // Program
                source.as_ptr(), // kernel string
                ptr::null(),     // Name of the file (there is no file)
                0,               // Number of headers
                ptr::null_mut(), // Header sources
                ptr::null_mut(), // Name of header files
            );
            assert_eq!(
                status, hiprtcResult_HIPRTC_SUCCESS,
                "Should create the program"
            );
        }

        // Step 2: Compile the program
        unsafe {
            let status = hiprtcCompileProgram(
                program,         // Program
                0,               // Number of options
                ptr::null_mut(), // Clang Options
            );
            if status != hiprtcResult_HIPRTC_SUCCESS {
                let mut log_size: usize = 0;
                let status = hiprtcGetProgramLogSize(program, &mut log_size as *mut usize);
                assert_eq!(
                    status, hiprtcResult_HIPRTC_SUCCESS,
                    "Should retrieve the compilation log size"
                );
                println!("Compilation log size: {log_size}");
                let mut log_buffer = vec![0i8; log_size];
                let status = hiprtcGetProgramLog(program, log_buffer.as_mut_ptr());
                assert_eq!(
                    status, hiprtcResult_HIPRTC_SUCCESS,
                    "Should retrieve the compilation log contents"
                );
                let log = std::ffi::CStr::from_ptr(log_buffer.as_ptr());
                println!("Compilation log: {}", log.to_string_lossy());
            }
            assert_eq!(
                status, hiprtcResult_HIPRTC_SUCCESS,
                "Should compile the program"
            );
        }

        // Step 3: Load compiled code
        let mut code_size: usize = 0;
        unsafe {
            let status = hiprtcGetCodeSize(program, &mut code_size);
            assert_eq!(
                status, hiprtcResult_HIPRTC_SUCCESS,
                "Should get size of compiled code"
            );
        }
        let mut code: Vec<u8> = vec![0; code_size];
        unsafe {
            let status = hiprtcGetCode(program, code.as_mut_ptr() as *mut _);
            assert_eq!(
                status, hiprtcResult_HIPRTC_SUCCESS,
                "Should load compiled code"
            );
        }

        // Step 4: Once the compiled code is loaded, the program can be destroyed
        unsafe {
            let status = hiprtcDestroyProgram(&mut program as *mut *mut _);
            assert_eq!(
                status, hiprtcResult_HIPRTC_SUCCESS,
                "Should destroy the program"
            );
        }
        assert!(!code.is_empty(), "Generated code should not be empty");

        // Step 5: Allocate Memory
        let n = 1024;
        let a = 2.0f32;
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
        let mut out: Vec<f32> = vec![0.0; n];
        // Allocate GPU memory for x, y, and out
        // There is no need to allocate memory for a and n as we can pass
        // host pointers directly to kernel launch function
        let mut device_x: *mut ::std::os::raw::c_void = std::ptr::null_mut();
        let mut device_b: *mut ::std::os::raw::c_void = std::ptr::null_mut();
        let mut device_out: *mut ::std::os::raw::c_void = std::ptr::null_mut();
        unsafe {
            let status_x = hipMalloc(&mut device_x, n * std::mem::size_of::<f32>());
            assert_eq!(status_x, HIP_SUCCESS, "Should allocate memory for device_x");
            let status_b = hipMalloc(&mut device_b, n * std::mem::size_of::<f32>());
            assert_eq!(status_b, HIP_SUCCESS, "Should allocate memory for device_b");
            let status_out = hipMalloc(&mut device_out, n * std::mem::size_of::<f32>());
            assert_eq!(
                status_out, HIP_SUCCESS,
                "Should allocate memory for device_out"
            );
        }

        // Step 6: Copy data to GPU memory
        unsafe {
            let status_device_x = hipMemcpy(
                device_x,
                x.as_ptr() as *const libc::c_void,
                n * std::mem::size_of::<f32>(),
                hipMemcpyKind_hipMemcpyHostToDevice,
            );
            assert_eq!(
                status_device_x, HIP_SUCCESS,
                "Should copy device_x successfully"
            );
            let status_device_b = hipMemcpy(
                device_b,
                b.as_ptr() as *const libc::c_void,
                n * std::mem::size_of::<f32>(),
                hipMemcpyKind_hipMemcpyHostToDevice,
            );
            assert_eq!(
                status_device_b, HIP_SUCCESS,
                "Should copy device_b successfully"
            );
            // Initialize the output memory on device to 0.0
            let status_device_out = hipMemcpy(
                device_out,
                out.as_ptr() as *const libc::c_void,
                n * std::mem::size_of::<f32>(),
                hipMemcpyKind_hipMemcpyHostToDevice,
            );
            assert_eq!(
                status_device_out, HIP_SUCCESS,
                "Should copy device_out successfully"
            );
        }

        // Step 7: Create the module containing the kernel and get the function that points to it
        let mut module: hipModule_t = ptr::null_mut();
        let mut function: hipFunction_t = ptr::null_mut();
        unsafe {
            let status_module =
                hipModuleLoadData(&mut module, code.as_ptr() as *const libc::c_void);
            assert_eq!(
                status_module, HIP_SUCCESS,
                "Should load compiled code into module"
            );
            let status_function = hipModuleGetFunction(&mut function, module, func_name.as_ptr());
            assert_eq!(
                status_function, HIP_SUCCESS,
                "Should return module function"
            );
        }

        // Step 8: Launch Kernel
        let start_time = Instant::now();
        // Create the array of arguments to pass to the kernel
        // They must be in the same order as the order of declaration of the kernel arguments
        let mut args: [*mut libc::c_void; 5] = [
            &a as *const _ as *mut libc::c_void,
            &device_x as *const _ as *mut libc::c_void,
            &device_b as *const _ as *mut libc::c_void,
            &device_out as *const _ as *mut libc::c_void,
            &n as *const _ as *mut libc::c_void,
        ];
        let block_dim_x: usize = 64;
        let grid_dim_x: usize = n / block_dim_x;
        // We could use the default stream by passing 0 to the launch kernel but for the sake of
        // coverage we create a stream explicitly
        let mut stream: hipStream_t = std::ptr::null_mut();
        unsafe {
            let stream_status = hipStreamCreate(&mut stream);
            assert_eq!(stream_status, HIP_SUCCESS, "Should create a stream");
        }
        unsafe {
            let status_launch = hipModuleLaunchKernel(
                function, // Kernel function
                block_dim_x as u32,
                1,
                1, // Grid dimensions (group of blocks)
                grid_dim_x as u32,
                1,
                1,                 // Block dimensions (group of threads)
                0,                 // Shared memory size
                stream,            // Created stream
                args.as_mut_ptr(), // Kernel arguments
                ptr::null_mut(),   // Extra options
            );
            assert_eq!(status_launch, HIP_SUCCESS, "Should launch the kernel");
        }
        // not strictly necessary but for the sake of coverage we sync here
        unsafe {
            let status = hipDeviceSynchronize();
            assert_eq!(status, HIP_SUCCESS, "Should sync with the device");
        }
        let duration = start_time.elapsed();
        println!("Execution time: {}µs", duration.as_micros());

        // Step 9: Copy the result back to host memory
        unsafe {
            hipMemcpy(
                out.as_mut_ptr() as *mut libc::c_void,
                device_out,
                n * std::mem::size_of::<f32>(),
                hipMemcpyKind_hipMemcpyDeviceToHost,
            );
        }

        // Step 10: Verify the results
        for i in 0..n {
            let result = out[i];
            let expected = a * x[i] + b[i];
            assert_eq!(result, expected, "Output mismatch at index {}", i);
        }

        // Step 11: Free up allocated memory on GPU device
        unsafe {
            let status = hipFree(device_x);
            assert_eq!(status, HIP_SUCCESS, "Should free device_x successfully");
            let status = hipFree(device_b);
            assert_eq!(status, HIP_SUCCESS, "Should free device_b successfully");
            let status = hipFree(device_out);
            assert_eq!(status, HIP_SUCCESS, "Should free device_out successfully");
        }
    }
}
