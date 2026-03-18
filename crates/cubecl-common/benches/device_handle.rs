use criterion::{Criterion, criterion_group, criterion_main};
use cubecl_common::{device::DeviceService, device_handle::DeviceHandle, stub::Mutex};

use std::{hint::black_box, sync::Arc};

#[derive(Default)]
struct TestService {
    id: u64,
    items: Vec<usize>,
}

static DATA_SIZE: usize = 16;

impl TestService {
    pub fn compute(&mut self, data: [u32; DATA_SIZE]) {
        let count = 100;
        if self.items.is_empty() {
            for i in 0..black_box(count) {
                self.items.push(i);
            }
        }

        for i in 0..black_box(count) {
            let index = i % DATA_SIZE;
            let value = data[index];

            if i % 5 == 0 {
                self.items[i] += value as usize * 3;
            } else {
                self.items[i] += value as usize + 2;
            }
        }
    }
}

impl DeviceService for TestService {
    fn init(_device_id: cubecl_common::device::DeviceId) -> Self {
        Self::default()
    }

    fn utilities(&self) -> Arc<dyn std::any::Any + Send + Sync> {
        Arc::new(())
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let device_handle = DeviceHandle::<TestService>::new(cubecl_common::device::DeviceId {
        type_id: 0,
        index_id: 0,
    });
    c.bench_function("device handle single thread", |b| {
        let device = device_handle.clone();
        b.iter(|| {
            for _ in 0..black_box(1000) {
                let data = [1; DATA_SIZE];
                device.submit(move |service| service.compute(data));
            }
            let total = device.submit_blocking(|service| service.id).unwrap();
            black_box(total);
        })
    });

    c.bench_function("Mutex single thread", |b| {
        let device = Arc::new(Mutex::new(TestService::default()));
        b.iter(|| {
            for _ in 0..black_box(1000) {
                let data = [1; DATA_SIZE];
                let mut device = device.lock().unwrap();
                device.compute(data);
            }
            black_box(device.lock().unwrap().id);
        })
    });
    c.bench_function("device handle multi-threads", |b| {
        let device = device_handle.clone();
        b.iter(|| {
            let count = 5000;
            let num_threads = 4;
            let mut handles = Vec::with_capacity(num_threads);
            for _ in 0..num_threads {
                let device_cloned = device.clone();
                let thread = std::thread::spawn(move || {
                    for _ in 0..black_box(count) {
                        let data = [1; DATA_SIZE];
                        device_cloned.submit(move |service| service.compute(data));
                    }
                });
                handles.push(thread);
            }

            for handle in handles {
                handle.join().unwrap();
            }

            let total = device.submit_blocking(|service| service.id).unwrap();
            black_box(total);
        })
    });

    c.bench_function("Mutex multi-threads", |b| {
        let device = Arc::new(Mutex::new(TestService::default()));
        b.iter(|| {
            let count = 5000;
            let num_threads = 4;
            let mut handles = Vec::with_capacity(num_threads);
            for _ in 0..num_threads {
                let device_cloned = device.clone();
                let thread = std::thread::spawn(move || {
                    for _ in 0..black_box(count) {
                        let mut device = device_cloned.lock().unwrap();
                        let data = [1; DATA_SIZE];
                        device.compute(data);
                    }
                });
                handles.push(thread);
            }

            for handle in handles {
                handle.join().unwrap();
            }

            black_box(device.lock().unwrap().id);
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

// fn main() {
//     println!("Start");
//     let device_handle = DeviceHandle::<TestService>::new(cubecl_common::device::DeviceId {
//         type_id: 0,
//         index_id: 0,
//     });
//     let device = device_handle.clone();
//     for _ in 0..black_box(1000) {
//         device.submit(|service| service.compute());
//     }
//     let total = device.submit_blocking(|service| service.id).unwrap();
//     black_box(total);
//     println!("Completed");
// }
