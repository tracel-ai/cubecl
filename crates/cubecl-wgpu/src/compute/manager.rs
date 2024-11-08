use crate::compiler::base::WgpuCompiler;

use super::{stream::WgpuStream, WgpuServer};
use cubecl_common::stream::StreamId;
use cubecl_runtime::storage::BindingResource;
use std::{collections::BTreeMap, future::Future};

pub struct WgpuStreamManager {
    streams: BTreeMap<StreamId, WgpuStream>,
    buffer2stream: BTreeMap<wgpu::Id<wgpu::Buffer>, StreamId>,
    stream2buffer: BTreeMap<StreamId, Vec<wgpu::Id<wgpu::Buffer>>>,
}

impl WgpuStreamManager {
    pub fn read_buffer(
        &mut self,
        buffer: &wgpu::Buffer,
        offset: u64,
        size: u64,
        stream_id: StreamId,
    ) -> impl Future<Output = Vec<u8>> + 'static {
        if let Some(stream_id) = self.buffer2stream.get(&buffer.global_id()) {
            let stream = self.streams.get_mut(stream_id).unwrap();

            stream.flush();

            if let Some(bufs) = self.stream2buffer.remove(stream_id) {
                for buf in bufs {
                    self.buffer2stream.remove(&buf);
                }
            }
        }

        self.streams
            .get_mut(&stream_id)
            .unwrap()
            .read_buffer(buffer, offset, size)
    }
}
