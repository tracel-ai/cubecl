use cubecl_common::stream_id::StreamId;

use crate::server::ProfilingToken;

#[derive(Debug)]
pub struct ProfilingPriotityQueue {
    items: spin::Mutex<Vec<ProfilingItem>>,
}

#[derive(new, Debug)]
struct ProfilingItem {
    stream_id: StreamId,
    num_allowed: u32,
    init_token: InitToken,
}

#[derive(Debug, PartialEq, Eq)]
enum InitToken {
    Uninitialized,
    Initialized(ProfilingToken),
    Finished,
}

impl ProfilingPriotityQueue {
    pub fn new() -> Self {
        Self {
            items: spin::Mutex::new(Vec::new()),
        }
    }

    /// In a multi-threaded environment, you don't want your profiling tasks to include jobs other than
    /// those from the current thread. If the current thread uses recursive profiling, you want
    /// all of those jobs to be completed before the ones from another thread.
    ///
    /// We stop prioritizing the current thread when the first profiling is done, to allow other
    /// threads to also perform recursive profiling.
    pub fn aquire_profile_priority(&self) -> (StreamId, bool) {
        let stream_id = StreamId::current();
        let mut items = self.items.lock();

        // Current stream is allowed.
        if let Some(current) = items.get_mut(0) {
            if current.stream_id == stream_id && !matches!(current.init_token, InitToken::Finished) {
                current.num_allowed += 1;
                // Aquired profile for stream.
                return (stream_id, false);
            }
        }

        // If we succeeded to register the current task in one of the existing profiling items.
        let mut registered = false;

        for current in items.iter_mut() {
            if current.stream_id == stream_id && !matches!(current.init_token, InitToken::Finished) {
                current.num_allowed += 1;
                registered = true;
                break;
            }
        }
        if !registered {
            // Register profile
            items.push(ProfilingItem::new(stream_id, 1, InitToken::Uninitialized));
        }

        // We init only if the current stream is the one that added the new
        // profiling item.
        let should_init = !registered;

        if !registered && items.len() == 1 {
            // Current stream is new, should also init the token.
            (stream_id, should_init)
        } else {
            core::mem::drop(items);

            loop {
                // Stream is waiting for priority
                std::thread::sleep(core::time::Duration::from_millis(10));

                let mut items = self.items.lock();
                if let Some(current) = items.get_mut(0) {
                    if current.stream_id == stream_id {
                        return (stream_id, should_init);
                    }
                }
            }
        }
    }

    pub fn set_profile_priotity_init_token(&self, init_token: ProfilingToken) {
        let mut items = self.items.lock();
        let current = items.get_mut(0).unwrap();

        if matches!(current.init_token, InitToken::Uninitialized) {
            current.init_token = InitToken::Initialized(init_token);
        }
    }

    pub fn release_profile_priotity(&self, stream_id: StreamId, token: ProfilingToken) {
        let mut items = self.items.lock();
        let current = items.get_mut(0).unwrap();

        if current.stream_id != stream_id {
            panic!("Wrong releasing priority.");
        }

        current.num_allowed -= 1;

        if current.init_token == InitToken::Initialized(token) {
            current.init_token = InitToken::Finished;
        }

        if current.num_allowed == 0 {
            items.remove(0);
        }
    }
}
