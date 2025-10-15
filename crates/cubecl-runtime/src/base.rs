use crate::{client::ComputeClient, server::ComputeServer};
use core::ops::DerefMut;
use hashbrown::HashMap;

/// The compute type has the responsibility to retrieve the correct compute client based on the
/// given device.
pub struct ComputeRuntime<Device, Server: ComputeServer> {
    clients: spin::Mutex<Option<HashMap<Device, ComputeClient<Server>>>>,
}

impl<Device, Server> Default for ComputeRuntime<Device, Server>
where
    Device: core::hash::Hash + PartialEq + Eq + Clone + core::fmt::Debug,
    Server: ComputeServer,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<Device, Server> ComputeRuntime<Device, Server>
where
    Device: core::hash::Hash + PartialEq + Eq + Clone + core::fmt::Debug,
    Server: ComputeServer,
{
    /// Create a new compute.
    pub const fn new() -> Self {
        Self {
            clients: spin::Mutex::new(None),
        }
    }

    /// Get the compute client for the given device.
    ///
    /// Provide the init function to create a new client if it isn't already initialized.
    pub fn client<Init>(&self, device: &Device, init: Init) -> ComputeClient<Server>
    where
        Init: Fn() -> ComputeClient<Server>,
    {
        let mut clients = self.clients.lock();

        if clients.is_none() {
            Self::register_inner(device, init(), &mut clients);
        }

        match clients.deref_mut() {
            Some(clients) => match clients.get(device) {
                Some(client) => client.clone(),
                None => {
                    let client = init();
                    clients.insert(device.clone(), client.clone());
                    client
                }
            },
            _ => unreachable!(),
        }
    }

    /// Register the compute client for the given device.
    ///
    /// # Note
    ///
    /// This function is mostly useful when the creation of the compute client can't be done
    /// synchronously and require special context.
    ///
    /// # Panics
    ///
    /// If a client is already registered for the given device.
    pub fn register(&self, device: &Device, client: ComputeClient<Server>) {
        let mut clients = self.clients.lock();

        Self::register_inner(device, client, &mut clients);
    }

    fn register_inner(
        device: &Device,
        client: ComputeClient<Server>,
        clients: &mut Option<HashMap<Device, ComputeClient<Server>>>,
    ) {
        if clients.is_none() {
            *clients = Some(HashMap::new());
        }

        if let Some(clients) = clients {
            if clients.contains_key(device) {
                panic!("Client already created for device {device:?}");
            }

            clients.insert(device.clone(), client);
        }
    }
}
