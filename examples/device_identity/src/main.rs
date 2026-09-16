//! Prints every device this build reaches, then which of them are the same card.

use cubecl::{
    Device,
    device::{WgpuBackend, WgpuDeviceKind},
    ir::{DeviceIdentity, PhysicalDevice},
};

fn main() {
    let mut devices: Vec<(String, Device)> = (0..)
        .map_while(|index| Some((format!("cuda:{index}"), Device::cuda(index).ok()?)))
        .collect();

    for api in [WgpuBackend::Vulkan, WgpuBackend::Dx12] {
        let kinds = [
            WgpuDeviceKind::DiscreteGpu,
            WgpuDeviceKind::IntegratedGpu,
            WgpuDeviceKind::VirtualGpu,
        ];
        for kind in kinds {
            for index in 0.. {
                let device = match api {
                    WgpuBackend::Vulkan => Device::vulkan(kind(index)),
                    _ => Device::dx12(kind(index)),
                };
                let Ok(device) = device else {
                    break;
                };
                devices.push((format!("{api:?}:{:?}", kind(index)), device));
            }
        }
    }

    let identities: Vec<(String, DeviceIdentity)> = devices
        .into_iter()
        .map(|(label, device)| (label, device.client().properties().identity.clone()))
        .collect();

    for (label, identity) in &identities {
        report(label, identity);
    }

    let mut cards: Vec<(&PhysicalDevice, Vec<&str>)> = Vec::new();
    for (label, identity) in &identities {
        let Some(physical) = &identity.physical else {
            continue;
        };
        match cards
            .iter_mut()
            .find(|(card, _)| card.is_same_card(physical))
        {
            Some((_, labels)) => labels.push(label),
            None => cards.push((physical, vec![label])),
        }
    }
    for (_, labels) in cards {
        println!("one card: {}", labels.join(", "));
    }
}

fn report(device: &str, identity: &DeviceIdentity) {
    println!("{device}: {} ({})", identity.name, identity.fingerprint);
    let Some(physical) = &identity.physical else {
        println!("  no card");
        return;
    };
    let unknown = || "?".to_string();
    let pci_address = physical
        .pci_address
        .map_or_else(unknown, |address| address.to_string());
    let luid = physical.luid.map_or_else(unknown, |luid| {
        luid.bytes()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    });
    let vendor = physical
        .vendor
        .map_or_else(unknown, |vendor| vendor.to_string());
    println!("  pci {pci_address}  luid {luid}  vendor {vendor}");
}
