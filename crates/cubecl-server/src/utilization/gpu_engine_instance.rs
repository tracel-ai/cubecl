use cubecl_ir::AdapterLuid;

/// One instance of Windows' `\GPU Engine(*)` counters, named like
/// `pid_1234_luid_0x00000000_0x0000D1C2_phys_0_eng_3_engtype_3D`: one process's share of one
/// engine of one adapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuEngineInstance {
    pub adapter: AdapterLuid,
    pub engine: GpuEngine,
}

/// An engine of an adapter, across every process using it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct GpuEngine {
    /// Which card of a linked adapter; 0 for the rest.
    pub physical_adapter: u32,
    pub engine: u32,
}

impl GpuEngineInstance {
    /// The LUID is written high part first, and neither the process id nor the engine type is
    /// kept: an engine is shared by every process, and its type names it no better than its index.
    pub fn parse(name: &str) -> Option<Self> {
        let mut fields = name.split('_');
        let mut adapter = None;
        let mut physical_adapter = None;
        let mut engine = None;
        while let Some(field) = fields.next() {
            match field {
                "luid" => {
                    let high_part = Self::parse_hex(fields.next()?)?;
                    let low_part = Self::parse_hex(fields.next()?)?;
                    adapter = Some(AdapterLuid::from_parts(low_part, high_part as i32));
                }
                "phys" => physical_adapter = Some(fields.next()?.parse().ok()?),
                "eng" => engine = Some(fields.next()?.parse().ok()?),
                _ => {}
            }
        }
        Some(Self {
            adapter: adapter?,
            engine: GpuEngine {
                physical_adapter: physical_adapter?,
                engine: engine?,
            },
        })
    }

    fn parse_hex(field: &str) -> Option<u32> {
        let digits = field
            .strip_prefix("0x")
            .or_else(|| field.strip_prefix("0X"))?;
        u32::from_str_radix(digits, 16).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ADAPTER_LOW_PART: u32 = 0x0000_d1c2;

    fn instance_of(physical_adapter: u32, engine: u32) -> GpuEngineInstance {
        GpuEngineInstance {
            adapter: AdapterLuid::from_parts(ADAPTER_LOW_PART, 0),
            engine: GpuEngine {
                physical_adapter,
                engine,
            },
        }
    }

    #[test]
    fn a_name_gives_the_adapter_and_the_engine() {
        assert_eq!(
            GpuEngineInstance::parse("pid_1234_luid_0x00000000_0x0000D1C2_phys_0_eng_3_engtype_3D"),
            Some(instance_of(0, 3))
        );
    }

    #[test]
    fn an_engine_type_holding_an_underscore_or_nothing_still_parses() {
        for name in [
            "pid_1234_luid_0x00000000_0x0000D1C2_phys_1_eng_5_engtype_Compute_0",
            "pid_1234_luid_0x00000000_0x0000D1C2_phys_1_eng_5_engtype_",
        ] {
            assert_eq!(
                GpuEngineInstance::parse(name),
                Some(instance_of(1, 5)),
                "{name}"
            );
        }
    }

    #[test]
    fn hex_digits_parse_in_either_case() {
        assert_eq!(
            GpuEngineInstance::parse("pid_1234_luid_0x00000000_0x0000d1c2_phys_0_eng_3_engtype_3D"),
            Some(instance_of(0, 3))
        );
    }

    #[test]
    fn a_high_part_with_its_top_bit_set_is_the_negative_one_dxgi_reports() {
        let instance =
            GpuEngineInstance::parse("pid_1234_luid_0xFFFFFFFF_0x0000D1C2_phys_0_eng_3_engtype_3D");

        assert_eq!(
            instance.map(|instance| instance.adapter),
            Some(AdapterLuid::from_parts(ADAPTER_LOW_PART, -1))
        );
    }

    #[test]
    fn a_name_missing_a_field_is_rejected() {
        for name in [
            "pid_1234_phys_0_eng_3_engtype_3D",
            "pid_1234_luid_0x00000000_0x0000D1C2_eng_3_engtype_3D",
            "pid_1234_luid_0x00000000_0x0000D1C2_phys_0_engtype_3D",
        ] {
            assert_eq!(GpuEngineInstance::parse(name), None, "{name}");
        }
    }
}
