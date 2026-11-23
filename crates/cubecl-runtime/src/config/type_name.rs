use alloc::format;
use alloc::string::{String, ToString};

/// How much of a type name a report keeps.
#[derive(Clone, Copy, Debug)]
pub enum TypeNameFormatLevel {
    /// The whole name, paths and generic arguments included.
    Full,
    /// The last path segment of the type, without its generic arguments.
    Short,
    /// The last path segment of the type and of each generic argument, joined by ` | `.
    Balanced,
}

impl TypeNameFormatLevel {
    /// `name` with only what this level keeps.
    pub fn format(self, name: &str) -> String {
        match self {
            Self::Full => name.to_string(),
            Self::Short => short_type_name(name).to_string(),
            Self::Balanced => balanced_type_name(name),
        }
    }
}

/// The last path segment of the type `name` names, without its generic arguments.
pub fn short_type_name(name: &str) -> &str {
    let outer = name.split('<').next().unwrap_or(name);
    outer.rsplit("::").next().unwrap_or(outer)
}

fn balanced_type_name(name: &str) -> String {
    let Some((outer, generics)) = name.split_once('<') else {
        return last_segment(name);
    };
    format!("{}{}", last_segment(outer), balanced_generics(generics))
}

/// The last path segment of `path`, without what closes an enclosing generic list.
fn last_segment(path: &str) -> String {
    path.rsplit("::")
        .next()
        .unwrap_or(path)
        .trim()
        .replace('>', "")
}

/// Each generic argument of `generics`, shortened, after a ` | `.
fn balanced_generics(generics: &str) -> String {
    generics
        .split(", ")
        .map(|argument| format!(" | {}", balanced_type_name(argument)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const KERNEL: &str = "burn_cubecl::kernel::unary_numeric::unary_numeric::UnaryNumeric<f32, burn_cubecl::tensor::base::CubeTensor<_>::copy::Copy, cubecl_cuda::runtime::CudaRuntime>";

    #[test]
    fn full_keeps_the_whole_name() {
        assert_eq!(TypeNameFormatLevel::Full.format(KERNEL), KERNEL);
    }

    #[test]
    fn short_keeps_the_last_segment_of_the_type() {
        assert_eq!(TypeNameFormatLevel::Short.format(KERNEL), "UnaryNumeric");
    }

    #[test]
    fn balanced_keeps_the_last_segment_of_each_argument() {
        assert_eq!(
            TypeNameFormatLevel::Balanced.format(KERNEL),
            "UnaryNumeric | f32 | CubeTensor | Copy | CudaRuntime"
        );
    }

    #[test]
    fn balanced_keeps_a_name_without_generics() {
        assert_eq!(
            TypeNameFormatLevel::Balanced.format("a::b::Kernel"),
            "Kernel"
        );
    }
}
