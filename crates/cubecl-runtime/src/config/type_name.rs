use alloc::format;
use alloc::string::{String, ToString};

#[derive(Clone, Copy, Debug)]
/// How to format cubecl type names.
pub enum TypeNameFormatLevel {
    /// No formatting apply, full information is included.
    Full,
    /// Most information is removed for a small formatted name.
    Short,
    /// Balanced info is kept.
    Balanced,
}

/// Format a type name with different options.
pub fn type_name_format(name: &str, level: TypeNameFormatLevel) -> String {
    match level {
        TypeNameFormatLevel::Full => name.to_string(),
        TypeNameFormatLevel::Short => {
            if let Some(val) = name.split("<").next() {
                val.split("::").last().unwrap_or(name).to_string()
            } else {
                name.to_string()
            }
        }
        TypeNameFormatLevel::Balanced => {
            let mut split = name.split("<");
            let before_generic = split.next();
            let after_generic = split.next();

            let before_generic = match before_generic {
                None => return name.to_string(),
                Some(val) => val
                    .split("::")
                    .last()
                    .unwrap_or(val)
                    .trim()
                    .replace(">", "")
                    .to_string(),
            };
            let inside_generic = match after_generic {
                None => return before_generic.to_string(),
                Some(val) => {
                    let mut val = val.to_string();
                    for s in split {
                        val += "<";
                        val += s;
                    }
                    val
                }
            };

            let inside = type_name_list_format(&inside_generic, level);

            format!("{before_generic}{inside}")
        }
    }
}

fn type_name_list_format(name: &str, level: TypeNameFormatLevel) -> String {
    let mut acc = String::new();
    let splits = name.split(", ");

    for a in splits {
        acc += " | ";
        acc += &type_name_format(a, level);
    }

    acc
}

#[cfg(test)]
mod test {
    use super::*;

    #[test_log::test]
    fn test_format_name() {
        let full_name = "burn_cubecl::kernel::unary_numeric::unary_numeric::UnaryNumeric<f32, burn_cubecl::tensor::base::CubeTensor<_>::copy::Copy, cubecl_cuda::runtime::CudaRuntime>";
        let name = type_name_format(full_name, TypeNameFormatLevel::Balanced);

        assert_eq!(name, "UnaryNumeric | f32 | CubeTensor | Copy | CudaRuntime");
    }
}
