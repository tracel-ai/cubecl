/// Represent the quantization of a f32 into an i8 using the symmetric scheme.

#[derive(Clone, Copy)]
pub struct SymQ8;

impl SymQ8 {
    pub fn quantize(val: f32, scaling: f32) -> i8 {
        let min = -scaling * (i8::MIN as f32);
        let max = min + 255.0 * scaling;
        if val < min {
            i8::MIN
        } else if val > max {
            i8::MAX
        } else {
            ((val - min) / scaling).round() as i8
        }
    }

    pub fn dequantize(val: i8, scaling: f32) -> f32 {
        val as f32 * scaling
    }
}
