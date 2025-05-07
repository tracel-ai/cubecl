# Quantized matrix multiplication

To make matrix multiplication faster,
we replace floating-point arithmetic using `f32`
with integer arithmetic using a mix of `u8`, `u16` and `i32`.
The benefits are twofold.
First,
we replace `Tensor<f32>` with `Tensor<u8>` to reduce memory cost by a factor of 4.
This leads to faster read and write operations into global memory.
Second,
integer operations are often faster than their floating-point counterparts.

In this section,
we start by presenting a more mathematical overview of the algorithm,
before discussing implementation.

## Mathematical formulation

### Scalar quantization

A real number \\(a\\) can be approximated by an integer \\(q\\) using the formula
\\[
    a \approx s(q - z).
\\]
In this equation \\(s\\) is a scaling factor and is also a real number,
while \\(z\\) is called the zero-offset and is an integer.
In theory,
with this approximation,
we can represent exactly all real numbers that are integral multiples of \\(s\\).
All other real numbers are rounded up to the closest representable value.
However, in practice, the range of \\(q\\) is limited by its representation (e.g. `u8`, `i32`).
Hence, the zero-offset \\(z\\) allows us to slide the interval of representable numbers toward
an interval we are interested in a particular application.
Also, by using the same type for \\(q\\) and \\(z\\),
we assure that 0 is exactly representable.

The multiplication of two real numbers is equivalent to
\\[
  a b = s_a s_b (q_a - z_a) (q_b - z_b).
\\]
However,
we are more interested in the quantized version \\(q_c\\) of \\(c = ab \\).
Given we want to approximate \\(c\\) with scaling \\(s_c\\) and zero-offset \\(z_c\\),
we have
\\[
  q_c =
  z_c + \frac{s_a s_b}{s_c} (q_a - z_a) (q_b - z_b).
\\]
Except for the factor \\( (s_a s_b) / s_c \\), the above equation involves only integer arithmetic.
However,
we can always find two integers \\(u, v\\) such that
\\[
  \frac uv \approx \frac{s_a s_b}{s_c}
\\]
is a satisfying approximation.
This leads to the final approximation for quantized multiplication
\\[
  q_c \approx z_c + \frac uv (q_a - z_a)(q_b - z_b)
\\]
requiring only integer arithmetic.

### Matrix quantization

The same idea holds for matrix multiplication.
To distinguish matrices from scalars,
we use capital letters for the former and lower letters for the latter.

A real matrix \\( A \\) is approximated by an integer matrix \\( Q \\) using
\\[
  A \approx s (Q - z N).
\\]
Here \\( N \\) is a matrix of ones the same size as \\( A \\).
For two matrices \\(A \\) and \\( B \\) with respective shape \\(m \times k\\)
and \\(k \times n\\) and their product \\( C \\) of shape \\( m \times n \\),
we have, similar to the scalar case that
\\[
  Q_c \approx z_c N_c + \frac uv (Q_a - z_a N_a)(Q_b - z_b N_b).
\\]

## Implementation

As an example,
we describe how to implement the quantized matrix multiplication
where the elements of \\(Q_a\\), \\(Q_b\\) and \\(Q_c\\) and the zero-offsets are represented as `u8`.

To compute \\(Q_a - z_a N_a \\),
we first convert the values to `i16` before performing the subtraction.
Then, we can compute the product \\((Q_a - z_a N_a)(Q_b - z_b N_b)\\)
by converting the values to `i32` before multiplying.
Of course,
in practice, we perform all these conversions on-the-fly to avoid wastefully allocating new matrices.

Now, suppose that \\(x\\) is a single element in the resulting matrix and \\(y\\)
is the element with the same position in \\(Q_c\\).
We still need to compute the following
\\[
  y = z_c + \frac uv \cdot x.
\\]
The tricky part here is the product.
First,
we impose that \\( v \\) is a power of 2 so that dividing by \\( v \\)
is equivalent to right-shifting the product \\( u x \\).
Then, we need to find the best values \\( u \\) and \\( v \\)
for the scaling factor \\( \sigma = \frac{s_a s_b}{s_c} \\).
The trick is to cleverly multiply \\( \sigma \\) by 1, to get a form that allows us to work with powers of 2:
\\[
  \sigma = \frac{2^{31 - f}}{2^{31 - f}} \sigma
\\]
where \\(2^f\\) is the smallest power of 2 larger than \\(\sigma\\).
For example, if \\(\sigma = 0.3\\), then \\(f = -1\\) as \\(2^{-1} = 0.5 > 0.3 \\)
and \\(2^{-2} = 0.25 < 0.3\\).
From this, we deduce we that we can use \\(u = 2^{31 - f} \sigma\\) rounded to the
nearest `i64` value and \\(v = 2^{31 - f}\\).
This gives us a 31-bit approximation for multiplying by \\(\sigma\\), which is the best
we can achieve when the other multiplicand is an `i32`.
Indeed, we need to keep one bit for the sign.
To properly round the product,
one can add \\(\frac v 2\\) to the product before right shifting.

A naive implementation of the above algorithm looks like the following.
```rust
fn scaling_ratio(sigma: f32) -> (i64, u32) {
    let log = x.log2().ceil() as i32;
    let u = (x * 2.0_f32.powi(31 - log)).round() as i64;
    let v_shift = (31 - log) as u32;
    (u, v_shift)
}

fn approx_mul(x: i32, u: i64, v_shift: u32) -> i32 {
    let prod = (x as i64) * u;
    let rounding: i64 = 1 << (v_shift - 1);
    let prod_with_rounding = prod + self.rounding;
    (prod_with_rounding >> self.shift) as i32
}

fn clamp_to_u8(x: i32) -> u8 {
    if x < 0 {
      0
    } else if x > u8::MAX as i32 {
      u8::Max
    } else {
      x as u8
    }
}

struct Matrix {
  scaling: f32,
  zero_offset: u8,
  // ... other fields to store the matrix elements.
}

impl Matrix {
  fn quantized_mul(&self, other: &Self, output: &mut Self) -> Self {
      // assume the shapes of the matrices match.

      let sigma = self.scaling * other.scaling / output.scaling;
      let (u, v_shift) = scaling_ratio(sigma);

      for row in 0..self.row_count() {
          for col in 0..other.col_count() {
              let mut sum: i32 = 0;
              for middle in 0..self.col_count() {
                  let a = self.get(row, middle) as i16 - self.zero_offset as i16;
                  let b = other.get(middle, col) as i16 - other.zero_offset as i16;
                  sum += (a as i32) * (b as i32);
              }
              sum = approx_mul(sum, u, v_shift);

              output.update(row, col, clamp_to_u8(sum + output.zero_offset as i32))
          }
      }
  }

  // return the value at (row, col)
  fn get(&self, row: usize, col: usize) -> u8 { /* ... */ }

  // replace the value at (row, col) with the given value.
  fn update(&mut self, row: usize, col: usize, value: u8) { /* ... */ }

  // return the number of rows of the matrix.
  fn row_count(&self) -> usize { /* ... */ }

  // return the number of columns of the matrix.
  fn col_count(&self) -> usize { /* ... */ }
}
```
Of course,
in CubeCL, we stride to provide the fastest implementation for GPU devices.
As such, the example emphasizes the correct type casting to demonstrate how this is achieved in CubeCL.
