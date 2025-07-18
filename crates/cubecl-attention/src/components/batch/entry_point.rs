use crate::components::args;
use crate::components::args::AttentionArgs;
use crate::components::args::TensorInput;
use crate::components::args::TensorOutput;
use crate::components::batch::BatchAttentionFamily;
use crate::components::batch::CubeCountInput;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

type Input<Args, EI> = <Args as AttentionArgs>::Input<EI>;
type Output<Args, EO> = <Args as AttentionArgs>::Output<EO>;

#[cube(launch_unchecked)]
/// Launches the attention kernel
pub(crate) fn attention<
    Args: AttentionArgs,
    EI: Numeric,
    ES: Numeric,
    EM: Numeric,
    EA: Numeric,
    EO: Numeric,
    BMMF: BatchAttentionFamily,
>(
    inputs: &Input<Args, EI>,
    output: &mut Output<Args, EO>,
    cube_count_args: CubeCountInput,
    #[comptime] config: BMMF::Config,
) {
    let mut state = Args::init_state(inputs, output);

    let query = TensorInput::<EI, EO, Args>::new(&state, args::TensorInputIdent::Query);
    let key = TensorInput::<EI, EO, Args>::new(&state, args::TensorInputIdent::Key);
    let value = TensorInput::<EI, EO, Args>::new(&state, args::TensorInputIdent::Value);
    let mut out = TensorOutput::<EI, EO, Args>::new(&mut state);

    let query = VirtualTensor::<EI>::new::<TensorInput<EI, EO, Args>>(&query);
    let key = VirtualTensor::<EI>::new::<TensorInput<EI, EO, Args>>(&key);
    let value = VirtualTensor::<EI>::new::<TensorInput<EI, EO, Args>>(&value);
    let out = VirtualTensor::<EO, ReadWrite>::new::<TensorOutput<EI, EO, Args>>(&mut out);

    very_naive_attention(query, key, value, out);
}

#[cube]
/// The goal is only correctness, no effort in performance
fn very_naive_attention<EI: Numeric, EA: Float, EO: Numeric>(
    query: VirtualTensor<EI>,
    key: VirtualTensor<EI>,
    value: VirtualTensor<EI>,
    out: VirtualTensor<EO, ReadWrite>,
) {
    if UNIT_POS != 0 {
        terminate!()
    }

    // let problem = AttentionProblem {
    //     batch: query.shape[0],
    //     seq_q: query.shape[1],
    //     seq_k: key.shape[1],
    //     num_heads: query.shape[2],
    //     head_dim: query.shape[3],
    //     masked: false,
    // };

    let batch = query.shape(0);
    let seq_q = query.shape(1);
    let seq_k = key.shape(1);
    let num_heads = query.shape(2);
    let head_dim = query.shape(3);
    let masked = false;

    // Constant for scaling
    let scale = EA::from_int(1) / EA::sqrt(EA::cast_from(head_dim));

    for b in 0..batch {
        for h in 0..num_heads {
            for i in 0..seq_q {
                // Step 1: Compute attention scores (A[i,j]) for all j in seq_k
                let mut scores = Array::new(seq_k);
                for j in 0..seq_k {
                    // Dot product Q[b,i,h,:] · K[b,j,h,:]
                    let mut dot = EA::from_int(0);
                    for d in 0..head_dim {
                        let q_idx = b * query.stride(0)
                            + i * query.stride(1)
                            + h * query.stride(2)
                            + d * query.stride(3);
                        let k_idx = b * key.stride(0)
                            + j * key.stride(1)
                            + h * key.stride(2)
                            + d * key.stride(3);
                        let q_val = ES::cast_from(query[q_idx]);
                        let k_val = ES::cast_from(key[k_idx]);
                        dot += (q_val * k_val).cast_into();
                    }
                    dot *= scale;

                    // Apply mask if present
                    scores[j] = if masked {
                        let m_idx = b * mask_strides[0]
                            + i * mask_strides[1]
                            + h * mask_strides[2]
                            + j * mask_strides[3];
                        let m_val = mask.unwrap()[m_idx].cast_into();
                        if m_val != P::EM::from_int(0) {
                            P::EA::new(f32::NEG_INFINITY)
                        } else {
                            dot
                        }
                    } else {
                        dot
                    };
                }

                // Step 2: Compute softmax over scores
                let mut max_score = P::EA::new(f32::NEG_INFINITY);
                for i in 0..seq_k {
                    let val = scores[i];
                    if val > max_score {
                        max_score = val;
                    }
                }
                let mut sum_exp = P::EA::from_int(0);
                let mut exp_scores = vec![P::EA::from_int(0); seq_k];
                for j in 0..seq_k {
                    let e = P::EA::exp(scores[j] - max_score);
                    exp_scores[j] = e;
                    sum_exp += e;
                }

                // Step 3: Compute context vector: sum_j softmax_j * V[b,j,h,:]
                for d in 0..head_dim {
                    let mut ctx = P::EA::from_int(0);
                    for j in 0..seq_k {
                        let v_idx = b * value_strides[0]
                            + j * value_strides[1]
                            + h * value_strides[2]
                            + d * value_strides[3];
                        let v_val = value[v_idx].cast_into().cast_into();
                        let softmax_j = exp_scores[j] / sum_exp;
                        ctx += softmax_j * v_val;
                    }
                    let out_idx = b * out_strides[0]
                        + i * out_strides[1]
                        + h * out_strides[2]
                        + d * out_strides[3];
                    out[out_idx] = ctx.cast_into();
                }
            }
        }
    }

    out
}
