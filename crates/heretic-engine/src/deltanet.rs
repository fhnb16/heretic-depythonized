// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Gated DeltaNet linear attention (Qwen3.5), ported from
//! `transformers/models/qwen3_5/modeling_qwen3_5.py`: `torch_chunk_gated_delta_rule`,
//! `torch_recurrent_gated_delta_rule`, `causal_conv1d_fn`/`causal_conv1d_update` and
//! `Qwen3_5GatedDeltaNet.forward`.

use anyhow::Result;
use candle_core::{D, DType, Device, Tensor};

/// `l2norm` of the reference (aligned with the FLA library): `x * rsqrt(sum(x^2) + eps)`.
fn l2norm(x: &Tensor) -> Result<Tensor> {
    let inv = (x.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?.recip()?;
    Ok(x.broadcast_mul(&inv)?)
}

/// `F.softplus` (threshold behavior is numerically identical): `max(x, 0) + log1p(exp(-|x|))`.
pub fn softplus(x: &Tensor) -> Result<Tensor> {
    Ok((x.relu()? + (x.abs()?.neg()?.exp()? + 1.0)?.log()?)?)
}

/// Prepares q/k/v/g/beta like the reference: float32, shape (batch, heads, seq, ...),
/// L2-normalized queries and keys, queries scaled by `1/sqrt(k_head_dim)`.
fn prepare(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    g: &Tensor,
    beta: &Tensor,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    // `force_contiguous`: with a single token, a transposed view counts as contiguous for
    // candle (size-1 dimensions are ignored) but its strides still upset the matmul kernels.
    let f32 = |t: &Tensor| -> Result<Tensor> { Ok(t.transpose(1, 2)?.to_dtype(DType::F32)?.force_contiguous()?) };
    let q = l2norm(&f32(query)?)?;
    let k = l2norm(&f32(key)?)?;
    let scale = (q.dim(D::Minus1)? as f64).powf(-0.5);
    Ok(((q * scale)?, k, f32(value)?, f32(g)?, f32(beta)?))
}

/// Inverse of the unit lower triangular matrix `I + L` (`L` strictly lower triangular),
/// by forward substitution over the rows: `X_i = e_i - L[i, :i] @ X[:i]`.
///
/// This replaces `torch.linalg.solve_triangular(..., unitriangular=True)` (the reference's
/// own fallback when exporting uses the same substitution).
fn unit_lower_inverse(l: &Tensor) -> Result<Tensor> {
    let dims = l.dims().to_vec();
    let c = dims[dims.len() - 1];
    let batch: usize = dims[..dims.len() - 2].iter().product();
    let l = l.reshape((batch, c, c))?;
    let eye = Tensor::eye(c, DType::F32, l.device())?;

    let mut inv = eye
        .narrow(0, 0, 1)?
        .unsqueeze(0)?
        .broadcast_as((batch, 1, c))?
        .contiguous()?;
    for i in 1..c {
        let row = l.narrow(1, i, 1)?.narrow(2, 0, i)?.contiguous()?;
        let e = eye.narrow(0, i, 1)?.unsqueeze(0)?;
        let next = e.broadcast_sub(&row.matmul(&inv)?)?;
        inv = Tensor::cat(&[&inv, &next], 1)?;
    }
    Ok(inv.reshape(dims)?)
}

/// Batched matrix product for inputs of any rank ≥ 3 (candle's kernels want 3-D operands).
fn bmm(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_dims = a.dims().to_vec();
    let b_dims = b.dims().to_vec();
    let rank = a_dims.len();
    let batch: usize = a_dims[..rank - 2].iter().product();
    let (m, k, n) = (a_dims[rank - 2], a_dims[rank - 1], b_dims[rank - 1]);
    let a3 = a.force_contiguous()?.reshape((batch, m, k))?;
    let b3 = b.force_contiguous()?.reshape((batch, k, n))?;
    let mut out_dims = a_dims;
    out_dims[rank - 1] = n;
    Ok(a3.matmul(&b3)?.reshape(out_dims)?)
}

/// Masks for a chunk: (upper triangle = -inf, else 0) and (strictly lower = 1, else 0).
fn chunk_masks(c: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let mut upper = vec![0f32; c * c];
    let mut strict_lower = vec![0f32; c * c];
    for i in 0..c {
        for j in 0..c {
            if j > i {
                upper[i * c + j] = f32::NEG_INFINITY;
            }
            if j < i {
                strict_lower[i * c + j] = 1.0;
            }
        }
    }
    Ok((
        Tensor::from_vec(upper, (c, c), device)?,
        Tensor::from_vec(strict_lower, (c, c), device)?,
    ))
}

/// `torch_chunk_gated_delta_rule` with `use_qk_l2norm_in_kernel=True`.
///
/// Inputs: query/key (batch, seq, heads, k_dim), value (batch, seq, heads, v_dim),
/// g/beta (batch, seq, heads). Returns the output (batch, seq, heads, v_dim) in float32 and
/// the final recurrent state (batch, heads, k_dim, v_dim).
pub fn chunk_gated_delta_rule(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    initial_state: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let (q, k, v, decay, beta) = prepare(query, key, value, g, beta)?;
    let (b, h, seq, dk) = q.dims4()?;
    let dv = v.dim(D::Minus1)?;
    let device = q.device().clone();

    // Any chunk size gives the same result; short prompts use a single chunk.
    let c = seq.clamp(1, 64);
    let pad = (c - seq % c) % c;
    let pad_seq = |t: &Tensor| -> Result<Tensor> {
        Ok(if pad > 0 {
            t.pad_with_zeros(2, 0, pad)?
        } else {
            t.clone()
        })
    };
    let (q, k, v, beta, decay) = (
        pad_seq(&q)?,
        pad_seq(&k)?,
        pad_seq(&v)?,
        pad_seq(&beta)?,
        pad_seq(&decay)?,
    );
    let n = (seq + pad) / c;

    let v_beta = v.broadcast_mul(&beta.unsqueeze(D::Minus1)?)?;
    let k_beta = k.broadcast_mul(&beta.unsqueeze(D::Minus1)?)?;

    let chunked = |t: &Tensor, d: usize| -> Result<Tensor> { Ok(t.reshape((b, h, n, c, d))?) };
    let q = chunked(&q, dk)?;
    let k = chunked(&k, dk)?;
    let k_beta = chunked(&k_beta, dk)?;
    let v_beta = chunked(&v_beta, dv)?;
    let decay = decay.reshape((b, h, n, c))?;

    let (upper_mask, strict_lower) = chunk_masks(c, &device)?;

    let cum = decay.cumsum(3)?;
    let pairwise = cum
        .unsqueeze(4)?
        .broadcast_sub(&cum.unsqueeze(3)?)?
        .broadcast_add(&upper_mask)?
        .exp()?;

    let k_t = k.transpose(3, 4)?.contiguous()?;
    let ut_system = (bmm(&k_beta, &k_t)? * &pairwise)?;
    let intra = (bmm(&q, &k_t)? * &pairwise)?;
    let cum_exp = cum.exp()?.unsqueeze(D::Minus1)?;
    let decayed_k_beta = k_beta.broadcast_mul(&cum_exp)?;

    let inverse = unit_lower_inverse(&ut_system.broadcast_mul(&strict_lower)?)?;
    let new_values = bmm(&inverse, &v_beta)?;
    let k_cumdecay = bmm(&inverse, &decayed_k_beta)?;

    let q = q.broadcast_mul(&cum_exp)?;
    let last = cum.narrow(3, c - 1, 1)?;
    let k = k.broadcast_mul(&last.broadcast_sub(&cum)?.exp()?.unsqueeze(D::Minus1)?)?;
    let chunk_decay = last.exp()?.unsqueeze(D::Minus1)?;

    let mut state = match initial_state {
        Some(s) => s.to_dtype(DType::F32)?,
        None => Tensor::zeros((b, h, dk, dv), DType::F32, &device)?,
    };
    let mut outputs = Vec::with_capacity(n);
    let at = |t: &Tensor, i: usize| -> Result<Tensor> { Ok(t.narrow(2, i, 1)?.squeeze(2)?.contiguous()?) };
    for i in 0..n {
        let v_new = (at(&new_values, i)? - at(&k_cumdecay, i)?.matmul(&state)?)?;
        let inter = at(&q, i)?.matmul(&state)?;
        outputs.push((inter + at(&intra, i)?.matmul(&v_new)?)?);
        let k_i = at(&k, i)?.transpose(2, 3)?.contiguous()?;
        state = (state.broadcast_mul(&at(&chunk_decay, i)?)? + k_i.matmul(&v_new)?)?;
    }

    let out = Tensor::cat(&outputs, 2)?.narrow(2, 0, seq)?;
    Ok((out.transpose(1, 2)?.contiguous()?, state))
}

/// `torch_recurrent_gated_delta_rule` with `use_qk_l2norm_in_kernel=True` (token by token).
pub fn recurrent_gated_delta_rule(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    initial_state: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let (q, k, v, decay, beta) = prepare(query, key, value, g, beta)?;
    let (b, h, seq, dk) = q.dims4()?;
    let dv = v.dim(D::Minus1)?;

    let mut state = match initial_state {
        Some(s) => s.to_dtype(DType::F32)?,
        None => Tensor::zeros((b, h, dk, dv), DType::F32, q.device())?,
    };
    let mut outputs = Vec::with_capacity(seq);
    for t in 0..seq {
        let q_t = q.narrow(2, t, 1)?.contiguous()?;
        let k_t = k.narrow(2, t, 1)?.contiguous()?;
        let v_t = v.narrow(2, t, 1)?;
        let g_t = decay.narrow(2, t, 1)?.exp()?.unsqueeze(D::Minus1)?;
        let beta_t = beta.narrow(2, t, 1)?.unsqueeze(D::Minus1)?;
        state = state.broadcast_mul(&g_t)?;
        let kv_mem = k_t.matmul(&state)?;
        let delta = (v_t - kv_mem)?.broadcast_mul(&beta_t)?;
        state = (&state + k_t.transpose(2, 3)?.contiguous()?.matmul(&delta)?)?;
        outputs.push(q_t.matmul(&state)?);
    }
    let out = Tensor::cat(&outputs, 2)?;
    Ok((out.transpose(1, 2)?.contiguous()?, state))
}

/// Depthwise causal convolution over the time axis followed by SiLU
/// (`causal_conv1d_fn` / `causal_conv1d_update`).
///
/// `x`: (batch, channels, seq) float32; `weight`: (channels, kernel) float32;
/// `state`: the previous `kernel - 1` inputs (batch, channels, kernel - 1), or zeros.
/// Returns the activated output (batch, channels, seq) and the new state.
pub fn causal_conv1d(x: &Tensor, weight: &Tensor, state: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
    let (b, channels, seq) = x.dims3()?;
    let kernel = weight.dim(1)?;
    let state = match state {
        Some(s) => s.clone(),
        None => Tensor::zeros((b, channels, kernel - 1), DType::F32, x.device())?,
    };
    let padded = Tensor::cat(&[&state, x], 2)?;
    let mut out = Tensor::zeros((b, channels, seq), DType::F32, x.device())?;
    for j in 0..kernel {
        let w = weight.narrow(1, j, 1)?.reshape((1, channels, 1))?;
        out = (out + padded.narrow(2, j, seq)?.broadcast_mul(&w)?)?;
    }
    let new_state = padded.narrow(2, seq, kernel - 1)?.contiguous()?;
    Ok((out.silu()?, new_state))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random(shape: &[usize], seed: u64) -> Result<Tensor> {
        // Small deterministic pseudo-random values.
        let n: usize = shape.iter().product();
        let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let data: Vec<f32> = (0..n)
            .map(|_| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((x >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect();
        Ok(Tensor::from_vec(data, shape, &Device::Cpu)?)
    }

    #[test]
    fn chunked_matches_recurrent() -> Result<()> {
        let (b, s, h, dk, dv) = (2, 70, 3, 8, 5);
        let q = random(&[b, s, h, dk], 1)?;
        let k = random(&[b, s, h, dk], 2)?;
        let v = random(&[b, s, h, dv], 3)?;
        let g = (random(&[b, s, h], 4)?.abs()? * -0.5)?;
        let beta = candle_nn::ops::sigmoid(&random(&[b, s, h], 5)?)?;
        let init = random(&[b, h, dk, dv], 6)?;

        let (out_c, state_c) = chunk_gated_delta_rule(&q, &k, &v, &g, &beta, Some(&init))?;
        let (out_r, state_r) = recurrent_gated_delta_rule(&q, &k, &v, &g, &beta, Some(&init))?;
        let diff = (out_c - out_r)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let state_diff = (state_c - state_r)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-4, "output difference {diff}");
        assert!(state_diff < 1e-4, "state difference {state_diff}");

        // Prefill then single-token decoding equals one long pass.
        let (_, state_prefix) = chunk_gated_delta_rule(
            &q.narrow(1, 0, s - 1)?,
            &k.narrow(1, 0, s - 1)?,
            &v.narrow(1, 0, s - 1)?,
            &g.narrow(1, 0, s - 1)?,
            &beta.narrow(1, 0, s - 1)?,
            Some(&init),
        )?;
        let (_, state_step) = recurrent_gated_delta_rule(
            &q.narrow(1, s - 1, 1)?,
            &k.narrow(1, s - 1, 1)?,
            &v.narrow(1, s - 1, 1)?,
            &g.narrow(1, s - 1, 1)?,
            &beta.narrow(1, s - 1, 1)?,
            Some(&state_prefix),
        )?;
        let (_, state_full) = recurrent_gated_delta_rule(&q, &k, &v, &g, &beta, Some(&init))?;
        let diff = (state_step - state_full)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-4, "incremental state difference {diff}");
        Ok(())
    }

    #[test]
    fn unit_lower_inverse_inverts() -> Result<()> {
        let c = 6;
        let (_, strict) = chunk_masks(c, &Device::Cpu)?;
        let l = random(&[2, c, c], 9)?.broadcast_mul(&strict)?;
        let inv = unit_lower_inverse(&l)?;
        let eye = Tensor::eye(c, DType::F32, &Device::Cpu)?;
        let product = l.broadcast_add(&eye)?.matmul(&inv)?;
        let diff = product.broadcast_sub(&eye)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-5);
        Ok(())
    }

    #[test]
    fn causal_conv_streams() -> Result<()> {
        let x = random(&[1, 4, 7], 11)?;
        let w = random(&[4, 4], 12)?;
        let (full, _) = causal_conv1d(&x, &w, None)?;
        let (first, state) = causal_conv1d(&x.narrow(2, 0, 5)?, &w, None)?;
        let (second, _) = causal_conv1d(&x.narrow(2, 5, 2)?, &w, Some(&state))?;
        let streamed = Tensor::cat(&[&first, &second], 2)?;
        let diff = (full - streamed)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-6);
        Ok(())
    }
}
