// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Randomized low-rank SVD, ported from `torch.svd_lowrank` (Halko et al. 2009,
//! Algorithm 5). Matrix products run on the tensor's device; QR and SVD of the
//! small factors run on the CPU.

use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use faer::Mat;
use rand::RngExt;
use rand_chacha::ChaCha8Rng;

fn standard_normal(rng: &mut ChaCha8Rng, count: usize) -> Vec<f32> {
    // Box-Muller transform.
    let mut values = Vec::with_capacity(count + 1);
    while values.len() < count {
        let u1: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
        let u2: f64 = rng.random::<f64>();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * u2;
        values.push((radius * angle.cos()) as f32);
        values.push((radius * angle.sin()) as f32);
    }
    values.truncate(count);
    values
}

fn to_faer(tensor: &Tensor) -> Result<Mat<f64>> {
    let (rows, cols) = tensor.dims2()?;
    let data: Vec<f32> = tensor
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1()?;
    Ok(Mat::from_fn(rows, cols, |i, j| data[i * cols + j] as f64))
}

fn from_faer(matrix: &Mat<f64>, device: &Device) -> Result<Tensor> {
    let (rows, cols) = (matrix.nrows(), matrix.ncols());
    let mut data = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            data.push(matrix[(i, j)] as f32);
        }
    }
    Ok(Tensor::from_vec(data, (rows, cols), device)?)
}

fn orthonormal_basis(x: &Tensor) -> Result<Tensor> {
    let q = to_faer(x)?.qr().compute_thin_Q();
    from_faer(&q, x.device())
}

/// Computes an orthonormal basis Q of shape (m, q) whose range approximates the range of A.
fn approximate_basis(a: &Tensor, q: usize, niter: usize, rng: &mut ChaCha8Rng) -> Result<Tensor> {
    let n = a.dim(1)?;
    let r = Tensor::from_vec(standard_normal(rng, n * q), (n, q), a.device())?;
    let mut basis = orthonormal_basis(&a.matmul(&r)?)?;
    let a_t = a.t()?.contiguous()?;
    for _ in 0..niter {
        basis = orthonormal_basis(&a_t.matmul(&basis)?)?;
        basis = orthonormal_basis(&a.matmul(&basis)?)?;
    }
    Ok(basis)
}

/// Returns (U, S, V) with U of shape (m, q), S of length q and V of shape (n, q),
/// such that A ≈ U diag(S) V^T.
pub fn svd_lowrank(a: &Tensor, q: usize, niter: usize, rng: &mut ChaCha8Rng) -> Result<(Tensor, Vec<f32>, Tensor)> {
    let a = a.to_dtype(DType::F32)?;
    let (m, n) = a.dims2()?;
    let device = a.device().clone();
    let q = q.min(m).min(n);

    if m < n || n > q {
        // Work with the transpose to keep the factors small.
        let a_t = a.t()?.contiguous()?;
        let basis = approximate_basis(&a_t, q, niter, rng)?; // (n, q)
        let b_t = a.matmul(&basis)?; // (m, q)
        let svd = to_faer(&b_t)?.thin_svd().map_err(|e| anyhow!("SVD failed: {e:?}"))?;
        let u = from_faer(&svd.U().to_owned(), &device)?;
        let s: Vec<f32> = (0..q).map(|i| svd.S().column_vector()[i] as f32).collect();
        let v = basis.matmul(&from_faer(&svd.V().to_owned(), &device)?)?;
        Ok((u, s, v))
    } else {
        let basis = approximate_basis(&a, q, niter, rng)?; // (m, q)
        let b = basis.t()?.contiguous()?.matmul(&a)?; // (q, n)
        let svd = to_faer(&b)?.thin_svd().map_err(|e| anyhow!("SVD failed: {e:?}"))?;
        let u = basis.matmul(&from_faer(&svd.U().to_owned(), &device)?)?;
        let s: Vec<f32> = (0..q).map(|i| svd.S().column_vector()[i] as f32).collect();
        let v = from_faer(&svd.V().to_owned(), &device)?;
        Ok((u, s, v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn recovers_low_rank_matrix() -> Result<()> {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let device = Device::Cpu;
        let (m, n, rank) = (60, 200, 3);
        let left = Tensor::from_vec(standard_normal(&mut rng, m * rank), (m, rank), &device)?;
        let right = Tensor::from_vec(standard_normal(&mut rng, rank * n), (rank, n), &device)?;
        let a = left.matmul(&right)?;

        let (u, s, v) = svd_lowrank(&a, 2 * rank + 4, 6, &mut rng)?;
        assert_eq!(u.dims(), &[m, 2 * rank + 4]);
        assert_eq!(v.dims(), &[n, 2 * rank + 4]);
        assert!(s.windows(2).all(|w| w[0] >= w[1]));
        assert!(s[rank] < 1e-3 * s[0], "{s:?}");

        // Truncate to the true rank and reconstruct.
        let u = u.narrow(1, 0, rank)?;
        let v = v.narrow(1, 0, rank)?;
        let s = Tensor::from_vec(s[..rank].to_vec(), (1, rank), &device)?;
        let reconstructed = u.broadcast_mul(&s)?.matmul(&v.t()?)?;
        let error: f32 = (reconstructed - &a)?.abs()?.max_all()?.to_scalar()?;
        let scale: f32 = a.abs()?.max_all()?.to_scalar()?;
        assert!(error < 1e-4 * scale, "{error}");
        Ok(())
    }

    #[test]
    fn tall_matrices() -> Result<()> {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let device = Device::Cpu;
        let column = Tensor::from_vec(standard_normal(&mut rng, 100), (100, 1), &device)?;
        let row = Tensor::from_vec(standard_normal(&mut rng, 8), (1, 8), &device)?;
        let a = column.matmul(&row)?;
        let (u, s, v) = svd_lowrank(&a, 10, 2, &mut rng)?;
        assert_eq!(u.dim(1)?, 8);
        assert_eq!(v.dims(), &[8, 8]);
        assert!(s[1] < 1e-4 * s[0]);
        Ok(())
    }
}
