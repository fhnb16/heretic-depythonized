// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Linear layers with an optional low-rank adapter.

use std::sync::RwLock;

use anyhow::{Result, bail};
use candle_core::{DType, Tensor};

/// A low-rank adapter: `delta W = B @ A` (applied at full strength, i.e. alpha = rank).
#[derive(Debug, Clone)]
pub struct Adapter {
    /// Shape (rank, in_features), float32.
    pub a: Tensor,
    /// Shape (out_features, rank), float32.
    pub b: Tensor,
}

/// A linear layer whose weight can be modified through a LoRA adapter without
/// touching the base weight, so it can be reset instantly between trials.
///
/// Like PEFT with `autocast_adapter_dtype=True`, the adapter is stored and evaluated in
/// float32: the base output is computed in the model's dtype, the adapter output is
/// added in float32, and the sum is cast back to the model's dtype.
pub struct LoraLinear {
    /// Checkpoint name of the module (without ".weight"), e.g. "model.layers.3.self_attn.o_proj".
    pub name: String,
    weight: Tensor,
    bias: Option<Tensor>,
    adapter: RwLock<Option<Adapter>>,
}

impl std::fmt::Debug for LoraLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoraLinear")
            .field("name", &self.name)
            .field("shape", &self.weight.dims())
            .field("bias", &self.bias.is_some())
            .finish()
    }
}

/// `x @ w.T` for inputs of any rank.
pub fn matmul_t(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let in_features = *dims.last().unwrap();
    let rows: usize = dims[..dims.len() - 1].iter().product();
    let y = x.reshape((rows, in_features))?.matmul(&w.t()?)?;
    let mut out_dims = dims;
    *out_dims.last_mut().unwrap() = w.dim(0)?;
    Ok(y.reshape(out_dims)?)
}

impl LoraLinear {
    pub fn new(name: String, weight: Tensor, bias: Option<Tensor>) -> Self {
        Self {
            name,
            weight,
            bias,
            adapter: RwLock::new(None),
        }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    pub fn out_features(&self) -> usize {
        self.weight.dims()[0]
    }

    pub fn in_features(&self) -> usize {
        self.weight.dims()[1]
    }

    /// The base weight as float32, shape (out_features, in_features).
    pub fn weight_f32(&self) -> Result<Tensor> {
        Ok(self.weight.to_dtype(DType::F32)?)
    }

    pub fn set_adapter(&self, a: Tensor, b: Tensor) -> Result<()> {
        let (rank_a, in_features) = a.dims2()?;
        let (out_features, rank_b) = b.dims2()?;
        if rank_a != rank_b || in_features != self.in_features() || out_features != self.out_features() {
            bail!(
                "Adapter shapes A={:?}, B={:?} don't match {} with weight {:?}",
                a.dims(),
                b.dims(),
                self.name,
                self.weight.dims()
            );
        }
        let device = self.weight.device();
        let adapter = Adapter {
            a: a.to_dtype(DType::F32)?.to_device(device)?,
            b: b.to_dtype(DType::F32)?.to_device(device)?,
        };
        *self.adapter.write().unwrap() = Some(adapter);
        Ok(())
    }

    /// Removes the adapter (identity transformation).
    pub fn clear_adapter(&self) {
        *self.adapter.write().unwrap() = None;
    }

    pub fn adapter(&self) -> Option<Adapter> {
        self.adapter.read().unwrap().clone()
    }

    pub fn has_adapter(&self) -> bool {
        self.adapter.read().unwrap().is_some()
    }

    /// The effective weight `W + B @ A` as float32.
    pub fn merged_weight_f32(&self) -> Result<Tensor> {
        let w = self.weight_f32()?;
        Ok(match self.adapter() {
            Some(adapter) => (w + adapter.b.matmul(&adapter.a)?)?,
            None => w,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_with(x, true)
    }

    /// Forward pass; `use_adapter = false` evaluates the original module.
    pub fn forward_with(&self, x: &Tensor, use_adapter: bool) -> Result<Tensor> {
        let mut y = matmul_t(x, &self.weight)?;
        if let Some(bias) = &self.bias {
            y = y.broadcast_add(bias)?;
        }

        if use_adapter && let Some(adapter) = self.adapter.read().unwrap().as_ref() {
            let dtype = y.dtype();
            let x32 = x.to_dtype(DType::F32)?;
            let delta = matmul_t(&matmul_t(&x32, &adapter.a)?, &adapter.b)?;
            y = (y.to_dtype(DType::F32)? + delta)?.to_dtype(dtype)?;
        }

        Ok(y)
    }
}

/// A plain linear layer.
#[derive(Debug, Clone)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = matmul_t(x, &self.weight)?;
        Ok(match &self.bias {
            Some(bias) => y.broadcast_add(bias)?,
            None => y,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn adapter_adds_low_rank_delta() -> Result<()> {
        let device = Device::Cpu;
        let w = Tensor::new(&[[1f32, 0.0], [0.0, 1.0], [1.0, 1.0]], &device)?;
        let layer = LoraLinear::new("test".into(), w, None);
        let x = Tensor::new(&[[2f32, 3.0]], &device)?;

        assert_eq!(layer.forward(&x)?.to_vec2::<f32>()?, vec![vec![2.0, 3.0, 5.0]]);

        // delta W = B @ A with A = [[1, 0]], B = [[1], [0], [-1]].
        layer.set_adapter(
            Tensor::new(&[[1f32, 0.0]], &device)?,
            Tensor::new(&[[1f32], [0.0], [-1.0]], &device)?,
        )?;
        assert_eq!(layer.forward(&x)?.to_vec2::<f32>()?, vec![vec![4.0, 3.0, 3.0]]);
        assert_eq!(
            layer.forward_with(&x, false)?.to_vec2::<f32>()?,
            vec![vec![2.0, 3.0, 5.0]]
        );
        assert_eq!(
            layer.merged_weight_f32()?.to_vec2::<f32>()?,
            vec![vec![2.0, 0.0], vec![0.0, 1.0], vec![0.0, 1.0]]
        );

        layer.clear_adapter();
        assert_eq!(layer.forward(&x)?.to_vec2::<f32>()?, vec![vec![2.0, 3.0, 5.0]]);
        Ok(())
    }

    #[test]
    fn rejects_mismatched_adapter() -> Result<()> {
        let device = Device::Cpu;
        let layer = LoraLinear::new("test".into(), Tensor::zeros((3, 2), DType::F32, &device)?, None);
        assert!(
            layer
                .set_adapter(
                    Tensor::zeros((1, 3), DType::F32, &device)?,
                    Tensor::zeros((3, 1), DType::F32, &device)?
                )
                .is_err()
        );
        Ok(())
    }
}
