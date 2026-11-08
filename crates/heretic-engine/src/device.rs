// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

use anyhow::{Result, bail};
use candle_core::{DType, Device};
use heretic_core::config::DeviceMap;

/// Whether this build of Heretic can use CUDA.
pub fn cuda_compiled() -> bool {
    cfg!(feature = "cuda")
}

/// Resolves the `device_map` setting to a device.
pub fn resolve_device(device_map: &DeviceMap) -> Result<Device> {
    let name = match device_map {
        DeviceMap::Device(name) => name.trim().to_ascii_lowercase(),
        DeviceMap::Map(_) => {
            bail!("Explicit device maps are not supported yet; use \"auto\", \"cpu\", \"cuda\", or \"cuda:<index>\"")
        }
    };

    match name.as_str() {
        "auto" => {
            if cuda_compiled() && candle_core::utils::cuda_is_available() {
                Ok(Device::new_cuda(0)?)
            } else {
                Ok(Device::Cpu)
            }
        }
        "cpu" => Ok(Device::Cpu),
        "cuda" | "gpu" => cuda_device(0),
        other => match other.strip_prefix("cuda:").and_then(|i| i.parse::<usize>().ok()) {
            Some(index) => cuda_device(index),
            None => bail!("Unknown device \"{other}\" (expected \"auto\", \"cpu\", \"cuda\", or \"cuda:<index>\")"),
        },
    }
}

fn cuda_device(index: usize) -> Result<Device> {
    if !cuda_compiled() {
        bail!(
            "This build of Heretic has no CUDA support. Download the CUDA build (\"cuda12\" in the file name) from \
             the releases page, or select the CPU device"
        );
    }
    Ok(Device::new_cuda(index)?)
}

pub fn device_name(device: &Device) -> String {
    match device {
        Device::Cpu => "CPU".into(),
        Device::Cuda(_) => "CUDA".into(),
        Device::Metal(_) => "Metal".into(),
    }
}

/// Parses a dtype name from the `dtypes` setting. `"auto"` resolves to the dtype the
/// checkpoint was saved in.
pub fn resolve_dtype(name: &str, checkpoint_dtype: Option<&str>) -> Result<DType> {
    let name = if name == "auto" {
        checkpoint_dtype.unwrap_or("float32")
    } else {
        name
    };
    Ok(match name.trim_start_matches("torch.") {
        "float32" | "float" | "f32" => DType::F32,
        "float16" | "half" | "f16" => DType::F16,
        "bfloat16" | "bf16" => DType::BF16,
        "float64" | "double" => DType::F64,
        other => bail!("Unsupported dtype \"{other}\""),
    })
}

pub fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "float32",
        DType::F16 => "float16",
        DType::BF16 => "bfloat16",
        DType::F64 => "float64",
        DType::U8 => "uint8",
        DType::U32 => "uint32",
        DType::I64 => "int64",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtypes() {
        assert_eq!(resolve_dtype("auto", Some("bfloat16")).unwrap(), DType::BF16);
        assert_eq!(resolve_dtype("auto", None).unwrap(), DType::F32);
        assert_eq!(resolve_dtype("float16", Some("bfloat16")).unwrap(), DType::F16);
        assert!(resolve_dtype("int4", None).is_err());
    }

    #[test]
    fn devices() {
        assert!(matches!(
            resolve_device(&DeviceMap::Device("cpu".into())).unwrap(),
            Device::Cpu
        ));
        assert!(resolve_device(&DeviceMap::Device("tpu".into())).is_err());
        if !cuda_compiled() {
            assert!(matches!(
                resolve_device(&DeviceMap::Device("auto".into())).unwrap(),
                Device::Cpu
            ));
            assert!(resolve_device(&DeviceMap::Device("cuda:0".into())).is_err());
        }
    }
}
