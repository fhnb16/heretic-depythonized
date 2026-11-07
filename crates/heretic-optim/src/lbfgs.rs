// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! L-BFGS with strong Wolfe line search, ported line by line from `torch.optim.LBFGS`.
//!
//! Parameters are a flat float32 vector; the objective returns the loss and its gradient.
//! The optimizer keeps its state across calls to [`Lbfgs::step`], like PyTorch's.

use anyhow::Result;
use candle_core::{DType, Tensor};

#[derive(Debug, Clone)]
pub struct LbfgsConfig {
    pub lr: f64,
    pub max_iter: usize,
    /// Defaults to `max_iter * 5 / 4`.
    pub max_eval: Option<usize>,
    pub tolerance_grad: f64,
    pub tolerance_change: f64,
    pub history_size: usize,
}

impl Default for LbfgsConfig {
    fn default() -> Self {
        Self {
            lr: 1.0,
            max_iter: 20,
            max_eval: None,
            tolerance_grad: 1e-7,
            tolerance_change: 1e-9,
            history_size: 100,
        }
    }
}

/// Objective: loss and gradient at the given (flat) parameters.
pub trait Objective {
    fn evaluate(&mut self, x: &Tensor) -> Result<(f64, Tensor)>;
}

impl<F: FnMut(&Tensor) -> Result<(f64, Tensor)>> Objective for F {
    fn evaluate(&mut self, x: &Tensor) -> Result<(f64, Tensor)> {
        self(x)
    }
}

pub struct Lbfgs {
    pub config: LbfgsConfig,
    /// Current parameters.
    pub x: Tensor,
    func_evals: usize,
    n_iter: usize,
    d: Option<Tensor>,
    t: f64,
    old_dirs: Vec<Tensor>,
    old_stps: Vec<Tensor>,
    ro: Vec<f64>,
    h_diag: f64,
    prev_flat_grad: Option<Tensor>,
    prev_loss: Option<f64>,
}

fn dot(a: &Tensor, b: &Tensor) -> Result<f64> {
    Ok((a * b)?.sum_all()?.to_dtype(DType::F64)?.to_scalar::<f64>()?)
}

fn abs_max(a: &Tensor) -> Result<f64> {
    Ok(a.abs()?.max_all()?.to_dtype(DType::F64)?.to_scalar::<f64>()?)
}

fn abs_sum(a: &Tensor) -> Result<f64> {
    Ok(a.abs()?.sum_all()?.to_dtype(DType::F64)?.to_scalar::<f64>()?)
}

/// Minimizer of the cubic interpolating two points with function values and derivatives.
fn cubic_interpolate(x1: f64, f1: f64, g1: f64, x2: f64, f2: f64, g2: f64, bounds: Option<(f64, f64)>) -> f64 {
    let (xmin_bound, xmax_bound) = bounds.unwrap_or(if x1 <= x2 { (x1, x2) } else { (x2, x1) });
    let d1 = g1 + g2 - 3.0 * (f1 - f2) / (x1 - x2);
    let d2_square = d1 * d1 - g1 * g2;
    if d2_square >= 0.0 {
        let d2 = d2_square.sqrt();
        let min_pos = if x1 <= x2 {
            x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2.0 * d2))
        } else {
            x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2.0 * d2))
        };
        min_pos.max(xmin_bound).min(xmax_bound)
    } else {
        (xmin_bound + xmax_bound) / 2.0
    }
}

struct LineSearchResult {
    loss: f64,
    grad: Tensor,
    t: f64,
    evals: usize,
}

#[allow(clippy::too_many_arguments)]
fn strong_wolfe(
    objective: &mut dyn Objective,
    x: &Tensor,
    mut t: f64,
    d: &Tensor,
    f: f64,
    g: &Tensor,
    gtd: f64,
    tolerance_change: f64,
) -> Result<LineSearchResult> {
    const C1: f64 = 1e-4;
    const C2: f64 = 0.9;
    const MAX_LS: usize = 25;

    let mut evaluate = |t: f64| -> Result<(f64, Tensor)> { objective.evaluate(&(x + (d * t)?)?) };

    let d_norm = abs_max(d)?;
    let (mut f_new, mut g_new) = evaluate(t)?;
    let mut evals = 1;
    let mut gtd_new = dot(&g_new, d)?;

    let (mut t_prev, mut f_prev, mut g_prev, mut gtd_prev) = (0.0, f, g.clone(), gtd);
    let mut done = false;
    let mut ls_iter = 0;

    let mut bracket: Vec<f64> = Vec::new();
    let mut bracket_f: Vec<f64> = Vec::new();
    let mut bracket_g: Vec<Tensor> = Vec::new();
    let mut bracket_gtd: Vec<f64> = Vec::new();

    while ls_iter < MAX_LS {
        if f_new > f + C1 * t * gtd || (ls_iter > 1 && f_new >= f_prev) {
            bracket = vec![t_prev, t];
            bracket_f = vec![f_prev, f_new];
            bracket_g = vec![g_prev.clone(), g_new.clone()];
            bracket_gtd = vec![gtd_prev, gtd_new];
            break;
        }
        if gtd_new.abs() <= -C2 * gtd {
            bracket = vec![t];
            bracket_f = vec![f_new];
            bracket_g = vec![g_new.clone()];
            done = true;
            break;
        }
        if gtd_new >= 0.0 {
            bracket = vec![t_prev, t];
            bracket_f = vec![f_prev, f_new];
            bracket_g = vec![g_prev.clone(), g_new.clone()];
            bracket_gtd = vec![gtd_prev, gtd_new];
            break;
        }

        // Interpolate.
        let min_step = t + 0.01 * (t - t_prev);
        let max_step = t * 10.0;
        let tmp = t;
        t = cubic_interpolate(t_prev, f_prev, gtd_prev, t, f_new, gtd_new, Some((min_step, max_step)));

        t_prev = tmp;
        f_prev = f_new;
        g_prev = g_new.clone();
        gtd_prev = gtd_new;
        (f_new, g_new) = evaluate(t)?;
        evals += 1;
        gtd_new = dot(&g_new, d)?;
        ls_iter += 1;
    }

    if ls_iter == MAX_LS {
        bracket = vec![0.0, t];
        bracket_f = vec![f, f_new];
        bracket_g = vec![g.clone(), g_new.clone()];
    }

    // Zoom phase.
    let mut insuf_progress = false;
    let (mut low_pos, mut high_pos) = if bracket_f[0] <= *bracket_f.last().unwrap() {
        (0, 1)
    } else {
        (1, 0)
    };

    while !done && ls_iter < MAX_LS {
        if (bracket[1] - bracket[0]).abs() * d_norm < tolerance_change {
            break;
        }

        t = cubic_interpolate(
            bracket[0],
            bracket_f[0],
            bracket_gtd[0],
            bracket[1],
            bracket_f[1],
            bracket_gtd[1],
            None,
        );

        let bracket_max = bracket[0].max(bracket[1]);
        let bracket_min = bracket[0].min(bracket[1]);
        let eps = 0.1 * (bracket_max - bracket_min);
        if (bracket_max - t).min(t - bracket_min) < eps {
            if insuf_progress || t >= bracket_max || t <= bracket_min {
                t = if (t - bracket_max).abs() < (t - bracket_min).abs() {
                    bracket_max - eps
                } else {
                    bracket_min + eps
                };
                insuf_progress = false;
            } else {
                insuf_progress = true;
            }
        } else {
            insuf_progress = false;
        }

        (f_new, g_new) = evaluate(t)?;
        evals += 1;
        gtd_new = dot(&g_new, d)?;
        ls_iter += 1;

        if f_new > f + C1 * t * gtd || f_new >= bracket_f[low_pos] {
            bracket[high_pos] = t;
            bracket_f[high_pos] = f_new;
            bracket_g[high_pos] = g_new.clone();
            bracket_gtd[high_pos] = gtd_new;
            (low_pos, high_pos) = if bracket_f[0] <= bracket_f[1] { (0, 1) } else { (1, 0) };
        } else {
            if gtd_new.abs() <= -C2 * gtd {
                done = true;
            } else if gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0.0 {
                bracket[high_pos] = bracket[low_pos];
                bracket_f[high_pos] = bracket_f[low_pos];
                bracket_g[high_pos] = bracket_g[low_pos].clone();
                bracket_gtd[high_pos] = bracket_gtd[low_pos];
            }
            bracket[low_pos] = t;
            bracket_f[low_pos] = f_new;
            bracket_g[low_pos] = g_new.clone();
            bracket_gtd[low_pos] = gtd_new;
        }
    }

    Ok(LineSearchResult {
        loss: bracket_f[low_pos],
        grad: bracket_g[low_pos].clone(),
        t: bracket[low_pos],
        evals,
    })
}

impl Lbfgs {
    pub fn new(x: Tensor, config: LbfgsConfig) -> Self {
        Self {
            config,
            x,
            func_evals: 0,
            n_iter: 0,
            d: None,
            t: 0.0,
            old_dirs: Vec::new(),
            old_stps: Vec::new(),
            ro: Vec::new(),
            h_diag: 1.0,
            prev_flat_grad: None,
            prev_loss: None,
        }
    }

    /// Performs one optimization step (up to `max_iter` iterations).
    /// Returns the loss at the parameters the step started from.
    pub fn step(&mut self, objective: &mut dyn Objective) -> Result<f64> {
        let lr = self.config.lr;
        let max_iter = self.config.max_iter;
        let max_eval = self.config.max_eval.unwrap_or(max_iter * 5 / 4);
        let tolerance_grad = self.config.tolerance_grad;
        let tolerance_change = self.config.tolerance_change;

        let (orig_loss, mut flat_grad) = objective.evaluate(&self.x)?;
        let mut loss = orig_loss;
        let mut current_evals = 1;
        self.func_evals += 1;

        if abs_max(&flat_grad)? <= tolerance_grad {
            return Ok(orig_loss);
        }

        let mut n_iter = 0;
        while n_iter < max_iter {
            n_iter += 1;
            self.n_iter += 1;

            let d = if self.n_iter == 1 {
                self.old_dirs.clear();
                self.old_stps.clear();
                self.ro.clear();
                self.h_diag = 1.0;
                flat_grad.neg()?
            } else {
                let prev = self.prev_flat_grad.as_ref().expect("previous gradient");
                let y = (&flat_grad - prev)?;
                let s = (self.d.as_ref().expect("previous direction") * self.t)?;
                let ys = dot(&y, &s)?;
                if ys > 1e-10 {
                    if self.old_dirs.len() == self.config.history_size {
                        self.old_dirs.remove(0);
                        self.old_stps.remove(0);
                        self.ro.remove(0);
                    }
                    self.h_diag = ys / dot(&y, &y)?;
                    self.old_dirs.push(y);
                    self.old_stps.push(s);
                    self.ro.push(1.0 / ys);
                }

                // Two-loop recursion for the approximate inverse Hessian.
                let num_old = self.old_dirs.len();
                let mut al = vec![0.0; num_old];
                let mut q = flat_grad.neg()?;
                for i in (0..num_old).rev() {
                    al[i] = dot(&self.old_stps[i], &q)? * self.ro[i];
                    q = (q - (&self.old_dirs[i] * al[i])?)?;
                }
                let mut r = (q * self.h_diag)?;
                #[allow(clippy::needless_range_loop)] // Mirrors PyTorch's indexing.
                for i in 0..num_old {
                    let be_i = dot(&self.old_dirs[i], &r)? * self.ro[i];
                    r = (r + (&self.old_stps[i] * (al[i] - be_i))?)?;
                }
                r
            };

            self.prev_flat_grad = Some(flat_grad.clone());
            self.prev_loss = Some(loss);

            // Step length.
            let t = if self.n_iter == 1 {
                (1.0f64).min(1.0 / abs_sum(&flat_grad)?) * lr
            } else {
                lr
            };

            let gtd = dot(&flat_grad, &d)?;
            if gtd > -tolerance_change {
                self.d = Some(d);
                break;
            }

            let result = strong_wolfe(objective, &self.x, t, &d, loss, &flat_grad, gtd, tolerance_change)?;
            loss = result.loss;
            flat_grad = result.grad;
            self.t = result.t;
            self.x = (&self.x + (&d * self.t)?)?;
            let opt_cond = abs_max(&flat_grad)? <= tolerance_grad;

            current_evals += result.evals;
            self.func_evals += result.evals;

            let step_size = abs_max(&(&d * self.t)?)?;
            self.d = Some(d);

            if n_iter == max_iter
                || current_evals >= max_eval
                || opt_cond
                || step_size <= tolerance_change
                || (loss - self.prev_loss.unwrap_or(loss)).abs() < tolerance_change
            {
                break;
            }
        }

        Ok(orig_loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn minimizes_rosenbrock() -> Result<()> {
        let mut rosenbrock = |x: &Tensor| -> Result<(f64, Tensor)> {
            let v: Vec<f32> = x.to_vec1()?;
            let (a, b) = (v[0] as f64, v[1] as f64);
            let loss = (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2);
            let grad = vec![
                (-2.0 * (1.0 - a) - 400.0 * a * (b - a * a)) as f32,
                (200.0 * (b - a * a)) as f32,
            ];
            Ok((loss, Tensor::new(grad.as_slice(), &Device::Cpu)?))
        };

        let x0 = Tensor::new(&[-1.5f32, 2.0], &Device::Cpu)?;
        let mut optimizer = Lbfgs::new(
            x0,
            LbfgsConfig {
                history_size: 10,
                ..LbfgsConfig::default()
            },
        );
        for _ in 0..10 {
            optimizer.step(&mut rosenbrock)?;
        }
        let x: Vec<f32> = optimizer.x.to_vec1()?;
        assert!((x[0] - 1.0).abs() < 1e-3 && (x[1] - 1.0).abs() < 1e-3, "{x:?}");
        Ok(())
    }

    #[test]
    fn minimizes_quadratic_with_autograd() -> Result<()> {
        // f(x) = sum((x - c)^2 * w), gradient via candle's autograd.
        let device = Device::Cpu;
        let c = Tensor::new(&[3f32, -1.0, 0.5, 2.0], &device)?;
        let w = Tensor::new(&[1f32, 10.0, 0.1, 5.0], &device)?;
        let mut objective = |x: &Tensor| -> Result<(f64, Tensor)> {
            let var = candle_core::Var::from_tensor(x)?;
            let loss = ((var.as_tensor() - &c)?.sqr()? * &w)?.sum_all()?;
            let grads = loss.backward()?;
            let grad = grads.get(var.as_tensor()).unwrap().clone();
            Ok((loss.to_dtype(DType::F64)?.to_scalar()?, grad))
        };
        let mut optimizer = Lbfgs::new(Tensor::zeros(4, DType::F32, &device)?, LbfgsConfig::default());
        optimizer.step(&mut objective)?;
        let x: Vec<f32> = optimizer.x.to_vec1()?;
        for (a, b) in x.iter().zip([3f32, -1.0, 0.5, 2.0]) {
            assert!((a - b).abs() < 1e-4, "{x:?}");
        }
        Ok(())
    }

    #[test]
    fn cubic_interpolation() {
        // f(x) = (x - 1)^2: f(0) = 1, f'(0) = -2, f(3) = 4, f'(3) = 4 -> minimum at 1.
        let t = cubic_interpolate(0.0, 1.0, -2.0, 3.0, 4.0, 4.0, None);
        assert!((t - 1.0).abs() < 1e-12);
    }
}
