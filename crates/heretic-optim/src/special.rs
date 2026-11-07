// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Special functions for the (truncated) normal distribution, implemented in a
//! numerically stable way following Optuna's `_truncnorm` module.

use std::f64::consts::{LN_2, PI, SQRT_2};

const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_8;

/// Standard normal CDF.
pub fn ndtr(x: f64) -> f64 {
    0.5 * libm::erfc(-x / SQRT_2)
}

/// Logarithm of the standard normal CDF.
pub fn log_ndtr(x: f64) -> f64 {
    if x > 6.0 {
        // log(1 - sf) ≈ -sf for tiny sf.
        -0.5 * libm::erfc(x / SQRT_2)
    } else if x > -20.0 {
        (0.5 * libm::erfc(-x / SQRT_2)).ln()
    } else {
        // Asymptotic expansion of the Mills ratio.
        let x2 = x * x;
        let mut sum = 1.0;
        let mut term = 1.0;
        for k in 1..8 {
            term *= -((2 * k - 1) as f64) / x2;
            sum += term;
        }
        -0.5 * x2 - (-x).ln() - LN_SQRT_2PI + sum.ln()
    }
}

/// log(1 - exp(x)) for x <= 0.
fn log1mexp(x: f64) -> f64 {
    if x > -LN_2 {
        (-x.exp_m1()).ln()
    } else {
        (-x.exp()).ln_1p()
    }
}

fn logaddexp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

/// log(Φ(b) - Φ(a)) for a <= b.
pub fn log_gauss_mass(a: f64, b: f64) -> f64 {
    if b <= 0.0 {
        let log_b = log_ndtr(b);
        if a == f64::NEG_INFINITY {
            return log_b;
        }
        log_b + log1mexp(log_ndtr(a) - log_b)
    } else if a > 0.0 {
        log_gauss_mass(-b, -a)
    } else {
        // Central case: 1 - Φ(a) - Φ(-b).
        (-ndtr(a) - ndtr(-b)).ln_1p()
    }
}

/// Inverse of the standard normal CDF (Acklam's approximation, refined with Halley's method).
pub fn ndtri(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];

    let p_low = 0.02425;
    let mut x = if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - p_low {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };

    // One step of Halley's method brings the result to full double precision.
    let e = 0.5 * libm::erfc(-x / SQRT_2) - p;
    let u = e * (2.0 * PI).sqrt() * (x * x / 2.0).exp();
    x -= u / (1.0 + x * u / 2.0);
    x
}

/// ndtri(exp(y)), accurate for very negative y.
pub fn ndtri_exp(y: f64) -> f64 {
    if y == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    if y >= 0.0 {
        return f64::INFINITY;
    }
    if y > -LN_2 {
        // exp(y) close to 1: use the upper tail for precision.
        return -ndtri(-y.exp_m1());
    }
    if y > -700.0 {
        let x = ndtri(y.exp());
        if x.is_finite() && y > -30.0 {
            return x;
        }
    }

    // Newton's method on log Φ(x) = y, starting from the asymptotic solution.
    let mut x = -(-2.0 * y).sqrt();
    for _ in 0..50 {
        let log_phi = -0.5 * x * x - LN_SQRT_2PI;
        let f = log_ndtr(x) - y;
        let derivative = (log_phi - log_ndtr(x)).exp();
        let step = f / derivative;
        x -= step;
        if step.abs() < 1e-14 * x.abs().max(1.0) {
            break;
        }
    }
    x
}

/// Percent point function (inverse CDF) of the standard normal truncated to [a, b].
pub fn truncnorm_ppf(q: f64, a: f64, b: f64) -> f64 {
    if a == b {
        return f64::NAN;
    }
    if q <= 0.0 {
        return a;
    }
    if q >= 1.0 {
        return b;
    }
    if a < 0.0 {
        let log_phi_x = logaddexp(log_ndtr(a), q.ln() + log_gauss_mass(a, b));
        ndtri_exp(log_phi_x)
    } else {
        let log_phi_x = logaddexp(log_ndtr(-b), (-q).ln_1p() + log_gauss_mass(a, b));
        -ndtri_exp(log_phi_x)
    }
}

/// Log density of a normal(loc, scale) distribution truncated to
/// [loc + a * scale, loc + b * scale], evaluated at x.
pub fn truncnorm_logpdf(x: f64, a: f64, b: f64, loc: f64, scale: f64) -> f64 {
    let z = (x - loc) / scale;
    if z < a || z > b {
        return f64::NEG_INFINITY;
    }
    -0.5 * z * z - LN_SQRT_2PI - log_gauss_mass(a, b) - scale.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() <= tolerance * b.abs().max(1.0)
    }

    #[test]
    fn normal_cdf_values() {
        assert!(close(ndtr(0.0), 0.5, 1e-15));
        assert!(close(ndtr(1.96), 0.975_002_104_851_780, 1e-12));
        assert!(close(log_ndtr(-30.0), -454.321_243_957, 1e-11));
        assert!(close(log_ndtr(-10.0), -53.231_285_150_512_08, 1e-10));
    }

    #[test]
    fn inverse_cdf_round_trip() {
        for &p in &[1e-300, 1e-20, 1e-5, 0.01, 0.3, 0.5, 0.9, 0.999_999] {
            let x = ndtri(p);
            assert!(close(ndtr(x), p, 1e-10), "p={p}, x={x}, back={}", ndtr(x));
        }
        for &y in &[-1000.0, -100.0, -5.0, -0.5, -1e-6] {
            let x = ndtri_exp(y);
            assert!(close(log_ndtr(x), y, 1e-9), "y={y}, x={x}");
        }
    }

    #[test]
    fn gauss_mass() {
        assert!(close(log_gauss_mass(-1.0, 1.0), (ndtr(1.0) - ndtr(-1.0)).ln(), 1e-12));
        assert!(close(log_gauss_mass(2.0, 3.0), (ndtr(3.0) - ndtr(2.0)).ln(), 1e-10));
        assert!(close(log_gauss_mass(-3.0, -2.0), (ndtr(-2.0) - ndtr(-3.0)).ln(), 1e-10));
        assert!(log_gauss_mass(30.0, 31.0).is_finite());
    }

    #[test]
    fn truncated_normal() {
        // Median of the symmetric truncation is 0.
        assert!(truncnorm_ppf(0.5, -1.0, 1.0).abs() < 1e-12);
        // Samples stay within the bounds, even far in the tails.
        for &q in &[1e-9, 0.25, 0.75, 1.0 - 1e-9] {
            let x = truncnorm_ppf(q, 10.0, 12.0);
            assert!((10.0..=12.0).contains(&x), "{x}");
            let x = truncnorm_ppf(q, -12.0, -10.0);
            assert!((-12.0..=-10.0).contains(&x), "{x}");
        }
        // Density integrates to 1.
        let n = 20_000;
        let (a, b) = (-0.5, 2.0);
        let width = (b - a) / n as f64;
        let integral: f64 = (0..n)
            .map(|i| truncnorm_logpdf(a + (i as f64 + 0.5) * width, a, b, 0.0, 1.0).exp() * width)
            .sum();
        assert!(close(integral, 1.0, 1e-6), "{integral}");
    }
}
