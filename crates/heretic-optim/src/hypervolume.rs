// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

//! Pareto dominance, non-domination ranks and hypervolume (all for minimization).

/// Whether `a` dominates `b` (no worse in all objectives, better in at least one).
pub fn dominates(a: &[f64], b: &[f64]) -> bool {
    let mut better = false;
    for (x, y) in a.iter().zip(b) {
        if x > y {
            return false;
        }
        if x < y {
            better = true;
        }
    }
    better
}

/// Non-domination rank of each point (0 = Pareto front). If `n_below` is given,
/// ranking stops once at least that many points have been ranked; the remaining
/// points get the next rank (like Optuna's `_fast_non_domination_rank`).
pub fn non_domination_ranks(points: &[Vec<f64>], n_below: Option<usize>) -> Vec<usize> {
    let n = points.len();
    let mut ranks = vec![usize::MAX; n];
    let mut remaining: Vec<usize> = (0..n).collect();
    let mut rank = 0;
    let mut ranked = 0;

    while !remaining.is_empty() {
        if n_below.is_some_and(|limit| ranked >= limit) {
            for &i in &remaining {
                ranks[i] = rank;
            }
            break;
        }

        let front: Vec<usize> = remaining
            .iter()
            .copied()
            .filter(|&i| !remaining.iter().any(|&j| j != i && dominates(&points[j], &points[i])))
            .collect();

        for &i in &front {
            ranks[i] = rank;
        }
        ranked += front.len();
        remaining.retain(|i| !front.contains(i));
        rank += 1;
    }

    ranks
}

/// Hypervolume dominated by `points` and bounded by `reference`.
pub fn hypervolume(points: &[Vec<f64>], reference: &[f64]) -> f64 {
    // Points outside the reference box contribute nothing.
    let points: Vec<Vec<f64>> = points
        .iter()
        .filter(|p| p.iter().zip(reference).all(|(x, r)| x < r))
        .cloned()
        .collect();
    if points.is_empty() {
        return 0.0;
    }
    hypervolume_recursive(points, reference)
}

fn hypervolume_recursive(mut points: Vec<Vec<f64>>, reference: &[f64]) -> f64 {
    let d = reference.len();
    if points.is_empty() {
        return 0.0;
    }
    if d == 1 {
        let best = points.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
        return reference[0] - best;
    }
    if d == 2 {
        points.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
        let mut volume = 0.0;
        let mut previous_y = reference[1];
        for p in &points {
            if p[1] < previous_y {
                volume += (reference[0] - p[0]) * (previous_y - p[1]);
                previous_y = p[1];
            }
        }
        return volume;
    }

    // Slice along the last dimension.
    points.sort_by(|a, b| a[d - 1].total_cmp(&b[d - 1]));
    let mut volume = 0.0;
    for i in 0..points.len() {
        let next = if i + 1 < points.len() {
            points[i + 1][d - 1]
        } else {
            reference[d - 1]
        };
        let depth = next - points[i][d - 1];
        if depth <= 0.0 {
            continue;
        }
        let slice: Vec<Vec<f64>> = points[..=i].iter().map(|p| p[..d - 1].to_vec()).collect();
        volume += hypervolume_recursive(slice, &reference[..d - 1]) * depth;
    }
    volume
}

/// Reference point used by Optuna: slightly worse than the worst point in each objective.
pub fn reference_point(points: &[Vec<f64>]) -> Vec<f64> {
    let d = points[0].len();
    (0..d)
        .map(|k| {
            let worst = points.iter().map(|p| p[k]).fold(f64::NEG_INFINITY, f64::max);
            let r = (1.1 * worst).max(0.9 * worst);
            if r == 0.0 { f64::EPSILON } else { r }
        })
        .collect()
}

/// Greedily selects `subset_size` points that (approximately) maximize the hypervolume
/// (the hypervolume subset selection problem). Returns indices into `points`.
pub fn solve_hssp(points: &[Vec<f64>], subset_size: usize, reference: &[f64]) -> Vec<usize> {
    if subset_size >= points.len() {
        return (0..points.len()).collect();
    }
    if !reference.iter().all(|r| r.is_finite()) {
        return (0..subset_size).collect();
    }

    let mut selected: Vec<usize> = Vec::with_capacity(subset_size);
    let mut selected_points: Vec<Vec<f64>> = Vec::with_capacity(subset_size);
    let mut current = 0.0;

    while selected.len() < subset_size {
        let mut best: Option<(usize, f64)> = None;
        for (i, point) in points.iter().enumerate() {
            if selected.contains(&i) {
                continue;
            }
            selected_points.push(point.clone());
            let contribution = hypervolume(&selected_points, reference) - current;
            selected_points.pop();
            if best.is_none_or(|(_, c)| contribution > c) {
                best = Some((i, contribution));
            }
        }
        let (index, contribution) = best.expect("there are unselected points");
        selected.push(index);
        selected_points.push(points[index].clone());
        current += contribution;
    }

    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dominance_and_ranks() {
        let points = vec![
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
            vec![1.0, 2.0],
        ];
        assert!(dominates(&points[0], &points[2]));
        assert!(!dominates(&points[0], &points[1]));
        assert!(!dominates(&points[0], &points[4]));
        assert_eq!(non_domination_ranks(&points, None), vec![0, 0, 1, 2, 0]);
        assert_eq!(non_domination_ranks(&points, Some(3)), vec![0, 0, 1, 1, 0]);
    }

    #[test]
    fn hypervolume_2d_and_3d() {
        let points = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        assert_eq!(hypervolume(&points, &[3.0, 3.0]), 3.0);

        let cube = vec![vec![0.0, 0.0, 0.0]];
        assert_eq!(hypervolume(&cube, &[1.0, 2.0, 3.0]), 6.0);

        // Two overlapping boxes: 2*2*2 + 2*2*2 - 1*1*2 ... computed by inclusion-exclusion.
        let points = vec![vec![0.0, 1.0, 0.0], vec![1.0, 0.0, 0.0]];
        // Box A: [0,2]x[1,2]x[0,2] = 2*1*2 = 4; box B: [1,2]x[0,2]x[0,2] = 1*2*2 = 4; overlap [1,2]x[1,2]x[0,2] = 2.
        assert_eq!(hypervolume(&points, &[2.0, 2.0, 2.0]), 6.0);
    }

    #[test]
    fn hssp_prefers_spread() {
        let points = vec![vec![0.0, 3.0], vec![0.1, 2.9], vec![3.0, 0.0], vec![1.5, 1.5]];
        let selected = solve_hssp(&points, 2, &[4.0, 4.0]);
        assert_eq!(selected.len(), 2);
        assert!(!(selected.contains(&0) && selected.contains(&1)));
    }
}
