use super::spillpoints::Spillpoint;
use ndarray::Array2;

/// Diagnostic capacity of every trap, as `sum over cells (z_spill - z_top)`.
///
/// The units are metres times cell count: there is no `dx`, no `dy`, no
/// porosity and no cap at the base of the sand, so this is neither a bulk
/// volume nor a pore volume. `TrapFill` does not use it; it computes its own
/// capacities with `geomvol`. This is reported through `rust.spillanalysis`
/// only, for comparison against the reference implementation.
pub fn trapvolumes(
    grid: &Array2<f64>,
    regions: &Array2<i64>,
    spillpoints: &[Spillpoint],
    lowest_regions: &[Vec<usize>],
) -> Vec<f64> {
    let (nx, ny) = (grid.shape()[0], grid.shape()[1]);
    let num_low = regions.iter().copied().max().unwrap_or(0).max(0) as usize;
    let num_all = lowest_regions.len();
    if num_all == 0 {
        return Vec::new();
    }

    let mut supertraps: Vec<Vec<usize>> = (1..=num_low).map(|r| vec![r]).collect();
    for sreg in (num_low + 1)..=num_all {
        for &reg in &lowest_regions[sreg - 1] {
            supertraps[reg - 1].push(sreg);
        }
    }

    let elevations: Vec<f64> = spillpoints.iter().map(|s| s.elevation).collect();
    let mut vol = vec![0.0f64; num_all];

    for j in 0..ny {
        for i in 0..nx {
            let lowreg = regions[[i, j]];
            if lowreg > 0 {
                let z = grid[[i, j]];
                for &reg in &supertraps[(lowreg - 1) as usize] {
                    let dz = elevations[reg - 1] - z;
                    if dz > 0.0 {
                        vol[reg - 1] += dz;
                    }
                }
            }
        }
    }

    vol
}
