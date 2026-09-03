use super::graph::DiGraph;
use super::spillfield::spillfield;
use super::spillpoints::{Spillpoint, spillpoints};
use super::spillregions::spillregions;
use super::sshierarchy::sshierarchy;
use super::trapvolumes::trapvolumes;
use ndarray::Array2;
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub struct SpillOptions {
    pub usediags: bool,
    pub closed: bool,
    pub lengths: Option<(f64, f64)>,
}

pub struct TrapStructure {
    pub spillfield: Array2<i8>,
    pub regions: Array2<i64>,
    pub spillpoints: Vec<Spillpoint>,
    pub trapvolumes: Vec<f64>,
    pub subvolumes: Vec<f64>,
    pub footprints: Vec<Vec<usize>>,
    pub supertraps_of: Vec<Vec<usize>>,
    pub agglomerations: DiGraph,
}

impl TrapStructure {
    pub const fn numtraps(&self) -> usize {
        self.spillpoints.len()
    }

    pub const fn numregions(&self) -> usize {
        self.supertraps_of.len()
    }
}

fn compute_supertraps_of(lowest_regions: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let num_low = lowest_regions
        .iter()
        .flat_map(|v| v.iter().copied())
        .max()
        .unwrap_or(0);
    let mut result: Vec<Vec<usize>> = vec![Vec::new(); num_low];
    for (i, lr) in lowest_regions.iter().enumerate() {
        for &k in lr {
            result[k - 1].push(i + 1);
        }
    }
    result
}

fn compute_subvolumes(trapvols: &[f64], subtrapgraph: &DiGraph) -> Vec<f64> {
    let mut svols = vec![0.0f64; trapvols.len()];
    for i in 1..=trapvols.len() {
        for &j in subtrapgraph.in_neighbors(i) {
            svols[i - 1] += trapvols[j - 1];
        }
    }
    svols
}

fn compute_footprints(
    grid: &Array2<f64>,
    lowest_regions: &[Vec<usize>],
    regions: &Array2<i64>,
    spillpoints: &[Spillpoint],
    num_regions: usize,
) -> Vec<Vec<usize>> {
    let (nx, ny) = (grid.shape()[0], grid.shape()[1]);
    let num_traps = spillpoints.len();
    let mut footprints: Vec<Vec<usize>> = vec![Vec::new(); num_traps];

    let mut supertraps_for: Vec<Vec<usize>> = vec![Vec::new(); num_regions];
    for i in 1..=num_traps {
        for &j in &lowest_regions[i - 1] {
            supertraps_for[j - 1].push(i);
        }
    }
    for v in &mut supertraps_for {
        v.reverse();
    }

    for j in 0..ny {
        for i in 0..nx {
            let r = regions[[i, j]];
            if r <= 0 {
                continue;
            }
            let z = grid[[i, j]];
            let cell = i + j * nx;
            for &tr in &supertraps_for[(r - 1) as usize] {
                if z <= spillpoints[tr - 1].elevation {
                    footprints[tr - 1].push(cell);
                } else {
                    break;
                }
            }
        }
    }

    footprints
}

fn close_regions(regions: &mut Array2<i64>) {
    let mut next = regions.iter().copied().max().unwrap_or(0).max(0);
    let mut remap: HashMap<i64, i64> = HashMap::new();
    for r in regions.iter_mut() {
        if *r < 0 {
            *r = *remap.entry(*r).or_insert_with(|| {
                next += 1;
                next
            });
        }
    }
}

pub fn spillanalysis(grid: &Array2<f64>, options: SpillOptions) -> TrapStructure {
    let usediags = options.usediags;
    let (field, _slope) = spillfield(grid, usediags, options.lengths);
    let mut regions = spillregions(&field, usediags);
    if options.closed {
        close_regions(&mut regions);
    }
    let (mut spoints, mut regbnd) = spillpoints(grid, &regions, usediags);
    let (subtrapgraph, lowest_regions) = sshierarchy(grid, &regions, &mut spoints, &mut regbnd);

    let num_regions = regions.iter().copied().max().unwrap_or(0).max(0) as usize;
    let supertraps_of = compute_supertraps_of(&lowest_regions);
    let trapvols = trapvolumes(grid, &regions, &spoints, &lowest_regions);
    let footprints = compute_footprints(grid, &lowest_regions, &regions, &spoints, num_regions);
    let subvolumes = compute_subvolumes(&trapvols, &subtrapgraph);

    TrapStructure {
        spillfield: field,
        regions,
        spillpoints: spoints,
        trapvolumes: trapvols,
        subvolumes,
        footprints,
        supertraps_of,
        agglomerations: subtrapgraph,
    }
}
