use super::graph::{DiGraph, functional_cycles, sibling_loops};
use super::spillpoints::Spillpoint;
use ndarray::Array2;
use std::collections::HashSet;

fn identify_highest_level_trap_of(lowest_trap: i64, subtrapgraph: &DiGraph) -> usize {
    if lowest_trap <= 0 {
        return 0;
    }
    let mut cur = lowest_trap as usize;
    loop {
        let neigh = subtrapgraph.out_neighbors(cur);
        if neigh.is_empty() {
            return cur;
        }
        cur = neigh[0];
    }
}

fn identify_new_spillpoint(
    grid: &Array2<f64>,
    nx: usize,
    subtraps: &[usize],
    regions: &Array2<i64>,
    boundaries: &mut Vec<Vec<(usize, usize)>>,
) -> Spillpoint {
    let z = |c: usize| -> f64 { grid[[c % nx, c / nx]] };
    let regof = |c: usize| -> i64 { regions[[c % nx, c / nx]] };

    let k = subtraps.len();
    let mut outer: Vec<Vec<(usize, usize)>> = Vec::with_capacity(k);

    for (i, &si) in subtraps.iter().enumerate() {
        let mut forbidden: HashSet<(usize, usize)> = HashSet::new();
        for (j, &sj) in subtraps.iter().enumerate() {
            if i == j {
                continue;
            }
            for &(a, b) in &boundaries[sj - 1] {
                forbidden.insert((b, a));
            }
        }
        let mut seen: HashSet<(usize, usize)> = HashSet::new();
        let mut reduced: Vec<(usize, usize)> = Vec::new();
        for &p in &boundaries[si - 1] {
            if forbidden.contains(&p) {
                continue;
            }
            if seen.insert(p) {
                reduced.push(p);
            }
        }
        outer.push(reduced);
    }

    let mut boundary: Vec<(usize, usize)> = Vec::new();
    for o in &outer {
        boundary.extend_from_slice(o);
    }

    boundaries.push(boundary.clone());

    if boundary.is_empty() {
        return Spillpoint::default();
    }

    let mut best_ix = 0usize;
    let mut best_z = f64::INFINITY;
    for (idx, &(a, b)) in boundary.iter().enumerate() {
        let zv = z(a).max(z(b));
        if zv < best_z {
            best_z = zv;
            best_ix = idx;
        }
    }

    let (a, b) = boundary[best_ix];
    let reg_ix = if a == b { 0 } else { regof(b) };
    Spillpoint {
        downstream_region: reg_ix,
        current_region_cell: a as i64,
        downstream_region_cell: b as i64,
        elevation: best_z,
    }
}

pub fn sshierarchy(
    grid: &Array2<f64>,
    regions: &Array2<i64>,
    spillpoints: &mut Vec<Spillpoint>,
    boundaries: &mut Vec<Vec<(usize, usize)>>,
) -> (DiGraph, Vec<Vec<usize>>) {
    let nx = grid.shape()[0];
    let maxpos = regions.iter().copied().max().unwrap_or(0).max(0) as usize;

    let mut cur_num_traps = maxpos;
    let mut subtrapgraph = DiGraph::new(maxpos);

    let mut out = vec![0i64; maxpos + 1];
    for i in 1..=maxpos {
        let ds = spillpoints[i - 1].downstream_region;
        if ds >= 1 && ds <= maxpos as i64 {
            out[i] = ds;
        }
    }

    let mut lowest_regions: Vec<Vec<usize>> = (1..=maxpos).map(|r| vec![r]).collect();

    let mut mergers = functional_cycles(&out, maxpos);

    while !mergers.is_empty() {
        subtrapgraph.add_vertices(mergers.len());

        for m in &mergers {
            cur_num_traps += 1;
            let spoint = identify_new_spillpoint(grid, nx, m, regions, boundaries);
            spillpoints.push(spoint);
            for &subtrap in m {
                subtrapgraph.add_edge(subtrap, cur_num_traps);
            }
            let mut lr: Vec<usize> = Vec::new();
            for &s in m {
                lr.extend_from_slice(&lowest_regions[s - 1]);
            }
            lowest_regions.push(lr);
        }

        let mut m_ix = cur_num_traps - mergers.len();
        let mut sibling_connections: Vec<(usize, usize)> = Vec::new();
        let mut sib_set: HashSet<(usize, usize)> = HashSet::new();

        for _m in &mergers {
            m_ix += 1;
            let spoint_elev = spillpoints[m_ix - 1].elevation;
            let spoint_ds = spillpoints[m_ix - 1].downstream_region;
            let top_dstrap = identify_highest_level_trap_of(spoint_ds, &subtrapgraph);

            #[expect(clippy::float_cmp)]
            if top_dstrap != 0 && spillpoints[top_dstrap - 1].elevation == spoint_elev {
                let mut visited = vec![m_ix];
                let mut prev = m_ix;
                let mut next = top_dstrap;
                while next != 0 && spillpoints[next - 1].elevation == spoint_elev {
                    let pair = (prev, next);
                    if sib_set.insert(pair) {
                        sibling_connections.push(pair);
                    }
                    if visited.contains(&next) {
                        break;
                    }
                    visited.push(next);
                    prev = next;
                    next = identify_highest_level_trap_of(
                        spillpoints[next - 1].downstream_region,
                        &subtrapgraph,
                    );
                }
            }
        }

        mergers = sibling_loops(&sibling_connections);
    }

    (subtrapgraph, lowest_regions)
}
