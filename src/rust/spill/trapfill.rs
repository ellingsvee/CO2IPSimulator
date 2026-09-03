use super::trapstructure::{SpillOptions, TrapStructure, spillanalysis};
use ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const EPS: f64 = 1e-9;
const PAD: usize = 1;

struct LayerModel {
    nx: usize,
    ny: usize,
    ts: TrapStructure,
    topo: Vec<f64>,
    base: Vec<f64>,
    area: f64,
    density_co2: f64,
    porosity: f64,
    num_low: usize,
    num_traps: usize,
    critical_height: f64,
    seal_thickness: f64,
    seal_speed: f64,
    foot_by_top: Vec<Vec<(f64, f64, usize)>>,
    foot_by_base: Vec<Vec<(f64, f64)>>,
    seal_cells: Vec<Vec<usize>>,
    seal_topo: Vec<Vec<f64>>,
    seal_topo_prefix: Vec<Vec<f64>>,
    seal_base: Vec<Vec<f64>>,
    seal_base_prefix: Vec<Vec<f64>>,
    spill: Vec<f64>,
    downstream: Vec<i64>,
    outlet: Vec<f64>,
    is_break: Vec<bool>,
    own_cap: Vec<f64>,
    own_cap_spill: Vec<f64>,
    parent: Vec<usize>,
    children: Vec<Vec<usize>>,
    vz_z: Vec<Vec<f64>>,
    vz_v: Vec<Vec<f64>>,
    region_of_cell: Vec<i64>,
}

/// Mass that left the trap graph, in kg.
///
/// `stalled` is the part of `escaped` that was dropped by the cascade guard in
/// `route` rather than by reaching a trap with no downstream neighbour. It is
/// zero for a well-formed spill graph, and is carried out so that a run in
/// which it is not can be identified instead of silently reading as lateral
/// outflow.
#[derive(Clone, Copy, Default)]
struct Losses {
    escaped: f64,
    stalled: f64,
}

impl Losses {
    fn add(&mut self, other: Self) {
        self.escaped += other.escaped;
        self.stalled += other.stalled;
    }
}

/// The seal above one sand unit, uniform over the unit.
#[derive(Clone, Copy)]
struct SealState {
    critical_height: f64,
    thickness: f64,
    speed: f64,
}

fn seal_states(
    mobility: Option<&PyReadonlyArray1<f64>>,
    critical_height: &[f64],
    thickness: &[f64],
    delta_rho_g: &[f64],
    nlayers: usize,
) -> Vec<SealState> {
    (0..nlayers)
        .map(|k| SealState {
            critical_height: critical_height[k],
            thickness: thickness[k],
            speed: mobility.map_or(0.0, |a| a.as_array()[k] * delta_rho_g[k]),
        })
        .collect()
}

fn breach_slot(regions: &[i64], cell: usize) -> usize {
    regions[cell].max(0) as usize
}

/// The trap region a cell drains into.
///
/// `spillanalysis` runs with `closed: true`, so `close_regions` has relabelled
/// every component that leaves the domain as an ordinary trap and every cell
/// carries a positive region. Nothing has to be followed downslope.
fn drain_region(ts: &TrapStructure, nx: usize, cell: usize) -> i64 {
    ts.regions[[cell % nx, cell / nx]]
}

fn pad_surface(s: &ArrayView2<f64>, wall: f64) -> Array2<f64> {
    let (nx, ny) = (s.shape()[0], s.shape()[1]);
    let mut out = Array2::from_elem((nx + 2 * PAD, ny + 2 * PAD), wall);
    for j in 0..ny {
        for i in 0..nx {
            out[[i + PAD, j + PAD]] = s[[i, j]];
        }
    }
    out
}

fn flatten(a: &Array2<f64>) -> Vec<f64> {
    let (nx, ny) = (a.shape()[0], a.shape()[1]);
    let mut v = vec![0.0; nx * ny];
    for j in 0..ny {
        for i in 0..nx {
            v[i + j * nx] = a[[i, j]];
        }
    }
    v
}

fn geomvol(cells: &[usize], topo: &[f64], base: &[f64], area: f64, z: f64) -> f64 {
    let mut s = 0.0;
    for &c in cells {
        s += (z.min(base[c]) - topo[c]).max(0.0);
    }
    s * area
}

fn volume_depth_table(
    cells: &[usize],
    topo: &[f64],
    base: &[f64],
    area: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut events: Vec<(f64, i64)> = Vec::with_capacity(2 * cells.len());
    for &c in cells {
        events.push((topo[c], 1));
        events.push((base[c], -1));
    }
    events.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut prev_z = events.first().map_or(0.0, |e| e.0);
    let mut zk: Vec<f64> = vec![prev_z];
    let mut vk: Vec<f64> = vec![0.0];
    let mut active: i64 = 0;
    let mut v = 0.0;
    for (z, d) in events {
        if z > prev_z {
            v += area * active as f64 * (z - prev_z);
            zk.push(z);
            vk.push(v);
            prev_z = z;
        }
        active += d;
    }
    (zk, vk)
}

impl LayerModel {
    fn build(
        top2d: &ArrayView2<f64>,
        base2d: &ArrayView2<f64>,
        seal: SealState,
        density_co2: f64,
        porosity: f64,
        cell: (f64, f64),
        usediags: bool,
    ) -> Self {
        let (dx, dy) = cell;
        let area = dx * dy;
        let wall = top2d.iter().copied().fold(f64::MIN, f64::max) + 1000.0;
        let top_padded = pad_surface(top2d, wall);
        let base_padded = pad_surface(base2d, wall);
        let nx = top_padded.shape()[0];
        let ny = top_padded.shape()[1];
        let ts = spillanalysis(
            &top_padded,
            SpillOptions {
                usediags,
                closed: true,
                lengths: Some((dx * nx as f64, dy * ny as f64)),
            },
        );
        let topo = flatten(&top_padded);
        let base = flatten(&base_padded);

        let num_traps = ts.numtraps();
        let num_low = ts.numregions();

        let spill: Vec<f64> = ts.spillpoints.iter().map(|s| s.elevation).collect();
        let downstream: Vec<i64> = ts.spillpoints.iter().map(|s| s.downstream_region).collect();

        let mut parent = vec![0usize; num_traps + 1];
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); num_traps + 1];
        for t in 1..=num_traps {
            if let Some(&p) = ts.agglomerations.out_neighbors(t).first() {
                parent[t] = p;
            }
            for &c in ts.agglomerations.in_neighbors(t) {
                children[t].push(c);
            }
        }

        let cap_spill: Vec<f64> = (0..num_traps)
            .map(|t| geomvol(&ts.footprints[t], &topo, &base, area, spill[t]))
            .collect();
        let mut own_cap_spill = cap_spill.clone();
        for t in 1..=num_traps {
            for &c in &children[t] {
                own_cap_spill[t - 1] -= cap_spill[c - 1];
            }
        }
        for v in &mut own_cap_spill {
            if *v < 0.0 {
                *v = 0.0;
            }
        }

        let (vz_z, vz_v): (Vec<Vec<f64>>, Vec<Vec<f64>>) = (0..num_traps)
            .map(|t| volume_depth_table(&ts.footprints[t], &topo, &base, area))
            .unzip();

        let region_of_cell: Vec<i64> = (0..nx * ny).map(|c| drain_region(&ts, nx, c)).collect();

        // Sorting a footprint by the depth of the sand top, and by the depth of
        // its base, is independent of the seal parameters: a uniform critical
        // height shifts every breach elevation by the same amount and leaves
        // both orders untouched. Only which cells are thick enough to hold a
        // critical column depends on the seal, and that is a linear filter over
        // these orders, applied in `apply_seals`.
        let mut foot_by_top = Vec::with_capacity(num_traps);
        let mut foot_by_base = Vec::with_capacity(num_traps);
        for t in 0..num_traps {
            let mut by_top: Vec<(f64, f64, usize)> = ts.footprints[t]
                .iter()
                .map(|&c| (topo[c], base[c] - topo[c], c))
                .collect();
            by_top.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            let mut by_base: Vec<(f64, f64)> = ts.footprints[t]
                .iter()
                .map(|&c| (base[c], base[c] - topo[c]))
                .collect();
            by_base.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            foot_by_top.push(by_top);
            foot_by_base.push(by_base);
        }

        let mut model = Self {
            nx,
            ny,
            ts,
            topo,
            base,
            area,
            density_co2,
            porosity,
            num_low,
            num_traps,
            critical_height: f64::INFINITY,
            seal_thickness: f64::INFINITY,
            seal_speed: 0.0,
            foot_by_top,
            foot_by_base,
            seal_cells: vec![Vec::new(); num_traps],
            seal_topo: vec![Vec::new(); num_traps],
            seal_topo_prefix: vec![Vec::new(); num_traps],
            seal_base: vec![Vec::new(); num_traps],
            seal_base_prefix: vec![Vec::new(); num_traps],
            spill,
            downstream,
            outlet: vec![0.0; num_traps],
            is_break: vec![false; num_traps],
            own_cap: vec![0.0; num_traps],
            own_cap_spill,
            parent,
            children,
            vz_z,
            vz_v,
            region_of_cell,
        };
        model.apply_seals(seal);
        model
    }

    /// Re-point the seal. Everything that had to be sorted was sorted in
    /// `build`, so this is one linear pass over each footprint.
    fn apply_seals(&mut self, seal: SealState) {
        self.critical_height = seal.critical_height;
        self.seal_thickness = seal.thickness;
        self.seal_speed = seal.speed;
        self.rebuild_seal_tables();

        let nt = self.num_traps;
        for t in 0..nt {
            let breach = self.breach_elevation(t + 1);
            self.is_break[t] = breach < self.spill[t];
            self.outlet[t] = if self.is_break[t] {
                breach
            } else {
                self.spill[t]
            };
        }

        let cap_total: Vec<f64> = (1..=nt)
            .map(|t| self.geomvol_at(t, self.outlet[t - 1]))
            .collect();
        for t in 1..=nt {
            let mut own = cap_total[t - 1];
            for &c in &self.children[t] {
                own -= cap_total[c - 1];
            }
            self.own_cap[t - 1] = own.max(0.0);
        }
    }

    /// Restrict each footprint to the cells that can hold a critical column.
    ///
    /// A cell whose sand is thinner than `h_c` is full of CO2 before the
    /// buoyancy pressure reaches the entry pressure, so it never breaches and
    /// never passes flux. Dropping it here keeps the closed-form flux of
    /// `leak_speed_with` exact, and makes `breach_elevation` the shallowest
    /// elevation at which the trap can actually breach.
    fn rebuild_seal_tables(&mut self) {
        let hc = self.critical_height;
        for t in 0..self.num_traps {
            let (cells, topo, prefix) = (
                &mut self.seal_cells[t],
                &mut self.seal_topo[t],
                &mut self.seal_topo_prefix[t],
            );
            cells.clear();
            topo.clear();
            prefix.clear();
            prefix.push(0.0);
            let mut running = 0.0;
            for &(top, thickness, cell) in &self.foot_by_top[t] {
                if thickness < hc {
                    continue;
                }
                cells.push(cell);
                topo.push(top);
                running += top;
                prefix.push(running);
            }

            let (bases, base_prefix) = (&mut self.seal_base[t], &mut self.seal_base_prefix[t]);
            bases.clear();
            base_prefix.clear();
            base_prefix.push(0.0);
            let mut running = 0.0;
            for &(base, thickness) in &self.foot_by_base[t] {
                if thickness < hc {
                    continue;
                }
                bases.push(base);
                running += base;
                base_prefix.push(running);
            }
        }
    }

    /// Number of seal cells of trap `t` that have been breached at contact `z`.
    fn seal_breached(&self, t: usize, z: f64) -> usize {
        self.seal_topo[t - 1].partition_point(|&top| top + self.critical_height <= z)
    }

    /// Number of seal cells of trap `t` whose sand is full at contact `z`, so
    /// that a deeper contact adds no further CO2 column above them.
    fn seal_saturated(&self, t: usize, z: f64) -> usize {
        self.seal_base[t - 1].partition_point(|&base| base <= z)
    }

    /// Contact elevation at which trap `t` first breaches: the shallowest top
    /// among the cells thick enough to hold a critical column, plus `h_c`.
    /// Infinite when no cell of the trap is thick enough.
    fn breach_elevation(&self, t: usize) -> f64 {
        self.seal_topo[t - 1]
            .first()
            .map_or(f64::INFINITY, |&top| top + self.critical_height)
    }

    /// Cell through which trap `t` first breaches: the shallowest of its
    /// footprint cells that can hold a critical column.
    fn breach_cell(&self, t: usize) -> usize {
        self.seal_cells[t - 1].first().copied().unwrap_or(0)
    }

    /// Flux of one breached seal cell, `U (h - h_c)/L`.
    ///
    /// The seal passes flux in proportion to the capillary pressure in excess of
    /// its entry pressure, so the flux vanishes at breakthrough and grows with
    /// the surplus column. An infinitely thick seal passes nothing.
    ///
    /// The column `h` that supplies the buoyancy is the CO2 standing in this
    /// cell, `min(z, base) - top`, not the depth of the trap contact below the
    /// cell top: once the sand here is full, a deeper contact elsewhere in the
    /// trap puts no further CO2 above this cell.
    ///
    /// Whether the cell is breached at all is `seal_breached`'s predicate,
    /// `top + h_c <= contact`, and it is asked that way here too. Asking it as
    /// `contact - top < h_c` instead rounds the other way for a cell sitting on
    /// its breach elevation, so `leak_speed_with` would count a cell this
    /// zeroes. `distribute_leak` splits the drained mass by these per-cell
    /// fluxes but normalises by that prefix total, so the difference would be
    /// dropped rather than passed up. Under this law the head at the gate is
    /// zero and the two agree on the answer whichever way they round, but any
    /// law with a non-zero head there would lose real mass. The terms are
    /// ordered as in `leak_speed_with` for the same reason.
    fn seal_cell_speed(&self, cell: usize, z: f64) -> f64 {
        let contact = z.min(self.base[cell]);
        let top = self.topo[cell];
        if top + self.critical_height > contact {
            return 0.0;
        }
        self.seal_speed * ((contact - self.critical_height) - top) / self.seal_thickness
    }

    /// Shallowest breach elevation of trap `t`: drainage stops here.
    fn leak_floor(&self, t: usize) -> f64 {
        if self.seal_speed > 0.0 {
            self.breach_elevation(t)
        } else {
            f64::INFINITY
        }
    }

    fn geomvol_at(&self, t: usize, z: f64) -> f64 {
        let zz = &self.vz_z[t - 1];
        let vv = &self.vz_v[t - 1];
        let last = zz.len() - 1;
        if z <= zz[0] {
            return 0.0;
        }
        if z >= zz[last] {
            return vv[last];
        }
        let k = zz.partition_point(|&zi| zi <= z) - 1;
        let dz = zz[k + 1] - zz[k];
        let frac = if dz > 0.0 { (z - zz[k]) / dz } else { 0.0 };
        vv[k] + frac * (vv[k + 1] - vv[k])
    }

    fn activatable(&self, t: usize, full: &[bool], vertical: bool) -> bool {
        if t <= self.num_low {
            return true;
        }
        self.children[t]
            .iter()
            .all(|&c| full[c] && (!vertical || !self.is_break[c - 1]))
    }

    fn active_container(&self, r: usize, full: &[bool], vertical: bool) -> Option<usize> {
        let mut t = r;
        loop {
            if !full[t] {
                return if self.activatable(t, full, vertical) {
                    Some(t)
                } else {
                    None
                };
            }
            let p = self.parent[t];
            if p == 0 {
                return None;
            }
            t = p;
        }
    }

    fn top_full(&self, r: usize, full: &[bool]) -> usize {
        let mut t = r;
        loop {
            let p = self.parent[t];
            if p == 0 || !full[p] {
                return t;
            }
            t = p;
        }
    }

    fn route(
        &self,
        sources: &[f64],
        kpg: f64,
        held: &mut [f64],
        full: &mut [bool],
        own_cap: &[f64],
        mut up: Option<(&[i64], &mut [f64])>,
    ) -> Losses {
        let vertical = up.is_some();
        let mut escaped = 0.0;
        let mut stalled = 0.0;
        let guard = 8 * (self.num_traps + 1) + 16;

        for (region, &mass) in sources.iter().enumerate() {
            if mass <= 0.0 {
                continue;
            }
            let mut r = region as i64;
            let mut v = mass / kpg;
            let mut steps = 0;
            while v > 0.0 {
                steps += 1;
                if r <= 0 {
                    escaped += v;
                    break;
                }
                if steps > guard {
                    stalled += v;
                    escaped += v;
                    break;
                }
                let ru = r as usize;
                let container = match self.active_container(ru, full, vertical) {
                    None => self.top_full(ru, full),
                    Some(c) => {
                        let space = (own_cap[c - 1] - held[c]).max(0.0);
                        if v <= space {
                            held[c] += v;
                            break;
                        }
                        held[c] = own_cap[c - 1];
                        full[c] = true;
                        v -= space;
                        c
                    }
                };
                if let Some((regions, acc)) = up.as_mut()
                    && self.is_break[container - 1]
                {
                    acc[breach_slot(regions, self.breach_cell(container))] += v * kpg;
                    break;
                }
                let ds = self.downstream[container - 1];
                if ds <= 0 {
                    escaped += v;
                    break;
                }
                r = ds;
            }
        }

        Losses {
            escaped: escaped * kpg,
            stalled: stalled * kpg,
        }
    }

    fn fill(&self, sources: &[f64], kpg: f64, up: (&[i64], &mut [f64])) -> (Vec<f64>, Losses) {
        let nt = self.num_traps;
        let mut held = vec![0.0f64; nt + 1];
        let mut full = vec![false; nt + 1];
        let losses = self.route(sources, kpg, &mut held, &mut full, &self.own_cap, Some(up));
        (held, losses)
    }

    fn invert_contact(&self, t: usize, target: f64) -> f64 {
        let zz = &self.vz_z[t - 1];
        let vv = &self.vz_v[t - 1];
        let last = vv.len() - 1;
        if target <= 0.0 {
            return zz[0];
        }
        if target >= vv[last] {
            return zz[last];
        }
        let k = vv.partition_point(|&vi| vi <= target) - 1;
        let dv = vv[k + 1] - vv[k];
        let frac = if dv > 0.0 { (target - vv[k]) / dv } else { 0.0 };
        zz[k] + frac * (zz[k + 1] - zz[k])
    }

    fn column_heights(&self, held: &[f64]) -> Vec<f64> {
        let nt = self.num_traps;
        let mut total_in = vec![0.0f64; nt + 1];
        for t in 1..=nt {
            let mut s = held[t];
            for &c in &self.children[t] {
                s += total_in[c];
            }
            total_in[t] = s;
        }

        let mut hmap = vec![0.0f64; self.nx * self.ny];
        for t in 1..=nt {
            if held[t] <= EPS {
                continue;
            }
            let mut anc = self.parent[t];
            let mut ancestor_holds = false;
            while anc != 0 {
                if held[anc] > EPS {
                    ancestor_holds = true;
                    break;
                }
                anc = self.parent[anc];
            }
            if ancestor_holds {
                continue;
            }
            let contact = self.invert_contact(t, total_in[t]);
            for &c in &self.ts.footprints[t - 1] {
                let col = (contact.min(self.base[c]) - self.topo[c]).max(0.0);
                if col > hmap[c] {
                    hmap[c] = col;
                }
            }
        }
        hmap
    }

    fn unpad_columns(&self, hmap: &[f64]) -> Array2<f64> {
        let (nxr, nyr) = (self.nx - 2 * PAD, self.ny - 2 * PAD);
        let mut out = Array2::<f64>::zeros((nxr, nyr));
        for j in 0..nyr {
            for i in 0..nxr {
                out[[i, j]] = hmap[(i + PAD) + (j + PAD) * self.nx];
            }
        }
        out
    }

    fn kg_per_geom(&self, connate_water_saturation: f64) -> f64 {
        self.density_co2 * self.porosity * (1.0 - connate_water_saturation)
    }

    /// Total flux out of trap `t` when its contact stands at `z`, given that the
    /// `seal_k` shallowest seal cells are breached and the `sat_k` shallowest-based
    /// ones are full. Writing `a = top + h_c`, the per-cell column
    /// `min(z, base) - a` telescopes into two prefix sums,
    ///
    /// ```text
    /// sum_c clamp(min(z, base) - a, 0) = sum_{a <= z} (z - a) - sum_{base <= z} (z - base),
    /// ```
    ///
    /// which holds because every cell kept by `rebuild_seal_tables` has
    /// `base >= a`. Both sums are closed form, so the flux stays `O(1)` once the
    /// two counts are known.
    fn leak_speed_with(&self, t: usize, z: f64, seal_k: usize, sat_k: usize) -> f64 {
        if seal_k == 0 {
            return 0.0;
        }
        let breached =
            seal_k as f64 * (z - self.critical_height) - self.seal_topo_prefix[t - 1][seal_k];
        let saturated = sat_k as f64 * z - self.seal_base_prefix[t - 1][sat_k];
        self.seal_speed * (breached - saturated) / self.seal_thickness
    }

    fn area_leak_speed(&self, t: usize, z: f64) -> f64 {
        self.leak_speed_with(t, z, self.seal_breached(t, z), self.seal_saturated(t, z))
    }

    fn area_leak_rate_geom(&self, t: usize, contact: f64, kpg: f64) -> f64 {
        self.area_leak_speed(t, contact) * self.density_co2 * self.area / kpg
    }

    fn subtree_volume(&self, t: usize, held: &[f64]) -> f64 {
        let mut volume = held[t];
        for &child in &self.children[t] {
            volume += self.subtree_volume(child, held);
        }
        volume
    }

    fn distribute_leak(
        &self,
        t: usize,
        contact: f64,
        mass: f64,
        regions: &[i64],
        acc: &mut [f64],
    ) -> bool {
        if mass <= 0.0 {
            return false;
        }
        let total = self.area_leak_speed(t, contact);
        if total <= 0.0 {
            return false;
        }
        let breached = self.seal_breached(t, contact);
        for &c in &self.seal_cells[t - 1][..breached] {
            acc[breach_slot(regions, c)] += mass * self.seal_cell_speed(c, contact) / total;
        }
        true
    }

    fn drain_container(
        &self,
        t: usize,
        held: &mut [f64],
        kpg: f64,
        dt: f64,
        regions: &[i64],
        acc: &mut [f64],
    ) {
        if dt <= 0.0 {
            return;
        }
        let v_n = self.subtree_volume(t, held);
        if v_n <= 0.0 {
            return;
        }
        let contact = self.invert_contact(t, v_n);
        let floor = self.leak_floor(t);
        if contact <= floor + EPS {
            return;
        }
        let z_star = self.solve_leak_contact(t, contact, floor, v_n, kpg, dt);
        let requested = v_n - self.geomvol_at(t, z_star);
        if requested <= 0.0 {
            return;
        }
        if requested <= held[t] {
            if self.distribute_leak(t, z_star, requested * kpg, regions, acc) {
                held[t] -= requested;
            }
            return;
        }

        let own = held[t];
        let boundary_volume = v_n - own;
        let boundary_contact = self.invert_contact(t, boundary_volume);
        let boundary_rate = self.area_leak_rate_geom(t, boundary_contact, kpg);
        if boundary_rate <= 0.0 {
            return;
        }
        let event_dt = (own / boundary_rate).min(dt);
        if !self.distribute_leak(t, boundary_contact, own * kpg, regions, acc) {
            return;
        }
        held[t] = 0.0;
        let remaining = dt - event_dt;
        if remaining <= 0.0 {
            return;
        }
        for &child in &self.children[t] {
            if self.subtree_volume(child, held) > 0.0 {
                self.drain_container(child, held, kpg, remaining, regions, acc);
            }
        }
    }

    fn vertical_leak(
        &self,
        held: &mut [f64],
        kpg: f64,
        dt: f64,
        regions: &[i64],
        acc: &mut [f64],
        active: &mut Vec<usize>,
    ) {
        active.clear();
        active.extend((1..=self.num_traps).filter(|&t| {
            if held[t] <= 0.0 {
                return false;
            }
            let mut ancestor = self.parent[t];
            while ancestor != 0 {
                if held[ancestor] > 0.0 {
                    return false;
                }
                ancestor = self.parent[ancestor];
            }
            true
        }));
        for &trap in active.iter() {
            self.drain_container(trap, held, kpg, dt, regions, acc);
        }
    }

    fn solve_leak_contact(
        &self,
        t: usize,
        contact: f64,
        floor: f64,
        v_n: f64,
        kpg: f64,
        dt: f64,
    ) -> f64 {
        let scale = self.density_co2 * self.area / kpg;
        let residual =
            |z: f64| (v_n - self.geomvol_at(t, z)) - self.area_leak_speed(t, z) * scale * dt;

        if residual(contact) >= 0.0 {
            return contact;
        }
        if residual(floor) < 0.0 {
            return floor;
        }

        // Between two consecutive breakpoints both the stored volume and the leak
        // rate are linear in z, so the solve is exact once the interval is known.
        // The residual decreases with z, so on each sorted breakpoint list the
        // breakpoints that still hold are a prefix; the bracket is the highest
        // holding breakpoint and the next one above it, over both lists.
        // The elevations at which a seal cell fills to its base are breakpoints
        // of the leak rate too, but `volume_depth_table` already emits every
        // base of the footprint as a breakpoint of `vz_z`, so bracketing on the
        // two lists below covers them.
        // Drainage usually stops just below the current contact, so the prefix
        // length is found by galloping down from the top of the window.
        let bracket = |levels: &[f64], offset: f64| {
            let start = levels.partition_point(|&z| z + offset <= floor);
            let stop = levels.partition_point(|&z| z + offset < contact);
            let window = &levels[start..stop];
            let (mut lo, mut hi) = (0, window.len());
            let mut step = 1;
            while hi > 0 {
                let probe = hi - step.min(hi);
                if residual(window[probe] + offset) >= 0.0 {
                    lo = probe + 1;
                    break;
                }
                hi = probe;
                step *= 2;
            }
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if residual(window[mid] + offset) >= 0.0 {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            (
                if lo > 0 {
                    window[lo - 1] + offset
                } else {
                    floor
                },
                window.get(lo).map_or(contact, |&z| z + offset),
            )
        };
        let (seal_lo, seal_hi) = bracket(&self.seal_topo[t - 1], self.critical_height);
        let (level_lo, level_hi) = bracket(&self.vz_z[t - 1], 0.0);
        let z_lo = seal_lo.max(level_lo);
        let z_hi = seal_hi.min(level_hi);
        let seal_k = self.seal_breached(t, z_lo);
        let sat_k = self.seal_saturated(t, z_lo);
        let linear = |z: f64| {
            (v_n - self.geomvol_at(t, z)) - self.leak_speed_with(t, z, seal_k, sat_k) * scale * dt
        };
        let f_hi = linear(z_hi);
        if f_hi >= 0.0 {
            return z_hi;
        }
        let f_lo = linear(z_lo);
        let denom = f_lo - f_hi;
        if denom > 0.0 {
            z_lo + (z_hi - z_lo) * (f_lo / denom)
        } else {
            z_lo
        }
    }
}

#[pyclass]
pub struct TrapFill {
    layers: Vec<LayerModel>,
    source_regions: Vec<usize>,
    escape_regions: Vec<i64>,
    connate_water_saturation: f64,
    delta_rho_g: Vec<f64>,
    leak_enabled: bool,
    step_seconds: f64,
    time_rtol: f64,
    max_substeps: usize,
    held: Vec<Vec<f64>>,
    losses: Losses,
}

#[pymethods]
impl TrapFill {
    #[new]
    #[pyo3(signature = (
        tops, bases, critical_height, seal_thickness, density_co2, porosity, delta_rho_g, dx, dy,
        sources, connate_water_saturation, usediags=true, mobility=None,
        step_seconds=31_557_600.0, time_rtol=1.0e-4, max_substeps=16384
    ))]
    #[expect(clippy::needless_pass_by_value)]
    #[expect(clippy::too_many_arguments)]
    fn new(
        tops: PyReadonlyArray3<f64>,
        bases: PyReadonlyArray3<f64>,
        critical_height: PyReadonlyArray1<f64>,
        seal_thickness: PyReadonlyArray1<f64>,
        density_co2: PyReadonlyArray1<f64>,
        porosity: PyReadonlyArray1<f64>,
        delta_rho_g: PyReadonlyArray1<f64>,
        dx: f64,
        dy: f64,
        sources: Vec<(usize, usize)>,
        connate_water_saturation: f64,
        usediags: bool,
        mobility: Option<PyReadonlyArray1<f64>>,
        step_seconds: f64,
        time_rtol: f64,
        max_substeps: usize,
    ) -> PyResult<Self> {
        if !time_rtol.is_finite() || time_rtol <= 0.0 {
            return Err(PyValueError::new_err("time_rtol must be finite and > 0"));
        }
        if max_substeps < 8 || !max_substeps.is_power_of_two() {
            return Err(PyValueError::new_err(
                "max_substeps must be a power of two and at least 8",
            ));
        }
        if !step_seconds.is_finite() || step_seconds <= 0.0 {
            return Err(PyValueError::new_err("step_seconds must be finite and > 0"));
        }
        let tops = tops.as_array();
        let bases = bases.as_array();
        let critical_height = critical_height.as_array().to_vec();
        let seal_thickness = seal_thickness.as_array().to_vec();
        let density = density_co2.as_array();
        let porosity = porosity.as_array();
        let delta_rho_g = delta_rho_g.as_array().to_vec();
        let nlayers = tops.shape()[0];
        let nx = tops.shape()[1];

        if seal_thickness.iter().any(|l| l.is_nan() || *l <= 0.0) {
            return Err(PyValueError::new_err(
                "seal_thickness must be > 0 (use inf for a non-capillary pathway)",
            ));
        }
        let leak_enabled = mobility.is_some();
        let seals = seal_states(
            mobility.as_ref(),
            &critical_height,
            &seal_thickness,
            &delta_rho_g,
            nlayers,
        );

        let mut layers = Vec::with_capacity(nlayers);
        for k in 0..nlayers {
            layers.push(LayerModel::build(
                &tops.index_axis(ndarray::Axis(0), k),
                &bases.index_axis(ndarray::Axis(0), k),
                seals[k],
                density[k],
                porosity[k],
                (dx, dy),
                usediags,
            ));
        }

        let held: Vec<Vec<f64>> = layers.iter().map(|m| vec![0.0; m.num_traps + 1]).collect();
        if sources.is_empty() {
            return Err(PyValueError::new_err("at least one source is required"));
        }
        let source_regions: Vec<usize> = sources
            .iter()
            .map(|&(i, j)| {
                let cell = (i + PAD) + (j + PAD) * (nx + 2 * PAD);
                breach_slot(&layers[0].region_of_cell, cell)
            })
            .collect();
        let escape_regions = vec![0i64; layers[0].region_of_cell.len()];

        Ok(Self {
            layers,
            source_regions,
            escape_regions,
            connate_water_saturation,
            delta_rho_g,
            leak_enabled,
            step_seconds,
            time_rtol,
            max_substeps,
            held,
            losses: Losses::default(),
        })
    }

    fn fill(&self, mass_kg: f64) -> (Vec<f64>, f64, f64) {
        let swc = self.connate_water_saturation;
        let (held_per_layer, losses) = self.run(mass_kg);
        let mass_per_layer = self
            .layers
            .iter()
            .zip(&held_per_layer)
            .map(|(m, held)| held.iter().sum::<f64>() * m.kg_per_geom(swc))
            .collect();
        (mass_per_layer, losses.escaped, losses.stalled)
    }

    fn column_heights<'py>(&self, py: Python<'py>, mass_kg: f64) -> Vec<Bound<'py, PyArray2<f64>>> {
        let (held_per_layer, _losses) = self.run(mass_kg);
        self.layers
            .iter()
            .zip(&held_per_layer)
            .map(|(m, held)| m.unpad_columns(&m.column_heights(held)).into_pyarray(py))
            .collect()
    }

    fn reset(&mut self) {
        for h in &mut self.held {
            h.fill(0.0);
        }
        self.losses = Losses::default();
    }

    #[pyo3(signature = (critical_height, mobility=None))]
    #[expect(clippy::needless_pass_by_value)]
    fn update_seals(
        &mut self,
        critical_height: PyReadonlyArray1<f64>,
        mobility: Option<PyReadonlyArray1<f64>>,
    ) {
        self.leak_enabled = mobility.is_some();
        let thickness: Vec<f64> = self.layers.iter().map(|m| m.seal_thickness).collect();
        let seals = seal_states(
            mobility.as_ref(),
            &critical_height.as_array().to_vec(),
            &thickness,
            &self.delta_rho_g,
            self.layers.len(),
        );
        for (m, seal) in self.layers.iter_mut().zip(seals) {
            m.apply_seals(seal);
        }
        self.reset();
    }

    fn step(&mut self, inj_mass: f64) -> PyResult<(usize, f64)> {
        if !inj_mass.is_finite() || inj_mass < 0.0 {
            return Err(PyValueError::new_err("inj_mass must be finite and >= 0"));
        }
        if !self.leak_enabled {
            return Err(PyValueError::new_err("step requires a seal mobility field"));
        }
        Ok(self.advance_adaptive(inj_mass))
    }

    fn mass_per_layer(&self) -> Vec<f64> {
        let swc = self.connate_water_saturation;
        self.layers
            .iter()
            .zip(&self.held)
            .map(|(m, h)| h.iter().sum::<f64>() * m.kg_per_geom(swc))
            .collect()
    }

    const fn escaped_kg(&self) -> f64 {
        self.losses.escaped
    }

    /// Part of `escaped_kg` that the cascade guard dropped. Non-zero only for
    /// a malformed spill graph.
    const fn stalled_kg(&self) -> f64 {
        self.losses.stalled
    }

    fn state_column_heights<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<f64>>> {
        self.layers
            .iter()
            .zip(&self.held)
            .map(|(m, h)| m.unpad_columns(&m.column_heights(h)).into_pyarray(py))
            .collect()
    }
}

struct Scratch {
    transfer: Vec<Vec<f64>>,
    injection: Vec<f64>,
    full: Vec<bool>,
    active: Vec<usize>,
}

impl TrapFill {
    fn regions_above(&self, layer: usize) -> &[i64] {
        self.layers
            .get(layer + 1)
            .map_or(self.escape_regions.as_slice(), |m| {
                m.region_of_cell.as_slice()
            })
    }

    fn transfer_len(&self, layer: usize) -> usize {
        self.layers.get(layer + 1).map_or(1, |m| m.num_low + 1)
    }

    fn scratch(&self) -> Scratch {
        Scratch {
            transfer: (0..self.layers.len())
                .map(|layer| vec![0.0; self.transfer_len(layer)])
                .collect(),
            injection: vec![0.0; self.layers[0].num_low + 1],
            full: Vec::new(),
            active: Vec::new(),
        }
    }

    fn route_mass_into_layer(
        &self,
        layer: usize,
        inflow: &[f64],
        held: &mut [Vec<f64>],
        full: &mut Vec<bool>,
    ) -> Losses {
        if inflow.iter().all(|&mass| mass <= 0.0) {
            return Losses::default();
        }
        let model = &self.layers[layer];
        full.clear();
        full.push(false);
        full.extend(
            model
                .own_cap_spill
                .iter()
                .zip(&held[layer][1..])
                .map(|(&cap, &stored)| stored >= cap - EPS),
        );
        model.route(
            inflow,
            model.kg_per_geom(self.connate_water_saturation),
            &mut held[layer],
            full,
            &model.own_cap_spill,
            None,
        )
    }

    fn route_injection(&self, mass: f64, held: &mut [Vec<f64>], scratch: &mut Scratch) -> Losses {
        if mass <= 0.0 {
            return Losses::default();
        }
        scratch.injection.fill(0.0);
        let share = mass / self.source_regions.len() as f64;
        for &region in &self.source_regions {
            scratch.injection[region] += share;
        }
        self.route_mass_into_layer(0, &scratch.injection, held, &mut scratch.full)
    }

    fn microstep_state(
        &self,
        held: &mut [Vec<f64>],
        scratch: &mut Scratch,
        losses: &mut Losses,
        inj_mass: f64,
        dt: f64,
    ) {
        losses.add(self.route_injection(0.5 * inj_mass, held, scratch));
        for layer in 0..self.layers.len() {
            let kpg = self.layers[layer].kg_per_geom(self.connate_water_saturation);
            scratch.transfer[layer].fill(0.0);
            self.layers[layer].vertical_leak(
                &mut held[layer],
                kpg,
                dt,
                self.regions_above(layer),
                &mut scratch.transfer[layer],
                &mut scratch.active,
            );
            if layer + 1 < self.layers.len() {
                losses.add(self.route_mass_into_layer(
                    layer + 1,
                    &scratch.transfer[layer],
                    held,
                    &mut scratch.full,
                ));
            } else {
                losses.escaped += scratch.transfer[layer].iter().sum::<f64>();
            }
        }
        losses.add(self.route_injection(0.5 * inj_mass, held, scratch));
    }

    fn integrate_state(
        &self,
        start_held: &[Vec<f64>],
        start_losses: Losses,
        inj_mass: f64,
        duration: f64,
        substeps: usize,
    ) -> (Vec<Vec<f64>>, Losses) {
        let mut held = start_held.to_vec();
        let mut scratch = self.scratch();
        let mut losses = start_losses;
        let sub_mass = inj_mass / substeps as f64;
        let dt = duration / substeps as f64;
        for _ in 0..substeps {
            self.microstep_state(&mut held, &mut scratch, &mut losses, sub_mass, dt);
        }
        (held, losses)
    }

    fn state_error(
        &self,
        coarse_held: &[Vec<f64>],
        coarse_losses: Losses,
        fine_held: &[Vec<f64>],
        fine_losses: Losses,
    ) -> f64 {
        let mut max_difference = (fine_losses.escaped - coarse_losses.escaped).abs();
        let mut total_mass = fine_losses.escaped.abs();
        for ((coarse, fine), model) in coarse_held.iter().zip(fine_held).zip(&self.layers) {
            let kpg = model.kg_per_geom(self.connate_water_saturation);
            total_mass += fine.iter().sum::<f64>() * kpg;
            let redistributed = coarse
                .iter()
                .zip(fine)
                .map(|(&coarse_volume, &fine_volume)| (fine_volume - coarse_volume).abs())
                .sum::<f64>()
                * kpg;
            max_difference = max_difference.max(redistributed);
        }
        max_difference / total_mass.max(1.0)
    }

    fn advance_adaptive(&mut self, inj_mass: f64) -> (usize, f64) {
        let start_held = self.held.clone();
        let start_losses = self.losses;
        let (mut coarse_held, mut coarse_losses) =
            self.integrate_state(&start_held, start_losses, inj_mass, self.step_seconds, 1);
        let mut substeps = 2;
        loop {
            let (fine_held, fine_losses) = self.integrate_state(
                &start_held,
                start_losses,
                inj_mass,
                self.step_seconds,
                substeps,
            );
            let error = self.state_error(&coarse_held, coarse_losses, &fine_held, fine_losses);
            if error <= self.time_rtol || substeps >= self.max_substeps {
                self.held = fine_held;
                self.losses = fine_losses;
                return (substeps, error);
            }
            coarse_held = fine_held;
            coarse_losses = fine_losses;
            substeps *= 2;
        }
    }

    fn run(&self, mass_kg: f64) -> (Vec<Vec<f64>>, Losses) {
        let swc = self.connate_water_saturation;
        let mut inflow = vec![0.0; self.layers[0].num_low + 1];
        let share = mass_kg / self.source_regions.len() as f64;
        for &region in &self.source_regions {
            inflow[region] += share;
        }
        let mut held_per_layer = Vec::with_capacity(self.layers.len());
        let mut total = Losses::default();

        for (layer, m) in self.layers.iter().enumerate() {
            let mut up = vec![0.0; self.transfer_len(layer)];
            let (held, losses) = m.fill(
                &inflow,
                m.kg_per_geom(swc),
                (self.regions_above(layer), &mut up),
            );
            held_per_layer.push(held);
            total.add(losses);
            inflow = up;
        }
        total.escaped += inflow.iter().sum::<f64>();

        (held_per_layer, total)
    }
}

#[cfg(test)]
mod tests {
    use super::{Array2, LayerModel, PAD, SealState};

    const N: usize = 5;
    const SWC: f64 = 0.3;
    const DELTA_RHO_G: f64 = 300.0 * 9.81;

    fn model(hmax: f64, thickness: f64, mobility: f64) -> LayerModel {
        model_with_sand(hmax, thickness, mobility, |_, _| 50.0)
    }

    /// The dome, with the sand thickness under the seal chosen by the caller so
    /// that a test can make cells fill to their base, or be too thin to breach.
    fn model_with_sand(
        hmax: f64,
        thickness: f64,
        mobility: f64,
        sand: impl Fn(usize, usize) -> f64,
    ) -> LayerModel {
        let mut top = Array2::<f64>::zeros((N, N));
        let mut base = Array2::<f64>::zeros((N, N));
        for j in 0..N {
            for i in 0..N {
                let di = i as f64 - 2.0;
                let dj = j as f64 - 2.0;
                top[[i, j]] = 800.0 + di.hypot(dj);
                base[[i, j]] = top[[i, j]] + sand(i, j);
            }
        }
        LayerModel::build(
            &top.view(),
            &base.view(),
            SealState {
                critical_height: hmax,
                thickness,
                speed: mobility * DELTA_RHO_G,
            },
            700.0,
            0.36,
            (50.0, 50.0),
            true,
        )
    }

    fn centre(m: &LayerModel) -> usize {
        (2 + PAD) + (2 + PAD) * m.nx
    }

    /// Flux of the single seal cell `c`, computed the slow way and independently
    /// of the prefix sums: the buoyancy column is the CO2 standing in this cell,
    /// so it is capped by the thickness of the sand under the seal.
    fn cell_speed(m: &LayerModel, c: usize, z: f64) -> f64 {
        let contact = z.min(m.base[c]);
        if m.topo[c] + m.critical_height > contact {
            return 0.0;
        }
        m.seal_speed * ((contact - m.critical_height) - m.topo[c]) / m.seal_thickness
    }

    #[test]
    fn cell_flux_vanishes_at_breakthrough_and_grows_with_the_surplus_column() {
        let m = model(10.0, 4.0, 1.0e-9);
        let c = centre(&m);
        let breach = m.topo[c] + m.critical_height;
        let speed = m.seal_speed;
        assert!(speed > 0.0);
        assert!(cell_speed(&m, c, breach - 1.0) == 0.0);
        assert!(cell_speed(&m, c, breach).abs() < 1.0e-18);
        assert!((cell_speed(&m, c, breach + 4.0) - speed).abs() < 1.0e-18);
        assert!((cell_speed(&m, c, breach + 12.0) - 3.0 * speed).abs() < 1.0e-18);
    }

    #[test]
    fn an_infinitely_thick_seal_passes_nothing() {
        let m = model(10.0, f64::INFINITY, 1.0e-9);
        let c = centre(&m);
        let breach = m.topo[c] + m.critical_height;
        assert!(cell_speed(&m, c, breach + 100.0) == 0.0);
        assert!(m.area_leak_speed(1, breach + 100.0) == 0.0);
    }

    #[test]
    fn the_column_driving_a_seal_cell_stops_growing_once_its_sand_is_full() {
        // 12 m of sand under a seal that holds a 10 m column: the cell breaches
        // 10 m below its top and saturates 2 m later, after which a deeper trap
        // contact adds nothing.
        let m = model_with_sand(10.0, 4.0, 1.0e-9, |_, _| 12.0);
        let c = centre(&m);
        let top = m.topo[c];
        let speed = m.seal_speed;
        assert!(cell_speed(&m, c, top + 10.0).abs() < 1.0e-18);
        assert!((cell_speed(&m, c, top + 12.0) - 0.5 * speed).abs() < 1.0e-18);
        assert!((cell_speed(&m, c, top + 40.0) - 0.5 * speed).abs() < 1.0e-18);
        assert!((m.seal_cell_speed(c, top + 40.0) - 0.5 * speed).abs() < 1.0e-18);
    }

    #[test]
    fn breached_prefix_reproduces_the_brute_force_trap_flux() {
        // A thick sand no contact can fill, a thin uniform one every breached
        // cell fills, and a wedge in which the shallow cells are too thin to
        // hold a critical column at all.
        let models = [
            model(10.0, 4.0, 1.0e-9),
            model_with_sand(10.0, 4.0, 1.0e-9, |_, _| 12.0),
            model_with_sand(10.0, 4.0, 1.0e-9, |i, _| 4.0 + 2.0 * i as f64),
        ];
        for m in &models {
            for t in 1..=m.num_traps {
                for step in 0..300 {
                    let z = 805.0 + 0.1 * f64::from(step);
                    let brute: f64 = m.ts.footprints[t - 1]
                        .iter()
                        .map(|&c| cell_speed(m, c, z))
                        .sum();
                    let fast = m.area_leak_speed(t, z);
                    assert!(
                        (fast - brute).abs() <= 1.0e-9 * brute.max(1.0e-12),
                        "trap {t} at z={z}: {fast} vs {brute}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_seal_too_strong_to_breach_passes_nothing() {
        let sealed = model(1.0e4, 4.0, 1.0e-9);
        assert!(sealed.area_leak_speed(1, 802.0) == 0.0);
        assert!(sealed.area_leak_speed(1, 1.0e5) == 0.0);
        // No cell holds a 10 km column, so there is no elevation at which the
        // trap breaches and drainage never starts.
        assert!(sealed.leak_floor(1).is_infinite());
    }

    #[test]
    fn a_cell_too_thin_to_hold_a_critical_column_never_breaches() {
        // 8 m of sand under a seal that needs 10 m: nothing can ever cross.
        let thin = model_with_sand(10.0, 4.0, 1.0e-9, |_, _| 8.0);
        assert!(thin.area_leak_speed(1, 1.0e5) == 0.0);
        assert!(thin.leak_floor(1).is_infinite());
        assert!(!thin.is_break[0]);

        // Thicken only the crest: the trap now breaches there and nowhere else.
        let crest = model_with_sand(
            10.0,
            4.0,
            1.0e-9,
            |i, j| {
                if i == 2 && j == 2 { 30.0 } else { 8.0 }
            },
        );
        let c = centre(&crest);
        assert!(crest.seal_cells[0] == vec![c]);
        assert!((crest.leak_floor(1) - (crest.topo[c] + 10.0)).abs() < 1.0e-12);
    }

    #[test]
    fn drainage_stops_at_the_shallowest_breach_elevation() {
        let m = model(10.0, 4.0, 1.0e-6);
        let kpg = m.kg_per_geom(SWC);
        let floor = m.leak_floor(1);
        let stored = m.own_cap_spill[0];
        let mut held = vec![0.0; m.num_traps + 1];
        held[1] = stored;
        let regions = vec![0i64; m.nx * m.ny];
        let mut out = vec![0.0; 1];
        m.drain_container(1, &mut held, kpg, 1.0e12, &regions, &mut out);

        let retained = m.geomvol_at(1, floor);
        assert!(
            (held[1] - retained).abs() < 1.0e-6 * retained.max(1.0),
            "held {} vs retained {retained}",
            held[1]
        );
        let moved: f64 = out.iter().sum();
        let expected = (stored - held[1]) * kpg;
        assert!((moved - expected).abs() < 1.0e-6 * expected.max(1.0));
        assert!(moved > 0.0);
    }

    #[test]
    fn a_thinner_seal_transfers_more_at_equal_mobility() {
        let transferred = |thickness: f64| {
            let m = model(10.0, thickness, 1.0e-12);
            let kpg = m.kg_per_geom(SWC);
            let mut held = vec![0.0; m.num_traps + 1];
            held[1] = m.own_cap_spill[0];
            let regions = vec![0i64; m.nx * m.ny];
            let mut out = vec![0.0; 1];
            m.drain_container(1, &mut held, kpg, 31_557_600.0, &regions, &mut out);
            out.iter().sum::<f64>()
        };
        let thin = transferred(1.0);
        let thick = transferred(8.0);
        assert!(thin > thick, "thin {thin} should exceed thick {thick}");
        assert!(thick > 0.0);
    }
}
