use ndarray::Array2;

#[derive(Clone, Copy, Debug)]
pub struct Spillpoint {
    pub downstream_region: i64,
    pub current_region_cell: i64,
    pub downstream_region_cell: i64,
    pub elevation: f64,
}

impl Default for Spillpoint {
    fn default() -> Self {
        Self {
            downstream_region: 0,
            current_region_cell: -1,
            downstream_region_cell: -1,
            elevation: f64::INFINITY,
        }
    }
}

pub fn spillpoints(
    grid: &Array2<f64>,
    regions: &Array2<i64>,
    usediags: bool,
) -> (Vec<Spillpoint>, Vec<Vec<(usize, usize)>>) {
    let (nx, ny) = (grid.shape()[0], grid.shape()[1]);
    let nxi = nx as i64;
    let nyi = ny as i64;

    let maxpos = regions.iter().copied().max().unwrap_or(0).max(0);
    let np = maxpos as usize;

    let mut result = vec![Spillpoint::default(); np];
    let mut boundaries: Vec<Vec<(usize, usize)>> = vec![Vec::new(); np];

    let lin0 = |i: i64, j: i64| -> usize { ((i - 1) + (j - 1) * nxi) as usize };
    let reg = |i: i64, j: i64| -> i64 { regions[[(i - 1) as usize, (j - 1) as usize]] };
    let z = |i: i64, j: i64| -> f64 { grid[[(i - 1) as usize, (j - 1) as usize]] };

    let mut shifts: Vec<(i64, i64)> = vec![(1, 0), (0, 1)];
    if usediags {
        shifts.push((1, 1));
        shifts.push((1, -1));
    }

    for &(s0, s1) in &shifts {
        let start_i = 1 - s0.min(0);
        let end_i = nxi - s0.max(0);
        let start_j = 1 - s1.min(0);
        let end_j = nyi - s1.max(0);

        let mut j = start_j;
        while j <= end_j {
            let mut i = start_i;
            while i <= end_i {
                let (i2, j2) = (i + s0, j + s1);
                let reg1 = reg(i, j);
                let reg2 = reg(i2, j2);
                if reg1 != 0 && reg2 != 0 && reg1 != reg2 {
                    let lpos1 = lin0(i, j);
                    let lpos2 = lin0(i2, j2);
                    if reg1 > 0 {
                        boundaries[(reg1 - 1) as usize].push((lpos1, lpos2));
                    }
                    if reg2 > 0 {
                        boundaries[(reg2 - 1) as usize].push((lpos2, lpos1));
                    }
                    let zval = z(i, j).max(z(i2, j2));
                    if reg1 > 0 && zval < result[(reg1 - 1) as usize].elevation {
                        result[(reg1 - 1) as usize] = Spillpoint {
                            downstream_region: reg2,
                            current_region_cell: lpos1 as i64,
                            downstream_region_cell: lpos2 as i64,
                            elevation: zval,
                        };
                    }
                    if reg2 > 0 && zval < result[(reg2 - 1) as usize].elevation {
                        result[(reg2 - 1) as usize] = Spillpoint {
                            downstream_region: reg1,
                            current_region_cell: lpos2 as i64,
                            downstream_region_cell: lpos1 as i64,
                            elevation: zval,
                        };
                    }
                }
                i += 1;
            }
            j += 1;
        }
    }

    let mut update_outer = |region: i64, i: i64, j: i64| {
        if region <= 0 {
            return;
        }
        let cell = lin0(i, j);
        boundaries[(region - 1) as usize].push((cell, cell));
        let zval = z(i, j);
        if zval < result[(region - 1) as usize].elevation {
            result[(region - 1) as usize] = Spillpoint {
                downstream_region: 0,
                current_region_cell: cell as i64,
                downstream_region_cell: cell as i64,
                elevation: zval,
            };
        }
    };

    for i in 1..=nxi {
        update_outer(reg(i, 1), i, 1);
        update_outer(reg(i, nyi), i, nyi);
    }
    for j in 1..=nyi {
        update_outer(reg(1, j), 1, j);
        update_outer(reg(nxi, j), nxi, j);
    }

    (result, boundaries)
}
