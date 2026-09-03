use super::index::{Index2, dir_offset};
use super::unionfind::UnionFind;
use ndarray::Array2;

pub fn spillregions(field: &Array2<i8>, usediags: bool) -> Array2<i64> {
    let (nx, ny) = (field.shape()[0], field.shape()[1]);
    let idx = Index2::new(nx, ny);
    let n = idx.len();
    let mut uf = UnionFind::new(n);

    let flat_shifts: &[(i64, i64)] = if usediags {
        &[(1, 0), (-1, 1), (0, 1), (1, 1)]
    } else {
        &[(1, 0), (0, 1)]
    };

    for j in 0..ny {
        for i in 0..nx {
            if field[[i, j]] == -1 {
                let c = idx.lin(i, j);
                for &(di, dj) in flat_shifts {
                    let (ni, nj) = (i as i64 + di, j as i64 + dj);
                    if idx.in_bounds(ni, nj) && field[[ni as usize, nj as usize]] == -1 {
                        uf.union(c, idx.lin(ni as usize, nj as usize));
                    }
                }
            }
        }
    }

    for j in 0..ny {
        for i in 0..nx {
            let d = field[[i, j]];
            if d == -1 {
                continue;
            }
            if let Some((di, dj)) = dir_offset(d) {
                let (ni, nj) = (i as i64 + di, j as i64 + dj);
                if idx.in_bounds(ni, nj) {
                    uf.union(idx.lin(i, j), idx.lin(ni as usize, nj as usize));
                }
            }
        }
    }

    let mut exit_cells: Vec<usize> = Vec::new();
    for i in 0..nx {
        let d = field[[i, 0]];
        if d == -1 || d == 2 || d == 4 || d == 6 {
            exit_cells.push(idx.lin(i, 0));
        }
        let d = field[[i, ny - 1]];
        if d == -1 || d == 3 || d == 5 || d == 7 {
            exit_cells.push(idx.lin(i, ny - 1));
        }
    }
    for j in 0..ny {
        let d = field[[0, j]];
        if d == -1 || d == 0 || d == 4 || d == 7 {
            exit_cells.push(idx.lin(0, j));
        }
        let d = field[[nx - 1, j]];
        if d == -1 || d == 1 || d == 5 || d == 6 {
            exit_cells.push(idx.lin(nx - 1, j));
        }
    }
    let mut is_exit_root = vec![false; n];
    for &c in &exit_cells {
        let r = uf.find(c);
        is_exit_root[r] = true;
    }

    let mut label = vec![0i64; n];
    let mut regions = Array2::<i64>::zeros((nx, ny));
    let mut next_pos = 1i64;
    let mut next_neg = -1i64;
    for j in 0..ny {
        for i in 0..nx {
            let r = uf.find(idx.lin(i, j));
            if label[r] == 0 {
                if is_exit_root[r] {
                    label[r] = next_neg;
                    next_neg -= 1;
                } else {
                    label[r] = next_pos;
                    next_pos += 1;
                }
            }
            regions[[i, j]] = label[r];
        }
    }

    regions
}
