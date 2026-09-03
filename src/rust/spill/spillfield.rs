use ndarray::Array2;

struct OffGrid {
    data: Vec<f64>,
    ni: usize,
    oi: i64,
    oj: i64,
}

impl OffGrid {
    fn new(ni: usize, nj: usize, oi: i64, oj: i64) -> Self {
        Self {
            data: vec![f64::NAN; ni * nj],
            ni,
            oi,
            oj,
        }
    }

    #[inline]
    const fn idx(&self, i: i64, j: i64) -> usize {
        let si = (i - self.oi - 1) as usize;
        let sj = (j - self.oj - 1) as usize;
        si + sj * self.ni
    }

    #[inline]
    fn get(&self, i: i64, j: i64) -> f64 {
        self.data[self.idx(i, j)]
    }

    #[inline]
    fn set(&mut self, i: i64, j: i64, v: f64) {
        let k = self.idx(i, j);
        self.data[k] = v;
    }

    #[inline]
    const fn first_i(&self) -> i64 {
        self.oi + 1
    }

    #[inline]
    const fn first_j(&self) -> i64 {
        self.oj + 1
    }
}

#[inline]
fn g(grid: &Array2<f64>, i: i64, j: i64) -> f64 {
    grid[[(i - 1) as usize, (j - 1) as usize]]
}

fn diffgrid(grid: &Array2<f64>, nx: usize, ny: usize, delta: f64, shift: (i64, i64)) -> OffGrid {
    let (s0, s1) = shift;
    let deltainv = 1.0 / delta;
    let nxi = nx as i64;
    let nyi = ny as i64;

    let r_is = 1.min(1 - s0);
    let r_ie = nxi.max(nxi - s0);
    let r_js = 1.min(1 - s1);
    let r_je = nyi.max(nyi - s1);

    let in_is = 1.max(1 - s0);
    let in_ie = nxi.min(nxi - s0);
    let in_js = 1.max(1 - s1);
    let in_je = nyi.min(nyi - s1);

    let ni = (nxi + s0.abs()) as usize;
    let nj = (nyi + s1.abs()) as usize;
    let oi = r_is - 1;
    let oj = r_js - 1;

    let mut r = OffGrid::new(ni, nj, oi, oj);

    let mut j = in_js;
    while j <= in_je {
        let mut i = in_is;
        while i <= in_ie {
            let v = (g(grid, i + s0, j + s1) - g(grid, i, j)) * deltainv;
            r.set(i, j, v);
            i += 1;
        }
        j += 1;
    }

    let sgn = s0 * s1;

    if r_is < in_is {
        let jt_s = r_js - sgn.min(0);
        let jt_e = r_je - sgn.max(0);
        let js_s = jt_s + sgn;
        let mut k = 0;
        while jt_s + k <= jt_e {
            let v = r.get(r_is + 1, js_s + k);
            r.set(r_is, jt_s + k, v);
            k += 1;
        }
    }
    if r_ie > in_ie {
        let jt_s = r_js + sgn.max(0);
        let jt_e = r_je + sgn.min(0);
        let js_s = jt_s - sgn;
        let mut k = 0;
        while jt_s + k <= jt_e {
            let v = r.get(r_ie - 1, js_s + k);
            r.set(r_ie, jt_s + k, v);
            k += 1;
        }
    }
    if r_js < in_js {
        let it_s = r_is - sgn.min(0);
        let it_e = r_ie - sgn.max(0);
        let is_s = it_s + sgn;
        let mut k = 0;
        while it_s + k <= it_e {
            let v = r.get(is_s + k, r_js + 1);
            r.set(it_s + k, r_js, v);
            k += 1;
        }
    }
    if r_je > in_je {
        let it_s = r_is + sgn.max(0);
        let it_e = r_ie + sgn.min(0);
        let is_s = it_s - sgn;
        let mut k = 0;
        while it_s + k <= it_e {
            let v = r.get(is_s + k, r_je - 1);
            r.set(it_s + k, r_je, v);
            k += 1;
        }
    }

    r
}

fn compare_slopes(g1: &OffGrid, g2: &OffGrid, fac1: f64, off: (i64, i64)) -> (Vec<bool>, OffGrid) {
    let (o0, o1) = off;
    let sh0 = (g1.ni as i64 - o0.abs()) as usize;
    let nj1 = g1.data.len() / g1.ni;
    let sh1 = (nj1 as i64 - o1.abs()) as usize;

    let c1i = g1.first_i() + o0.max(0) - 1;
    let c1j = g1.first_j() + o1.max(0) - 1;
    let c2i = g2.first_i() + o0.max(0) - 1;
    let c2j = g2.first_j() + o1.max(0) - 1;

    let mut choice = vec![false; sh0 * sh1];
    let mut minslope = OffGrid::new(sh0, sh1, 0, 0);

    let mut col: i64 = 1;
    while col <= sh1 as i64 {
        let mut row: i64 = 1;
        while row <= sh0 as i64 {
            let lval = g1.get(row + c1i - o0, col + c1j - o1) * fac1;
            let rval = g2.get(row + c2i, col + c2j);
            let pick = rval < lval;
            choice[(row - 1) as usize + (col - 1) as usize * sh0] = pick;
            minslope.set(row, col, if pick { rval } else { lval });
            row += 1;
        }
        col += 1;
    }

    (choice, minslope)
}

fn find_downslopes(
    grid: &Array2<f64>,
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    diag: bool,
) -> (Vec<i8>, OffGrid) {
    let dxy = dx.hypot(dy);
    let (delta1, delta2, step1, step2) = if diag {
        (dxy, dxy, (1i64, 1i64), (-1i64, 1i64))
    } else {
        (dx, dy, (1i64, 0i64), (0i64, 1i64))
    };

    let d1 = diffgrid(grid, nx, ny, delta1, step1);
    let d2 = diffgrid(grid, nx, ny, delta2, step2);

    let (orient1, slopes1) = compare_slopes(&d1, &d1, -1.0, step1);
    let (orient2, slopes2) = compare_slopes(&d2, &d2, -1.0, step2);
    let (orient, slope) = compare_slopes(&slopes1, &slopes2, 1.0, (0, 0));

    let n = nx * ny;
    let mut dir = vec![0i8; n];
    for c in 0..n {
        dir[c] = if orient[c] {
            2 + i8::from(orient2[c])
        } else {
            i8::from(orient1[c])
        };
    }

    (dir, slope)
}

pub fn spillfield(
    grid: &Array2<f64>,
    usediags: bool,
    lengths: Option<(f64, f64)>,
) -> (Array2<i8>, Array2<f64>) {
    let (nx, ny) = (grid.shape()[0], grid.shape()[1]);

    let (lx, ly) = lengths.unwrap_or((nx as f64, ny as f64));
    let dx = lx / nx as f64;
    let dy = ly / ny as f64;

    let (mut dir, slope) = find_downslopes(grid, nx, ny, dx, dy, false);
    let mut slope = slope.data;

    if usediags {
        let (dir_d, slope_d) = find_downslopes(grid, nx, ny, dx, dy, true);
        let slope_d = slope_d.data;
        for c in 0..nx * ny {
            if slope_d[c] < slope[c] {
                dir[c] = dir_d[c] + 4;
                slope[c] = slope_d[c];
            }
        }
    }

    for c in 0..nx * ny {
        if slope[c] >= 0.0 {
            dir[c] = -1;
        }
    }

    let mut dir_arr = Array2::<i8>::zeros((nx, ny));
    let mut slope_arr = Array2::<f64>::zeros((nx, ny));
    for j in 0..ny {
        for i in 0..nx {
            let c = i + j * nx;
            dir_arr[[i, j]] = dir[c];
            slope_arr[[i, j]] = slope[c];
        }
    }

    (dir_arr, slope_arr)
}
