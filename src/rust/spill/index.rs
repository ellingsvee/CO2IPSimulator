#[derive(Clone, Copy, Debug)]
pub struct Index2 {
    pub nx: usize,
    pub ny: usize,
}

impl Index2 {
    pub const fn new(nx: usize, ny: usize) -> Self {
        Self { nx, ny }
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.nx * self.ny
    }

    #[inline]
    pub const fn lin(&self, i: usize, j: usize) -> usize {
        i + j * self.nx
    }

    #[inline]
    pub const fn in_bounds(&self, i: i64, j: i64) -> bool {
        i >= 0 && j >= 0 && (i as usize) < self.nx && (j as usize) < self.ny
    }
}

pub const DIR_OFFSETS: [(i64, i64); 8] = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
];

#[inline]
pub fn dir_offset(dir: i8) -> Option<(i64, i64)> {
    if (0..8).contains(&dir) {
        Some(DIR_OFFSETS[dir as usize])
    } else {
        None
    }
}
