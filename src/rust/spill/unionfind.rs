use std::cmp::Ordering;

pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    pub fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        let mut cur = x;
        while self.parent[cur] != root {
            let next = self.parent[cur];
            self.parent[cur] = root;
            cur = next;
        }
        root
    }

    pub fn union(&mut self, a: usize, b: usize) -> usize {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return ra;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            Ordering::Less => {
                self.parent[ra] = rb;
                rb
            }
            Ordering::Greater => {
                self.parent[rb] = ra;
                ra
            }
            Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
                ra
            }
        }
    }
}
