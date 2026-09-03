use std::collections::HashMap;

pub struct DiGraph {
    nv: usize,
    out_adj: Vec<Vec<usize>>,
    in_adj: Vec<Vec<usize>>,
}

impl DiGraph {
    pub fn new(n: usize) -> Self {
        Self {
            nv: n,
            out_adj: vec![Vec::new(); n + 1],
            in_adj: vec![Vec::new(); n + 1],
        }
    }

    pub fn add_vertices(&mut self, k: usize) {
        self.nv += k;
        self.out_adj.resize(self.nv + 1, Vec::new());
        self.in_adj.resize(self.nv + 1, Vec::new());
    }

    pub fn add_edge(&mut self, s: usize, d: usize) {
        if s == 0 || d == 0 || s > self.nv || d > self.nv {
            return;
        }
        if self.out_adj[s].contains(&d) {
            return;
        }
        self.out_adj[s].push(d);
        self.in_adj[d].push(s);
    }

    pub fn out_neighbors(&self, v: usize) -> &[usize] {
        &self.out_adj[v]
    }

    pub fn in_neighbors(&self, v: usize) -> &[usize] {
        &self.in_adj[v]
    }

    pub fn ne(&self) -> usize {
        self.out_adj.iter().map(Vec::len).sum()
    }
}

pub fn functional_cycles(out: &[i64], nv: usize) -> Vec<Vec<usize>> {
    let mut state = vec![0u8; nv + 1];
    let mut pos = vec![0usize; nv + 1];
    let mut cycles: Vec<Vec<usize>> = Vec::new();

    for s in 1..=nv {
        if state[s] != 0 {
            continue;
        }
        let mut path: Vec<usize> = Vec::new();
        let mut v = s;
        loop {
            if v == 0 || state[v] == 2 {
                break;
            }
            if state[v] == 1 {
                let start = pos[v];
                cycles.push(path[start..].to_vec());
                break;
            }
            state[v] = 1;
            pos[v] = path.len();
            path.push(v);
            v = out[v] as usize;
        }
        for &p in &path {
            state[p] = 2;
        }
    }
    cycles
}

fn scc(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = adj.len();
    let mut radj = vec![Vec::new(); n];
    for (u, neighbors) in adj.iter().enumerate() {
        for &w in neighbors {
            radj[w].push(u);
        }
    }

    let mut visited = vec![false; n];
    let mut order: Vec<usize> = Vec::new();
    for s in 0..n {
        if visited[s] {
            continue;
        }
        visited[s] = true;
        let mut stack: Vec<(usize, usize)> = vec![(s, 0)];
        while let Some((u, idx)) = stack.pop() {
            if let Some(&w) = adj[u].get(idx) {
                stack.push((u, idx + 1));
                if !visited[w] {
                    visited[w] = true;
                    stack.push((w, 0));
                }
            } else {
                order.push(u);
            }
        }
    }

    let mut comp = vec![usize::MAX; n];
    let mut comps: Vec<Vec<usize>> = Vec::new();
    for &s in order.iter().rev() {
        if comp[s] != usize::MAX {
            continue;
        }
        let cid = comps.len();
        comp[s] = cid;
        let mut c = Vec::new();
        let mut stack = vec![s];
        while let Some(u) = stack.pop() {
            c.push(u);
            for &w in &radj[u] {
                if comp[w] == usize::MAX {
                    comp[w] = cid;
                    stack.push(w);
                }
            }
        }
        comps.push(c);
    }
    comps
}

pub fn sibling_loops(connections: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut id: HashMap<usize, usize> = HashMap::new();
    let mut uniq: Vec<usize> = Vec::new();
    for x in connections.iter().flat_map(|c| [c.0, c.1]) {
        id.entry(x).or_insert_with(|| {
            uniq.push(x);
            uniq.len() - 1
        });
    }
    let n = uniq.len();
    let mut adj = vec![Vec::new(); n];
    for &(a, b) in connections {
        adj[id[&a]].push(id[&b]);
    }
    scc(&adj)
        .into_iter()
        .filter(|c| c.len() > 1)
        .map(|c| c.into_iter().map(|i| uniq[i]).collect())
        .collect()
}
