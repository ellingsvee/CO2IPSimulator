#![expect(clippy::needless_pass_by_value)]

use super::trapstructure::{SpillOptions, spillanalysis as run_spillanalysis};
use super::{spillfield, spillpoints, spillregions};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
#[pyo3(name = "spillfield", signature = (topography, usediags=true))]
fn spillfield_py<'py>(
    py: Python<'py>,
    topography: PyReadonlyArray2<'py, f64>,
    usediags: bool,
) -> (Bound<'py, PyArray2<i8>>, Bound<'py, PyArray2<f64>>) {
    let topo = topography.as_array().to_owned();
    let (dir, slope) = spillfield::spillfield(&topo, usediags, None);
    (dir.into_pyarray(py), slope.into_pyarray(py))
}

#[pyfunction]
#[pyo3(name = "spillregions", signature = (spillfield, usediags=true))]
fn spillregions_py<'py>(
    py: Python<'py>,
    spillfield: PyReadonlyArray2<'py, i8>,
    usediags: bool,
) -> Bound<'py, PyArray2<i64>> {
    let field = spillfield.as_array().to_owned();
    let regions = spillregions::spillregions(&field, usediags);
    regions.into_pyarray(py)
}

type SpillpointArrays<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
);

const fn cell_to_1based(c: i64) -> i64 {
    if c < 0 { 0 } else { c + 1 }
}

#[pyfunction]
#[pyo3(name = "spillpoints", signature = (grid, regions, usediags=true))]
fn spillpoints_py<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray2<'py, f64>,
    regions: PyReadonlyArray2<'py, i64>,
    usediags: bool,
) -> SpillpointArrays<'py> {
    let g = grid.as_array().to_owned();
    let r = regions.as_array().to_owned();
    let (sps, _boundaries) = spillpoints::spillpoints(&g, &r, usediags);
    let downstream: Vec<i64> = sps.iter().map(|s| s.downstream_region).collect();
    let current: Vec<i64> = sps
        .iter()
        .map(|s| cell_to_1based(s.current_region_cell))
        .collect();
    let downcell: Vec<i64> = sps
        .iter()
        .map(|s| cell_to_1based(s.downstream_region_cell))
        .collect();
    let elev: Vec<f64> = sps.iter().map(|s| s.elevation).collect();
    (
        PyArray1::from_vec(py, downstream),
        PyArray1::from_vec(py, current),
        PyArray1::from_vec(py, downcell),
        PyArray1::from_vec(py, elev),
    )
}

#[pyfunction]
#[pyo3(name = "spillanalysis", signature = (grid, usediags=true))]
fn spillanalysis_py<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray2<'py, f64>,
    usediags: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let g = grid.as_array().to_owned();
    let ts = run_spillanalysis(
        &g,
        SpillOptions {
            usediags,
            closed: false,
            lengths: None,
        },
    );

    let elev: Vec<f64> = ts.spillpoints.iter().map(|s| s.elevation).collect();
    let fp_len: Vec<i64> = ts.footprints.iter().map(|f| f.len() as i64).collect();

    let d = PyDict::new(py);
    d.set_item("num_spoints", ts.spillpoints.len() as i64)?;
    d.set_item("num_regions", ts.supertraps_of.len() as i64)?;
    d.set_item("agglom_ne", ts.agglomerations.ne() as i64)?;
    d.set_item("regions", ts.regions.into_pyarray(py))?;
    d.set_item("spillfield", ts.spillfield.into_pyarray(py))?;
    d.set_item("trapvolumes", PyArray1::from_vec(py, ts.trapvolumes))?;
    d.set_item("subvolumes", PyArray1::from_vec(py, ts.subvolumes))?;
    d.set_item("sp_elevation", PyArray1::from_vec(py, elev))?;
    d.set_item("footprint_lengths", PyArray1::from_vec(py, fp_len))?;
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(spillfield_py, m)?)?;
    m.add_function(wrap_pyfunction!(spillregions_py, m)?)?;
    m.add_function(wrap_pyfunction!(spillpoints_py, m)?)?;
    m.add_function(wrap_pyfunction!(spillanalysis_py, m)?)?;
    Ok(())
}
