use pyo3::prelude::*;

mod spill;

#[pymodule]
fn rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<spill::TrapFill>()?;
    spill::register(m)?;
    Ok(())
}
