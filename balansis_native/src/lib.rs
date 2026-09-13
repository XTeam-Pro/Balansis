use pyo3::prelude::*;

#[pyfunction]
fn add_absolute(m1: f64, d1: i8, m2: f64, d2: i8) -> (f64, i8) {
    if d1 == d2 {
        (m1 + m2, d1)
    } else if m1 > m2 {
        (m1 - m2, d1)
    } else if m2 > m1 {
        (m2 - m1, d2)
    } else {
        (0.0, 1)
    }
}

#[pymodule]
fn balansis_native(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add_absolute, m)?)?;
    Ok(())
}
