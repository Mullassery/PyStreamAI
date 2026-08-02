use pyo3::prelude::*;

#[pyfunction]
fn hello() -> String {
    "hello".to_string()
}

#[pymodule]
fn pystreamai(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    Ok(())
}
