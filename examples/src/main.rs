use std::result::Result;
use std::error::Error;

extern crate label_propagation;
extern crate ndarray;
extern crate ndarray_stats;

use ndarray::prelude::*;
use label_propagation::CAMLP;
use ndarray::Array;

pub fn main() -> Result<(), Box<dyn Error>> {
    // origin matrix data
    let data = vec![
        vec![0.0, 0.3, 0.0],
        vec![0.3, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
    ];
    let row_len = data.len();
    let col_len = data[0].len();
    let flat_data = data.into_iter().flatten().collect::<Vec<_>>();
    let graph = Array::from_shape_vec((row_len, col_len), flat_data).unwrap();

    let x = array![0, 1];  // node index
    let y = array![0, 1];  // label index

    let mut model = CAMLP::new(graph);
    model.fit(&x, &y)?;

    let target = array![0, 1];
    let result = model.predict_proba(&target);
    println!("{:?}", result);

    Ok(())
}
