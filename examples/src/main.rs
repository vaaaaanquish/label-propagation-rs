use std::result::Result;
use std::error::Error;

extern crate label_propagation;
extern crate ndarray;
extern crate ndarray_stats;

use ndarray::prelude::*;
use label_propagation::CAMLP;
use ndarray::Array;

pub fn main() -> Result<(), Box<dyn Error>> {
    let graph = Array::from_shape_vec((3, 3), vec![0.0, 0.3, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();  // graph matrix data
    let x = array![0, 1];  // node index
    let y = array![0, 1];  // label index

    let mut model = CAMLP::new(graph).iter(2).beta(0.1);
    model.fit(&x, &y)?;

    let target = array![0, 1];
    let result = model.predict_proba(&target);
    println!("{:?}", result);

    Ok(())
}
