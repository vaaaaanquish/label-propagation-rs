use std::result::Result;
use std::error::Error;

extern crate label_propagation;
extern crate ndarray;
extern crate ndarray_stats;

use ndarray::prelude::*;
use label_propagation::camlp::CAMLP;

pub fn main() -> Result<(), Box<dyn Error>> {
    // origin matrix data
    let data = vec![
        vec![0.0, 0.3, 0.0],
        vec![0.3, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
    ];
    let x = array![0, 1];
    let y = array![0, 1];

    let mut model = CAMLP::new(data);
    model.fit(&x, &y)?;

    let target = array![0, 1];
    let result = model.predict_proba(&target);
    println!("{:?}", result);

    Ok(())
}
