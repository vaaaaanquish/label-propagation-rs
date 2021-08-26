use std::result::Result;
use std::error::Error;

extern crate label_propagation;
extern crate ndarray;
extern crate ndarray_stats;
extern crate smartcore;
extern crate rand;

use label_propagation::CAMLP;
use ndarray::Array;
use ndarray::prelude::*;
use ndarray_stats::{DeviationExt, QuantileExt};
use smartcore::dataset::iris;
use rand::{thread_rng, seq::IteratorRandom};


pub fn main() -> Result<(), Box<dyn Error>> {
    let iris = iris::load_dataset();

    let node = (0..iris.num_samples).collect::<Array<usize, _>>();
    let mut label = Array::from_shape_vec(iris.num_samples, iris.target.iter().map(|x| *x as usize).collect())?;
    let mut graph = Array::<f32, _>::zeros((iris.num_samples, iris.num_samples));

    // convert to euclid distance matrix
    let data = Array::from_shape_vec((iris.num_samples, iris.num_features), iris.data)?;
    for i in 0..iris.num_samples {
        for j in 0..iris.num_samples {
            if i != j {
                let weight = 1. / (*&data.slice(s![i, ..]).sq_l2_dist(&data.slice(s![j, ..]))? + 1.);  // reciprocal
                if weight > 0.5 {
                    graph[[i, j]] = weight;
                }
            }
        }
    }

    // destroy target labels
    let target_num = 10;
    let mut rng = thread_rng();
    let target = (0..iris.num_samples).choose_multiple(&mut rng, target_num).iter().map(|x| *x).collect::<Array<usize, _>>();
    for i in &target {
        label[*i] = 0;
    }

    // model fit
    let mut model = CAMLP::new(graph).iter(100).beta(0.1);
    model.fit(&node, &label)?;

    // predict
    let result = model.predict_proba(&target);

    // print
    for (i, x) in target.iter().enumerate() {
        println!("node: {:?}, label: {:?}, result: {:?}", *x, iris.target[*x], result.slice(s![i, ..]).argmax()?);
    }

    Ok(())
}
