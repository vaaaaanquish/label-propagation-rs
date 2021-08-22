/// Core model in Confidence-Aware Modulated Label Propagation (CAMLP)
use std::result::Result;
use std::error::Error;

use ndarray::{Array, ArrayBase, Axis, OwnedRepr};
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;

pub struct CAMLP {
    graph: ArrayBase::<OwnedRepr<f32>, Ix2>,
    result: Option<ArrayBase::<OwnedRepr<f32>, Ix2>>,
}


impl CAMLP {
    pub fn new(graph: ArrayBase::<OwnedRepr<f32>, Ix2>) -> Self {
        CAMLP { graph: graph, result: None }
    }

    pub fn fit(&mut self, x: &ArrayBase::<OwnedRepr<usize>, Ix1>, y: &ArrayBase::<OwnedRepr<usize>, Ix1>) -> Result<(), Box<dyn Error>> {
        let c = *x.max()? + 1;
        let s = self.graph.shape()[0];
        let d = self.graph.sum_axis(Axis(1));

        let z = Array::from_diag(&(1.0 / (1.0 + d * 0.1)));  // param beta=0.1

        let p = z.dot(&(0.1 * &self.graph));
        let mut b = Array::ones((s, c)) / (c as f32);

        // numpy -> b[[0,1]] = 0.
        for x_i in x {
            b.slice_mut(s![*x_i, ..]).fill(0.);
        }

        // numpy -> b[[0,1], [0,1]] = 1.
        for (x_i, y_i) in x.iter().zip(y.iter()) {
            b.slice_mut(s![*x_i, *y_i]).fill(1.);
        }

        let mut f = z.dot(&b);

        for _ in 0..2 {  // param iter=2
            f = p.dot(&f) + &b;
        }

        self.result = Some(f);
        Ok(())
    }

    pub fn predict_proba(&mut self, target_node: &ArrayBase::<OwnedRepr<usize>, Ix1>) -> ArrayBase::<OwnedRepr<f32>, Ix2> {
        let mut result = Array::zeros((target_node.shape()[0], self.result.as_mut().unwrap().shape()[1]));
        for i in target_node {
            result.slice_mut(s![*i, ..]).assign(&self.result.as_mut().unwrap().slice_mut(s![*i, ..]));
        }
        result
    }
}
