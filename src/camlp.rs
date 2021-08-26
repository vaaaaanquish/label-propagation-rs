use std::error::Error;
use std::result::Result;

use ndarray::prelude::*;
use ndarray::{Array, ArrayBase, Axis, OwnedRepr};
use ndarray_stats::QuantileExt;

/// Core model in Confidence-Aware Modulated Label Propagation (CAMLP) [Yamaguchi+, SDM'16]
/// https://epubs.siam.org/doi/pdf/10.1137/1.9781611974348.58
pub struct CAMLP {
    graph: ArrayBase<OwnedRepr<f32>, Ix2>,
    iter: Option<usize>,
    beta: Option<f32>,
    result: Option<ArrayBase<OwnedRepr<f32>, Ix2>>,
}

impl CAMLP {
    pub fn new(graph: ArrayBase<OwnedRepr<f32>, Ix2>) -> Self {
        CAMLP {
            graph: graph,
            result: None,
            beta: None,
            iter: None,
        }
    }

    /// Set the parameter for the iteration count
    pub fn iter(mut self, num: usize) -> Self {
        self.iter = Some(num);
        self
    }

    /// Set the parameter for the beta
    pub fn beta(mut self, num: f32) -> Self {
        self.beta = Some(num);
        self
    }

    /// Fit the model to the data.
    /// x: index of node for label
    /// y: label index
    pub fn fit(
        &mut self,
        x: &ArrayBase<OwnedRepr<usize>, Ix1>,
        y: &ArrayBase<OwnedRepr<usize>, Ix1>,
    ) -> Result<(), Box<dyn Error>> {
        let c = *y.max()? + 1;
        let s = self.graph.shape()[0];
        let d = self.graph.sum_axis(Axis(1));

        let z = Array::from_diag(&(1.0 / (1.0 + d * self.beta.unwrap_or(0.1))));

        let p = z.dot(&(0.1 * &self.graph));
        let h = Array::eye(c);
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

        for _ in 0..self.iter.unwrap_or(30) {
            f = p.dot(&f).dot(&h) + &b;
        }

        self.result = Some(f);
        Ok(())
    }

    /// predict label score.
    /// target_node: node index
    pub fn predict_proba(
        &mut self,
        target_node: &ArrayBase<OwnedRepr<usize>, Ix1>,
    ) -> ArrayBase<OwnedRepr<f32>, Ix2> {
        let mut result = Array::zeros((
            target_node.shape()[0],
            self.result.as_mut().unwrap().shape()[1],
        ));
        for (i, n) in target_node.iter().enumerate() {
            result
                .slice_mut(s![i, ..])
                .assign(&self.result.as_mut().unwrap().slice_mut(s![*n, ..]));
        }
        result
    }
}
