use std::error::Error;
use std::result::Result;

use ndarray::prelude::*;
use ndarray::{Array, ArrayBase, Axis, OwnedRepr};
use ndarray_stats::QuantileExt;

/// Local and Global Consistency (LGC) [Zhou+, NIPS'04]
/// https://dennyzhou.github.io/papers/LLGC.pdf
pub struct LGC {
    graph: ArrayBase<OwnedRepr<f32>, Ix2>,
    iter: Option<usize>,
    alpha: Option<f32>,
    result: Option<ArrayBase<OwnedRepr<f32>, Ix2>>,
}

impl LGC {
    pub fn new(graph: ArrayBase<OwnedRepr<f32>, Ix2>) -> Self {
        LGC {
            graph: graph,
            result: None,
            iter: None,
            alpha: None,
        }
    }

    /// Set the parameter for the iteration count
    pub fn iter(mut self, num: usize) -> Self {
        self.iter = Some(num);
        self
    }

    /// Set the parameter for the alpha
    pub fn alpha(mut self, num: f32) -> Self {
        self.alpha = Some(num);
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
        let c = *x.max()? + 1;
        let s = self.graph.shape()[0];
        let d = self.graph.sum_axis(Axis(0));

        // Avoid division by 0
        let d_t = Array::from_iter(
            d.iter()
                .filter_map(|&b| if b == 0. { Some(1.) } else { Some(b) }),
        );

        let d_diag = Array::from_diag(&(1.0 / d_t));
        let d_sqrt = d_diag.mapv(f32::sqrt);
        let p = self.alpha.unwrap_or(0.99) * d_sqrt.dot(&self.graph).dot(&d_sqrt);

        let mut b = Array::zeros((s, c)) / (c as f32);

        // numpy -> b[[0,1], [0,1]] = 1.
        for (x_i, y_i) in x.iter().zip(y.iter()) {
            b.slice_mut(s![*x_i, *y_i]).fill(1.);
        }

        let mut f = (1. - self.alpha.unwrap_or(0.99)) * &b;

        for _ in 0..self.iter.unwrap_or(30) {
            f = p.dot(&f) + &b;
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
        for i in target_node {
            result
                .slice_mut(s![*i, ..])
                .assign(&self.result.as_mut().unwrap().slice_mut(s![*i, ..]));
        }
        result
    }
}
