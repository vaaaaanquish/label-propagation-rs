# label-propagation-rs

Label Propagation Algorithm by Rust.

Label propagation (LP) is graph-based semi-supervised learning (SSL).

A simple LGC and a more advanced CAMLP have been implemented.


# Usage

You can find the examples in the examples directory.

The label is a continuous value of `[0, class_n]`, and the result of `predict_proba` is the value of the label.

```rust
use std::result::Result;
use std::error::Error;

extern crate label_propagation;
extern crate ndarray;
extern crate ndarray_stats;

use ndarray::prelude::*;
use label_propagation::{CAMLP, LGC};
use ndarray::Array;

pub fn main() -> Result<(), Box<dyn Error>> {
    // make graph matrix ndarray
    let graph = Array::from_shape_vec(
        (3, 3),
        vec![0.0, 0.3, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();

    // node index for label
    let x = array![0, 1];
    // label
    let y = array![0, 1];

    // make model
    let mut model = CAMLP::new(graph).iter(2).beta(0.1);
    // let mut model = LGC::new(graph).iter(2).alpha(0.99);

    model.fit(&x, &y)?;

    let target = array![0, 1];
    let result = model.predict_proba(&target);
    println!("{:?}", result);

    Ok(())
}
```


# develop

```sh
docker build -t graph .
docker run -it -v $PWD:/app graph bash
```

# Thanks

- Local and Global Consistency (LGC) [Zhou+, NIPS'04] https://dennyzhou.github.io/papers/LLGC.pdf
- Core model in Confidence-Aware Modulated Label Propagation (CAMLP) [Yamaguchi+, SDM'16] https://epubs.siam.org/doi/pdf/10.1137/1.9781611974348.58
- [yamaguchiyuto/label_propagation](https://github.com/yamaguchiyuto/label_propagation) - Implementations of label propagation like algorithms, python
