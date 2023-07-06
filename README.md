Code for "GeoPhy: Differentiable Phylogenetic Inference via Geometric Gradients of Tree Topologies"
====

Setup
==

The code was tested with Python 3.9.5.
```
pip install -r requirements.txt
```

To download datasets and build libraries, run the following command:
```
$ make setup
```

Running an example
==

```
$ make example
```

Running benchmarks
==
```
$ make bench_init_all
$ make bench_model_cv__all
$ make bench_ds_all
```
