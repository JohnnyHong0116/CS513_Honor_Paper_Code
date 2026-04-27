### For dependencies run the following

``` bash
pip install numpy matplotlib scipy scikit-image pillow opencv-contrib-python
```

`experiment_coding.py`: Main file to run the experiment

`sweep_k.py`: Test for best k value for TSVD for camera, aruco tags, and text images

`sweep_mu.py`: Test for best mu value for Tikhonov regularization for camera, aruco tags, and text images

### Best parameter setting for TSVD and Tikhonov
Running `sweep_k.py` and `sweep_mu.py` provide following data
| Image Type | Best TSVD \(k\) | Best TSVD Relative Error | Best Tikhonov \(\mu\) | Best Tikhonov Relative Error |
|---|---:|---:|---:|---:|
| ArUco | 3300 | 0.05682378 | 0.05 | 0.05457128 |
| Text | 3200 | 0.04249999 | 0.05 | 0.03863419 |
| Camera | 2100 | 0.05582469 | 0.10 | 0.05037506 |