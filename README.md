# CNSF Factorization of dMRI data

Python implementation of convexity-constrained and nonnegativity-constrained spherical factorization.

## Method & use

This method will factorize multi-shell dMRI data into multiple (tissue) components, each represented by a response function and an orientation distribution function (ODF). The method jointly estimates the response functions and ODFs for any desired number of components, and is therefore fully unsupervised.

All input and output data is stored in the [MRtrix](www.mrtrix.org) `.mif` file format.

Typical use for a 3-component factorization with one anisotropic (Lmax=8) and two isotropic (Lmax=0) components:
```
dwifact -lmax 8,0,0 dmri.mif brainmask.mif output_directory 
```

## Please cite

Christiaens, D., Sunaert, S., Suetens, P., Maes, F. (2017). Convexity-constrained and nonnegativity-constrained spherical factorization in diffusion-weighted imaging. NeuroImage, 146, 507-517. doi: 10.1016/j.neuroimage.2016.10.040


