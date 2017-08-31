# iSDR_p (iterative source and dynamics reconstruction)
A solver of EEG/MEG inverse problem using a multivariate auto-regressive model on the source space


iSDR_p is a c++ package for solving the EEG/MEG inverse problem using structural/functional prior 
on the causality between brain regions/sources.

# Requirements
1-MATIO

2-HDF5

3-flens

4-MKL intel

5- BLAS

6- LAPACK


# Cite

(1) Brahim Belaoucha, Théodore Papadopoulo. Large brain effective network
from EEG/MEG data and dMR information. PRNI 2017 – 7th International
Workshop on Pattern Recognition in NeuroImaging, Jun 2017, Toronto, Canada. 

(2) Brahim Belaoucha, Mouloud Kachouane, Théodore Papadopoulo. Multivariate
Autoregressive Model Constrained by Anatomical Connectivity to Reconstruct
Focal Sources. 2016 38th Annual International Conference of the IEEE
Engineering in Medicine and Biology Society (EMBC), Aug 2016, Orlando,
United States. 2016.


# Author

Belaoucha Brahim 

# Installation
first modify the makefile to include the different folders of the required packages

make 

# Details
More details about the package input parameters can be obtained by:

./iSDR_p --help (-h).

# Example
An example of iSDR can be obtained by the running the following:
./iSDR/iSDR_p simulated_data_p2.mat 0.001 1e-4 iSDR_results_p2.mat

The results are saved in iSDR_results_p2.mat

