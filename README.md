# iSDR_p (iterative source and dynamics reconstruction)
A solver of EEG/MEG inverse problem using a multivariate auto-regressive model (MVAR) on the source space


iSDR_p is a c++ package for solving the EEG/MEG inverse problem using structural/functional prior 
on the causality between brain regions/sources.
iSDR_p solve the following functional:

\sum_t ||M_t-G \sum_i A_i J_{t-i}||_2^2+\alpha ||J||_{21}.

Where: 
     A_i: i=1,..,p are the matrices of the MVAR model of order p.

     M_t: EEG or/and MEG measurements at time t.

     G: Lead field matrix which project brain activity into sensor space.

     J_{t-i}: Brain activity (distributed source model with fixed position) at time t-i.

     alpha (>0): the regularization parameter.
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

The first argument of ./iSDR_p is a .mat file containing the following:
    
    M: measurements.
 
    G: Lead field matrix.

    GA: lead field matrix times initial values of A_i's, i=1,..,p i.e. GA = [GA1,..,GAp].

    SC: structural connectivity matrix (symmetric).

    n_c: size of sensor space.

    n_t: number of measurements samples.

    n_s: size of source space.

    m_p: order of MVAR.

# Example
An example of iSDR can be obtained by the running the following:
./iSDR/iSDR_p simulated_data_p2.mat 0.001 1e-4 iSDR_results_p2.mat

The results are saved in iSDR_results_p2.mat

