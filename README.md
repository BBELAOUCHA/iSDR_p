# iSDR_p (iterative source and dynamics reconstruction)
A solver of EEG/MEG inverse problem using a multivariate auto-regressive model (MVAR) on the source space


iSDR_p is a c++ package for solving the EEG/MEG inverse problem using structural/functional prior 
on the causality between brain regions/sources.
iSDR_p solve the following functional:

<img src="http://latex.codecogs.com/gif.latex?\Large&space;U(J)&space;=&space;\sum_{t=p}^T||M_t-G\sum_{i=1}^pA_iJ_{t-i}||_{2}^2&plus;\alpha&space;||J||_{21}" title="\Large U(J) = \sum_{t=p}^T||M_t-G\sum_{i=1}^pA_iJ_{t-i}||_{2}^2+\alpha ||J||_{21}" />

to obtain the brain activity with an initial MAR model A_i's. Then, brain regions/sources interactions are obtained by optimizing the following cost function:

<a href="http://www.codecogs.com/eqnedit.php?latex=\Large&space;L(A)&space;=&space;\sum_{t=1}^{T-1}||J_t-\sum_{i=1}^pA_iJ_{t-i}||_{2}^2" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\Large&space;L(A)&space;=&space;\sum_{t=1}^{T-1}||J_t-\sum_{i=1}^pA_iJ_{t-i}||_{2}^2" title="\Large L(A) = \sum_{t=1}^{T-1}||J_t-\sum_{i=1}^pA_iJ_{t-i}||_{2}^2" /></a>


Where: 

     * A_i: i=1,..,p are the matrices of the MVAR model of order p.

     * M_t: EEG or/and MEG measurements at time t.

     * G: Lead field matrix which project brain activity into sensor space.

     * J_{t-i}: Brain activity (distributed source model with fixed position) at time t-i.

     * alpha ]0,100[: percentage of the maximum alpha_max (reg parameter which results in zero brain activity).
# Requirements
1-MATIO > 1.5.2

2-HDF5

3-FLENS

4-MKL intel


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

Theodore Papadopoulo

# Installation
first modify the makefile to include the different folders of the required packages. Then compile the package:


     make 

# Details
More details about the package input parameters can be obtained by:

iSDR_p --help (-h).

The first argument of ./iSDR_p is a .mat file containing the following:
    
    * M: measurements.
 
    * G: Lead field matrix.

    * GA: lead field matrix times initial values of A_i's, i=1,..,p i.e. GA = [GA1,..,GAp].

    * SC: structural connectivity matrix (symmetric).

    * n_c: size of sensor space.

    * n_t: number of measurements samples.

    * n_s: size of source space.

    * m_p: order of MVAR.

# Example
An example of iSDR can be seeing in the examples folder.

# Test
To test the iSDR_p package, you can run the test module:

test_MxNE_iSDR

