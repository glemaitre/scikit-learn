from libc.math cimport exp, lgamma, log
import cython
cimport numpy as cnp

import numpy as np

cnp.import_array()

cpdef expected_mutual_information(contingency, cnp.int64_t n_samples):
    """Compute the expected mutual information.

    Parameters
    ----------
    contingency : sparse-matrix of shape (n_labels, n_labels), dtype=np.int64
        Contengency matrix storing the label counts. This is as sparse
        matrix at the CSR format.

    n_samples : int
        The number of samples which is the sum of the counts in the contengency
        matrix.

    Returns
    -------
    emi : float
        The expected mutual information between the two vectors of labels.
    """
    cdef:
        unsigned long long n_rows, n_cols
        cnp.ndarray[cnp.int64_t] a, b
        cnp.int64_t a_i, b_j
        cnp.intp_t i, j, nij, nij_nz
        cnp.intp_t start, end
        cnp.float64_t emi = 0.0
        double term1, term2, term3
        double log_n_samples = log(n_samples)

    n_rows, n_cols = contingency.shape
    a = np.ravel(contingency.sum(axis=1))
    b = np.ravel(contingency.sum(axis=0))

    if a.size == 1 or b.size == 1:
        return 0.0

    for i in range(n_rows):
        for j in range(n_cols):
            a_i, b_j = a[i], b[j]
            start = max(1, a_i - n_samples + b_j)
            end = min(a_i, b_j) + 1
            for nij in range(start, end):
                nij_nz = 1 if nij == 0 else nij
                term1 = nij_nz / <double>n_samples
                term2 = log_n_samples + log(nij_nz) - log(a_i) - log(b_j)
                term3 = exp(
                    lgamma(a_i + 1) + lgamma(b_j + 1)
                    + lgamma(n_samples - a_i + 1) + lgamma(n_samples - b_j + 1)
                    - lgamma(n_samples + 1) - lgamma(nij_nz + 1)
                    - lgamma(a_i - nij + 1)
                    - lgamma(b_j - nij + 1)
                    - lgamma(n_samples - a_i - b_j + nij + 1)
                )
                emi += (term1 * term2 * term3)
    return emi
