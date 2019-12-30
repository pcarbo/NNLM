#include "nnlm.h"
  
// Non-negative Matrix Factorization(NNMF) using alternating scheme
// Decompose matrix A such that A = W*H.
//
// Arguments:
//
//   A: Matrix to be decomposed
//
//   W, H: Initial matrices of W and H, where ncol(W) = nrow(H) = k. #
//   of rows/columns of W/H could be 0
//
//   inner_max_iter: Maximum number of iterations passed to each inner W
//   or H matrix updating loop
//
// Return:
//   A list (Rcpp::List) of 
//   W, H: resulting W and H matrices
//

// [[Rcpp::export]]
Rcpp::List nnmf_rcpp (const arma::mat& A, arma::mat& W, arma::mat& H, 
		      const unsigned int inner_max_iter) {

  inplace_trans(W);

  // update W
  update(W,H,A.t(),inner_max_iter);

  // update H
  update(H,W,A,inner_max_iter);

  return Rcpp::List::create(Rcpp::Named("W") = W.t(),
			    Rcpp::Named("H") = H);
}
