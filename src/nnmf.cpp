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
arma::mat nnmf_update_loadings_rcpp (const arma::mat& A, const arma::mat& W,
				     const arma::mat& H, uint inner_max_iter) {
  arma::mat Wnew = W;
  update(Wnew,H,A,inner_max_iter);
  return Wnew;
}

// [[Rcpp::export]]
arma::mat nnmf_update_factors_rcpp (const arma::mat& A, const arma::mat& W,
				    const arma::mat& H, uint inner_max_iter) {
  arma::mat Hnew = H;
  update(Hnew,W,A,inner_max_iter);
  return Hnew;
}
