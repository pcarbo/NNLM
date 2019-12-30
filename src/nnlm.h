#define ARMA_NO_DEBUG

//[[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <R.h>

void update (arma::mat& H, const arma::mat& Wt, const arma::mat& A,
	     uint max_iter);

void scd_kl_update (arma::subview_col<double> Hj, const arma::mat& Wt,
		    const arma::vec& Aj, const arma::vec& sumW,
		    uint max_iter);
