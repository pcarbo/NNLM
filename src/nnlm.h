#define ARMA_NO_DEBUG

//[[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <R.h>

#define TINY_NUM 1e-15

void update (arma::mat& H, const arma::mat& Wt, const arma::mat& A,
	     uint max_iter);

void scd_kl_update (arma::subview_col<double> Hj, const arma::mat& Wt,
		    const arma::vec& Aj, const arma::vec& sumW,
		    const uint& max_iter);
