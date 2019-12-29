//disable armadillo matrix/vector boundary checking
#define ARMA_NO_DEBUG

#ifdef _OPENMP
#include <omp.h>
#endif

//[[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

//[[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>

#include <Rcpp.h>
#include <R.h>

#define TINY_NUM 1e-15
#define NNLM_REL_TOL 1e-8
#define NNMF_REL_TOL 1e-6
#define MAX_ITER 500
#define NNMF_INNER_MAX_ITER 10
#define N_THREADS 1
#define TRACE_STEP 10
#define SHOW_WARNING true
#define DEFAULT_METHOD 1

using namespace arma;

Rcpp::List nnlm(const mat & x, const mat & y, const vec & alpha, const umat & mat, unsigned int max_iter = MAX_ITER,
	double rel_tol = NNLM_REL_TOL, int n_threads = N_THREADS, int method = DEFAULT_METHOD);

Rcpp::List nnmf (const mat & A, const unsigned int k, mat W, mat H, umat Wm,
		 const umat & Hm, const vec & alpha, const vec & beta,
		 unsigned int max_iter = MAX_ITER,
		 double rel_tol = NNMF_REL_TOL, int n_threads = N_THREADS,
		 const int verbose = 1, bool show_warning = SHOW_WARNING,
		 unsigned int inner_max_iter = NNMF_INNER_MAX_ITER,
		 double inner_rel_tol = NNLM_REL_TOL,
		 int method = DEFAULT_METHOD,
		 unsigned int trace = TRACE_STEP);

void update (arma::mat& H, const arma::mat& Wt, const arma::mat& A,
	     const arma::vec& beta, uint max_iter = NNMF_INNER_MAX_ITER);

void scd_kl_update (arma::subview_col<double> Hj, const arma::mat& Wt,
		    const arma::vec& Aj, const arma::vec& sumW,
		    const uint& max_iter);
