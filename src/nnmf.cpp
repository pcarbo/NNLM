#include "nnlm.h"

double cost (const mat& A, const mat& Ahat, double e);
  
//[[Rcpp::export]]
Rcpp::List nnmf(const mat & A, const unsigned int k, mat W, mat H, umat Wm, umat Hm,
	const vec & alpha, const vec & beta, const unsigned int max_iter, const double rel_tol, 
	const int n_threads, const int verbose, const bool show_warning, const unsigned int inner_max_iter, 
	const double inner_rel_tol, const int method, unsigned int trace)
{
	/******************************************************************************************************
	 *              Non-negative Matrix Factorization(NNMF) using alternating scheme
	 *              ----------------------------------------------------------------
	 * Description:
	 * 	Decompose matrix A such that
	 * 		A = W H
	 * Arguments:
	 * 	A              : Matrix to be decomposed
	 * 	W, H           : Initial matrices of W and H, where ncol(W) = nrow(H) = k. # of rows/columns of W/H could be 0
	 * 	Wm, Hm         : Masks of W and H, s.t. masked entries are no-updated and fixed to initial values
	 * 	alpha          : [L2, angle, L1] regularization on W (non-masked entries)
	 * 	beta           : [L2, angle, L1] regularization on H (non-masked entries)
	 * 	max_iter       : Maximum number of iteration
	 * 	rel_tol        : Relative tolerance between two successive iterations, = |e2-e1|/avg(e1, e2)
	 * 	n_threads      : Number of threads (openMP)
	 * 	verbose        : Either 0 = no any tracking, 1 == progression bar, 2 == print iteration info
	 * 	show_warning   : If to show warning if targeted `tol` is not reached
	 * 	inner_max_iter : Maximum number of iterations passed to each inner W or H matrix updating loop
	 * 	inner_rel_tol  : Relative tolerance passed to inner W or H matrix updating loop, = |e2-e1|/avg(e1, e2)
	 * 	method         : Integer of 1, 2, 3 or 4, which encodes methods
	 * 	               : 1 = sequential coordinate-wise minimization using square loss
	 * 	               : 2 = Lee's multiplicative update with square loss, which is re-scaled gradient descent
	 * 	               : 3 = sequentially quadratic approximated minimization with KL-divergence
	 * 	               : 4 = Lee's multiplicative update with KL-divergence, which is re-scaled gradient descent
	 * 	trace          : A positive integer, error will be checked very 'trace' iterations. Computing WH can be very expansive,
	 * 	               : so one may not want to check error A-WH every single iteration
	 * Return:
	 * 	A list (Rcpp::List) of 
	 * 		W, H          : resulting W and H matrices
	 * 		mse_error     : a vector of mean square error (divided by number of non-missings)
	 * 		mkl_error     : a vector (length = number of iterations) of mean KL-distance
	 * 		target_error  : a vector of loss (0.5*mse or mkl), plus constraints
	 * 		average_epoch : a vector of average epochs (one complete swap over W and H)
	 * Author:
	 * 	Eric Xihui Lin <xihuil.silence@gmail.com>
	 * Version:
	 * 	2015-12-11
	 ******************************************************************************************************/

	unsigned int n = A.n_rows;
	unsigned int m = A.n_cols;
	//int k = H.n_rows; // decomposition rank k
	unsigned int N_non_missing = n*m;

	if (trace < 1) trace = 1;
	unsigned int err_len = (unsigned int)std::ceil(double(max_iter)/double(trace)) + 1;
	vec mse_err(err_len), mkl_err(err_len), terr(err_len), ave_epoch(err_len);

	// check progression
	bool show_progress = false;
	if (verbose == 1) show_progress = true;
	Progress prgrss(max_iter, show_progress);

	double rel_err = rel_tol + 1;
	double terr_last = 1e99;
	uvec non_missing;
	bool any_missing = !A.is_finite();
	if (any_missing) 
	{
	  non_missing = find_finite(A);
	  N_non_missing = non_missing.n_elem;
	  mkl_err.fill(0);
	}
	else
	  mkl_err.fill(0);

	if (Wm.empty())
		Wm.resize(0, n);
	else
		inplace_trans(Wm);
	if (Hm.empty())
		Hm.resize(0, m);

	if (W.empty())
	{
		W.randu(k, n);
		W *= 0.01;
		if (!Wm.empty())
			W.elem(find(Wm > 0)).fill(0.0);
	}
	else
		inplace_trans(W);

	if (H.empty())
	{
		H.randu(k, m);
		H *= 0.01;
		if (!Hm.empty())
			H.elem(find(Hm > 0)).fill(0.0);
	}

	if (verbose == 2)
	{
		Rprintf("\n%10s | %10s | %10s | %10s | %10s\n", "Iteration", "MSE", "MKL", "Target", "Rel. Err.");
		Rprintf("--------------------------------------------------------------\n");
	}

	int total_raw_iter = 0;
	unsigned int i = 0;
	unsigned int i_e = 0; // index for error checking
	for(; i < max_iter && std::abs(rel_err) > rel_tol; i++) 
	{
		Rcpp::checkUserInterrupt();
		prgrss.increment();

		if (any_missing)
		{
			// update W
			total_raw_iter += update_with_missing(W, H, A.t(), Wm, alpha, inner_max_iter, inner_rel_tol, n_threads, method);
			// update H
			total_raw_iter += update_with_missing(H, W, A, Hm, beta, inner_max_iter, inner_rel_tol, n_threads, method);
		}
		else
		{
			// update W
			total_raw_iter += update(W, H, A.t(), Wm, alpha, inner_max_iter, inner_rel_tol, n_threads, method);
			// update H
			total_raw_iter += update(H, W, A, Hm, beta, inner_max_iter, inner_rel_tol, n_threads, method);

			if (i % trace == 0) {
			  const mat & Ahat = W.t()*H;
			  mse_err(i_e) = mean(mean(square((A - Ahat))));
			  mkl_err(i_e) = cost(A,Ahat,TINY_NUM);
			}
		}

		if (i % trace == 0)
		{
			ave_epoch(i_e) = double(total_raw_iter)/(n+m);
			if (method < 3) // mse based
				terr(i_e) = 0.5*mse_err(i_e);
			else // KL based
				terr(i_e) = mkl_err(i_e);

			add_penalty(i_e, terr, W, H, N_non_missing, alpha, beta);

			rel_err = 2*(terr_last - terr(i_e)) / (terr_last + terr(i_e) + TINY_NUM );
			terr_last = terr(i_e);
			if (verbose == 2)
				Rprintf("%10d | %10.4f | %10.4f | %10.4f | %10.g\n", i+1, mse_err(i_e), mkl_err(i_e), terr(i_e), rel_err);

			total_raw_iter = 0; // reset to 0
			++i_e;
		}
	}

	// compute error of the last iteration
	if ((i-1) % trace != 0)
	{
	  const mat & Ahat = W.t()*H;
	  mse_err(i_e) = mean(mean(square((A - Ahat))));
	  mkl_err(i_e) = cost(A,Ahat,TINY_NUM);

		ave_epoch(i_e) = double(total_raw_iter)/(n+m);
		if (method < 3) // mse based
			terr(i_e) = 0.5*mse_err(i_e);
		else // KL based
			terr(i_e) = mkl_err(i_e);
		add_penalty(i_e, terr, W, H, N_non_missing, alpha, beta);

		rel_err = 2*(terr_last - terr(i_e)) / (terr_last + terr(i_e) + TINY_NUM );
		terr_last = terr(i_e);
		if (verbose == 2)
			Rprintf("%10d | %10.4f | %10.4f | %10.4f | %10.g\n", i+1, mse_err(i_e), mkl_err(i_e), terr(i_e), rel_err);

		++i_e;
	}

	if (verbose == 2)
	{
		Rprintf("--------------------------------------------------------------\n");
		Rprintf("%10s | %10s | %10s | %10s | %10s\n\n", "Iteration", "MSE", "MKL", "Target", "Rel. Err.");
	}

	if (i_e < err_len)
	{
		mse_err.resize(i_e);
		mkl_err.resize(i_e);
		terr.resize(i_e);
		ave_epoch.resize(i_e);
	}

	if (show_warning && rel_err > rel_tol)
		Rcpp::warning("Target tolerance not reached. Try a larger max.iter.");

	return Rcpp::List::create(
		Rcpp::Named("W") = W.t(),
		Rcpp::Named("H") = H,
		Rcpp::Named("mse_error") = mse_err,
		Rcpp::Named("mkl_error") = mkl_err,
		Rcpp::Named("target_error") = terr,
		Rcpp::Named("average_epoch") = ave_epoch,
		Rcpp::Named("n_iteration") = i
		);
}

// Compute the value of the cost function for non-negative matrix
// factorization, in which matrix A is approximated by the matrix
// product W*H. Ths is equivalent to the negative Poisson
// log-likelihood after removing terms that do not depend on W or H.
double cost (const mat& A, const mat& Ahat, double e) {
  uint   m = A.n_cols;
  double f = 0;
  
  // Repeat for each column of A.
  for (uint j = 0; j < m; j++)
    f += sum(Ahat.col(j)) - sum(A.col(j) % log(Ahat.col(j) + e));
  
  return f;
}

// add_penalty to the target error 'terr'
void add_penalty(const unsigned int & i_e, vec & terr, const mat & W, const mat & H,
	const unsigned int & N_non_missing, const vec & alpha, const vec & beta)
{
	// add penalty term back to the loss function (terr)
	if (alpha(0) != alpha(1))
		terr(i_e) += 0.5*(alpha(0)-alpha(1))*accu(square(W))/N_non_missing;
	if (beta(0) != beta(1))
		terr(i_e) += 0.5*(beta(0)-beta(1))*accu(square(H))/N_non_missing;
	if (alpha(1) != 0)
		terr(i_e) += 0.5*alpha(1)*accu(W*W.t())/N_non_missing;
	if (beta(1) != 0)
		terr(i_e) += 0.5*beta(1)*accu(H*H.t())/N_non_missing;
	if (alpha(2) != 0)
		terr(i_e) += alpha(2)*accu(W)/N_non_missing;
	if (beta(2) != 0)
		terr(i_e) += beta(2)*accu(H)/N_non_missing;
}
