#include "nnlm.h"

using namespace arma;

// Problem:  Aj = W * Hj
// Wt = W^T
// sumW = column sum of W
// beta: a vector of 3, for L2, angle, L1 regularization
void scd_kl_update (subview_col<double> Hj, const mat& Wt, const vec& Aj,
		    const vec& sumW, const uint& max_iter) {

  double sumHj = sum(Hj);
  vec    Ajt   = Wt.t()*Hj;
  vec    mu;
  double a;
  double b;
  double tmp;

  for (uint t = 0; t < max_iter; t++) {
    for (uint k = 0; k < Wt.n_rows; k++) {
      mu  = Wt.row(k).t()/(Ajt + TINY_NUM);
      a   = dot(Aj, square(mu));
      b   = dot(Aj, mu) - sumW(k);
      b  += a*Hj(k);
      tmp = b/(a+TINY_NUM); 
      if (tmp < 0) tmp = 0;
      if (tmp != Hj(k)) {
	Ajt   += (tmp - Hj(k)) * Wt.row(k).t();
	sumHj += tmp - Hj(k);
	Hj(k)  = tmp;
      }
    }
  }
}
