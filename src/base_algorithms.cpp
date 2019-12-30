#include "nnlm.h"

#define maximum(a,b) ((a) > (b) ? (a) : (b))

using namespace arma;

// Problem: Aj = W * Hj
// Wt = W^T
// sumW = column sum of W
void scd_kl_update (subview_col<double> Hj, const mat& Wt, const vec& Aj,
		    const vec& sumW, uint max_iter) {
  double e   = 1e-15;
  vec    Ajt = Wt.t()*Hj;
  vec    mu;
  double a;
  double b;
  double x;

  for (uint t = 0; t < max_iter; t++)
    for (uint k = 0; k < Wt.n_rows; k++) {
      mu = Wt.row(k).t()/(Ajt + e);
      a  = dot(Aj,square(mu));
      b  = dot(Aj,mu) - sumW(k) + a*Hj(k);
      x  = b/(a + e);
      x  = maximum(x,0);
      if (x != Hj(k)) {
	Ajt   += (x - Hj(k)) * Wt.row(k).t();
	Hj(k)  = x;
      }
    }
}
