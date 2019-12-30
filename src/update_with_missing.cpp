#include "nnlm.h"

using namespace arma;

// A = W H, solve H
// No missing in A, Wt = W^T
void update (mat& H, const mat& Wt, const mat& A, uint max_iter) {
  uint m    = A.n_cols;
  vec  sumW = sum(Wt,1);

  // By columns of H.
  for (uint j = 0; j < m; j++) 
    scd_kl_update(H.col(j),Wt,A.col(j),sumW,max_iter);
}
