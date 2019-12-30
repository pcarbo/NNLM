#' @useDynLib NNLM
#' 
#' @export
#' 
nnmf_update_loadings <- function (A, W, H, inner_max_iter)
  t(nnmf_update_loadings_rcpp(t(A),t(W),H,inner_max_iter))

#' @export
nnmf_update_factors <- function (A, W, H, inner_max_iter)
  nnmf_update_factors_rcpp(A,t(W),H,inner_max_iter)
