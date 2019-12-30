nnmf <- function(
	A, k = 1L, alpha = rep(0,3), beta = rep(0,3), method = c('scd', 'lee'),
	loss = c('mse', 'mkl'), init = NULL, mask = NULL, W.norm = -1L, check.k = TRUE,
	max.iter = 500L, rel.tol = 1e-4, n.threads = 1L, trace = 100/inner.max.iter,
	verbose = 1L, show.warning = TRUE, inner.max.iter = ifelse('mse' == loss, 50L, 1L),
	inner.rel.tol = 1e-9
	) {
	method <- match.arg(method);
	loss <- match.arg(loss);
	check.matrix(A, input.name = 'A');
	if (!is.double(A))
		storage.mode(A) <- 'double';
	n <- nrow(A);
	m <- ncol(A);

	init.mask <- reformat.input(init, mask, n, m, k);
	k <- init.mask$K;

	alpha <- c(as.double(alpha), rep(0., 3))[1:3];
	beta <- c(as.double(beta), rep(0., 3))[1:3];
	method.code <- get.method.code(method, loss);

	min.k <- min(dim(A));
	A.isNA <- is.na(A);
	A.anyNA <- any(A.isNA); # anyNA is depreciated in new version of R
	if (A.anyNA) {
		min.k <- min(min.k, ncol(A) - rowSums(A.isNA), nrow(A) - colSums(A.isNA));
		}
	rm(A.isNA);
	if (check.k && k > min.k && all(c(alpha, beta) == 0))
		stop(paste("k larger than", min.k, "is not recommended, unless properly masked or regularized.
				Set check.k = FALSE if you want to skip this checking."));

	if (n.threads < 0L) n.threads <- 0L; # let openMP decide
	if (is.logical(verbose)) {
		verbose <- as.integer(verbose);
		}
	if (trace <= 0) {
		trace <- 999999L; # only compute error of the 1st and last iteration
		}

	## run.time <- system.time(
	## 	out <- .Call('NNLM_nnmf', A, as.integer(k),
	## 		init.mask$Wi, init.mask$Hi, init.mask$Wm, init.mask$Hm,
	## 		alpha, beta, as.integer(max.iter), as.double(rel.tol),
	## 		as.integer(n.threads), as.integer(verbose), as.logical(show.warning),
	## 		as.integer(inner.max.iter), as.double(inner.rel.tol), as.integer(method.code),
	## 		as.integer(trace), PACKAGE = 'NNLM'
	## 		)
	## 	);
	names(out) <- c('W', 'H', 'mse', 'mkl', 'target.loss', 'average.epochs', 'n.iteration');
	out$mse <- as.vector(out$mse);
	out$mkl <- as.vector(out$mkl);
	out$target.loss <- as.vector(out$target.loss);
	out$average.epochs <- as.vector(out$average.epochs);

	# add row/col names back
	colnames(out$W) <- colnames(init.mask$Wi);
	rownames(out$H) <- rownames(init.mask$Hi);
	if (!is.null(rownames(A))) rownames(out$W) <- rownames(A);
	if (!is.null(colnames(A))) colnames(out$H) <- colnames(A);
	rm(init.mask);

	if (W.norm > 0) {
		if (is.finite(W.norm)) {
			W.scale <- sapply(out$W, function(x) sum(x^W.norm)^(1./W.norm));
		} else {
			W.scale <- sapply(out$W, max);
			}
		out$W <- out$W %*% diag(1./W.scale);
		out$H <- diag(W.scale) %*% out$H
		}

	out$run.time <- run.time;
	out$options <- list(
		method = method,
		loss = loss,
		alpha = alpha,
		beta = beta,
		init = init,
		mask = mask,
		n.threads = n.threads,
		trace = trace,
		verbose = verbose,
		max.iter = max.iter,
		rel.tol = rel.tol,
		inner.max.iter = inner.max.iter,
		inner.rel.tol = inner.rel.tol
		);
	out$call <- match.call();
	return(structure(out, class = 'nnmf'));
	}
