#include <RcppArmadillo.h>
//#include <bits/stdc++.h>
#include <cmath>
#include <limits>
#include <time.h>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

//[[Rcpp::export]]
vec nnlsCpp(const mat &A, const vec &b, int max_iter = 500, double tol = 1e-6, bool verbose = false)
{
  /*
  * Description: sequential Coordinate-wise algorithm for non-negative least square regression A x = b, s.t. x >= 0
  * 	Reference: http://cmp.felk.cvut.cz/ftp/articles/franc/Franc-TR-2005-06.pdf 
  */
  
  arma::vec mu            = -A.t() * b;
  arma::mat H             = A.t() * A;
  arma::vec x(A.n_cols);
  arma::vec x0(A.n_cols);
  x.fill(0);
  x0.fill(-9999);
  
  int       i     = 0;
  double    tmp;
  while(i < max_iter && max(abs(x - x0)) > tol){
    x0 = x;
    for (uword k = 0; k < A.n_cols; k++){
      tmp = x[k] - mu[k] / H.at(k,k);
      if (tmp < 0) tmp = 0;
      if (tmp != x[k]) mu += (tmp - x[k]) * H.col(k);
      x[k] = tmp;
    }
    ++i;
  }
  return x;
}

arma::cube myreshape(arma::mat A, arma::uword n, arma::uword q, arma::uword r){
  arma::cube X         = zeros(r, q, n);
  arma::uvec idq       = regspace<uvec>(0, q - 1);
  for(uword i = 0; i < n; i++){
    X.slice(i) = A.cols(idq);
    idq        = idq + ones<uvec>(q, 1)*q;
  }
  return(X);
}

//[[Rcpp::export]]
arma::vec FCLS(arma::mat M, arma::vec y, arma::mat Sigma, arma::uword p, arma::uword q){
  arma::mat Qq;
  arma::mat Rr;
  arma::mat Q;
  arma::mat R;
  arma::mat Rinv;
  qr(Qq, Rr, Sigma*M);
  R                     = Rr.head_rows(q*p);
  Q                     = Qq.head_cols(q*p);
  Rinv                  = pinv(R);
  arma::vec   c         = Q.t()*Sigma*y;
  arma::mat   G         = join_vert(eye(p*q, p*q), ones(1, p*q))*Rinv;
  arma::vec   g         = join_vert(zeros(p*q, 1), ones(1, 1)) - join_vert(eye(p*q, p*q), ones(1, p*q))*Rinv*c;
  arma::mat   E         = join_vert(G.t(), g.t());
  arma::vec   f         = join_vert(zeros(p*q), ones(1, 1));
  arma::vec   u         = nnlsCpp(E, f);
  arma::vec   r         = E*u - f;
  arma::vec   z;        
  if(r(p*q) == 0){
    z         = -r.head(p*q)/(-1e-16);
  } else {
    z         = -r.head(p*q)/r(p*q);
  }
  arma::vec   x         = Rinv*(z + c);
  arma::uvec  idp       = regspace<uvec>(0, p - 1);
  arma::vec   thetahat  = zeros(p*q, 1);
  for(uword i = 0; i < q; i++){
    thetahat(idp)       = abs(x(idp))/sum(abs(x(idp)));
    idp                 = idp + p;
  }
  return(thetahat);
}

//[[Rcpp::export]] 
List FCLS_Rep(arma::mat M, arma::cube Y, double eps = 1e-6){
  arma::uword r         = M.n_rows;
  arma::uword p         = M.n_cols;
  arma::uword q         = Y.n_cols;
  arma::uword n         = Y.n_slices;
  arma::uword nIter     = 0;
  arma::uword maxIter   = 2e3;
  double      diff      = 1;
  double      c         = 1;
  double      ttime     = 0;
  clock_t 	  start;
  clock_t     end;
  arma::mat   Mtilde    = kron(eye(q, q), M);
  arma::mat   Omega     = eye(r, r)*100;
  arma::mat   Phi       = eye(q, q)*100;
  arma::mat   OmegaInv  = eye(r, r)*0.001;
  arma::mat   PhiInv    = eye(q, q)*0.001;
  arma::mat   SigmaInv  = kron(PhiInv, OmegaInv);
  arma::cube  Theta     = randu(p, q, n);
  arma::cube  ThetaOld  = randu(p, q, n);
  arma::vec   eigval;
  arma::mat   eigvec;
  eig_sym(eigval, eigvec, SigmaInv);
  arma::mat   Sigma_12  = diagmat(sqrt(eigval));
  Rcpp::List  Out;
  arma::mat   R;
  arma::mat   S;
  arma::vec   y;
  arma::mat   thetahat;
  arma::mat   Residual;
  arma::mat   AsympVar;
  arma::vec   se;
  arma::mat   tmp;
  /* Start timing */
  start 				= clock();
	while( (diff > eps) & (nIter < maxIter)){
    R         = zeros(r, r);
    S         = zeros(q, q);
    ThetaOld  = Theta;
    for(uword i = 0; i < n; i++){
      y               = vectorise(Y.slice(i));
      tmp             = FCLS(Mtilde, y, Sigma_12, p, q);
      thetahat        = reshape(tmp, p, q);
      Theta.slice(i)  = thetahat; 
      Residual        = Y.slice(i) - M*Theta.slice(i);
      S               = S + Residual.t()*OmegaInv*Residual;
      R               = R + Residual*PhiInv*Residual.t();
    }
    Phi               = S/(n*r);
    Omega             = R/(n*q);
	  c                 = 1/(trace(Phi)/q);
    PhiInv            = inv(Phi*c);
    OmegaInv          = inv(Omega);
    SigmaInv          = kron(PhiInv, OmegaInv);
    eig_sym(eigval, eigvec, SigmaInv);
    Sigma_12          = eigvec.t()*diagmat(sqrt(eigval));
    nIter             = nIter + 1;
    diff              = accu(abs(Theta - ThetaOld));
  }
  end 				        = clock();
  ttime               = ((double) (end - start)) / CLOCKS_PER_SEC;
  AsympVar            = inv(Mtilde.t()*SigmaInv*Mtilde);
  se                  = sqrt(AsympVar.diag()); 
  /* Printing message to console */
  Rcpp::Rcout << "                                               " << std::endl;
  Rcpp::Rcout << " Done! " << std::endl;
  Rcpp::Rcout << "                                               " << std::endl;
  Rcpp::Rcout << " Convergence criterion achieved (" << std::fixed << eps << ")" << std::endl;
  Rcpp::Rcout << nIter << " iterations in " << std::fixed << ttime << " seconds" << std::endl;
  /* Saving output */
  Out["Theta"]        = Theta;
  Out["s.e."]         = se;
  Out["Iterations"]   = nIter;
  Out["Time"]         = ttime;
  return(Out);
}

//[[Rcpp::export]] 
List FCLS_NRM(arma::mat M, arma::cube Y, arma::mat Ginv, double eps = 1e-6){
  arma::uword r         = M.n_rows;
  arma::uword p         = M.n_cols;
  arma::uword q         = Y.n_cols;
  arma::uword n         = Y.n_slices;
  arma::uword nIter     = 0;
  arma::uword maxIter   = 2e3;
  double      diff      = 1;
  double      c         = 1;
  arma::mat   Mtilde    = kron(eye(q*n, q*n), M);
  arma::mat   Omega     = eye(r, r)*100;
  arma::mat   Phi       = eye(q, q)*100;
  arma::mat   OmegaInv  = eye(r, r)*0.001;
  arma::mat   PhiInv    = eye(q, q)*0.001;
  arma::mat   SigmaInv  = kron(Ginv, kron(PhiInv, OmegaInv));
  arma::cube  Theta     = randu(p, q, n);
  arma::cube  ThetaOld  = randu(p, q, n);
  arma::cube  ResC      = zeros(q, r, n);
  arma::cube  ResR      = zeros(r, q, n);
  arma::mat   Er;
  arma::mat   Ec;
  arma::vec   eigval;
  arma::mat   eigvec;
  eig_sym(eigval, eigvec, SigmaInv);
  arma::mat   Sigma_12  = diagmat(sqrt(eigval));
  Rcpp::List  Out;
  arma::mat   R;
  arma::mat   S;
  arma::vec   y;
  arma::mat   thetahat;
  arma::mat   Residual;
  arma::mat   AsympVar;
  arma::vec   se;
  arma::mat   tmp;
  while((diff > eps) & (nIter < maxIter)){
    R         = zeros(r, r);
    S         = zeros(q, q);
    ThetaOld  = Theta;
	y         = vectorise(Y);
	thetahat  = FCLS(Mtilde, y, Sigma_12, p, q*n);
	Theta     = myreshape(thetahat, n, q, p);
	ResC      = zeros(q, r, n);
	ResR      = zeros(r, q, n);
    for(uword i = 0; i < n; i++){
      Residual        = Y.slice(i) - M*Theta.slice(i);
	  ResR.slice(i)   = Residual;
	  ResC.slice(i)   = Residual.t();
    }
	Er                = reshape(ResR, r, q*n, 1);
	Ec                = reshape(ResC, q, r*n, 1);
	S                 = Ec*kron(Ginv, OmegaInv)*Ec.t();
	R                 = Er*kron(Ginv, PhiInv)*Er.t();
    Phi               = S/(n*r);
	c                 = 1/(trace(Phi)/q);
    PhiInv            = inv(Phi*c);
    Omega             = R/(n*q);
    OmegaInv          = inv(Omega);
    SigmaInv          = kron(PhiInv, OmegaInv);
    eig_sym(eigval, eigvec, SigmaInv);
    Sigma_12          = eigvec.t()*diagmat(sqrt(eigval));
    nIter             = nIter + 1;
    diff              = accu(abs(Theta - ThetaOld));
  }
  AsympVar            = inv(Mtilde.t()*SigmaInv*Mtilde);
  se                  = sqrt(AsympVar.diag()); 
  Out["Theta"]        = Theta;
  Out["s.e."]         = se;
  Out["Iterations"]   = nIter;
  return(Out);
}
