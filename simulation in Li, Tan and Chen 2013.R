library(MASS)
library(Matrix)
library(numDeriv)
library(mvtnorm)
library(msm)
library(maxLik)
library(weights)
library(trust)

##############################
# Common functions and values
{##############################
colProd<-function(x){ # input a m*n matrix
 ncol_x<-dim(x)[2]; nrow_x<-dim(x)[1]
 results_prod<-rep(1,nrow_x)
 for(i in 1:ncol_x) results_prod<-results_prod*x[,i]
 return(results_prod)
}

# Estimating MLE var using IS
var_MLE_est<-function(alpha,q1_dens,q2_dens,pi_dens,q_gamma,g_dens){ 
 alpha1<-alpha
 alpha2<-1-alpha
 
 q_alpha<-alpha1*q1_dens+alpha2*q2_dens
 temp<-!((pi_dens==0)&(q_alpha==0))
 if(length(temp)>0){
 	 pi_dens<-pi_dens[temp]
	 q_gamma<-q_gamma[temp]
 	 q_alpha<-q_alpha[temp]
 	 g_dens<-g_dens[temp]
 }
 
 pi_over_gamma<-pi_dens/q_gamma
 pi_over_alpha<-pi_dens/q_alpha
 g_over_gamma<-g_dens/q_gamma
 g_over_alpha<-g_dens/q_alpha
 
 var_MLE_value<-mean(pi_over_gamma*pi_over_alpha)-mean(pi_over_alpha*g_over_gamma)^2/mean(g_over_gamma*g_over_alpha) 
 return(var_MLE_value)
}

var_MLE_expec_est<-function(alpha,h,mu_hat,q1_dens,q2_dens,pi_dens,q_gamma,g_dens,beta_output){ 
 alpha1<-alpha
 alpha2<-1-alpha
 
 q_alpha<-alpha1*q1_dens+alpha2*q2_dens
 temp<-!((pi_dens==0)&(q_alpha==0))
 pi_dens<-pi_dens[temp]
 q_gamma<-q_gamma[temp]
 q_alpha<-q_alpha[temp]
 g_dens<-g_dens[temp]

 beta_MLE<-mean((h-mu_hat)*pi_dens*g_dens/q_alpha/q_gamma)/mean(g_dens^2/q_alpha/q_gamma)
 var_MLE_value<-mean(((h-mu_hat)*pi_dens-beta_MLE*g_dens)^2/q_alpha/q_gamma)
 
 if(beta_output) return(list(var_MLE_value,beta_MLE))
 if(!beta_output) return(var_MLE_value)
}

var_DIS_est<-function(alpha,q1_dens,q2_dens,pi_dens,q_gamma,n1){ 
 alpha1<-alpha
 alpha2<-1-alpha
 n<-length(pi_dens)
 
 if(n1>1) q_alpha<-alpha1*q1_dens+alpha2*q2_dens
 if(n1<=1) q_alpha<-q2_dens 
 temp<-!((pi_dens==0)&(q_alpha==0))
 pi_dens<-pi_dens[temp]
 q_gamma<-q_gamma[temp]
 q_alpha<-q_alpha[temp]
 
 #var_DIS_value<-mean(pi_dens^2/q_alpha/q_gamma)
 if(n1>1) var_DIS_value<-alpha1*var((pi_dens/q_alpha)[1:n1])+alpha2*var((pi_dens/q_alpha)[(n1+1):n])
 if(n1<=1) var_DIS_value<-var(pi_dens/q_alpha)
 
 return(var_DIS_value)
}

var_DIS_expec_est<-function(alpha,h,mu_hat,q1_dens,q2_dens,pi_dens,q_gamma,n1){ 
 alpha1<-alpha
 alpha2<-1-alpha
 n<-length(pi_dens)
 
 if(n1>1) q_alpha<-alpha1*q1_dens+alpha2*q2_dens
 if(n1<=1) q_alpha<-q2_dens 
 temp<-!((pi_dens==0)&(q_alpha==0))
 pi_dens<-pi_dens[temp]
 q_gamma<-q_gamma[temp]
 q_alpha<-q_alpha[temp]
 
 if(n1>1) var_DIS_value<-alpha1*var(((h-mu_hat)*pi_dens/q_alpha)[1:n1])+alpha2*var(((h-mu_hat)*pi_dens/q_alpha)[(n1+1):n])
 if(n1<=1) var_DIS_value<-var((h-mu_hat)*pi_dens/q_alpha)

 return(var_DIS_value)
}

# Estimating MLE var using IID
var_MLE_est_1<-function(alpha,q1_dens,q2_dens,pi_dens,q_gamma,g_dens,beta_output,n1){ 
 alpha1<-alpha
 alpha2<-1-alpha
 
 q_alpha<-alpha1*q1_dens+alpha2*q2_dens
 temp<-!((pi_dens==0)&(q_alpha==0))
 pi_dens<-pi_dens[temp]
 q_gamma<-q_gamma[temp]
 q_alpha<-q_alpha[temp]
 g_dens<-g_dens[temp]
 
 beta_MLE<-(alpha1*mean((pi_dens*g_dens/q_alpha^2)[1:n1])+alpha2*mean((pi_dens*g_dens/q_alpha^2)[(n1+1):n]))/(alpha1*mean((g_dens^2/q_alpha^2)[1:n1])+alpha2*mean((g_dens^2/q_alpha^2)[(n1+1):n]))
 var_MLE_value<-alpha1*mean(((pi_dens-beta_MLE*g_dens)/q_alpha)[1:n1]^2)+alpha2*mean(((pi_dens-beta_MLE*g_dens)/q_alpha)[(n1+1):n]^2)

 if(beta_output) return(list(var_MLE_value,beta_MLE))
 if(!beta_output) return(var_MLE_value)
}

# Calculating densities
t_dens_joint<-function(x,mu_t,df_t){
  joint_dens_log<-log(1+(x-mu_t)^2/df_t)*(-(df_t+1)/2)+log(gamma((df_t+1)/2))-log(df_t*pi)/2-log(gamma(df_t/2))
  joint_dens<-colProd(exp(joint_dens_log))
  return(joint_dens)
}

norm_dens_joint<-function(x,mu_norm,sigma_norm){
  n<-dim(x)[1]
  joint_dens_log<-(x-rep(1,n)%*%t(mu_norm))^2/(-2*sigma_norm^2)-log(2*pi)/2-log(sigma_norm)
  joint_dens<-colProd(exp(joint_dens_log))
  return(joint_dens)
}

pi_dens_joint<-function(x,mu_norm_pi,sigma_norm_pi,mu_t_pi,df_t_pi,alpha_pi,target_component){

 if(target_component==1){
  # normal target
   pi_dens<-norm_dens_joint(x,mu_norm_pi,sigma_norm_pi)
 }

 if(target_component==2){
  # target normal dens  
  pi_dens_part1<-norm_dens_joint(x,mu_norm_pi,sigma_norm_pi)
  # target t dens  
  pi_dens_part2<-t_dens_joint(x,mu_t_pi,df_t_pi)
  # mixture target
  pi_dens<-alpha_pi*pi_dens_part1+(1-alpha_pi)*pi_dens_part2
 }
 
 return(pi_dens)
}

# MLE likelihood
l_fun_neg<-function(zeta,dens_all,prop){
  q1_dens<-dens_all[[1]]; q2_dens<-dens_all[[2]]
  if(sum((prop-zeta)*q1_dens+(1-prop+zeta)*q2_dens<0)>0) return(Inf)
  l_funval<-sum(log((prop-zeta)*q1_dens+(1-prop+zeta)*q2_dens))
  return(-l_funval)
}

delta_lowbound<-10^(-3)
}

##########################################
# asymptotic performance A1
{##########################################
mu_norm_pi<-rep(0,10); sigma_norm_pi<-rep(1,10);
mu_norm_q2<-rep(0,10); sigma_norm_q2<-rep(1.1,10); 
mu_t_q1<-0; df_t_q1<-1

n=2000000; dimension<-10; replic<-10

var_half_DIS<-list(0)
alpha_opt_DIS<-list(0)
var_opt_DIS<-list(0)
var_0_DIS<-list(0)
var_half_MLE<-list(0)
alpha_opt_MLE<-list(0)
var_opt_MLE<-list(0)

gamma_vec<-c(0.5,0.5)
 n1<-n*gamma_vec[1]
 n2<-n*gamma_vec[2]

save.time<-proc.time()
for(i in 1:replic){
 x1<-matrix(rt(n*dimension,df_t_q1)+mu_t_q1,ncol=dimension)  
 x2<-mvrnorm(n,mu=mu_norm_q2,Sigma=diag(sigma_norm_q2^2))
 data_all<-list(0,0)
 data_all[[1]]<-x1
 data_all[[2]]<-x2

 aa1<-as.matrix(data_all[[1]][1:n1,])
 aa2<-as.matrix(data_all[[2]][1:n2,])
 
 # proposal t dens
 q1_dens<-t_dens_joint(rbind(aa1,aa2),mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(rbind(aa1,aa2),mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens<-norm_dens_joint(rbind(aa1,aa2),mu_norm_pi,sigma_norm_pi)

 q_gamma<-gamma_vec[1]*q1_dens+gamma_vec[2]*q2_dens
 g_dens<-q2_dens-q1_dens
 

 var_half_DIS_results<-var_DIS_est(.5,q1_dens,q2_dens,pi_dens,q_gamma,n1)
 var_half_DIS[[i]]<-var_half_DIS_results

 alpha_results_DIS<-optimize(var_DIS_est,interval=c(delta_lowbound,1),q1_dens=q1_dens,q2_dens=q2_dens,pi_dens=pi_dens,q_gamma=q_gamma,n1=n1)
 alpha_opt_DIS[[i]]<-alpha_results_DIS$minimum
 var_opt_DIS[[i]]<-alpha_results_DIS$objective

 var_half_MLE_results<-var_MLE_est(.5,q1_dens,q2_dens,pi_dens,q_gamma,g_dens,0)
 var_half_MLE[[i]]<-var_half_MLE_results-1
  
 alpha_results_MLE<-optimize(var_MLE_est,interval=c(delta_lowbound,1),q1_dens=q1_dens,q2_dens=q2_dens,pi_dens=pi_dens,q_gamma=q_gamma,g_dens=g_dens,beta_output=0)
 alpha_opt_MLE[[i]]<-alpha_results_MLE$minimum
 var_opt_MLE[[i]]<-alpha_results_MLE$objective-1

  aa0<-as.matrix(data_all[[2]][1:n,])
 
 # proposal Normal dens
 q2_dens_0<-norm_dens_joint(aa0,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_0<-norm_dens_joint(aa0,mu_norm_pi,sigma_norm_pi)

 var_0_DIS_results<-var_DIS_est(0,q1_dens,q2_dens_0,pi_dens_0,q_gamma,0)
 var_0_DIS[[i]]<-var_0_DIS_results
 
 }
proc.time()-save.time

c(0.5,var_half_DIS=mean(unlist(var_half_DIS)))
c(mean(unlist(alpha_opt_DIS)),var_opt_DIS=mean(unlist(var_opt_DIS)))
c(0.5,var_half_MLE=mean(unlist(var_half_MLE)))
c(mean(unlist(alpha_opt_MLE)),var_opt_MLE=mean(unlist(var_opt_MLE)))
c(0,var_0_DIS=mean(unlist(var_0_DIS)))
}

###########################################
# two-stage performance A1
# use var_est, 400 for 1st stage
{###########################################
n=4000; replic=1000; dimension<-10; n0<-400

mu_norm_pi<-rep(0,10); sigma_norm_pi<-rep(1,10);
mu_norm_q2<-rep(0,10); sigma_norm_q2<-rep(1.1,10); 
mu_t_q1<-0; df_t_q1<-1

# fix mixture proportions
estimation_MLE<-function(alpha){
 n1<-floor(n*alpha); n2<-floor(n*(1-alpha));
 
 x1<-data_all[[1]][1:n1,]
 x2<-data_all[[2]][1:n2,]
 x<-rbind(x1,x2)
 
 # proposal t dens
 q1_dens<-t_dens_joint(x,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(x,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens<-norm_dens_joint(x,mu_norm_pi,sigma_norm_pi)
 
 q_alpha<-alpha*q1_dens+(1-alpha)*q2_dens
 g_dens<-q1_dens-q2_dens

 dens_all<-list(q1_dens,q2_dens)
 zeta_optim_results<-optimize(l_fun_neg,interval=c(-1,1),dens_all=dens_all,prop=alpha)
 zeta_optim<-zeta_optim_results$minimum
 estimator_MLE<-mean(pi_dens/((alpha-zeta_optim)*q1_dens+(1-alpha+zeta_optim)*q2_dens))

 return(estimator_MLE)
}

estimation_DIS<-function(alpha){
 n1<-floor(n*alpha); n2<-floor(n*(1-alpha));
 
 x1<-data_all[[1]][1:n1,]
 x2<-data_all[[2]][1:n2,]
 x<-rbind(x1,x2)
 
 # proposal t dens
 q1_dens<-t_dens_joint(x,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(x,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens<-norm_dens_joint(x,mu_norm_pi,sigma_norm_pi)
 
 q_alpha<-alpha*q1_dens+(1-alpha)*q2_dens
 g_dens<-q1_dens-q2_dens

 estimator_DIS<-mean(pi_dens/q_alpha)

 return(estimator_DIS)
}

estimation_q2<-function(n){ 
 x2<-data_all[[2]][1:n,]
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(x2,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens<-norm_dens_joint(x2,mu_norm_pi,sigma_norm_pi)

 estimator_DIS<-mean(pi_dens/q2_dens)

 return(estimator_DIS)
}


# two-stage mixture proportions 
estimation_MLE_twostages<-function(n0,gamma_vec){
 n01<-floor(n0*gamma_vec[1]); n02<-floor(n0*gamma_vec[2]);
 
 x11<-as.matrix(data_all[[1]][1:n01,])
 x12<-as.matrix(data_all[[2]][1:n02,])
 x1<-rbind(x11,x12)
 
 # proposal t dens
 q1_dens_stage1<-t_dens_joint(x1,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage1<-norm_dens_joint(x1,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage1<-norm_dens_joint(x1,mu_norm_pi,sigma_norm_pi)

 q_gamma_stage1<-gamma_vec[1]*q1_dens_stage1+gamma_vec[2]*q2_dens_stage1
 g_dens_stage1<-q1_dens_stage1-q2_dens_stage1
 
alpha_hat_MLE<-optimize(var_MLE_est,interval=c(delta_lowbound,1),q1_dens=q1_dens_stage1,q2_dens=q2_dens_stage1,pi_dens=pi_dens_stage1,q_gamma=q_gamma_stage1,g_dens=g_dens_stage1)$minimum

 n11<-floor((n-n0)*alpha_hat_MLE); n12<-floor((n-n0)*(1-alpha_hat_MLE));
 if(n11>1) x21<-as.matrix(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==1) x21<-t(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==0) x21<-NULL
 if(n12>1) x22<-as.matrix(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==1) x22<-t(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==0) x22<-NULL
 x2<-rbind(x21,x22)
 
 # proposal t dens
 q1_dens_stage2<-t_dens_joint(x2,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage2<-norm_dens_joint(x2,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage2<-norm_dens_joint(x2,mu_norm_pi,sigma_norm_pi)
 
 alpha_tilde<-n0/n*gamma_vec[1]+(n-n0)/n*alpha_hat_MLE
 q1_dens<-c(q1_dens_stage1,q1_dens_stage2)
 q2_dens<-c(q2_dens_stage1,q2_dens_stage2)
 pi_dens<-c(pi_dens_stage1,pi_dens_stage2)
 q_alpha<-alpha_tilde*q1_dens+(1-alpha_tilde)*q2_dens
 g_dens<-q1_dens-q2_dens
 
 dens_all<-list(q1_dens,q2_dens)
 zeta_optim_results<-optimize(l_fun_neg,interval=c(-1,1),dens_all=dens_all,prop=alpha_tilde)
 zeta_optim<-zeta_optim_results$minimum
 estimator_MLE<-mean(pi_dens/((alpha_tilde-zeta_optim)*q1_dens+(1-alpha_tilde+zeta_optim)*q2_dens))
  
 return(c(alpha_hat_MLE,estimator_MLE))
}

estimation_DIS_twostages<-function(n0,gamma_vec){
 n01<-floor(n0*gamma_vec[1]); n02<-floor(n0*gamma_vec[2]);
 
 x11<-as.matrix(data_all[[1]][1:n01,])
 x12<-as.matrix(data_all[[2]][1:n02,])
 x1<-rbind(x11,x12)
 
 # proposal t dens
 q1_dens_stage1<-t_dens_joint(x1,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage1<-norm_dens_joint(x1,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage1<-norm_dens_joint(x1,mu_norm_pi,sigma_norm_pi)

 q_gamma_stage1<-gamma_vec[1]*q1_dens_stage1+gamma_vec[2]*q2_dens_stage1
 g_dens_stage1<-q1_dens_stage1-q2_dens_stage1

 alpha_hat_DIS<-optimize(var_DIS_est,interval=c(delta_lowbound,1),q1_dens=q1_dens_stage1,q2_dens=q2_dens_stage1,pi_dens=pi_dens_stage1,q_gamma=q_gamma_stage1,n1=n01,tol=1e-05)$minimum

 n11<-floor((n-n0)*alpha_hat_DIS); n12<-floor((n-n0)*(1-alpha_hat_DIS));
 if(n11>1) x21<-as.matrix(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==1) x21<-t(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==0) x21<-NULL
 if(n12>1) x22<-as.matrix(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==1) x22<-t(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==0) x22<-NULL
 x2<-rbind(x21,x22)
 
 # proposal t dens
 q1_dens_stage2<-t_dens_joint(x2,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage2<-norm_dens_joint(x2,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage2<-norm_dens_joint(x2,mu_norm_pi,sigma_norm_pi)
 
 alpha_tilde<-n0/n*gamma_vec[1]+(n-n0)/n*alpha_hat_DIS
 q1_dens<-c(q1_dens_stage1,q1_dens_stage2)
 q2_dens<-c(q2_dens_stage1,q2_dens_stage2)
 pi_dens<-c(pi_dens_stage1,pi_dens_stage2)
 q_alpha<-alpha_tilde*q1_dens+(1-alpha_tilde)*q2_dens
 g_dens<-q1_dens-q2_dens
 
 estimator_DIS<-mean(pi_dens/q_alpha)
  
 return(c(alpha_hat_DIS,estimator_DIS))
}

est_DIS<-0
est_MLE<-0
est_q2<-0
alpha_hat_DIS_twostages<-0
est_DIS_twostages<-0
alpha_hat_MLE_twostages<-0
est_MLE_twostages<-0


save.time<-proc.time()
for(i in 1:replic){
  set.seed(2001+i)
  x1<-matrix(rt(n*dimension,df_t_q1)+mu_t_q1,ncol=dimension)
  x2<-mvrnorm(n,mu=mu_norm_q2,Sigma=diag(sigma_norm_q2^2))
  data_all<-list(x1,x2)

  results_est_DIS<-estimation_DIS(0.5)
  est_DIS[i]<-results_est_DIS
}
proc.time()-save.time

save.time<-proc.time()
for(i in 1:replic){
  set.seed(2001+i)
  x1<-matrix(rt(n*dimension,df_t_q1)+mu_t_q1,ncol=dimension)
  x2<-mvrnorm(n,mu=mu_norm_q2,Sigma=diag(sigma_norm_q2^2))
  data_all<-list(x1,x2)

  results_est_DIS_twostages<-estimation_DIS_twostages(n0,c(0.5,0.5))
  alpha_hat_DIS_twostages[i]<-results_est_DIS_twostages[1]
  est_DIS_twostages[i]<-results_est_DIS_twostages[2]
}
proc.time()-save.time


save.time<-proc.time()
for(i in 1:replic){
  set.seed(2001+i)
  x1<-matrix(rt(n*dimension,df_t_q1)+mu_t_q1,ncol=dimension)
  x2<-mvrnorm(n,mu=mu_norm_q2,Sigma=diag(sigma_norm_q2^2))
  data_all<-list(x1,x2)

  results_est_MLE<-estimation_MLE(0.5) 
  est_MLE[i]<-results_est_MLE
}
proc.time()-save.time

save.time<-proc.time()
for(i in 1:replic){
  set.seed(2001+i)
  x1<-matrix(rt(n*dimension,df_t_q1)+mu_t_q1,ncol=dimension)
  x2<-mvrnorm(n,mu=mu_norm_q2,Sigma=diag(sigma_norm_q2^2))
  data_all<-list(x1,x2)

  results_est_MLE_twostages<-estimation_MLE_twostages(n0,c(0.5,0.5))
  alpha_hat_MLE_twostages[i]<-results_est_MLE_twostages[1]
  est_MLE_twostages[i]<-results_est_MLE_twostages[2]
}
proc.time()-save.time


for(i in 1:replic){
  x2<-mvrnorm(n,mu=mu_norm_q2,Sigma=diag(sigma_norm_q2^2))
  data_all<-list(NULL,x2)

  results_est_0<-estimation_q2(n)
  est_q2[i]<-results_est_0
}

alpha_hat_MLE<-mean(alpha_hat_MLE_twostages)
var_MLE_twostages<-var(est_MLE_twostages)*n
mean_MLE_twostages<-mean(est_MLE_twostages)

alpha_hat_DIS<-mean(alpha_hat_DIS_twostages)
var_DIS_twostages<-var(est_DIS_twostages)*n
mean_DIS_twostages<-mean(est_DIS_twostages)
 
var_DIS<-var(est_DIS)*n 
mean_DIS<-mean(est_DIS)

var_q2<-var(est_q2)*n
mean_q2<-mean(est_q2)

var_MLE<-var(est_MLE)*n
mean_MLE<-mean(est_MLE)

results<-matrix(c(0.5,0.5,alpha_hat_DIS,alpha_hat_MLE,0,var_DIS,var_MLE,var_DIS_twostages,var_MLE_twostages,var_q2,mean_DIS,mean_MLE,mean_DIS_twostages,mean_MLE_twostages,mean_q2),5,3)
colnames(results)<-c("alpha","n*var","mean")
rownames(results)<-c("DIS","MLE","DIS_twostage","MLE_twostage","q2")
results
}

########################################################
# asymptotic performance A1 expectation
{#######################################################
mu_norm_pi<-rep(0,10); sigma_norm_pi<-rep(1,10);
mu_norm_q2<-rep(0,10); sigma_norm_q2<-rep(1.1,10); 
mu_t_q1<-0; df_t_q1<-1

n=2000000; dimension<-10; replic<-10

var_half_DIS_expec<-list(0)
alpha_opt_DIS_expec<-list(0)
var_opt_DIS_expec<-list(0)
var_half_MLE_expec<-list(0)
alpha_opt_MLE_expec<-list(0)
var_opt_MLE_expec<-list(0)
var_0_DIS_expec<-list(0)

gamma_vec<-c(.5,.5)
n1<-n*gamma_vec[1]
n2<-n*gamma_vec[2]

for(i in 1:replic){
 x1<-matrix(rt(n*dimension,df_t_q1)+mu_t_q1,ncol=dimension)  
 x2<-mvrnorm(n,mu=mu_norm_q2,Sigma=diag(sigma_norm_q2^2))
 data_all<-list(0,0)
 data_all[[1]]<-x1
 data_all[[2]]<-x2

 aa1<-as.matrix(data_all[[1]][1:n1,])
 aa2<-as.matrix(data_all[[2]][1:n2,])
 x<-rbind(aa1,aa2)
 
  # proposal t dens
 q1_dens<-t_dens_joint(x,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(x,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens<-norm_dens_joint(x,mu_norm_pi,sigma_norm_pi)

 q_gamma<-gamma_vec[1]*q1_dens+gamma_vec[2]*q2_dens
 g_dens<-q1_dens-q2_dens
 h<-rowSums(x^2)/dimension
 temp<-!((pi_dens==0)&(q_gamma==0))
 mu_hat<-mean((h*pi_dens/q_gamma)[temp])/mean((pi_dens/q_gamma)[temp])
 
 var_half_DIS_expec_results<-var_DIS_expec_est(.5,h,mu_hat,q1_dens,q2_dens,pi_dens,q_gamma,n1)
 var_half_DIS_expec[[i]]<-var_half_DIS_expec_results

 alpha_results_DIS_expec<-optimize(var_DIS_expec_est,interval=c(delta_lowbound,1),h=h,mu_hat=mu_hat,q1_dens=q1_dens,q2_dens=q2_dens,pi_dens=pi_dens,q_gamma=q_gamma,n1=n1)
 alpha_opt_DIS_expec[[i]]<-alpha_results_DIS_expec$minimum
 var_opt_DIS_expec[[i]]<-alpha_results_DIS_expec$objective

 var_half_MLE_expec_results<-var_MLE_expec_est(.5,h,mu_hat,q1_dens,q2_dens,pi_dens,q_gamma,g_dens,0)
 var_half_MLE_expec[[i]]<-var_half_MLE_expec_results
  
 alpha_results_MLE_expec<-optimize(var_MLE_expec_est,interval=c(delta_lowbound,1),h=h,mu_hat=mu_hat,q1_dens=q1_dens,q2_dens=q2_dens,pi_dens=pi_dens,q_gamma=q_gamma,g_dens=g_dens,beta_output=0)
 alpha_opt_MLE_expec[[i]]<-alpha_results_MLE_expec$minimum
 var_opt_MLE_expec[[i]]<-alpha_results_MLE_expec$objective

 aa0<-as.matrix(data_all[[2]][1:n,])
 h0<-rowSums(aa0^2)/dimension

 # proposal Normal dens
 q2_dens_0<-norm_dens_joint(aa0,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_0<-norm_dens_joint(aa0,mu_norm_pi,sigma_norm_pi)

 mu_hat0<-mean(h0*pi_dens_0/q2_dens_0)/mean(pi_dens_0/q2_dens_0)

 var_0_DIS_expec_results<-var_DIS_expec_est(0,h0,mu_hat0,q1_dens,q2_dens_0,pi_dens_0,q_gamma,0)
 var_0_DIS_expec[[i]]<-var_0_DIS_expec_results
}

c(0.5,var_half_DIS_expec=mean(unlist(var_half_DIS_expec)))
c(mean(unlist(alpha_opt_DIS_expec)),var_opt_DIS_expec=mean(unlist(var_opt_DIS_expec)))
c(0.5,var_half_MLE_expec=mean(unlist(var_half_MLE_expec)))
c(mean(unlist(alpha_opt_MLE_expec)),var_opt_MLE_expec=mean(unlist(var_opt_MLE_expec)))
c(0,var_0_DIS_expec=mean(unlist(var_0_DIS_expec)))

}

###########################################################
# two-stage performance A1 expectation
# use var_est, 400 for 1st stage
{##########################################################
n=4000; replic=1000; dimension<-10; n0<-400

mu_norm_pi<-rep(0,10); sigma_norm_pi<-rep(1,10);
mu_norm_q2<-rep(0,10); sigma_norm_q2<-rep(1.1,10); 
mu_t_q1<-0; df_t_q1<-1

# fix mixture proportions
estimation_MLE<-function(alpha){
 n1<-floor(n*alpha); n2<-floor(n*(1-alpha));
 
 x1<-data_all[[1]][1:n1,]
 x2<-data_all[[2]][1:n2,]
 x<-rbind(x1,x2)

 # proposal t dens
 q1_dens<-t_dens_joint(x,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(x,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens<-norm_dens_joint(x,mu_norm_pi,sigma_norm_pi)
 
 q_alpha<-alpha*q1_dens+(1-alpha)*q2_dens
 g_dens<-q1_dens-q2_dens
 h<-rowSums(x^2)/dimension

 dens_all<-list(q1_dens,q2_dens)
 zeta_optim_results<-optimize(l_fun_neg,interval=c(-1,1),dens_all=dens_all,prop=alpha)
 zeta_optim<-zeta_optim_results$minimum
 q_alpha_zeta<-(alpha-zeta_optim)*q1_dens+(1-alpha+zeta_optim)*q2_dens
 estimator_MLE<-mean(h*pi_dens/q_alpha_zeta)/mean(pi_dens/q_alpha_zeta) 

 return(estimator_MLE)
}

estimation_DIS<-function(alpha){
 n1<-floor(n*alpha); n2<-floor(n*(1-alpha));
 
 x1<-data_all[[1]][1:n1,]
 x2<-data_all[[2]][1:n2,]
 x<-rbind(x1,x2)
 
 # proposal t dens
 q1_dens<-t_dens_joint(x,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(x,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens<-norm_dens_joint(x,mu_norm_pi,sigma_norm_pi)

 q_alpha<-alpha*q1_dens+(1-alpha)*q2_dens
 g_dens<-q1_dens-q2_dens
 h<-rowSums(x^2)/dimension

 estimator_DIS<-mean(h*pi_dens/q_alpha)/mean(pi_dens/q_alpha)

 return(estimator_DIS)
}

estimation_q2<-function(n){ 
 x2<-data_all[[2]][1:n,]
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(x2,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens<-norm_dens_joint(x2,mu_norm_pi,sigma_norm_pi)
 
 h<-rowSums(x2^2)/dimension

 estimator_DIS<-mean(h*pi_dens/q2_dens)/mean(pi_dens/q2_dens)

 return(estimator_DIS)
}

# two-stage mixture proportions 
estimation_MLE_twostages<-function(n0,gamma_vec){
 n01<-floor(n0*gamma_vec[1]); n02<-floor(n0*gamma_vec[2]);
 
 x11<-as.matrix(data_all[[1]][1:n01,])
 x12<-as.matrix(data_all[[2]][1:n02,])
 x1<-rbind(x11,x12)

 # proposal t dens
 q1_dens_stage1<-t_dens_joint(x1,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage1<-norm_dens_joint(x1,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage1<-norm_dens_joint(x1,mu_norm_pi,sigma_norm_pi)

 q_gamma_stage1<-gamma_vec[1]*q1_dens_stage1+gamma_vec[2]*q2_dens_stage1
 g_dens_stage1<-q1_dens_stage1-q2_dens_stage1

 temp<-!((pi_dens_stage1==0)&(q_gamma_stage1==0))
 h_stage1<-rowSums(x1^2)/dimension
 mu_hat_stage1<-mean((h_stage1*pi_dens_stage1/q_gamma_stage1)[temp])/mean((pi_dens_stage1/q_gamma_stage1)[temp])
  
 alpha_hat_MLE<-optimize(var_MLE_expec_est,interval=c(delta_lowbound,1),h=h_stage1,mu_hat=mu_hat_stage1,q1_dens=q1_dens_stage1,q2_dens=q2_dens_stage1,pi_dens=pi_dens_stage1,q_gamma=q_gamma_stage1,g_dens=g_dens_stage1,beta_output=0)$minimum

 n11<-floor((n-n0)*alpha_hat_MLE); n12<-floor((n-n0)*(1-alpha_hat_MLE));
 if(n11>1) x21<-as.matrix(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==1) x21<-t(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==0) x21<-NULL
 if(n12>1) x22<-as.matrix(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==1) x22<-t(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==0) x22<-NULL
 x2<-rbind(x21,x22)

 # proposal t dens
 q1_dens_stage2<-t_dens_joint(x2,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage2<-norm_dens_joint(x2,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage2<-norm_dens_joint(x2,mu_norm_pi,sigma_norm_pi)

 h<-rowSums(rbind(x11,x12,x21,x22)^2)/dimension
 
 alpha_tilde<-n0/n*gamma_vec[1]+(n-n0)/n*alpha_hat_MLE
 q1_dens<-c(q1_dens_stage1,q1_dens_stage2)
 q2_dens<-c(q2_dens_stage1,q2_dens_stage2)
 pi_dens<-c(pi_dens_stage1,pi_dens_stage2)
 q_alpha<-alpha_tilde*q1_dens+(1-alpha_tilde)*q2_dens
 g_dens<-q1_dens-q2_dens
 
 dens_all<-list(q1_dens,q2_dens)
 zeta_optim_results<-optimize(l_fun_neg,interval=c(-1,1),dens_all=dens_all,prop=alpha_tilde)
 zeta_optim<-zeta_optim_results$minimum
 q_alpha_zeta<-(alpha_tilde-zeta_optim)*q1_dens+(1-alpha_tilde+zeta_optim)*q2_dens
 estimator_MLE<-mean(h*pi_dens/q_alpha_zeta)/mean(pi_dens/q_alpha_zeta)

 return(c(alpha_hat_MLE,estimator_MLE))
}

estimation_DIS_twostages<-function(n0,gamma_vec){
 n01<-floor(n0*gamma_vec[1]); n02<-floor(n0*gamma_vec[2]);
 
 x11<-as.matrix(data_all[[1]][1:n01,])
 x12<-as.matrix(data_all[[2]][1:n02,])
 x1<-rbind(x11,x12)

 # proposal t dens
 q1_dens_stage1<-t_dens_joint(x1,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage1<-norm_dens_joint(x1,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage1<-norm_dens_joint(x1,mu_norm_pi,sigma_norm_pi)

 q_gamma_stage1<-gamma_vec[1]*q1_dens_stage1+gamma_vec[2]*q2_dens_stage1
 g_dens_stage1<-q1_dens_stage1-q2_dens_stage1

 temp<-!((pi_dens_stage1==0)&(q_gamma_stage1==0))
 h_stage1<-rowSums(x1^2)/dimension
 mu_hat_stage1<-mean((h_stage1*pi_dens_stage1/q_gamma_stage1)[temp])/mean((pi_dens_stage1/q_gamma_stage1)[temp])
  
 alpha_hat_DIS<-optimize(var_DIS_expec_est,interval=c(delta_lowbound,1),h=h_stage1,mu_hat=mu_hat_stage1,q1_dens=q1_dens_stage1,q2_dens=q2_dens_stage1,pi_dens=pi_dens_stage1,q_gamma=q_gamma_stage1,n1=n01)$minimum

 n11<-floor((n-n0)*alpha_hat_DIS); n12<-floor((n-n0)*(1-alpha_hat_DIS));
 if(n11>1) x21<-as.matrix(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==1) x21<-t(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==0) x21<-NULL
 if(n12>1) x22<-as.matrix(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==1) x22<-t(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==0) x22<-NULL
 x2<-rbind(x21,x22)

 # proposal t dens
 q1_dens_stage2<-t_dens_joint(x2,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage2<-norm_dens_joint(x2,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage2<-norm_dens_joint(x2,mu_norm_pi,sigma_norm_pi)

 h<-rowSums(rbind(x11,x12,x21,x22)^2)/dimension
 
 alpha_tilde<-n0/n*gamma_vec[1]+(n-n0)/n*alpha_hat_DIS
 q1_dens<-c(q1_dens_stage1,q1_dens_stage2)
 q2_dens<-c(q2_dens_stage1,q2_dens_stage2)
 pi_dens<-c(pi_dens_stage1,pi_dens_stage2)
 q_alpha<-alpha_tilde*q1_dens+(1-alpha_tilde)*q2_dens
 g_dens<-q1_dens-q2_dens
 
 estimator_DIS<-mean(h*pi_dens/q_alpha)/mean(pi_dens/q_alpha)

 return(c(alpha_hat_DIS,estimator_DIS))
}

est_DIS<-0
est_MLE<-0
est_q2<-0
alpha_hat_DIS_twostages<-0
est_DIS_twostages<-0
alpha_hat_MLE_twostages<-0
est_MLE_twostages<-0

save.time<-proc.time()
for(i in 1:replic){
  x1<-matrix(rt(n*dimension,df_t_q1)+mu_t_q1,ncol=dimension)
  x2<-mvrnorm(n,mu=mu_norm_q2,Sigma=diag(sigma_norm_q2^2))
  data_all<-list(x1,x2)
  
  results_est_DIS<-estimation_DIS(0.5)
  est_DIS[i]<-results_est_DIS
  
  results_est_DIS_twostages<-estimation_DIS_twostages(n0,c(0.5,0.5))
  alpha_hat_DIS_twostages[i]<-results_est_DIS_twostages[1]
  est_DIS_twostages[i]<-results_est_DIS_twostages[2]

  results_est_MLE<-estimation_MLE(0.5) 
  est_MLE[i]<-results_est_MLE

  results_est_MLE_twostages<-estimation_MLE_twostages(n0,c(0.5,0.5))
  alpha_hat_MLE_twostages[i]<-results_est_MLE_twostages[1]
  est_MLE_twostages[i]<-results_est_MLE_twostages[2]
}
proc.time()-save.time

for(i in 1:replic){
  x2<-mvrnorm(n,mu=mu_norm_q2,Sigma=diag(sigma_norm_q2^2))
  data_all<-list(NULL,x2)

  results_est_0<-estimation_q2(n)
  est_q2[i]<-results_est_0
}

alpha_hat_MLE<-mean(alpha_hat_MLE_twostages)
var_MLE_twostages<-var(est_MLE_twostages)*n
mean_MLE_twostages<-mean(est_MLE_twostages)

alpha_hat_DIS<-mean(alpha_hat_DIS_twostages)
var_DIS_twostages<-var(est_DIS_twostages)*n
mean_DIS_twostages<-mean(est_DIS_twostages)
 
var_DIS<-var(est_DIS)*n 
mean_DIS<-mean(est_DIS)

var_q2<-var(est_q2)*n
mean_q2<-mean(est_q2)

var_MLE<-var(est_MLE)*n
mean_MLE<-mean(est_MLE)

results<-matrix(c(0.5,0.5,alpha_hat_DIS,alpha_hat_MLE,0,var_DIS,var_MLE,var_DIS_twostages,var_MLE_twostages,var_q2,mean_DIS,mean_MLE,mean_DIS_twostages,mean_MLE_twostages,mean_q2),5,3)
colnames(results)<-c("alpha","n*var","mean")
rownames(results)<-c("DIS","MLE","DIS_twostage","MLE_twostage","q2")
results
}

##########################################
# asymptotic performance A2
{##########################################
mu_norm_pi<-rep(0,10); sigma_norm_pi<-rep(1,10);
mu_norm_q2<-rep(0,10); sigma_norm_q2<-rep(.4,10); 
mu_t_q1<-0; df_t_q1<-1

n=2000000; dimension<-10; replic<-10

var_half_DIS<-list(0)
alpha_opt_DIS<-list(0)
var_opt_DIS<-list(0)
var_half_MLE<-list(0)
alpha_opt_MLE<-list(0)
var_opt_MLE<-list(0)
var_0_DIS<-list(0)

gamma_vec<-c(0.5,0.5)
 n1<-n*gamma_vec[1]
 n2<-n*gamma_vec[2]

save.time<-proc.time()
for(i in 1:replic){
 x1<-matrix(rt(n*dimension,df_t_q1)+mu_t_q1,ncol=dimension)  
 x2<-mvrnorm(n,mu=mu_norm_q2,Sigma=diag(sigma_norm_q2^2))
 data_all<-list(0,0)
 data_all[[1]]<-x1
 data_all[[2]]<-x2

 aa1<-as.matrix(data_all[[1]][1:n1,])
 aa2<-as.matrix(data_all[[2]][1:n2,])
 
 # proposal t dens
 q1_dens<-t_dens_joint(rbind(aa1,aa2),mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(rbind(aa1,aa2),mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens<-norm_dens_joint(rbind(aa1,aa2),mu_norm_pi,sigma_norm_pi)

 q_gamma<-gamma_vec[1]*q1_dens+gamma_vec[2]*q2_dens
 g_dens<-q2_dens-q1_dens
 

 var_half_DIS_results<-var_DIS_est(.5,q1_dens,q2_dens,pi_dens,q_gamma,n1)
 var_half_DIS[[i]]<-var_half_DIS_results
 
 alpha_results_DIS<-optimize(var_DIS_est,interval=c(delta_lowbound,1),q1_dens=q1_dens,q2_dens=q2_dens,pi_dens=pi_dens,q_gamma=q_gamma,n1=n1)
 alpha_opt_DIS[[i]]<-alpha_results_DIS$minimum
 var_opt_DIS[[i]]<-alpha_results_DIS$objective

 var_half_MLE_results<-var_MLE_est(.5,q1_dens,q2_dens,pi_dens,q_gamma,g_dens,0)
 var_half_MLE[[i]]<-var_half_MLE_results-1
  
 alpha_results_MLE<-optimize(var_MLE_est,interval=c(delta_lowbound,1),q1_dens=q1_dens,q2_dens=q2_dens,pi_dens=pi_dens,q_gamma=q_gamma,g_dens=g_dens,beta_output=0)
 alpha_opt_MLE[[i]]<-alpha_results_MLE$minimum
 var_opt_MLE[[i]]<-alpha_results_MLE$objective-1

  aa0<-as.matrix(data_all[[2]][1:n,])
 
 # proposal Normal dens
 q2_dens_0<-norm_dens_joint(aa0,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_0<-norm_dens_joint(aa0,mu_norm_pi,sigma_norm_pi)

 var_0_DIS_results<-var_DIS_est(0,q1_dens,q2_dens_0,pi_dens_0,q_gamma,0)
 var_0_DIS[[i]]<-var_0_DIS_results
}
proc.time()-save.time

c(0.5,var_half_DIS=mean(unlist(var_half_DIS)))
c(mean(unlist(alpha_opt_DIS)),var_opt_DIS=mean(unlist(var_opt_DIS)))
c(0.5,var_half_MLE=mean(unlist(var_half_MLE)))
c(mean(unlist(alpha_opt_MLE)),var_opt_MLE=mean(unlist(var_opt_MLE)))
c(0,var_0_DIS=mean(unlist(var_0_DIS)))
}

###########################################
# two-stage performance A2
# use var_est, 400 for 1st stage
{###########################################
n=4000; replic=1000; dimension<-10; n0<-400

mu_norm_pi<-rep(0,10); sigma_norm_pi<-rep(1,10);
mu_norm_q2<-rep(0,10); sigma_norm_q2<-rep(.4,10); 
mu_t_q1<-0; df_t_q1<-1

# fix mixture proportions
estimation_MLE<-function(alpha){
 n1<-floor(n*alpha); n2<-floor(n*(1-alpha));
 
 x1<-data_all[[1]][1:n1,]
 x2<-data_all[[2]][1:n2,]
 x<-rbind(x1,x2)
 
 # proposal t dens
 q1_dens<-t_dens_joint(x,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(x,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens<-norm_dens_joint(x,mu_norm_pi,sigma_norm_pi)
 
 q_alpha<-alpha*q1_dens+(1-alpha)*q2_dens
 g_dens<-q1_dens-q2_dens

 dens_all<-list(q1_dens,q2_dens)
 zeta_optim_results<-optimize(l_fun_neg,interval=c(-1,1),dens_all=dens_all,prop=alpha)
 zeta_optim<-zeta_optim_results$minimum
 estimator_MLE<-mean(pi_dens/((alpha-zeta_optim)*q1_dens+(1-alpha+zeta_optim)*q2_dens))

 return(estimator_MLE)
}

estimation_DIS<-function(alpha){
 n1<-floor(n*alpha); n2<-floor(n*(1-alpha));
 
 x1<-data_all[[1]][1:n1,]
 x2<-data_all[[2]][1:n2,]
 x<-rbind(x1,x2)
 
 # proposal t dens
 q1_dens<-t_dens_joint(x,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(x,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens<-norm_dens_joint(x,mu_norm_pi,sigma_norm_pi)
 
 q_alpha<-alpha*q1_dens+(1-alpha)*q2_dens
 g_dens<-q1_dens-q2_dens

 estimator_DIS<-mean(pi_dens/q_alpha)

 return(estimator_DIS)
}

estimation_q2<-function(n){ 
 x2<-data_all[[2]][1:n,]
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(x2,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens<-norm_dens_joint(x2,mu_norm_pi,sigma_norm_pi)

 estimator_DIS<-mean(pi_dens/q2_dens)

 return(estimator_DIS)
}

# two-stage mixture proportions 
estimation_MLE_twostages<-function(n0,gamma_vec){
 n01<-floor(n0*gamma_vec[1]); n02<-floor(n0*gamma_vec[2]);
 
 x11<-as.matrix(data_all[[1]][1:n01,])
 x12<-as.matrix(data_all[[2]][1:n02,])
 x1<-rbind(x11,x12)
 
 # proposal t dens
 q1_dens_stage1<-t_dens_joint(x1,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage1<-norm_dens_joint(x1,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage1<-norm_dens_joint(x1,mu_norm_pi,sigma_norm_pi)

 q_gamma_stage1<-gamma_vec[1]*q1_dens_stage1+gamma_vec[2]*q2_dens_stage1
 g_dens_stage1<-q1_dens_stage1-q2_dens_stage1

 alpha_hat_MLE<-optimize(var_MLE_est,interval=c(delta_lowbound,1),q1_dens=q1_dens_stage1,q2_dens=q2_dens_stage1,pi_dens=pi_dens_stage1,q_gamma=q_gamma_stage1,g_dens=g_dens_stage1,beta_output=0)$minimum

 n11<-floor((n-n0)*alpha_hat_MLE); n12<-floor((n-n0)*(1-alpha_hat_MLE));
 if(n11>1) x21<-as.matrix(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==1) x21<-t(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==0) x21<-NULL
 if(n12>1) x22<-as.matrix(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==1) x22<-t(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==0) x22<-NULL
 x2<-rbind(x21,x22)
 
 # proposal t dens
 q1_dens_stage2<-t_dens_joint(x2,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage2<-norm_dens_joint(x2,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage2<-norm_dens_joint(x2,mu_norm_pi,sigma_norm_pi)
 
 alpha_tilde<-n0/n*gamma_vec[1]+(n-n0)/n*alpha_hat_MLE
 q1_dens<-c(q1_dens_stage1,q1_dens_stage2)
 q2_dens<-c(q2_dens_stage1,q2_dens_stage2)
 pi_dens<-c(pi_dens_stage1,pi_dens_stage2)
 q_alpha<-alpha_tilde*q1_dens+(1-alpha_tilde)*q2_dens
 g_dens<-q1_dens-q2_dens
 
 dens_all<-list(q1_dens,q2_dens)
 zeta_optim_results<-optimize(l_fun_neg,interval=c(-1,1),dens_all=dens_all,prop=alpha_tilde)
 zeta_optim<-zeta_optim_results$minimum
 estimator_MLE<-mean(pi_dens/((alpha_tilde-zeta_optim)*q1_dens+(1-alpha_tilde+zeta_optim)*q2_dens))
  
  browser()
 return(c(alpha_hat_MLE,estimator_MLE))
}

estimation_DIS_twostages<-function(n0,gamma_vec){
 n01<-floor(n0*gamma_vec[1]); n02<-floor(n0*gamma_vec[2]);
 
 x11<-as.matrix(data_all[[1]][1:n01,])
 x12<-as.matrix(data_all[[2]][1:n02,])
 x1<-rbind(x11,x12)
 
 # proposal t dens
 q1_dens_stage1<-t_dens_joint(x1,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage1<-norm_dens_joint(x1,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage1<-norm_dens_joint(x1,mu_norm_pi,sigma_norm_pi)

 q_gamma_stage1<-gamma_vec[1]*q1_dens_stage1+gamma_vec[2]*q2_dens_stage1
 g_dens_stage1<-q1_dens_stage1-q2_dens_stage1

 alpha_hat_DIS<-optimize(var_DIS_est,interval=c(delta_lowbound,1),q1_dens=q1_dens_stage1,q2_dens=q2_dens_stage1,pi_dens=pi_dens_stage1,q_gamma=q_gamma_stage1,n1=n01)$minimum

 n11<-floor((n-n0)*alpha_hat_DIS); n12<-floor((n-n0)*(1-alpha_hat_DIS));
 if(n11>1) x21<-as.matrix(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==1) x21<-t(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==0) x21<-NULL
 if(n12>1) x22<-as.matrix(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==1) x22<-t(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==0) x22<-NULL
 x2<-rbind(x21,x22)
 
 # proposal t dens
 q1_dens_stage2<-t_dens_joint(x2,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage2<-norm_dens_joint(x2,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage2<-norm_dens_joint(x2,mu_norm_pi,sigma_norm_pi)
 
 alpha_tilde<-n0/n*gamma_vec[1]+(n-n0)/n*alpha_hat_DIS
 q1_dens<-c(q1_dens_stage1,q1_dens_stage2)
 q2_dens<-c(q2_dens_stage1,q2_dens_stage2)
 pi_dens<-c(pi_dens_stage1,pi_dens_stage2)
 q_alpha<-alpha_tilde*q1_dens+(1-alpha_tilde)*q2_dens
 g_dens<-q1_dens-q2_dens
 
 estimator_DIS<-mean(pi_dens/q_alpha)
  
 return(c(alpha_hat_DIS,estimator_DIS))
}

est_DIS<-0
est_MLE<-0
est_q2<-0
alpha_hat_DIS_twostages<-0
est_DIS_twostages<-0
alpha_hat_MLE_twostages<-0
est_MLE_twostages<-0

save.time<-proc.time()
for(i in 1:replic){
  x1<-matrix(rt(n*dimension,df_t_q1)+mu_t_q1,ncol=dimension)
  x2<-mvrnorm(n,mu=mu_norm_q2,Sigma=diag(sigma_norm_q2^2))
  data_all<-list(x1,x2)
  
  results_est_DIS<-estimation_DIS(0.5)
  est_DIS[i]<-results_est_DIS
  
  results_est_DIS_twostages<-estimation_DIS_twostages(n0,c(0.5,0.5))
  alpha_hat_DIS_twostages[i]<-results_est_DIS_twostages[1]
  est_DIS_twostages[i]<-results_est_DIS_twostages[2]

  results_est_MLE<-estimation_MLE(0.5) 
  est_MLE[i]<-results_est_MLE

  results_est_MLE_twostages<-estimation_MLE_twostages(n0,c(0.5,0.5))
  alpha_hat_MLE_twostages[i]<-results_est_MLE_twostages[1]
  est_MLE_twostages[i]<-results_est_MLE_twostages[2]
}
proc.time()-save.time

for(i in 1:replic){
  x2<-mvrnorm(n,mu=mu_norm_q2,Sigma=diag(sigma_norm_q2^2))
  data_all<-list(NULL,x2)

  results_est_0<-estimation_q2(n)
  est_q2[i]<-results_est_0
}

alpha_hat_MLE<-mean(alpha_hat_MLE_twostages)
var_MLE_twostages<-var(est_MLE_twostages)*n
mean_MLE_twostages<-mean(est_MLE_twostages)

alpha_hat_DIS<-mean(alpha_hat_DIS_twostages)
var_DIS_twostages<-var(est_DIS_twostages)*n
mean_DIS_twostages<-mean(est_DIS_twostages)
 
var_DIS<-var(est_DIS)*n 
mean_DIS<-mean(est_DIS)

var_q2<-var(est_q2)*n
mean_q2<-mean(est_q2)

var_MLE<-var(est_MLE)*n
mean_MLE<-mean(est_MLE)

results<-matrix(c(0.5,0.5,alpha_hat_DIS,alpha_hat_MLE,0,var_DIS,var_MLE,var_DIS_twostages,var_MLE_twostages,var_q2,mean_DIS,mean_MLE,mean_DIS_twostages,mean_MLE_twostages,mean_q2),5,3)
colnames(results)<-c("alpha","n*var","mean")
rownames(results)<-c("DIS","MLE","DIS_twostage","MLE_twostage","q2")
results
}

########################################################
# asymptotic performance A2 expectation
{#######################################################
mu_norm_pi<-rep(0,10); sigma_norm_pi<-rep(1,10);
mu_norm_q2<-rep(0,10); sigma_norm_q2<-rep(.4,10); 
mu_t_q1<-0; df_t_q1<-1

n=2000000; dimension<-10; replic<-10

var_half_DIS_expec<-list(0)
alpha_opt_DIS_expec<-list(0)
var_opt_DIS_expec<-list(0)
var_half_MLE_expec<-list(0)
alpha_opt_MLE_expec<-list(0)
var_opt_MLE_expec<-list(0)
var_0_DIS_expec<-list(0)

gamma_vec<-c(.5,.5)
n1<-n*gamma_vec[1]
n2<-n*gamma_vec[2]



for(i in 1:replic){
 x1<-matrix(rt(n*dimension,df_t_q1)+mu_t_q1,ncol=dimension)  
 x2<-mvrnorm(n,mu=mu_norm_q2,Sigma=diag(sigma_norm_q2^2))
 data_all<-list(0,0)
 data_all[[1]]<-x1
 data_all[[2]]<-x2

 aa0<-as.matrix(data_all[[2]][1:n,])
 h0<-rowSums(aa0^2)/dimension
 
 # proposal Normal dens
 q2_dens_0<-norm_dens_joint(aa0,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_0<-norm_dens_joint(aa0,mu_norm_pi,sigma_norm_pi)

 mu_hat0<-mean(h0*pi_dens_0/q2_dens_0)/mean(pi_dens_0/q2_dens_0)

 var_0_DIS_expec_results<-var_DIS_expec_est(0,h0,mu_hat0,q1_dens,q2_dens_0,pi_dens_0,q_gamma,0)
 var_0_DIS_expec[[i]]<-var_0_DIS_expec_results
}

for(i in 1:replic){
 x1<-matrix(rt(n*dimension,df_t_q1)+mu_t_q1,ncol=dimension)  
 x2<-mvrnorm(n,mu=mu_norm_q2,Sigma=diag(sigma_norm_q2^2))
 data_all<-list(0,0)
 data_all[[1]]<-x1
 data_all[[2]]<-x2

 aa1<-as.matrix(data_all[[1]][1:n1,])
 aa2<-as.matrix(data_all[[2]][1:n2,])
 x<-rbind(aa1,aa2)
 
  # proposal t dens
 q1_dens<-t_dens_joint(x,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(x,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens<-norm_dens_joint(x,mu_norm_pi,sigma_norm_pi)

 q_gamma<-gamma_vec[1]*q1_dens+gamma_vec[2]*q2_dens
 g_dens<-q1_dens-q2_dens
 h<-rowSums(x^2)/dimension
 temp<-!((pi_dens==0)&(q_gamma==0))
 mu_hat<-mean((h*pi_dens/q_gamma)[temp])/mean((pi_dens/q_gamma)[temp])
 
 var_half_DIS_expec_results<-var_DIS_expec_est(.5,h,mu_hat,q1_dens,q2_dens,pi_dens,q_gamma,n1)
 var_half_DIS_expec[[i]]<-var_half_DIS_expec_results
 
 alpha_results_DIS_expec<-optimize(var_DIS_expec_est,interval=c(delta_lowbound,1),h=h,mu_hat=mu_hat,q1_dens=q1_dens,q2_dens=q2_dens,pi_dens=pi_dens,q_gamma=q_gamma,n1=n1)
 alpha_opt_DIS_expec[[i]]<-alpha_results_DIS_expec$minimum
 var_opt_DIS_expec[[i]]<-alpha_results_DIS_expec$objective

 var_half_MLE_expec_results<-var_MLE_expec_est(.5,h,mu_hat,q1_dens,q2_dens,pi_dens,q_gamma,g_dens,0)
 var_half_MLE_expec[[i]]<-var_half_MLE_expec_results
  
 alpha_results_MLE_expec<-optimize(var_MLE_expec_est,interval=c(delta_lowbound,1),h=h,mu_hat=mu_hat,q1_dens=q1_dens,q2_dens=q2_dens,pi_dens=pi_dens,q_gamma=q_gamma,g_dens=g_dens,beta_output=0)
 alpha_opt_MLE_expec[[i]]<-alpha_results_MLE_expec$minimum
 var_opt_MLE_expec[[i]]<-alpha_results_MLE_expec$objective

 aa0<-as.matrix(data_all[[2]][1:n,])
 h0<-rowSums(aa0^2)/dimension
 
 # proposal Normal dens
 q2_dens_0<-norm_dens_joint(aa0,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_0<-norm_dens_joint(aa0,mu_norm_pi,sigma_norm_pi)

 mu_hat0<-mean(h0*pi_dens_0/q2_dens_0)/mean(pi_dens_0/q2_dens_0)

 var_0_DIS_expec_results<-var_DIS_expec_est(0,h0,mu_hat0,q1_dens,q2_dens_0,pi_dens_0,q_gamma,0)
 var_0_DIS_expec[[i]]<-var_0_DIS_expec_results
}

c(0.5,var_half_DIS_expec=n*mean(unlist(var_half_DIS_expec)))
c(mean(unlist(alpha_opt_DIS_expec)),var_opt_DIS_expec=n*mean(unlist(var_opt_DIS_expec)))
c(0.5,var_half_MLE_expec=n*mean(unlist(var_half_MLE_expec)))
c(mean(unlist(alpha_opt_MLE_expec)),var_opt_MLE_expec=n*mean(unlist(var_opt_MLE_expec)))
c(0,var_0_DIS_expec=mean(unlist(var_0_DIS_expec)))

}

###########################################################
# two-stage performance A2 expectation
# use var_est, 400 for 1st stage
{##########################################################
n=4000; replic=1000; dimension<-10; n0<-400

mu_norm_pi<-rep(0,10); sigma_norm_pi<-rep(1,10);
mu_norm_q2<-rep(0,10); sigma_norm_q2<-rep(.4,10); 
mu_t_q1<-0; df_t_q1<-1

# fix mixture proportions
estimation_MLE<-function(alpha){
 n1<-floor(n*alpha); n2<-floor(n*(1-alpha));
 
 x1<-data_all[[1]][1:n1,]
 x2<-data_all[[2]][1:n2,]
 x<-rbind(x1,x2)

 # proposal t dens
 q1_dens<-t_dens_joint(x,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(x,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens<-norm_dens_joint(x,mu_norm_pi,sigma_norm_pi)
 
 q_alpha<-alpha*q1_dens+(1-alpha)*q2_dens
 g_dens<-q1_dens-q2_dens
 h<-rowSums(x^2)/dimension

 dens_all<-list(q1_dens,q2_dens)
 zeta_optim_results<-optimize(l_fun_neg,interval=c(-1,1),dens_all=dens_all,prop=alpha)
 zeta_optim<-zeta_optim_results$minimum
 q_alpha_zeta<-(alpha-zeta_optim)*q1_dens+(1-alpha+zeta_optim)*q2_dens
 estimator_MLE<-mean(h*pi_dens/q_alpha_zeta)/mean(pi_dens/q_alpha_zeta) 

 return(estimator_MLE)
}

estimation_DIS<-function(alpha){
 n1<-floor(n*alpha); n2<-floor(n*(1-alpha));
 
 x1<-data_all[[1]][1:n1,]
 x2<-data_all[[2]][1:n2,]
 x<-rbind(x1,x2)
 
 # proposal t dens
 q1_dens<-t_dens_joint(x,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(x,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens<-norm_dens_joint(x,mu_norm_pi,sigma_norm_pi)

 q_alpha<-alpha*q1_dens+(1-alpha)*q2_dens
 g_dens<-q1_dens-q2_dens
 h<-rowSums(x^2)/dimension

 estimator_DIS<-mean(h*pi_dens/q_alpha)/mean(pi_dens/q_alpha)

 return(estimator_DIS)
}

estimation_q2<-function(n){ 
 x2<-data_all[[2]][1:n,]
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(x2,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens<-norm_dens_joint(x2,mu_norm_pi,sigma_norm_pi)
 
 h<-rowSums(x2^2)/dimension

 estimator_DIS<-mean(h*pi_dens/q2_dens)/mean(pi_dens/q2_dens)

 return(estimator_DIS)
}

# two-stage mixture proportions 
estimation_MLE_twostages<-function(n0,gamma_vec){
 n01<-floor(n0*gamma_vec[1]); n02<-floor(n0*gamma_vec[2]);
 
 x11<-as.matrix(data_all[[1]][1:n01,])
 x12<-as.matrix(data_all[[2]][1:n02,])
 x1<-rbind(x11,x12)

 # proposal t dens
 q1_dens_stage1<-t_dens_joint(x1,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage1<-norm_dens_joint(x1,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage1<-norm_dens_joint(x1,mu_norm_pi,sigma_norm_pi)

 q_gamma_stage1<-gamma_vec[1]*q1_dens_stage1+gamma_vec[2]*q2_dens_stage1
 g_dens_stage1<-q1_dens_stage1-q2_dens_stage1

 temp<-!((pi_dens_stage1==0)&(q_gamma_stage1==0))
 h_stage1<-rowSums(x1^2)/dimension
 mu_hat_stage1<-mean((h_stage1*pi_dens_stage1/q_gamma_stage1)[temp])/mean((pi_dens_stage1/q_gamma_stage1)[temp])
  
 alpha_hat_MLE<-optimize(var_MLE_expec_est,interval=c(delta_lowbound,1),h=h_stage1,mu_hat=mu_hat_stage1,q1_dens=q1_dens_stage1,q2_dens=q2_dens_stage1,pi_dens=pi_dens_stage1,q_gamma=q_gamma_stage1,g_dens=g_dens_stage1,beta_output=0)$minimum

 n11<-floor((n-n0)*alpha_hat_MLE); n12<-floor((n-n0)*(1-alpha_hat_MLE));
 if(n11>1) x21<-as.matrix(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==1) x21<-t(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==0) x21<-NULL
 if(n12>1) x22<-as.matrix(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==1) x22<-t(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==0) x22<-NULL
 x2<-rbind(x21,x22)

 # proposal t dens
 q1_dens_stage2<-t_dens_joint(x2,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage2<-norm_dens_joint(x2,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage2<-norm_dens_joint(x2,mu_norm_pi,sigma_norm_pi)

 h<-rowSums(rbind(x11,x12,x21,x22)^2)/dimension
 
 alpha_tilde<-n0/n*gamma_vec[1]+(n-n0)/n*alpha_hat_MLE
 q1_dens<-c(q1_dens_stage1,q1_dens_stage2)
 q2_dens<-c(q2_dens_stage1,q2_dens_stage2)
 pi_dens<-c(pi_dens_stage1,pi_dens_stage2)
 q_alpha<-alpha_tilde*q1_dens+(1-alpha_tilde)*q2_dens
 g_dens<-q1_dens-q2_dens
 
 dens_all<-list(q1_dens,q2_dens)
 zeta_optim_results<-optimize(l_fun_neg,interval=c(-1,1),dens_all=dens_all,prop=alpha_tilde)
 zeta_optim<-zeta_optim_results$minimum
 q_alpha_zeta<-(alpha_tilde-zeta_optim)*q1_dens+(1-alpha_tilde+zeta_optim)*q2_dens
 estimator_MLE<-mean(h*pi_dens/q_alpha_zeta)/mean(pi_dens/q_alpha_zeta)

 return(c(alpha_hat_MLE,estimator_MLE))
}

estimation_DIS_twostages<-function(n0,gamma_vec){
 n01<-floor(n0*gamma_vec[1]); n02<-floor(n0*gamma_vec[2]);
 
 x11<-as.matrix(data_all[[1]][1:n01,])
 x12<-as.matrix(data_all[[2]][1:n02,])
 x1<-rbind(x11,x12)

 # proposal t dens
 q1_dens_stage1<-t_dens_joint(x1,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage1<-norm_dens_joint(x1,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage1<-norm_dens_joint(x1,mu_norm_pi,sigma_norm_pi)

 q_gamma_stage1<-gamma_vec[1]*q1_dens_stage1+gamma_vec[2]*q2_dens_stage1
 g_dens_stage1<-q1_dens_stage1-q2_dens_stage1

 temp<-!((pi_dens_stage1==0)&(q_gamma_stage1==0))
 h_stage1<-rowSums(x1^2)/dimension
 mu_hat_stage1<-mean((h_stage1*pi_dens_stage1/q_gamma_stage1)[temp])/mean((pi_dens_stage1/q_gamma_stage1)[temp])
  
 alpha_hat_DIS<-optimize(var_DIS_expec_est,interval=c(delta_lowbound,1),h=h_stage1,mu_hat=mu_hat_stage1,q1_dens=q1_dens_stage1,q2_dens=q2_dens_stage1,pi_dens=pi_dens_stage1,q_gamma=q_gamma_stage1,n1=n01)$minimum

 n11<-floor((n-n0)*alpha_hat_DIS); n12<-floor((n-n0)*(1-alpha_hat_DIS));
 if(n11>1) x21<-as.matrix(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==1) x21<-t(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==0) x21<-NULL
 if(n12>1) x22<-as.matrix(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==1) x22<-t(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==0) x22<-NULL
 x2<-rbind(x21,x22)

 # proposal t dens
 q1_dens_stage2<-t_dens_joint(x2,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage2<-norm_dens_joint(x2,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage2<-norm_dens_joint(x2,mu_norm_pi,sigma_norm_pi)

 h<-rowSums(rbind(x11,x12,x21,x22)^2)/dimension
 
 alpha_tilde<-n0/n*gamma_vec[1]+(n-n0)/n*alpha_hat_DIS
 q1_dens<-c(q1_dens_stage1,q1_dens_stage2)
 q2_dens<-c(q2_dens_stage1,q2_dens_stage2)
 pi_dens<-c(pi_dens_stage1,pi_dens_stage2)
 q_alpha<-alpha_tilde*q1_dens+(1-alpha_tilde)*q2_dens
 g_dens<-q1_dens-q2_dens
 
 estimator_DIS<-mean(h*pi_dens/q_alpha)/mean(pi_dens/q_alpha)

 return(c(alpha_hat_DIS,estimator_DIS))
}

est_DIS<-0
est_MLE<-0
est_q2<-0
alpha_hat_DIS_twostages<-0
est_DIS_twostages<-0
alpha_hat_MLE_twostages<-0
est_MLE_twostages<-0

save.time<-proc.time()
for(i in 1:replic){
  x1<-matrix(rt(n*dimension,df_t_q1)+mu_t_q1,ncol=dimension)
  x2<-mvrnorm(n,mu=mu_norm_q2,Sigma=diag(sigma_norm_q2^2))
  data_all<-list(x1,x2)
  
  results_est_DIS<-estimation_DIS(0.5)
  est_DIS[i]<-results_est_DIS
  
  results_est_DIS_twostages<-estimation_DIS_twostages(n0,c(0.5,0.5))
  alpha_hat_DIS_twostages[i]<-results_est_DIS_twostages[1]
  est_DIS_twostages[i]<-results_est_DIS_twostages[2]

  results_est_MLE<-estimation_MLE(0.5) 
  est_MLE[i]<-results_est_MLE

  results_est_MLE_twostages<-estimation_MLE_twostages(n0,c(0.5,0.5))
  alpha_hat_MLE_twostages[i]<-results_est_MLE_twostages[1]
  est_MLE_twostages[i]<-results_est_MLE_twostages[2]
}
proc.time()-save.time

for(i in 1:replic){
  x2<-mvrnorm(n,mu=mu_norm_q2,Sigma=diag(sigma_norm_q2^2))
  data_all<-list(NULL,x2)

  results_est_0<-estimation_q2(n)
  est_q2[i]<-results_est_0
}

alpha_hat_MLE<-mean(alpha_hat_MLE_twostages)
var_MLE_twostages<-var(est_MLE_twostages)*n
mean_MLE_twostages<-mean(est_MLE_twostages)

alpha_hat_DIS<-mean(alpha_hat_DIS_twostages)
var_DIS_twostages<-var(est_DIS_twostages)*n
mean_DIS_twostages<-mean(est_DIS_twostages)
 
var_DIS<-var(est_DIS)*n 
mean_DIS<-mean(est_DIS)

var_MLE<-var(est_MLE)*n
mean_MLE<-mean(est_MLE)

var_q2<-var(est_q2)*n
mean_q2<-mean(est_q2)

results<-matrix(c(0.5,0.5,alpha_hat_DIS,alpha_hat_MLE,0,var_DIS,var_MLE,var_DIS_twostages,var_MLE_twostages,var_q2,mean_DIS,mean_MLE,mean_DIS_twostages,mean_MLE_twostages,mean_q2),5,3)
colnames(results)<-c("alpha","n*var","mean")
rownames(results)<-c("DIS","MLE","DIS_twostage","MLE_twostage","q2")
results
}

####################################
# asymptotic performance B1
{###################################
mu_norm_pi<-rep(0,10); sigma_norm_pi<-rep(1,10); #sigma_norm_pi<-rep(0.8,10);
mu_norm_q2<-rep(0,10); sigma_norm_q2<-rep(1,10); 
mu_t_pi<-0; df_t_pi<-4; 
mu_t_q1<-0; df_t_q1<-1

n=2000000; dimension<-10; replic<-10; alpha_pi<-.8

var_half_DIS<-list(0)
alpha_opt_DIS<-list(0)
var_opt_DIS<-list(0)
var_half_MLE<-list(0)
alpha_opt_MLE<-list(0)
var_opt_MLE<-list(0)
var_0_DIS<-list(0)

gamma_vec<-c(.5,.5)
n1<-n*gamma_vec[1]
n2<-n*gamma_vec[2]


save.time<-proc.time()
for(i in 1:replic){
 x1<-matrix(rt(n*dimension,df_t_q1)+mu_t_q1,ncol=dimension)  
 x2<-mvrnorm(n,mu=mu_norm_q2,Sigma=diag(sigma_norm_q2^2))
 data_all<-list(0,0)
 data_all[[1]]<-x1
 data_all[[2]]<-x2

 aa1<-as.matrix(data_all[[1]][1:n1,])
 aa2<-as.matrix(data_all[[2]][1:n2,])
 x<-rbind(aa1,aa2)
 
 # proposal t dens
 q1_dens<-t_dens_joint(x,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(x,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_part1<-norm_dens_joint(x,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_part2<-t_dens_joint(x,mu_t_pi,df_t_pi)
 
 pi_dens<-alpha_pi*pi_dens_part1+(1-alpha_pi)*pi_dens_part2
 q_gamma<-gamma_vec[1]*q1_dens+gamma_vec[2]*q2_dens
 g_dens<-q1_dens-q2_dens
 
 var_half_DIS_results<-var_DIS_est(.5,q1_dens,q2_dens,pi_dens,q_gamma,n1)
 var_half_DIS[[i]]<-var_half_DIS_results
 
 alpha_results_DIS<-optimize(var_DIS_est,interval=c(delta_lowbound,1),q1_dens=q1_dens,q2_dens=q2_dens,pi_dens=pi_dens,q_gamma=q_gamma,n1=n1)
 alpha_opt_DIS[[i]]<-alpha_results_DIS$minimum
 var_opt_DIS[[i]]<-alpha_results_DIS$objective

 var_half_MLE_results<-var_MLE_est(.5,q1_dens,q2_dens,pi_dens,q_gamma,g_dens,0)
 var_half_MLE[[i]]<-var_half_MLE_results-1
  
 alpha_results_MLE<-optimize(var_MLE_est,interval=c(delta_lowbound,1),q1_dens=q1_dens,q2_dens=q2_dens,pi_dens=pi_dens,q_gamma=q_gamma,g_dens=g_dens,beta_output=0)
 alpha_opt_MLE[[i]]<-alpha_results_MLE$minimum
 var_opt_MLE[[i]]<-alpha_results_MLE$objective-1

  aa0<-as.matrix(data_all[[2]][1:n,])
 
 # proposal Normal dens
 q2_dens_0<-norm_dens_joint(aa0,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_0_part1<-norm_dens_joint(aa0,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_0_part2<-t_dens_joint(aa0,mu_t_pi,df_t_pi)
 
 pi_dens_0<-alpha_pi*pi_dens_0_part1+(1-alpha_pi)*pi_dens_0_part2

 var_0_DIS_results<-var_DIS_est(0,q1_dens,q2_dens_0,pi_dens_0,q_gamma,0)
 var_0_DIS[[i]]<-var_0_DIS_results
}
proc.time()-save.time

c(0.5,var_half_DIS=mean(unlist(var_half_DIS)))
c(mean(unlist(alpha_opt_DIS)),var_opt_DIS=mean(unlist(var_opt_DIS)))
c(0.5,var_half_MLE=mean(unlist(var_half_MLE)))
c(mean(unlist(alpha_opt_MLE)),var_opt_MLE=mean(unlist(var_opt_MLE)))
c(0,var_0_DIS=mean(unlist(var_0_DIS)))
}

############################################
# two-stage performance B1
# use var_est, 400 for 1st stage
{###########################################
n=4000; replic=1000; dimension<-10; n0<-400

mu_norm_pi<-rep(0,10); sigma_norm_pi<-rep(1,10);
mu_norm_q2<-rep(0,10); sigma_norm_q2<-rep(1,10); 
mu_t_pi<-0; df_t_pi<-4; 
mu_t_q1<-0; df_t_q1<-1


# fix mixture proportions
estimation_MLE<-function(alpha){
 n1<-floor(n*alpha); n2<-floor(n*(1-alpha));
 
 x1<-data_all[[1]][1:n1,]
 x2<-data_all[[2]][1:n2,]
 x<-rbind(x1,x2)
 
 # proposal t dens
 q1_dens<-t_dens_joint(x,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(x,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_part1<-norm_dens_joint(x,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_part2<-t_dens_joint(x,mu_t_pi,df_t_pi)
 
 pi_dens<-0.8*pi_dens_part1+0.2*pi_dens_part2
 
 q_alpha<-alpha*q1_dens+(1-alpha)*q2_dens
 g_dens<-q1_dens-q2_dens

 dens_all<-list(q1_dens,q2_dens)
 zeta_optim_results<-optimize(l_fun_neg,interval=c(-1,1),dens_all=dens_all,prop=alpha)
 zeta_optim<-zeta_optim_results$minimum
 estimator_MLE<-mean(pi_dens/((alpha-zeta_optim)*q1_dens+(1-alpha+zeta_optim)*q2_dens))

 return(estimator_MLE)
}

estimation_DIS<-function(alpha){
 n1<-floor(n*alpha); n2<-floor(n*(1-alpha));
 
 x1<-data_all[[1]][1:n1,]
 x2<-data_all[[2]][1:n2,]
 x<-rbind(x1,x2)
 
 # proposal t dens
 q1_dens<-t_dens_joint(x,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(x,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_part1<-norm_dens_joint(x,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_part2<-t_dens_joint(x,mu_t_pi,df_t_pi)
 
 pi_dens<-0.8*pi_dens_part1+0.2*pi_dens_part2
 
 q_alpha<-alpha*q1_dens+(1-alpha)*q2_dens
 g_dens<-q1_dens-q2_dens

 estimator_DIS<-mean(pi_dens/q_alpha)

 return(estimator_DIS)
}

estimation_q2<-function(n){ 
 x2<-data_all[[2]][1:n,]
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(x2,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_part1<-norm_dens_joint(x2,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_part2<-t_dens_joint(x2,mu_t_pi,df_t_pi)
 
 pi_dens<-0.8*pi_dens_part1+0.2*pi_dens_part2

 estimator_DIS<-mean(pi_dens/q2_dens)

 return(estimator_DIS)
}

# two-stage mixture proportions 
estimation_MLE_twostages<-function(n0,gamma_vec){
 n01<-floor(n0*gamma_vec[1]); n02<-floor(n0*gamma_vec[2]);
 
 x11<-as.matrix(data_all[[1]][1:n01,])
 x12<-as.matrix(data_all[[2]][1:n02,])
 x1<-rbind(x11,x12)
 
 # proposal t dens
 q1_dens_stage1<-t_dens_joint(x1,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage1<-norm_dens_joint(x1,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage1_part1<-norm_dens_joint(x1,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_stage1_part2<-t_dens_joint(x1,mu_t_pi,df_t_pi)
 
 pi_dens_stage1<-0.8*pi_dens_stage1_part1+0.2*pi_dens_stage1_part2
 q_gamma_stage1<-gamma_vec[1]*q1_dens_stage1+gamma_vec[2]*q2_dens_stage1
 g_dens_stage1<-q1_dens_stage1-q2_dens_stage1

 alpha_hat_MLE<-optimize(var_MLE_est,interval=c(delta_lowbound,1),q1_dens=q1_dens_stage1,q2_dens=q2_dens_stage1,pi_dens=pi_dens_stage1,q_gamma=q_gamma_stage1,g_dens=g_dens_stage1,beta_output=0)$minimum

 n11<-floor((n-n0)*alpha_hat_MLE); n12<-floor((n-n0)*(1-alpha_hat_MLE));
 if(n11>1) x21<-as.matrix(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==1) x21<-t(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==0) x21<-NULL
 if(n12>1) x22<-as.matrix(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==1) x22<-t(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==0) x22<-NULL
 x2<-rbind(x21,x22)
 
 # proposal t dens
 q1_dens_stage2<-t_dens_joint(x2,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage2<-norm_dens_joint(x2,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage2_part1<-norm_dens_joint(x2,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_stage2_part2<-t_dens_joint(x2,mu_t_pi,df_t_pi)
 
 pi_dens_stage2<-0.8*pi_dens_stage2_part1+0.2*pi_dens_stage2_part2
 
 alpha_tilde<-n0/n*gamma_vec[1]+(n-n0)/n*alpha_hat_MLE
 q1_dens<-c(q1_dens_stage1,q1_dens_stage2)
 q2_dens<-c(q2_dens_stage1,q2_dens_stage2)
 pi_dens<-c(pi_dens_stage1,pi_dens_stage2)
 q_alpha<-alpha_tilde*q1_dens+(1-alpha_tilde)*q2_dens
 g_dens<-q1_dens-q2_dens
 
 dens_all<-list(q1_dens,q2_dens)
 zeta_optim_results<-optimize(l_fun_neg,interval=c(-1,1),dens_all=dens_all,prop=alpha_tilde)
 zeta_optim<-zeta_optim_results$minimum
 estimator_MLE<-mean(pi_dens/((alpha_tilde-zeta_optim)*q1_dens+(1-alpha_tilde+zeta_optim)*q2_dens))

 return(c(alpha_hat_MLE,estimator_MLE))
}

estimation_DIS_twostages<-function(n0,gamma_vec){
 n01<-floor(n0*gamma_vec[1]); n02<-floor(n0*gamma_vec[2]);
 
 x11<-as.matrix(data_all[[1]][1:n01,])
 x12<-as.matrix(data_all[[2]][1:n02,])
 x1<-rbind(x11,x12)
 
 # proposal t dens
 q1_dens_stage1<-t_dens_joint(x1,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage1<-norm_dens_joint(x1,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage1_part1<-norm_dens_joint(x1,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_stage1_part2<-t_dens_joint(x1,mu_t_pi,df_t_pi)
 
 pi_dens_stage1<-0.8*pi_dens_stage1_part1+0.2*pi_dens_stage1_part2
 q_gamma_stage1<-gamma_vec[1]*q1_dens_stage1+gamma_vec[2]*q2_dens_stage1
 g_dens_stage1<-q1_dens_stage1-q2_dens_stage1

 alpha_hat_DIS<-optimize(var_DIS_est,interval=c(0,1),q1_dens=q1_dens_stage1,q2_dens=q2_dens_stage1,pi_dens=pi_dens_stage1,q_gamma=q_gamma_stage1,n1=n01)$minimum

 n11<-floor((n-n0)*alpha_hat_DIS); n12<-floor((n-n0)*(1-alpha_hat_DIS));
 if(n11>1) x21<-as.matrix(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==1) x21<-t(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==0) x21<-NULL
 if(n12>1) x22<-as.matrix(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==1) x22<-t(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==0) x22<-NULL
 x2<-rbind(x21,x22)
 
 # proposal t dens
 q1_dens_stage2<-t_dens_joint(x2,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage2<-norm_dens_joint(x2,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage2_part1<-norm_dens_joint(x2,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_stage2_part2<-t_dens_joint(x2,mu_t_pi,df_t_pi)
 
 pi_dens_stage2<-0.8*pi_dens_stage2_part1+0.2*pi_dens_stage2_part2
 
 alpha_tilde<-n0/n*gamma_vec[1]+(n-n0)/n*alpha_hat_DIS
 q1_dens<-c(q1_dens_stage1,q1_dens_stage2)
 q2_dens<-c(q2_dens_stage1,q2_dens_stage2)
 pi_dens<-c(pi_dens_stage1,pi_dens_stage2)
 q_alpha<-alpha_tilde*q1_dens+(1-alpha_tilde)*q2_dens
 g_dens<-q1_dens-q2_dens
 
 estimator_DIS<-mean(pi_dens/q_alpha)
  
 return(c(alpha_hat_DIS,estimator_DIS))
}


est_DIS<-0
est_MLE<-0
est_q2<-0
alpha_hat_DIS_twostages<-0
est_DIS_twostages<-0
alpha_hat_MLE_twostages<-0
est_MLE_twostages<-0

save.time<-proc.time()
for(i in 1:replic){
  x1<-matrix(rt(n*dimension,df_t_q1)+mu_t_q1,ncol=dimension)
  x2<-mvrnorm(n,mu=mu_norm_q2,Sigma=diag(sigma_norm_q2^2))
  data_all<-list(x1,x2)
  
  results_est_DIS<-estimation_DIS(0.5)
  est_DIS[i]<-results_est_DIS
  
  results_est_DIS_twostages<-estimation_DIS_twostages(n0,c(0.5,0.5))
  alpha_hat_DIS_twostages[i]<-results_est_DIS_twostages[1]
  est_DIS_twostages[i]<-results_est_DIS_twostages[2]

  results_est_MLE<-estimation_MLE(0.5) 
  est_MLE[i]<-results_est_MLE

  results_est_MLE_twostages<-estimation_MLE_twostages(n0,c(0.5,0.5))
  alpha_hat_MLE_twostages[i]<-results_est_MLE_twostages[1]
  est_MLE_twostages[i]<-results_est_MLE_twostages[2]
}
proc.time()-save.time

for(i in 1:replic){
  x2<-mvrnorm(n,mu=mu_norm_q2,Sigma=diag(sigma_norm_q2^2))

  results_est_0<-estimation_q2(n)
  est_q2[i]<-results_est_0
}

alpha_hat_MLE<-mean(alpha_hat_MLE_twostages)
var_MLE_twostages<-var(est_MLE_twostages)*n
mean_MLE_twostages<-mean(est_MLE_twostages)

alpha_hat_DIS<-mean(alpha_hat_DIS_twostages)
var_DIS_twostages<-var(est_DIS_twostages)*n
mean_DIS_twostages<-mean(est_DIS_twostages)
 
var_DIS<-var(est_DIS)*n 
mean_DIS<-mean(est_DIS)

var_MLE<-var(est_MLE)*n
mean_MLE<-mean(est_MLE)

results<-matrix(c(0.5,0.5,alpha_hat_DIS,alpha_hat_MLE,var_DIS,var_MLE,var_DIS_twostages,var_MLE_twostages,mean_DIS,mean_MLE,mean_DIS_twostages,mean_MLE_twostages),4,3)
colnames(results)<-c("alpha","n*var","mean")
rownames(results)<-c("DIS","MLE","DIS_twostage","MLE_twostage")
results

}

#####################################################
# asymptotic performance B1 expectation
{####################################################
mu_norm_pi<-rep(0,10); sigma_norm_pi<-rep(1,10);
mu_norm_q2<-rep(0,10); sigma_norm_q2<-rep(1,10); 
mu_t_pi<-0; df_t_pi<-4; 
mu_t_q1<-0; df_t_q1<-1

n=2000000; dimension<-10; replic<-10; alpha_pi<-.8

var_half_DIS_expec<-list(0)
alpha_opt_DIS_expec<-list(0)
var_opt_DIS_expec<-list(0)
var_half_MLE_expec<-list(0)
alpha_opt_MLE_expec<-list(0)
var_opt_MLE_expec<-list(0)
var_0_DIS_expec<-list(0)

gamma_vec<-c(.5,.5)
n1<-n*gamma_vec[1]
n2<-n*gamma_vec[2]

for(i in 1:replic){
 x1<-matrix(rt(n*dimension,df_t_q1)+mu_t_q1,ncol=dimension)  
 x2<-mvrnorm(n,mu=mu_norm_q2,Sigma=diag(sigma_norm_q2^2))
 data_all<-list(0,0)
 data_all[[1]]<-x1
 data_all[[2]]<-x2

 aa1<-as.matrix(data_all[[1]][1:n1,])
 aa2<-as.matrix(data_all[[2]][1:n2,])
 x<-rbind(aa1,aa2)

  # proposal t dens
 q1_dens<-t_dens_joint(x,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(x,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_part1<-norm_dens_joint(x,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_part2<-t_dens_joint(x,mu_t_pi,df_t_pi)

 pi_dens<-0.8*pi_dens_part1+0.2*pi_dens_part2
 q_gamma<-gamma_vec[1]*q1_dens+gamma_vec[2]*q2_dens
 g_dens<-q1_dens-q2_dens
 h<-rowSums(x^2)/dimension
 temp<-!((pi_dens==0)&(q_gamma==0))
 mu_hat<-mean((h*pi_dens/q_gamma)[temp])/mean((pi_dens/q_gamma)[temp])

 var_half_DIS_expec_results<-var_DIS_expec_est(.5,h,mu_hat,q1_dens,q2_dens,pi_dens,q_gamma,n1)
 var_half_DIS_expec[[i]]<-var_half_DIS_expec_results
 
 alpha_results_DIS_expec<-optimize(var_DIS_expec_est,interval=c(delta_lowbound,1),h=h,mu_hat=mu_hat,q1_dens=q1_dens,q2_dens=q2_dens,pi_dens=pi_dens,q_gamma=q_gamma,n1=n1)
 alpha_opt_DIS_expec[[i]]<-alpha_results_DIS_expec$minimum
 var_opt_DIS_expec[[i]]<-alpha_results_DIS_expec$objective

 var_half_MLE_expec_results<-var_MLE_expec_est(.5,h,mu_hat,q1_dens,q2_dens,pi_dens,q_gamma,g_dens,0)
 var_half_MLE_expec[[i]]<-var_half_MLE_expec_results
  
 alpha_results_MLE_expec<-optimize(var_MLE_expec_est,interval=c(delta_lowbound,1),h=h,mu_hat=mu_hat,q1_dens=q1_dens,q2_dens=q2_dens,pi_dens=pi_dens,q_gamma=q_gamma,g_dens=g_dens,beta_output=0)
 alpha_opt_MLE_expec[[i]]<-alpha_results_MLE_expec$minimum
 var_opt_MLE_expec[[i]]<-alpha_results_MLE_expec$objective

 aa0<-as.matrix(data_all[[2]][1:n,])
 h0<-rowSums(aa0^2)/dimension

 # proposal Normal dens
 q2_dens_0<-norm_dens_joint(aa0,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_0_part1<-norm_dens_joint(aa0,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_0_part2<-t_dens_joint(aa0,mu_t_pi,df_t_pi)
 
 pi_dens_0<-alpha_pi*pi_dens_0_part1+(1-alpha_pi)*pi_dens_0_part2

 mu_hat0<-mean(h0*pi_dens_0/q2_dens_0)/mean(pi_dens_0/q2_dens_0)

 var_0_DIS_expec_results<-var_DIS_expec_est(0,h0,mu_hat0,q1_dens,q2_dens_0,pi_dens_0,q_gamma,0)
 var_0_DIS_expec[[i]]<-var_0_DIS_expec_results
}

c(0.5,var_half_DIS_expec=mean(unlist(var_half_DIS_expec)))
c(mean(unlist(alpha_opt_DIS_expec)),var_opt_DIS_expec=mean(unlist(var_opt_DIS_expec)))
c(0.5,var_half_MLE_expec=mean(unlist(var_half_MLE_expec)))
c(mean(unlist(alpha_opt_MLE_expec)),var_opt_MLE_expec=mean(unlist(var_opt_MLE_expec)))
c(0,var_0_DIS_expec=mean(unlist(var_0_DIS_expec)))

}

##############################################################
# two-stage performance B1 expectation
# use var_est, 400 for 1st stage
{#############################################################
n=4000; replic=1000; dimension<-10; n0<-400

mu_norm_pi<-rep(0,10); sigma_norm_pi<-rep(1,10);
mu_norm_q2<-rep(0,10); sigma_norm_q2<-rep(1,10); 
mu_t_pi<-0; df_t_pi<-4; 
mu_t_q1<-0; df_t_q1<-1


# fix mixture proportions
estimation_MLE<-function(alpha){
 n1<-floor(n*alpha); n2<-floor(n*(1-alpha));
 
 x1<-data_all[[1]][1:n1,]
 x2<-data_all[[2]][1:n2,]
 x<-rbind(x1,x2)

 # proposal t dens
 q1_dens<-t_dens_joint(x,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(x,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_part1<-norm_dens_joint(x,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_part2<-t_dens_joint(x,mu_t_pi,df_t_pi)
 
 pi_dens<-0.8*pi_dens_part1+0.2*pi_dens_part2
 
 q_alpha<-alpha*q1_dens+(1-alpha)*q2_dens
 g_dens<-q1_dens-q2_dens
 h<-rowSums(x^2)/dimension

 dens_all<-list(q1_dens,q2_dens)
 zeta_optim_results<-optimize(l_fun_neg,interval=c(-1,1),dens_all=dens_all,prop=alpha)
 zeta_optim<-zeta_optim_results$minimum
 q_alpha_zeta<-(alpha-zeta_optim)*q1_dens+(1-alpha+zeta_optim)*q2_dens
 estimator_MLE<-mean(h*pi_dens/q_alpha_zeta)/mean(pi_dens/q_alpha_zeta) 

 return(estimator_MLE)
}

estimation_DIS<-function(alpha){
 n1<-floor(n*alpha); n2<-floor(n*(1-alpha));
 
 x1<-data_all[[1]][1:n1,]
 x2<-data_all[[2]][1:n2,]
 x<-rbind(x1,x2)
 
 # proposal t dens
 q1_dens<-t_dens_joint(x,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(x,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_part1<-norm_dens_joint(x,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_part2<-t_dens_joint(x,mu_t_pi,df_t_pi)
 
 pi_dens<-0.8*pi_dens_part1+0.2*pi_dens_part2
 
 q_alpha<-alpha*q1_dens+(1-alpha)*q2_dens
 g_dens<-q1_dens-q2_dens
 h<-rowSums(x^2)/dimension

 estimator_DIS<-mean(h*pi_dens/q_alpha)/mean(pi_dens/q_alpha)

 return(estimator_DIS)
}

estimation_q2<-function(n){ 
 x2<-data_all[[2]][1:n,]
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(x2,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_part1<-norm_dens_joint(x2,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_part2<-t_dens_joint(x2,mu_t_pi,df_t_pi)
 
 pi_dens<-0.8*pi_dens_part1+0.2*pi_dens_part2
 h<-rowSums(x2^2)/dimension

 estimator_DIS<-mean(h*pi_dens/q2_dens)/mean(pi_dens/q2_dens)

 return(estimator_DIS)
}


# two-stage mixture proportions 
estimation_MLE_twostages<-function(n0,gamma_vec){
 n01<-floor(n0*gamma_vec[1]); n02<-floor(n0*gamma_vec[2]);
 
 x11<-as.matrix(data_all[[1]][1:n01,])
 x12<-as.matrix(data_all[[2]][1:n02,])
 x1<-rbind(x11,x12)

 # proposal t dens
 q1_dens_stage1<-t_dens_joint(x1,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage1<-norm_dens_joint(x1,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage1_part1<-norm_dens_joint(x1,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_stage1_part2<-t_dens_joint(x1,mu_t_pi,df_t_pi)
 
 pi_dens_stage1<-0.8*pi_dens_stage1_part1+0.2*pi_dens_stage1_part2
 q_gamma_stage1<-gamma_vec[1]*q1_dens_stage1+gamma_vec[2]*q2_dens_stage1
 g_dens_stage1<-q1_dens_stage1-q2_dens_stage1

 temp<-!((pi_dens_stage1==0)&(q_gamma_stage1==0))
 h_stage1<-rowSums(x1^2)/dimension
 mu_hat_stage1<-mean((h_stage1*pi_dens_stage1/q_gamma_stage1)[temp])/mean((pi_dens_stage1/q_gamma_stage1)[temp])
 alpha_hat_MLE<-optimize(var_MLE_expec_est,interval=c(delta_lowbound,1),h=h_stage1,mu_hat=mu_hat_stage1,q1_dens=q1_dens_stage1,q2_dens=q2_dens_stage1,pi_dens=pi_dens_stage1,q_gamma=q_gamma_stage1,g_dens=g_dens_stage1,beta_output=0)$minimum
  
 n11<-floor((n-n0)*alpha_hat_MLE); n12<-floor((n-n0)*(1-alpha_hat_MLE));
 if(n11>1) x21<-as.matrix(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==1) x21<-t(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==0) x21<-NULL
 if(n12>1) x22<-as.matrix(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==1) x22<-t(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==0) x22<-NULL
 x2<-rbind(x21,x22)

 # proposal t dens
 q1_dens_stage2<-t_dens_joint(x2,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage2<-norm_dens_joint(x2,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage2_part1<-norm_dens_joint(x2,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_stage2_part2<-t_dens_joint(x2,mu_t_pi,df_t_pi)

 pi_dens_stage2<-0.8*pi_dens_stage2_part1+0.2*pi_dens_stage2_part2

 h<-rowSums(rbind(x11,x12,x21,x22)^2)/dimension
 
 alpha_tilde<-n0/n*gamma_vec[1]+(n-n0)/n*alpha_hat_MLE
 q1_dens<-c(q1_dens_stage1,q1_dens_stage2)
 q2_dens<-c(q2_dens_stage1,q2_dens_stage2)
 pi_dens<-c(pi_dens_stage1,pi_dens_stage2)
 q_alpha<-alpha_tilde*q1_dens+(1-alpha_tilde)*q2_dens
 g_dens<-q1_dens-q2_dens
 
 dens_all<-list(q1_dens,q2_dens)
 zeta_optim_results<-optimize(l_fun_neg,interval=c(-1,1),dens_all=dens_all,prop=alpha_tilde)
 zeta_optim<-zeta_optim_results$minimum
 q_alpha_zeta<-(alpha_tilde-zeta_optim)*q1_dens+(1-alpha_tilde+zeta_optim)*q2_dens
 estimator_MLE<-mean(h*pi_dens/q_alpha_zeta)/mean(pi_dens/q_alpha_zeta)

 return(c(alpha_hat_MLE,estimator_MLE))
}

estimation_DIS_twostages<-function(n0,gamma_vec){
 n01<-floor(n0*gamma_vec[1]); n02<-floor(n0*gamma_vec[2]);
 
 x11<-as.matrix(data_all[[1]][1:n01,])
 x12<-as.matrix(data_all[[2]][1:n02,])
 x1<-rbind(x11,x12)

 # proposal t dens
 q1_dens_stage1<-t_dens_joint(x1,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage1<-norm_dens_joint(x1,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage1_part1<-norm_dens_joint(x1,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_stage1_part2<-t_dens_joint(x1,mu_t_pi,df_t_pi)
 
 pi_dens_stage1<-0.8*pi_dens_stage1_part1+0.2*pi_dens_stage1_part2
 q_gamma_stage1<-gamma_vec[1]*q1_dens_stage1+gamma_vec[2]*q2_dens_stage1
 g_dens_stage1<-q1_dens_stage1-q2_dens_stage1

 temp<-!((pi_dens_stage1==0)&(q_gamma_stage1==0))
 h_stage1<-rowSums(x1^2)/dimension
 mu_hat_stage1<-mean((h_stage1*pi_dens_stage1/q_gamma_stage1)[temp])/mean((pi_dens_stage1/q_gamma_stage1)[temp])
  
 alpha_hat_DIS<-optimize(var_DIS_expec_est,interval=c(delta_lowbound,1),h=h_stage1,mu_hat=mu_hat_stage1,q1_dens=q1_dens_stage1,q2_dens=q2_dens_stage1,pi_dens=pi_dens_stage1,q_gamma=q_gamma_stage1,n1=n01)$minimum

 n11<-floor((n-n0)*alpha_hat_DIS); n12<-floor((n-n0)*(1-alpha_hat_DIS));
 if(n11>1) x21<-as.matrix(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==1) x21<-t(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==0) x21<-NULL
 if(n12>1) x22<-as.matrix(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==1) x22<-t(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==0) x22<-NULL
 x2<-rbind(x21,x22)

 # proposal t dens
 q1_dens_stage2<-t_dens_joint(x2,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage2<-norm_dens_joint(x2,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage2_part1<-norm_dens_joint(x2,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_stage2_part2<-t_dens_joint(x2,mu_t_pi,df_t_pi)

 pi_dens_stage2<-0.8*pi_dens_stage2_part1+0.2*pi_dens_stage2_part2

 h<-rowSums(rbind(x11,x12,x21,x22)^2)/dimension
 
 alpha_tilde<-n0/n*gamma_vec[1]+(n-n0)/n*alpha_hat_DIS
 q1_dens<-c(q1_dens_stage1,q1_dens_stage2)
 q2_dens<-c(q2_dens_stage1,q2_dens_stage2)
 pi_dens<-c(pi_dens_stage1,pi_dens_stage2)
 q_alpha<-alpha_tilde*q1_dens+(1-alpha_tilde)*q2_dens
 g_dens<-q1_dens-q2_dens
 
 estimator_DIS<-mean(h*pi_dens/q_alpha)/mean(pi_dens/q_alpha)

 return(c(alpha_hat_DIS,estimator_DIS))
}

est_DIS<-0
est_MLE<-0
est_q2<-0
alpha_hat_DIS_twostages<-0
est_DIS_twostages<-0
alpha_hat_MLE_twostages<-0
est_MLE_twostages<-0

save.time<-proc.time()
for(i in 1:replic){
  x1<-matrix(rt(n*dimension,df_t_q1)+mu_t_q1,ncol=dimension)
  x2<-mvrnorm(n,mu=mu_norm_q2,Sigma=diag(sigma_norm_q2^2))
  data_all<-list(x1,x2)
  
  results_est_DIS<-estimation_DIS(0.5)
  est_DIS[i]<-results_est_DIS
  
  results_est_DIS_twostages<-estimation_DIS_twostages(n0,c(0.5,0.5))
  alpha_hat_DIS_twostages[i]<-results_est_DIS_twostages[1]
  est_DIS_twostages[i]<-results_est_DIS_twostages[2]

  results_est_MLE<-estimation_MLE(0.5) 
  est_MLE[i]<-results_est_MLE

  results_est_MLE_twostages<-estimation_MLE_twostages(n0,c(0.5,0.5))
  alpha_hat_MLE_twostages[i]<-results_est_MLE_twostages[1]
  est_MLE_twostages[i]<-results_est_MLE_twostages[2]
}
proc.time()-save.time

for(i in 1:replic){
  x2<-mvrnorm(n,mu=mu_norm_q2,Sigma=diag(sigma_norm_q2^2))
  data_all<-list(NULL,x2)

  results_est_0<-estimation_q2(n)
  est_q2[i]<-results_est_0
}

alpha_hat_MLE<-mean(alpha_hat_MLE_twostages)
var_MLE_twostages<-var(est_MLE_twostages)*n
mean_MLE_twostages<-mean(est_MLE_twostages)

alpha_hat_DIS<-mean(alpha_hat_DIS_twostages)
var_DIS_twostages<-var(est_DIS_twostages)*n
mean_DIS_twostages<-mean(est_DIS_twostages)
 
var_DIS<-var(est_DIS)*n 
mean_DIS<-mean(est_DIS)

var_MLE<-var(est_MLE)*n
mean_MLE<-mean(est_MLE)

var_q2<-var(est_q2)*n
mean_q2<-mean(est_q2)

results<-matrix(c(0,0.5,0.5,alpha_hat_DIS,alpha_hat_MLE,var_q2,var_DIS,var_MLE,var_DIS_twostages,var_MLE_twostages,mean_q2,mean_DIS,mean_MLE,mean_DIS_twostages,mean_MLE_twostages),ncol=3)
colnames(results)<-c("alpha","n*var","mean")
rownames(results)<-c("q2","DIS","MLE","DIS_twostage","MLE_twostage")
results

}

####################################
# asymptotic performance B2 
{###################################
mu_norm_pi<-rep(0,10); sigma_norm_pi<-rep(1,10); #sigma_norm_pi<-rep(0.8,10);
mu_norm_q2<-rep(0,10); sigma_norm_q2<-rep(1,10); 
mu_t_pi<-0; df_t_pi<-4; 
mu_t_q1<-0; df_t_q1<-2

n=2000000; dimension<-10; replic<-10; alpha_pi<-.8

var_half_DIS<-list(0)
alpha_opt_DIS<-list(0)
var_opt_DIS<-list(0)
var_half_MLE<-list(0)
alpha_opt_MLE<-list(0)
var_opt_MLE<-list(0)
var_0_DIS<-list(0)

gamma_vec<-c(.5,.5)
n1<-n*gamma_vec[1]
n2<-n*gamma_vec[2]

 save.time<-proc.time()
 proc.time()-save.time

for(i in 1:replic){
 x1<-matrix(rt(n*dimension,df_t_q1)+mu_t_q1,ncol=dimension)  
 x2<-mvrnorm(n,mu=mu_norm_q2,Sigma=diag(sigma_norm_q2^2))
 data_all<-list(0,0)
 data_all[[1]]<-x1
 data_all[[2]]<-x2

 aa1<-as.matrix(data_all[[1]][1:n1,])
 aa2<-as.matrix(data_all[[2]][1:n2,])
 x<-rbind(aa1,aa2)
 
 # proposal t dens
 q1_dens<-t_dens_joint(x,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(x,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_part1<-norm_dens_joint(x,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_part2<-t_dens_joint(x,mu_t_pi,df_t_pi)
 
 pi_dens<-0.8*pi_dens_part1+0.2*pi_dens_part2
 q_gamma<-gamma_vec[1]*q1_dens+gamma_vec[2]*q2_dens
 g_dens<-q1_dens-q2_dens
 
 var_half_DIS_results<-var_DIS_est(.5,q1_dens,q2_dens,pi_dens,q_gamma,n1)
 var_half_DIS[[i]]<-var_half_DIS_results
 
 alpha_results_DIS<-optimize(var_DIS_est,interval=c(delta_lowbound,1),q1_dens=q1_dens,q2_dens=q2_dens,pi_dens=pi_dens,q_gamma=q_gamma,n1=n1)
 alpha_opt_DIS[[i]]<-alpha_results_DIS$minimum
 var_opt_DIS[[i]]<-alpha_results_DIS$objective

 var_half_MLE_results<-var_MLE_est(.5,q1_dens,q2_dens,pi_dens,q_gamma,g_dens,0)
 var_half_MLE[[i]]<-var_half_MLE_results-1
  
 alpha_results_MLE<-optimize(var_MLE_est,interval=c(delta_lowbound,1),q1_dens=q1_dens,q2_dens=q2_dens,pi_dens=pi_dens,q_gamma=q_gamma,g_dens=g_dens,beta_output=0)
 alpha_opt_MLE[[i]]<-alpha_results_MLE$minimum
 var_opt_MLE[[i]]<-alpha_results_MLE$objective-1

  aa0<-as.matrix(data_all[[2]][1:n,])
 
 # proposal Normal dens
 q2_dens_0<-norm_dens_joint(aa0,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_0_part1<-norm_dens_joint(aa0,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_0_part2<-t_dens_joint(aa0,mu_t_pi,df_t_pi)
 
 pi_dens_0<-alpha_pi*pi_dens_0_part1+(1-alpha_pi)*pi_dens_0_part2

 var_0_DIS_results<-var_DIS_est(0,q1_dens,q2_dens_0,pi_dens_0,q_gamma,0)
 var_0_DIS[[i]]<-var_0_DIS_results
}

c(0.5,var_half_DIS=mean(unlist(var_half_DIS)))
c(mean(unlist(alpha_opt_DIS)),var_opt_DIS=mean(unlist(var_opt_DIS)))
c(0.5,var_half_MLE=mean(unlist(var_half_MLE)))
c(mean(unlist(alpha_opt_MLE)),var_opt_MLE=mean(unlist(var_opt_MLE)))
c(0,var_0_DIS=mean(unlist(var_0_DIS)))

}

############################################
# two-stage performance B2
# use var_est_1, 4000 for 1st stage
{###########################################
n=4000; replic=1000; dimension<-10; n0<-400

mu_norm_pi<-rep(0,10); sigma_norm_pi<-rep(1,10);
mu_norm_q2<-rep(0,10); sigma_norm_q2<-rep(1,10); 
mu_t_pi<-0; df_t_pi<-4; 
mu_t_q1<-0; df_t_q1<-2


# fix mixture proportions
estimation_MLE<-function(alpha){
 n1<-floor(n*alpha); n2<-floor(n*(1-alpha));
 
 x1<-data_all[[1]][1:n1,]
 x2<-data_all[[2]][1:n2,]
 x<-rbind(x1,x2)
 
 # proposal t dens
 q1_dens<-t_dens_joint(x,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(x,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_part1<-norm_dens_joint(x,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_part2<-t_dens_joint(x,mu_t_pi,df_t_pi)
 
 pi_dens<-0.8*pi_dens_part1+0.2*pi_dens_part2
 
 q_alpha<-alpha*q1_dens+(1-alpha)*q2_dens
 g_dens<-q1_dens-q2_dens

 dens_all<-list(q1_dens,q2_dens)
 zeta_optim_results<-optimize(l_fun_neg,interval=c(-1,1),dens_all=dens_all,prop=alpha)
 zeta_optim<-zeta_optim_results$minimum
 estimator_MLE<-mean(pi_dens/((alpha-zeta_optim)*q1_dens+(1-alpha+zeta_optim)*q2_dens))

 return(estimator_MLE)
}

estimation_DIS<-function(alpha){
 n1<-floor(n*alpha); n2<-floor(n*(1-alpha));
 
 x1<-data_all[[1]][1:n1,]
 x2<-data_all[[2]][1:n2,]
 x<-rbind(x1,x2)
 
 # proposal t dens
 q1_dens<-t_dens_joint(x,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(x,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_part1<-norm_dens_joint(x,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_part2<-t_dens_joint(x,mu_t_pi,df_t_pi)
 
 pi_dens<-0.8*pi_dens_part1+0.2*pi_dens_part2
 
 q_alpha<-alpha*q1_dens+(1-alpha)*q2_dens
 g_dens<-q1_dens-q2_dens

 estimator_DIS<-mean(pi_dens/q_alpha)

 return(estimator_DIS)
}

estimation_q2<-function(n){ 
 x2<-data_all[[2]][1:n,]
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(x2,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_part1<-norm_dens_joint(x2,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_part2<-t_dens_joint(x2,mu_t_pi,df_t_pi)
 
 pi_dens<-0.8*pi_dens_part1+0.2*pi_dens_part2

 estimator_DIS<-mean(pi_dens/q2_dens)

 return(estimator_DIS)
}

# two-stage mixture proportions 
estimation_MLE_twostages<-function(n0,gamma_vec){
 n01<-floor(n0*gamma_vec[1]); n02<-floor(n0*gamma_vec[2]);
 
 x11<-as.matrix(data_all[[1]][1:n01,])
 x12<-as.matrix(data_all[[2]][1:n02,])
 x1<-rbind(x11,x12)
 
 # proposal t dens
 q1_dens_stage1<-t_dens_joint(x1,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage1<-norm_dens_joint(x1,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage1_part1<-norm_dens_joint(x1,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_stage1_part2<-t_dens_joint(x1,mu_t_pi,df_t_pi)
 
 pi_dens_stage1<-0.8*pi_dens_stage1_part1+0.2*pi_dens_stage1_part2
 q_gamma_stage1<-gamma_vec[1]*q1_dens_stage1+gamma_vec[2]*q2_dens_stage1
 g_dens_stage1<-q1_dens_stage1-q2_dens_stage1

 alpha_hat_MLE<-optimize(var_MLE_est_1,interval=c(delta_lowbound,1),q1_dens=q1_dens_stage1,q2_dens=q2_dens_stage1,pi_dens=pi_dens_stage1,q_gamma=q_gamma_stage1,g_dens=g_dens_stage1,beta_output=0,n1=n01)$minimum

 n11<-floor((n-n0)*alpha_hat_MLE); n12<-floor((n-n0)*(1-alpha_hat_MLE));
 if(n11>1) x21<-as.matrix(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==1) x21<-t(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==0) x21<-NULL
 if(n12>1) x22<-as.matrix(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==1) x22<-t(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==0) x22<-NULL
 x2<-rbind(x21,x22)
 
 # proposal t dens
 q1_dens_stage2<-t_dens_joint(x2,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage2<-norm_dens_joint(x2,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage2_part1<-norm_dens_joint(x2,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_stage2_part2<-t_dens_joint(x2,mu_t_pi,df_t_pi)
 
 pi_dens_stage2<-0.8*pi_dens_stage2_part1+0.2*pi_dens_stage2_part2
 
 alpha_tilde<-n0/n*gamma_vec[1]+(n-n0)/n*alpha_hat_MLE
 q1_dens<-c(q1_dens_stage1,q1_dens_stage2)
 q2_dens<-c(q2_dens_stage1,q2_dens_stage2)
 pi_dens<-c(pi_dens_stage1,pi_dens_stage2)
 q_alpha<-alpha_tilde*q1_dens+(1-alpha_tilde)*q2_dens
 g_dens<-q1_dens-q2_dens
 
 dens_all<-list(q1_dens,q2_dens)
 zeta_optim_results<-optimize(l_fun_neg,interval=c(-1,1),dens_all=dens_all,prop=alpha_tilde)
 zeta_optim<-zeta_optim_results$minimum
 estimator_MLE<-mean(pi_dens/((alpha_tilde-zeta_optim)*q1_dens+(1-alpha_tilde+zeta_optim)*q2_dens))
  
 return(c(alpha_hat_MLE,estimator_MLE))
}

estimation_DIS_twostages<-function(n0,gamma_vec){
 n01<-floor(n0*gamma_vec[1]); n02<-floor(n0*gamma_vec[2]);
 
 x11<-as.matrix(data_all[[1]][1:n01,])
 x12<-as.matrix(data_all[[2]][1:n02,])
 x1<-rbind(x11,x12)
 
 # proposal t dens
 q1_dens_stage1<-t_dens_joint(x1,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage1<-norm_dens_joint(x1,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage1_part1<-norm_dens_joint(x1,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_stage1_part2<-t_dens_joint(x1,mu_t_pi,df_t_pi)
 
 pi_dens_stage1<-0.8*pi_dens_stage1_part1+0.2*pi_dens_stage1_part2
 q_gamma_stage1<-gamma_vec[1]*q1_dens_stage1+gamma_vec[2]*q2_dens_stage1
 g_dens_stage1<-q1_dens_stage1-q2_dens_stage1

 alpha_hat_DIS<-optimize(var_DIS_est,interval=c(0,1),q1_dens=q1_dens_stage1,q2_dens=q2_dens_stage1,pi_dens=pi_dens_stage1,q_gamma=q_gamma_stage1,n1=n01)$minimum

 n11<-floor((n-n0)*alpha_hat_DIS); n12<-floor((n-n0)*(1-alpha_hat_DIS));
 if(n11>1) x21<-as.matrix(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==1) x21<-t(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==0) x21<-NULL
 if(n12>1) x22<-as.matrix(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==1) x22<-t(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==0) x22<-NULL
 x2<-rbind(x21,x22)
 
 # proposal t dens
 q1_dens_stage2<-t_dens_joint(x2,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage2<-norm_dens_joint(x2,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage2_part1<-norm_dens_joint(x2,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_stage2_part2<-t_dens_joint(x2,mu_t_pi,df_t_pi)
 
 pi_dens_stage2<-0.8*pi_dens_stage2_part1+0.2*pi_dens_stage2_part2
 
 alpha_tilde<-n0/n*gamma_vec[1]+(n-n0)/n*alpha_hat_DIS
 q1_dens<-c(q1_dens_stage1,q1_dens_stage2)
 q2_dens<-c(q2_dens_stage1,q2_dens_stage2)
 pi_dens<-c(pi_dens_stage1,pi_dens_stage2)
 q_alpha<-alpha_tilde*q1_dens+(1-alpha_tilde)*q2_dens
 g_dens<-q1_dens-q2_dens
 
 estimator_DIS<-mean(pi_dens/q_alpha)
  
 return(c(alpha_hat_DIS,estimator_DIS))
}


est_DIS<-0
est_MLE<-0
est_q2<-0
alpha_hat_DIS_twostages<-0
est_DIS_twostages<-0
alpha_hat_MLE_twostages<-0
est_MLE_twostages<-0

for(i in 1:replic){
  x1<-matrix(rt(n*dimension,df_t_q1)+mu_t_q1,ncol=dimension)
  x2<-mvrnorm(n,mu=mu_norm_q2,Sigma=diag(sigma_norm_q2^2))
  data_all<-list(x1,x2)
  
  results_est_DIS<-estimation_DIS(0.5)
  est_DIS[i]<-results_est_DIS
  
  results_est_DIS_twostages<-estimation_DIS_twostages(n0,c(0.5,0.5))
  alpha_hat_DIS_twostages[i]<-results_est_DIS_twostages[1]
  est_DIS_twostages[i]<-results_est_DIS_twostages[2]

  results_est_MLE<-estimation_MLE(0.5) 
  est_MLE[i]<-results_est_MLE

  results_est_MLE_twostages<-estimation_MLE_twostages(n0,c(0.5,0.5))
  alpha_hat_MLE_twostages[i]<-results_est_MLE_twostages[1]
  est_MLE_twostages[i]<-results_est_MLE_twostages[2]
}

# proposal q2
for(i in 1:replic){
  x2<-mvrnorm(n,mu=mu_norm_q2,Sigma=diag(sigma_norm_q2^2))

  results_est_0<-estimation_q2(n)
  est_q2[i]<-results_est_0
}

alpha_hat_MLE<-mean(alpha_hat_MLE_twostages)
var_MLE_twostages<-var(est_MLE_twostages)*n
mean_MLE_twostages<-mean(est_MLE_twostages)

alpha_hat_DIS<-mean(alpha_hat_DIS_twostages)
var_DIS_twostages<-var(est_DIS_twostages)*n
mean_DIS_twostages<-mean(est_DIS_twostages)
 
var_DIS<-var(est_DIS)*n 
mean_DIS<-mean(est_DIS)

var_MLE<-var(est_MLE)*n
mean_MLE<-mean(est_MLE)

results<-matrix(c(0.5,0.5,alpha_hat_DIS,alpha_hat_MLE,var_DIS,var_MLE,var_DIS_twostages,var_MLE_twostages,mean_DIS,mean_MLE,mean_DIS_twostages,mean_MLE_twostages),4,3)
colnames(results)<-c("alpha","n*var","mean")
rownames(results)<-c("DIS","MLE","DIS_twostage","MLE_twostage")
results

}

#####################################################
# asymptotic performance B2 expectation
{####################################################
mu_norm_pi<-rep(0,10); sigma_norm_pi<-rep(1,10);
mu_norm_q2<-rep(0,10); sigma_norm_q2<-rep(1,10); 
mu_t_pi<-0; df_t_pi<-4; 
mu_t_q1<-0; df_t_q1<-2

n=2000000; dimension<-10; replic<-10; alpha_pi<-.8

var_half_DIS_expec<-list(0)
alpha_opt_DIS_expec<-list(0)
var_opt_DIS_expec<-list(0)
var_half_MLE_expec<-list(0)
alpha_opt_MLE_expec<-list(0)
var_opt_MLE_expec<-list(0)
var_0_DIS_expec<-list(0)

gamma_vec<-c(.5,.5)
n1<-n*gamma_vec[1]
n2<-n*gamma_vec[2]

for(i in 1:replic){
 x1<-matrix(rt(n*dimension,df_t_q1)+mu_t_q1,ncol=dimension)  
 x2<-mvrnorm(n,mu=mu_norm_q2,Sigma=diag(sigma_norm_q2^2))
 data_all<-list(0,0)
 data_all[[1]]<-x1
 data_all[[2]]<-x2

 aa1<-as.matrix(data_all[[1]][1:n1,])
 aa2<-as.matrix(data_all[[2]][1:n2,])
 x<-rbind(aa1,aa2)

  # proposal t dens
 q1_dens<-t_dens_joint(x,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(x,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_part1<-norm_dens_joint(x,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_part2<-t_dens_joint(x,mu_t_pi,df_t_pi)

 pi_dens<-0.8*pi_dens_part1+0.2*pi_dens_part2
 q_gamma<-gamma_vec[1]*q1_dens+gamma_vec[2]*q2_dens
 g_dens<-q1_dens-q2_dens
 h<-rowSums(x^2)/dimension
 temp<-!((pi_dens==0)&(q_gamma==0))
 mu_hat<-mean((h*pi_dens/q_gamma)[temp])/mean((pi_dens/q_gamma)[temp])

 var_half_DIS_expec_results<-var_DIS_expec_est(.5,h,mu_hat,q1_dens,q2_dens,pi_dens,q_gamma,n1)
 var_half_DIS_expec[[i]]<-var_half_DIS_expec_results
 
 alpha_results_DIS_expec<-optimize(var_DIS_expec_est,interval=c(delta_lowbound,1),h=h,mu_hat=mu_hat,q1_dens=q1_dens,q2_dens=q2_dens,pi_dens=pi_dens,q_gamma=q_gamma,n1=n1)
 alpha_opt_DIS_expec[[i]]<-alpha_results_DIS_expec$minimum
 var_opt_DIS_expec[[i]]<-alpha_results_DIS_expec$objective

 var_half_MLE_expec_results<-var_MLE_expec_est(.5,h,mu_hat,q1_dens,q2_dens,pi_dens,q_gamma,g_dens,0)
 var_half_MLE_expec[[i]]<-var_half_MLE_expec_results
  
 alpha_results_MLE_expec<-optimize(var_MLE_expec_est,interval=c(delta_lowbound,1),h=h,mu_hat=mu_hat,q1_dens=q1_dens,q2_dens=q2_dens,pi_dens=pi_dens,q_gamma=q_gamma,g_dens=g_dens,beta_output=0)
 alpha_opt_MLE_expec[[i]]<-alpha_results_MLE_expec$minimum
 var_opt_MLE_expec[[i]]<-alpha_results_MLE_expec$objective

 aa0<-as.matrix(data_all[[2]][1:n,])
 h0<-rowSums(aa0^2)/dimension
 
 # proposal Normal dens
 q2_dens_0<-norm_dens_joint(aa0,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_0_part1<-norm_dens_joint(aa0,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_0_part2<-t_dens_joint(aa0,mu_t_pi,df_t_pi)
 
 pi_dens_0<-alpha_pi*pi_dens_0_part1+(1-alpha_pi)*pi_dens_0_part2

 mu_hat0<-mean(h0*pi_dens_0/q2_dens_0)/mean(pi_dens_0/q2_dens_0)

 var_0_DIS_expec_results<-var_DIS_expec_est(0,h0,mu_hat0,q1_dens,q2_dens_0,pi_dens_0,q_gamma,0)
 var_0_DIS_expec[[i]]<-var_0_DIS_expec_results
}

c(0.5,var_half_DIS_expec=mean(unlist(var_half_DIS_expec)))
c(mean(unlist(alpha_opt_DIS_expec)),var_opt_DIS_expec=mean(unlist(var_opt_DIS_expec)))
c(0.5,var_half_MLE_expec=mean(unlist(var_half_MLE_expec)))
c(mean(unlist(alpha_opt_MLE_expec)),var_opt_MLE_expec=mean(unlist(var_opt_MLE_expec)))
c(0,var_0_DIS_expec=mean(unlist(var_0_DIS_expec)))

}

##############################################################
# two-stage performance B2 expectation
# use var_est, 400 for 1st stage
{#############################################################
n=4000; replic=1000; dimension<-10; n0<-400

mu_norm_pi<-rep(0,10); sigma_norm_pi<-rep(1,10);
mu_norm_q2<-rep(0,10); sigma_norm_q2<-rep(1,10); 
mu_t_pi<-0; df_t_pi<-4; 
mu_t_q1<-0; df_t_q1<-2


# fix mixture proportions
estimation_MLE<-function(alpha){
 n1<-floor(n*alpha); n2<-floor(n*(1-alpha));
 
 x1<-data_all[[1]][1:n1,]
 x2<-data_all[[2]][1:n2,]
 x<-rbind(x1,x2)

 # proposal t dens
 q1_dens<-t_dens_joint(x,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(x,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_part1<-norm_dens_joint(x,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_part2<-t_dens_joint(x,mu_t_pi,df_t_pi)
 
 pi_dens<-0.8*pi_dens_part1+0.2*pi_dens_part2
 
 q_alpha<-alpha*q1_dens+(1-alpha)*q2_dens
 g_dens<-q1_dens-q2_dens
 h<-rowSums(x^2)/dimension

 dens_all<-list(q1_dens,q2_dens)
 zeta_optim_results<-optimize(l_fun_neg,interval=c(-1,1),dens_all=dens_all,prop=alpha)
 zeta_optim<-zeta_optim_results$minimum
 q_alpha_zeta<-(alpha-zeta_optim)*q1_dens+(1-alpha+zeta_optim)*q2_dens
 estimator_MLE<-mean(h*pi_dens/q_alpha_zeta)/mean(pi_dens/q_alpha_zeta) 

 return(estimator_MLE)
}

estimation_DIS<-function(alpha){
 n1<-floor(n*alpha); n2<-floor(n*(1-alpha));
 
 x1<-data_all[[1]][1:n1,]
 x2<-data_all[[2]][1:n2,]
 x<-rbind(x1,x2)
 
 # proposal t dens
 q1_dens<-t_dens_joint(x,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(x,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_part1<-norm_dens_joint(x,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_part2<-t_dens_joint(x,mu_t_pi,df_t_pi)
 
 pi_dens<-0.8*pi_dens_part1+0.2*pi_dens_part2
 
 q_alpha<-alpha*q1_dens+(1-alpha)*q2_dens
 g_dens<-q1_dens-q2_dens
 h<-rowSums(x^2)/dimension

 estimator_DIS<-mean(h*pi_dens/q_alpha)/mean(pi_dens/q_alpha)

 return(estimator_DIS)
}

estimation_q2<-function(n){ 
 x2<-data_all[[2]][1:n,]
 
 # proposal Normal dens
 q2_dens<-norm_dens_joint(x2,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_part1<-norm_dens_joint(x2,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_part2<-t_dens_joint(x2,mu_t_pi,df_t_pi)
 
 pi_dens<-0.8*pi_dens_part1+0.2*pi_dens_part2
 h<-rowSums(x2^2)/dimension

 estimator_DIS<-mean(h*pi_dens/q2_dens)/mean(pi_dens/q2_dens)

 return(estimator_DIS)
}

for(i in 1:replic){
  x2<-mvrnorm(n,mu=mu_norm_q2,Sigma=diag(sigma_norm_q2^2))

  results_est_0<-estimation_q2(n)
  est_q2[i]<-results_est_0
}

# two-stage mixture proportions 
estimation_MLE_twostages<-function(n0,gamma_vec){
 n01<-floor(n0*gamma_vec[1]); n02<-floor(n0*gamma_vec[2]);
 
 x11<-as.matrix(data_all[[1]][1:n01,])
 x12<-as.matrix(data_all[[2]][1:n02,])
 x1<-rbind(x11,x12)

 # proposal t dens
 q1_dens_stage1<-t_dens_joint(x1,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage1<-norm_dens_joint(x1,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage1_part1<-norm_dens_joint(x1,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_stage1_part2<-t_dens_joint(x1,mu_t_pi,df_t_pi)
 
 pi_dens_stage1<-0.8*pi_dens_stage1_part1+0.2*pi_dens_stage1_part2
 q_gamma_stage1<-gamma_vec[1]*q1_dens_stage1+gamma_vec[2]*q2_dens_stage1
 g_dens_stage1<-q1_dens_stage1-q2_dens_stage1

 temp<-!((pi_dens_stage1==0)&(q_gamma_stage1==0))
 h_stage1<-rowSums(x1^2)/dimension
 mu_hat_stage1<-mean((h_stage1*pi_dens_stage1/q_gamma_stage1)[temp])/mean((pi_dens_stage1/q_gamma_stage1)[temp])
  
 alpha_hat_MLE<-optimize(var_MLE_expec_est,interval=c(delta_lowbound,1),h=h_stage1,mu_hat=mu_hat_stage1,q1_dens=q1_dens_stage1,q2_dens=q2_dens_stage1,pi_dens=pi_dens_stage1,q_gamma=q_gamma_stage1,g_dens=g_dens_stage1,beta_output=0)$minimum

 n11<-floor((n-n0)*alpha_hat_MLE); n12<-floor((n-n0)*(1-alpha_hat_MLE));
 if(n11>1) x21<-as.matrix(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==1) x21<-t(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==0) x21<-NULL
 if(n12>1) x22<-as.matrix(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==1) x22<-t(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==0) x22<-NULL
 x2<-rbind(x21,x22)

 # proposal t dens
 q1_dens_stage2<-t_dens_joint(x2,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage2<-norm_dens_joint(x2,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage2_part1<-norm_dens_joint(x2,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_stage2_part2<-t_dens_joint(x2,mu_t_pi,df_t_pi)

 pi_dens_stage2<-0.8*pi_dens_stage2_part1+0.2*pi_dens_stage2_part2

 h<-rowSums(rbind(x11,x12,x21,x22)^2)/dimension
 
 alpha_tilde<-n0/n*gamma_vec[1]+(n-n0)/n*alpha_hat_MLE
 q1_dens<-c(q1_dens_stage1,q1_dens_stage2)
 q2_dens<-c(q2_dens_stage1,q2_dens_stage2)
 pi_dens<-c(pi_dens_stage1,pi_dens_stage2)
 q_alpha<-alpha_tilde*q1_dens+(1-alpha_tilde)*q2_dens
 g_dens<-q1_dens-q2_dens
 
 dens_all<-list(q1_dens,q2_dens)
 zeta_optim_results<-optimize(l_fun_neg,interval=c(-1,1),dens_all=dens_all,prop=alpha_tilde)
 zeta_optim<-zeta_optim_results$minimum
 q_alpha_zeta<-(alpha_tilde-zeta_optim)*q1_dens+(1-alpha_tilde+zeta_optim)*q2_dens
 estimator_MLE<-mean(h*pi_dens/q_alpha_zeta)/mean(pi_dens/q_alpha_zeta)

 return(c(alpha_hat_MLE,estimator_MLE))
}

estimation_DIS_twostages<-function(n0,gamma_vec){
 n01<-floor(n0*gamma_vec[1]); n02<-floor(n0*gamma_vec[2]);
 
 x11<-as.matrix(data_all[[1]][1:n01,])
 x12<-as.matrix(data_all[[2]][1:n02,])
 x1<-rbind(x11,x12)

 # proposal t dens
 q1_dens_stage1<-t_dens_joint(x1,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage1<-norm_dens_joint(x1,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage1_part1<-norm_dens_joint(x1,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_stage1_part2<-t_dens_joint(x1,mu_t_pi,df_t_pi)
 
 pi_dens_stage1<-0.8*pi_dens_stage1_part1+0.2*pi_dens_stage1_part2
 q_gamma_stage1<-gamma_vec[1]*q1_dens_stage1+gamma_vec[2]*q2_dens_stage1
 g_dens_stage1<-q1_dens_stage1-q2_dens_stage1

 temp<-!((pi_dens_stage1==0)&(q_gamma_stage1==0))
 h_stage1<-rowSums(x1^2)/dimension
 mu_hat_stage1<-mean((h_stage1*pi_dens_stage1/q_gamma_stage1)[temp])/mean((pi_dens_stage1/q_gamma_stage1)[temp])
  
 alpha_hat_DIS<-optimize(var_DIS_expec_est,interval=c(delta_lowbound,1),h=h_stage1,mu_hat=mu_hat_stage1,q1_dens=q1_dens_stage1,q2_dens=q2_dens_stage1,pi_dens=pi_dens_stage1,q_gamma=q_gamma_stage1,n1=n01)$minimum

 n11<-floor((n-n0)*alpha_hat_DIS); n12<-floor((n-n0)*(1-alpha_hat_DIS));
 if(n11>1) x21<-as.matrix(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==1) x21<-t(data_all[[1]][(n01+1):(n01+n11),])
 if(n11==0) x21<-NULL
 if(n12>1) x22<-as.matrix(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==1) x22<-t(data_all[[2]][(n02+1):(n02+n12),])
 if(n12==0) x22<-NULL
 x2<-rbind(x21,x22)

 # proposal t dens
 q1_dens_stage2<-t_dens_joint(x2,mu_t_q1,df_t_q1)
 
 # proposal Normal dens
 q2_dens_stage2<-norm_dens_joint(x2,mu_norm_q2,sigma_norm_q2)

 # target normal dens  
 pi_dens_stage2_part1<-norm_dens_joint(x2,mu_norm_pi,sigma_norm_pi)

  # target t dens  
 pi_dens_stage2_part2<-t_dens_joint(x2,mu_t_pi,df_t_pi)

 pi_dens_stage2<-0.8*pi_dens_stage2_part1+0.2*pi_dens_stage2_part2

 h<-rowSums(rbind(x11,x12,x21,x22)^2)/dimension
 
 alpha_tilde<-n0/n*gamma_vec[1]+(n-n0)/n*alpha_hat_DIS
 q1_dens<-c(q1_dens_stage1,q1_dens_stage2)
 q2_dens<-c(q2_dens_stage1,q2_dens_stage2)
 pi_dens<-c(pi_dens_stage1,pi_dens_stage2)
 q_alpha<-alpha_tilde*q1_dens+(1-alpha_tilde)*q2_dens
 g_dens<-q1_dens-q2_dens
 
 estimator_DIS<-mean(h*pi_dens/q_alpha)/mean(pi_dens/q_alpha)

 return(c(alpha_hat_DIS,estimator_DIS))
}

est_DIS<-0
est_MLE<-0
est_q2<-0
alpha_hat_DIS_twostages<-0
est_DIS_twostages<-0
alpha_hat_MLE_twostages<-0
est_MLE_twostages<-0


save.time<-proc.time()
for(i in 1:replic){
  x1<-matrix(rt(n*dimension,df_t_q1)+mu_t_q1,ncol=dimension)
  x2<-mvrnorm(n,mu=mu_norm_q2,Sigma=diag(sigma_norm_q2^2))
  data_all<-list(x1,x2)
  
  results_est_DIS<-estimation_DIS(0.5)
  est_DIS[i]<-results_est_DIS
  
  results_est_DIS_twostages<-estimation_DIS_twostages(n0,c(0.5,0.5))
  alpha_hat_DIS_twostages[i]<-results_est_DIS_twostages[1]
  est_DIS_twostages[i]<-results_est_DIS_twostages[2]

  results_est_MLE<-estimation_MLE(0.5) 
  est_MLE[i]<-results_est_MLE

  results_est_MLE_twostages<-estimation_MLE_twostages(n0,c(0.5,0.5))
  alpha_hat_MLE_twostages[i]<-results_est_MLE_twostages[1]
  est_MLE_twostages[i]<-results_est_MLE_twostages[2]
}
proc.time()-save.time

# proposal q2
for(i in 1:replic){
  x2<-mvrnorm(n,mu=mu_norm_q2,Sigma=diag(sigma_norm_q2^2))

  results_est_0<-estimation_q2(n)
  est_q2[i]<-results_est_0
}

alpha_hat_MLE<-mean(alpha_hat_MLE_twostages)
var_MLE_twostages<-var(est_MLE_twostages)*n
mean_MLE_twostages<-mean(est_MLE_twostages)

alpha_hat_DIS<-mean(alpha_hat_DIS_twostages)
var_DIS_twostages<-var(est_DIS_twostages)*n
mean_DIS_twostages<-mean(est_DIS_twostages)
 
var_DIS<-var(est_DIS)*n 
mean_DIS<-mean(est_DIS)

var_MLE<-var(est_MLE)*n
mean_MLE<-mean(est_MLE)

var_q2<-var(est_q2)*n
mean_q2<-mean(est_q2)

results<-matrix(c(0.5,0.5,alpha_hat_DIS,alpha_hat_MLE,var_DIS,var_MLE,var_DIS_twostages,var_MLE_twostages,mean_DIS,mean_MLE,mean_DIS_twostages,mean_MLE_twostages),4,3)
colnames(results)<-c("alpha","n*var","mean")
rownames(results)<-c("DIS","MLE","DIS_twostage","MLE_twostage")
results

}


#######################
# Hesterberg's example
######################

# take random samples and proposal densities
gamma_dens_log<-function(x,shape,scale_gamma){
 c_gamma=log(gamma(shape))
 dens_log<--x/scale_gamma+(shape-1)*log(x)-c_gamma-shape*log(scale_gamma)
 return(dens_log)
}

gamma_tilt_dens_log<-function(x,shape,scale_gamma,alpha){
 c_gamma=log(gamma(shape))
 dens_log<--(alpha+1/scale_gamma)*x+(shape-1)*log(x)-c_gamma-shape*log(scale_gamma/(1+alpha*scale_gamma))
 return(dens_log)
}

exponen_dens_log<-function(x){ # truncated exponential
 c_exp=log(100*(exp(3)-1))
 dens_log<-x/100-c_exp
 return(dens_log)
}

exponen_tilt_dens_log<-function(x,alpha){ # truncated exponential tilted
 dens_log<-(1/100-alpha)*x+log((1/100-alpha)/(exp(3-300*alpha)-1))
 return(dens_log)
}

norm_dens_log<-function(x,mu,sigma){
 c_normal<-log(2*pi)/2+log(sigma)
 dens_log<--(x-mu)^2/2/sigma^2-c_normal
 return(dens_log)
}

norm_tilt_dens_log<-function(x,mu,sigma,alpha){
 c_normal<-log(2*pi)/2+log(sigma)
 dens_log<--(x-mu-alpha*sigma^2)^2/2/sigma^2-c_normal
 return(dens_log)
}

norm_temp_tilt_dens_log<-function(x,mu,sigma,alpha){
 c_normal<-log(2*pi)/2+log(sigma)
 dens_log<--(x-mu+alpha*(10+40)*sigma^2)^2/2/sigma^2-c_normal
 return(dens_log)
}


gamma_sample<-function(n,shape,scale_gamma){
 samples<-rgamma(n,shape=shape,scale=scale_gamma)
 return(samples)
}

gamma_tilt_sample<-function(n,shape,scale_gamma,alpha){
 samples<-rgamma(n,shape=shape,scale=1/(1/scale_gamma+alpha))
 return(samples)
}

exponen_sample<-function(n){
 samples_unif<-runif(n)
 samples<-300*log(samples_unif*(exp(3)-1)+1)/3
 return(samples)
}

exponen_tilt_sample<-function(n,alpha){
 samples_unif<-runif(n)
 samples<-300*log(samples_unif*(exp(3-alpha*300)-1)+1)/(3-alpha*300)
 return(samples)
}

norm_sample<-function(n,mu,sigma){
 samples<-rnorm(n,mu,sigma)
 return(samples)
}

norm_tilt_sample<-function(n,mu,sigma,alpha){
 samples<-rnorm(n,mu+alpha*sigma^2,sigma)
 return(samples)
}

norm_temp_tilt_sample<-function(n,mu,sigma,alpha){
 samples<-rnorm(n,mu-alpha*(10+40)*sigma^2,sigma)
 return(samples) 
}

sample_all<-function(n,par_m1,par_m2,par_m3){
 # output matrices are 3*n
 
 x_11_samples<-gamma_sample(n,par_m1[1,1],par_m1[1,2])
 x_12_samples<-gamma_sample(n,par_m2[1,1],par_m2[1,2])
 x_13_samples<-gamma_sample(n,par_m3[1,1],par_m3[1,2])
 
 x_21_samples<-exponen_sample(n)
 x_22_samples<-exponen_sample(n)
 x_23_samples<-exponen_sample(n)

 x_31_samples<-norm_sample(n,par_m1[3,1],par_m1[3,2])
 x_32_samples<-norm_sample(n,par_m2[3,1],par_m2[3,2])
 x_33_samples<-norm_sample(n,par_m3[3,1],par_m3[3,2])

 x_41_prime_samples<-norm_sample(n,par_m1[4,1],par_m1[4,2])
 x_42_prime_samples<-norm_sample(n,par_m2[4,1],par_m2[4,2])
 x_43_prime_samples<-norm_sample(n,par_m3[4,1],par_m3[4,2])
 x_41_samples<-x_41_prime_samples+10*(60-x_31_samples+abs(60-x_31_samples))/2
 x_42_samples<-x_42_prime_samples+10*(60-x_32_samples+abs(60-x_32_samples))/2
 x_43_samples<-x_43_prime_samples+10*(60-x_33_samples+abs(60-x_33_samples))/2

 x_51_prime_samples<-norm_sample(n,par_m1[5,1],par_m1[5,2])
 x_52_prime_samples<-norm_sample(n,par_m2[5,1],par_m2[5,2])
 x_53_prime_samples<-norm_sample(n,par_m3[5,1],par_m3[5,2])
 x_51_samples<-x_51_prime_samples+40*(60-x_31_samples+abs(60-x_31_samples))/2
 x_52_samples<-x_52_prime_samples+40*(60-x_32_samples+abs(60-x_32_samples))/2
 x_53_samples<-x_53_prime_samples+40*(60-x_33_samples+abs(60-x_33_samples))/2

 Hydro<-rbind(x_11_samples,x_12_samples,x_13_samples)
 Nuclear<-rbind(x_21_samples,x_22_samples,x_23_samples)
 Temperature<-rbind(x_31_samples,x_32_samples,x_33_samples)
 ElectricDemand<-rbind(x_41_samples,x_42_samples,x_43_samples)
 GasDemand<-rbind(x_51_samples,x_52_samples,x_53_samples)

 return(list(Hydro,Nuclear,Temperature,ElectricDemand,GasDemand))
}

sample_tilt_all<-function(n,par_m1,par_m2,par_m3,alpha){

 x_11_samples<-gamma_tilt_sample(n,par_m1[1,1],par_m1[1,2],alpha[1])
 x_12_samples<-gamma_tilt_sample(n,par_m2[1,1],par_m2[1,2],alpha[2])
 x_13_samples<-gamma_tilt_sample(n,par_m3[1,1],par_m3[1,2],alpha[3])
 
 x_21_samples<-exponen_tilt_sample(n,alpha[1])
 x_22_samples<-exponen_tilt_sample(n,alpha[2])
 x_23_samples<-exponen_tilt_sample(n,alpha[3])

 x_31_samples<-norm_temp_tilt_sample(n,par_m1[3,1],par_m1[3,2],alpha[1])
 x_32_samples<-norm_temp_tilt_sample(n,par_m2[3,1],par_m2[3,2],alpha[2])
 x_33_samples<-norm_temp_tilt_sample(n,par_m3[3,1],par_m3[3,2],alpha[3])

 x_41_prime_samples<-norm_tilt_sample(n,par_m1[4,1],par_m1[4,2],alpha[1])
 x_42_prime_samples<-norm_tilt_sample(n,par_m2[4,1],par_m2[4,2],alpha[2])
 x_43_prime_samples<-norm_tilt_sample(n,par_m3[4,1],par_m3[4,2],alpha[3])
 x_41_samples<-x_41_prime_samples+10*(60-x_31_samples+abs(60-x_31_samples))/2
 x_42_samples<-x_42_prime_samples+10*(60-x_32_samples+abs(60-x_32_samples))/2
 x_43_samples<-x_43_prime_samples+10*(60-x_33_samples+abs(60-x_33_samples))/2

 x_51_prime_samples<-norm_tilt_sample(n,par_m1[5,1],par_m1[5,2],alpha[1])
 x_52_prime_samples<-norm_tilt_sample(n,par_m2[5,1],par_m2[5,2],alpha[2])
 x_53_prime_samples<-norm_tilt_sample(n,par_m3[5,1],par_m3[5,2],alpha[3])
 x_51_samples<-x_51_prime_samples+40*(60-x_31_samples+abs(60-x_31_samples))/2
 x_52_samples<-x_52_prime_samples+40*(60-x_32_samples+abs(60-x_32_samples))/2
 x_53_samples<-x_53_prime_samples+40*(60-x_33_samples+abs(60-x_33_samples))/2

 Hydro<-rbind(x_11_samples,x_12_samples,x_13_samples)
 Nuclear<-rbind(x_21_samples,x_22_samples,x_23_samples)
 Temperature<-rbind(x_31_samples,x_32_samples,x_33_samples)
 ElectricDemand<-rbind(x_41_samples,x_42_samples,x_43_samples)
 GasDemand<-rbind(x_51_samples,x_52_samples,x_53_samples)

 return(list(Hydro,Nuclear,Temperature,ElectricDemand,GasDemand))
}

dens_all_log<-function(sample_all,par_m1,par_m2,par_m3){

 x_11_dens<-gamma_dens_log(sample_all[[1]][1,],par_m1[1,1],par_m1[1,2])
 x_12_dens<-gamma_dens_log(sample_all[[1]][2,],par_m2[1,1],par_m2[1,2])
 x_13_dens<-gamma_dens_log(sample_all[[1]][3,],par_m3[1,1],par_m3[1,2])
 
 x_21_dens<-exponen_dens_log(sample_all[[2]][1,])
 x_22_dens<-exponen_dens_log(sample_all[[2]][2,])
 x_23_dens<-exponen_dens_log(sample_all[[2]][3,])

 x_31_dens<-norm_dens_log(sample_all[[3]][1,],par_m1[3,1],par_m1[3,2])
 x_32_dens<-norm_dens_log(sample_all[[3]][2,],par_m2[3,1],par_m2[3,2])
 x_33_dens<-norm_dens_log(sample_all[[3]][3,],par_m3[3,1],par_m3[3,2])

 x_41_prime_samples<-sample_all[[4]][1,]-10*(60-sample_all[[3]][1,]+abs(60-sample_all[[3]][1,]))/2
 x_42_prime_samples<-sample_all[[4]][2,]-10*(60-sample_all[[3]][2,]+abs(60-sample_all[[3]][2,]))/2
 x_43_prime_samples<-sample_all[[4]][3,]-10*(60-sample_all[[3]][3,]+abs(60-sample_all[[3]][3,]))/2
 x_41_dens<-norm_dens_log(x_41_prime_samples,par_m1[4,1],par_m1[4,2])
 x_42_dens<-norm_dens_log(x_42_prime_samples,par_m2[4,1],par_m2[4,2])
 x_43_dens<-norm_dens_log(x_43_prime_samples,par_m3[4,1],par_m3[4,2])

 x_51_prime_samples<-sample_all[[5]][1,]-40*(60-sample_all[[3]][1,]+abs(60-sample_all[[3]][1,]))/2
 x_52_prime_samples<-sample_all[[5]][2,]-40*(60-sample_all[[3]][2,]+abs(60-sample_all[[3]][2,]))/2
 x_53_prime_samples<-sample_all[[5]][3,]-40*(60-sample_all[[3]][3,]+abs(60-sample_all[[3]][3,]))/2
 x_51_dens<-norm_dens_log(x_51_prime_samples,par_m1[5,1],par_m1[5,2])
 x_52_dens<-norm_dens_log(x_52_prime_samples,par_m2[5,1],par_m2[5,2])
 x_53_dens<-norm_dens_log(x_53_prime_samples,par_m3[5,1],par_m3[5,2])
 
 obs_dens<-list(cbind(x_11_dens,x_21_dens,x_31_dens,x_41_dens,x_51_dens),cbind(x_12_dens,x_22_dens,x_32_dens,x_42_dens,x_52_dens),cbind(x_13_dens,x_23_dens,x_33_dens,x_43_dens,x_53_dens))

 return(obs_dens)
}

dens_tilt_all_log<-function(sample_tilt_all,par_m1,par_m2,par_m3,alpha){
 # 0.2s
 x_11_dens<-gamma_tilt_dens_log(sample_tilt_all[[1]][1,],par_m1[1,1],par_m1[1,2],alpha[1])
 x_12_dens<-gamma_tilt_dens_log(sample_tilt_all[[1]][2,],par_m2[1,1],par_m2[1,2],alpha[2])
 x_13_dens<-gamma_tilt_dens_log(sample_tilt_all[[1]][3,],par_m3[1,1],par_m3[1,2],alpha[3])

  #0.13s
 x_21_dens<-exponen_tilt_dens_log(sample_tilt_all[[2]][1,],alpha[1])
 x_22_dens<-exponen_tilt_dens_log(sample_tilt_all[[2]][2,],alpha[2])
 x_23_dens<-exponen_tilt_dens_log(sample_tilt_all[[2]][3,],alpha[3])
 
 # 0.23s
 x_31_dens<-norm_temp_tilt_dens_log(sample_tilt_all[[3]][1,],par_m1[3,1],par_m1[3,2],alpha[1])
 x_32_dens<-norm_temp_tilt_dens_log(sample_tilt_all[[3]][2,],par_m2[3,1],par_m2[3,2],alpha[2])
 x_33_dens<-norm_temp_tilt_dens_log(sample_tilt_all[[3]][3,],par_m3[3,1],par_m3[3,2],alpha[3])

 # 0.35s (reduced from 1.92s) 
 x_41_prime_samples<-sample_tilt_all[[4]][1,]-10*((60-sample_tilt_all[[3]][1,])+abs(60-sample_tilt_all[[3]][1,]))/2
 x_42_prime_samples<-sample_tilt_all[[4]][2,]-10*((60-sample_tilt_all[[3]][2,])+abs(60-sample_tilt_all[[3]][2,]))/2
 x_43_prime_samples<-sample_tilt_all[[4]][3,]-10*((60-sample_tilt_all[[3]][3,])+abs(60-sample_tilt_all[[3]][3,]))/2

 x_41_dens<-norm_tilt_dens_log(x_41_prime_samples,par_m1[4,1],par_m1[4,2],alpha[1])
 x_42_dens<-norm_tilt_dens_log(x_42_prime_samples,par_m2[4,1],par_m2[4,2],alpha[2])
 x_43_dens<-norm_tilt_dens_log(x_43_prime_samples,par_m3[4,1],par_m3[4,2],alpha[3])

 x_51_prime_samples<-sample_tilt_all[[5]][1,]-40*((60-sample_tilt_all[[3]][1,])+abs(60-sample_tilt_all[[3]][1,]))/2
 x_52_prime_samples<-sample_tilt_all[[5]][2,]-40*((60-sample_tilt_all[[3]][2,])+abs(60-sample_tilt_all[[3]][2,]))/2
 x_53_prime_samples<-sample_tilt_all[[5]][3,]-40*((60-sample_tilt_all[[3]][3,])+abs(60-sample_tilt_all[[3]][3,]))/2

 x_51_dens<-norm_tilt_dens_log(x_51_prime_samples,par_m1[5,1],par_m1[5,2],alpha[1])
 x_52_dens<-norm_tilt_dens_log(x_52_prime_samples,par_m2[5,1],par_m2[5,2],alpha[2])
 x_53_dens<-norm_tilt_dens_log(x_53_prime_samples,par_m3[5,1],par_m3[5,2],alpha[3])
 
 obs_dens<-list(cbind(x_11_dens,x_21_dens,x_31_dens,x_41_dens,x_51_dens),cbind(x_12_dens,x_22_dens,x_32_dens,x_42_dens,x_52_dens),cbind(x_13_dens,x_23_dens,x_33_dens,x_43_dens,x_53_dens))

 return(obs_dens)
}

# physical model
max2<-function(y,x){ 
 col_length<-dim(x)[2]; 
 x_vec<-c(x)
 results<-matrix((x_vec+y+abs(x_vec-y))/2,ncol=col_length)
 return(results)
}

colsum<-function(x){
 row_no<-dim(x)[1]; col_no<-dim(x)[2]
 sum_over_col<-rep(0,col_no)
 for(i in 1:row_no) sum_over_col<-sum_over_col+x[i,]
 return(sum_over_col)
}

BlackBox<-function(GasDemand, ElectricDemand, Temperature, Hydro, Nuclear,OtherElectric=matrix(500,3,n), GasSupply=matrix(2500,3,n), OilInventory=1200,InventoryPrice=1, CurtailmentPrice=80){
  ##Input matrices should have dimension 3 (months) * n (replications)
  ##Other input is vectors, 3 monthly values
  ##Internal matrices are 3 * n
  ##Outputs are n * 3
  n<-dim(Temperature)[2]
  DegreeDays<-max(0,60-Temperature)
  NetElectricDemand<-max2( 0, ElectricDemand )  # 0.06s (reduced from 1.85s)
  GasFlow<- GasSupply -GasDemand
  NetOilDemand<-max2( 0, NetElectricDemand - OtherElectric -Hydro - Nuclear - GasFlow )
  CumOilDemand<-rbind(NetOilDemand[1,],colsum(NetOilDemand[1:2,]),colsum(NetOilDemand)) # 0.16s (reduced from 0.57s)
  CumCurtailment<-max2( 0, CumOilDemand - OilInventory )
  SOilInventory<-OilInventory - CumOilDemand + CumCurtailment
  OutageCost<-CurtailmentPrice * CumCurtailment[3,]
  InventoryCost<-InventoryPrice * colsum(SOilInventory)
  TotalCost<-InventoryCost + OutageCost
  dimnames(SOilInventory)<-list(paste("Invent",1:3),NULL)
  ShortageInd<-(CumCurtailment[3,]>0)

  list(OutageCost=OutageCost,InventoryCost=InventoryCost,TotalCost=TotalCost,OilInventory = t(SOilInventory),ShortageInd,CumOilDemand)
}


# take part of samples from the list type dataset
takesamples<-function(m_vec,data_all){
 data_take<-list(NULL)
 l<-length(m_vec)
 if(l>1){ 
   data_take[[1]]<-data_all[[1]][,m_vec]
   data_take[[2]]<-data_all[[2]][,m_vec]
   data_take[[3]]<-data_all[[3]][,m_vec]
   data_take[[4]]<-data_all[[4]][,m_vec]
   data_take[[5]]<-data_all[[5]][,m_vec]
 }
 if(l==1){
   data_take[[1]]<-as.matrix(data_all[[1]][,m_vec])
   data_take[[2]]<-as.matrix(data_all[[2]][,m_vec])
   data_take[[3]]<-as.matrix(data_all[[3]][,m_vec])
   data_take[[4]]<-as.matrix(data_all[[4]][,m_vec])
   data_take[[5]]<-as.matrix(data_all[[5]][,m_vec])
 }
 return(data_take)
}

# combine the list elements of several lists
list_cbind<-function(x){
 number_of_lists<-length(x); length_of_eachlist<-length(x[[1]])
 combine_list<-as.list(rep(1,length_of_eachlist))
 for(i in 1:length_of_eachlist){
  temp<-NULL
  for(j in 1:number_of_lists) temp<-cbind(temp,x[[j]][[i]])
  combine_list[[i]]<-temp
 }
 return(combine_list)
}

# calculate r_i
r_calculation<-function(obs){
 oil_req_month1<-apply(obs[[1]],1,sum)
 oil_req_month2<-apply(obs[[2]],1,sum)
 oil_req_month3<-apply(obs[[3]],1,sum)
 oil_req_total<-oil_req_month1+oil_req_month2+oil_req_month3
 r<-cor(oil_req_total,cbind(oil_req_month1,oil_req_month2,oil_req_month3))
 return(r)
}

# calculate the 99% quantile of oil_req under f
oil_req_quant<-function(n,par_m1,par_m2,par_m3){
 x_11_samples<-gamma_sample(n,par_m1[1,1],par_m1[1,2])
 x_12_samples<-gamma_sample(n,par_m2[1,1],par_m2[1,2])
 x_13_samples<-gamma_sample(n,par_m3[1,1],par_m3[1,2])
 
 x_21_samples<-exponen_sample(n,par_m1[2,1])
 x_22_samples<-exponen_sample(n,par_m2[2,1])
 x_23_samples<-exponen_sample(n,par_m3[2,1])

 x_31_samples<-norm_sample(n,par_m1[3,1],par_m1[3,2])
 x_32_samples<-norm_sample(n,par_m2[3,1],par_m2[3,2])
 x_33_samples<-norm_sample(n,par_m3[3,1],par_m3[3,2])

 x_41_prime_samples<-norm_sample(n,par_m1[4,1],par_m1[4,2])
 x_42_prime_samples<-norm_sample(n,par_m2[4,1],par_m2[4,2])
 x_43_prime_samples<-norm_sample(n,par_m3[4,1],par_m3[4,2])
 x_41_samples<-x_41_prime_samples+10*apply(cbind(60-x_31_samples,0),1,max)
 x_42_samples<-x_42_prime_samples+10*apply(cbind(60-x_32_samples,0),1,max)
 x_43_samples<-x_43_prime_samples+10*apply(cbind(60-x_33_samples,0),1,max)

 x_51_prime_samples<-norm_sample(n,par_m1[5,1],par_m1[5,2])
 x_52_prime_samples<-norm_sample(n,par_m2[5,1],par_m2[5,2])
 x_53_prime_samples<-norm_sample(n,par_m3[5,1],par_m3[5,2])
 x_51_samples<-x_51_prime_samples+40*apply(cbind(60-x_31_samples,0),1,max)
 x_52_samples<-x_52_prime_samples+40*apply(cbind(60-x_32_samples,0),1,max)
 x_53_samples<-x_53_prime_samples+40*apply(cbind(60-x_33_samples,0),1,max)

 samples_total<-cbind(-x_11_samples,-x_12_samples,-x_13_samples,-x_21_samples,-x_22_samples,-x_23_samples,x_41_samples,x_42_samples,x_43_samples,x_51_samples,x_52_samples,x_53_samples)
 oil_requir_samples<-apply(samples_total,1,sum)
 return(oil_requir_samples)
}

# calculate the mean value of oil reqir at each month
oil_req_mean<-function(alpha,par_m,OtherElectric=500,GasSupply=2500){
 #browser()
 mean_gamma<-par_m[1,1]/(1/par_m[1,2]+alpha)
 mean_exp<-1/(1/100-alpha)*((2-300*alpha)*exp(3-300*alpha)+1)/(exp(3-300*alpha)-1)
 mean_temp_tilt<-par_m[3,1]-alpha*(10+40)*par_m[3,2]^2
 temp<-(mean_temp_tilt-60)/par_m[3,2]
 mean_max_0_temp<-60-mean_temp_tilt+par_m[3,2]*dnorm(temp)/(1-pnorm(temp))
 mean_N2<-par_m[4,1]+alpha*par_m[4,2]^2+10*mean_max_0_temp
 mean_N3<-par_m[5,1]+alpha*par_m[5,2]^2+40*mean_max_0_temp
 oil_req<-mean_N2+mean_N3-mean_gamma-mean_exp-OtherElectric-GasSupply
 return(oil_req)
}

oil_req_mean_diff<-function(alpha,par_m1,par_m2,par_m3,whichmonth,mean_balance){
 oil_req_mean_month1<-0; oil_req_mean_month2<-0; oil_req_mean_month3<-0
 if(whichmonth[1]==1) oil_req_mean_month1<-oil_req_mean(alpha,par_m1)
 if(whichmonth[2]==1) oil_req_mean_month2<-oil_req_mean(alpha,par_m2)
 if(whichmonth[3]==1) oil_req_mean_month3<-oil_req_mean(alpha,par_m3)
 oil_req<-oil_req_mean_month1+oil_req_mean_month2+oil_req_mean_month3
 return(1200-oil_req-mean_balance)
}

# log-likelihood calculated in MLE method
l_fun_neg<-function(zeta_vec,q_dens,g_dens,alpha_vec){
 q_alpha_zeta<-t(alpha_vec)%*%q_dens+t(g_dens%*%zeta_vec)
 if(sum(q_alpha_zeta<0)>0) return(Inf)
 
 l_funval<-sum(log(q_alpha_zeta))
 return(-l_funval)
 }

colProd<-function(x){ # input a m*n matrix
 ncol_x<-dim(x)[2]; nrow_x<-dim(x)[1]
 results_prod<-rep(1,nrow_x)
 for(i in 1:ncol_x) results_prod<-results_prod*x[,i]
 return(results_prod)
}
colSum<-function(x){ # input a m*n matrix
 ncol_x<-dim(x)[2]; nrow_x<-dim(x)[1]
 results_sum<-rep(0,nrow_x)
 for(i in 1:ncol_x) results_sum<-results_sum+x[,i]
 return(results_sum)
}

# functions for optimization algorithm
 p<-8
 I_G1_1<-NULL
 for(i in 1:(p-1)) I_G1_1<-rbind(I_G1_1,diag(1,p-1))
 I_G1_2_ele<-list(0)
 for(i in 1:(p-1)) I_G1_2_ele[[i]]<-rep(1,p-1)
 I_G1_2<-bdiag(I_G1_2_ele)
 I_G2_1<-t(I_G1_1)
 I_G2_2<-t(I_G1_2)

 A1_index<-NULL; A1_re_index<-NULL # rearrange (p-1)*(p-1) k1*k2 matrices
 k1<-p-1; k2<-p-1
 for(i in 1:(p-1)){
  for(j in 1:(p-1)){
   A1_index<-rbind(A1_index,cbind(rep(1:k1,rep(k2,k1))+(i-1)*k1,rep(1:k2,k1)+(j-1)*k2))
   A1_re_index<-rbind(A1_re_index,cbind(rep(1:k1,rep(k2,k1))+(j-1)*k1,rep(1:k2,k1)+(i-1)*k2))
  }
 }
  
 A2_index<-NULL; A2_re_index<-NULL # rearrange (p-1)*(p-1) k1*k2 matrices
 k1<-p-1; k2<-1
 for(i in 1:(p-1)){
  for(j in 1:(p-1)){
   A2_index<-rbind(A2_index,cbind(rep(1:k1,rep(k2,k1))+(i-1)*k1,rep(1:k2,k1)+(j-1)*k2))
   A2_re_index<-rbind(A2_re_index,cbind(rep(1:k1,rep(k2,k1))+(j-1)*k1,rep(1:k2,k1)+(i-1)*k2))
  }
 }
 
times<-function(x,y) return(x*y) 
blockdiag<-function(A,k){
 temp<-as.list(rep(1,k))
 A_list<-lapply(temp,times,y=A)
 return(bdiag(A_list))
}

blockdiag<-function(A,k){ # bdiag is costly, need modification
 l1<-dim(A)[1]; l2<-dim(A)[2];
 A_diag<-matrix(0,k*l1,k*l2)
 for(i in 1:k) A_diag[((i-1)*l1+1):(i*l1),((i-1)*l2+1):(i*l2)]<-A
 return(A_diag)
}

var_MLE_value<-function(alpha_vec_excludefirst,pi_dens,q_dens,h,mu_hat,q_gamma,g_dens){
 if(sum(alpha_vec_excludefirst<0)>0||sum(alpha_vec_excludefirst)>0.999) return(list(value=Inf,gradient=Inf,hessian=Inf))
 n0<-length(pi_dens)  
 alpha_vec<-c(1-sum(alpha_vec_excludefirst),alpha_vec_excludefirst)
 q_alpha<-t(alpha_vec)%*%q_dens
 temp<-!((pi_dens==0)&(q_alpha==0))
 pi_dens<-pi_dens[temp]
 q_gamma<-q_gamma[temp]
 g_dens<-g_dens[,temp]
 q_alpha<-q_alpha[temp]
 h<-h[temp]
 #
 cc1<-min(q_gamma)
 q_gamma_temp<-q_gamma
 q_gamma<-q_gamma/cc1
 #
 f<-(h-mu_hat)*pi_dens
 part1_beta<-g_dens*(rep(1,p-1)%*%t(1/q_alpha))
 part2_beta<-t(g_dens*(rep(1,p-1)%*%t(1/q_gamma)))
 B_inv<-solve(part1_beta%*%part2_beta)
 #
 part4_beta<-f/q_gamma
 part5_beta<-part1_beta%*%part4_beta
 beta_MLE<-B_inv%*%part5_beta
 #
 G<-g_dens
 tau1<-(f-c(t(beta_MLE)%*%G))/q_alpha/q_gamma
 var_MLE_val<-mean(tau1^2*q_alpha*q_gamma) 
 var_MLE_val<-var_MLE_val/cc1
 return(var_MLE_val)
}

var_MLE_gradient<-function(alpha_vec_excludefirst,pi_dens,q_dens,h,mu_hat,q_gamma,g_dens){
 if(sum(alpha_vec_excludefirst<0)>0||sum(alpha_vec_excludefirst)>0.999) return(list(value=Inf,gradient=Inf,hessian=Inf))
 n0<-length(pi_dens)  
 alpha_vec<-c(1-sum(alpha_vec_excludefirst),alpha_vec_excludefirst)
 q_alpha<-t(alpha_vec)%*%q_dens
 temp<-!((pi_dens==0)&(q_alpha==0))
 pi_dens<-pi_dens[temp]
 q_gamma<-q_gamma[temp]
 g_dens<-g_dens[,temp]
 q_alpha<-q_alpha[temp]
 h<-h[temp]
 #
 cc1<-min(q_gamma)
 q_gamma_temp<-q_gamma
 q_gamma<-q_gamma/cc1
 #
 f<-(h-mu_hat)*pi_dens
 part1_beta<-g_dens*(rep(1,p-1)%*%t(1/q_alpha))
 part2_beta<-t(g_dens*(rep(1,p-1)%*%t(1/q_gamma)))
 B_inv<-solve(part1_beta%*%part2_beta)
 #
 part4_beta<-f/q_gamma
 part5_beta<-part1_beta%*%part4_beta
 beta_MLE<-B_inv%*%part5_beta
 #
 #Q<-q_dens
 G<-g_dens
 tau1<-(f-c(t(beta_MLE)%*%G))/q_alpha/q_gamma
 #tau2<-tau1/q_alpha
 #tau3<-tau1^2/q_alpha*q_gamma
 #
 #q_alpha1<-1/q_alpha
 q_alpha2_gamma<-1/q_alpha^2/q_gamma #could be Inf
 C<-part5_beta
 B_inv_C_diag<-blockdiag(B_inv%*%C,p-1)
 #G1<-(I_G1_1%*%(G*(rep(1,p-1)%*%t(q_alpha1))))*(I_G1_2%*%G)
 G2<-(t(G)%*%I_G2_1)*(((q_alpha2_gamma%*%t(rep(1,p-1)))*t(G))%*%I_G2_2)
 G3<-((f*q_alpha2_gamma)%*%t(rep(1,p-1)))*t(G)
 #G4<-((f*q_alpha1)%*%t(rep(1,p-1)))*t(G)
 #
 B_p<--G%*%G2
 C_p<--G%*%G3
 #
 beta_gradient<-t(-B_inv%*%B_p%*%B_inv_C_diag+B_inv%*%C_p)

 var_gradient_part1<-G%*%(tau1^2*q_gamma)/n0
 var_gradient_part2<-beta_gradient%*%(G*(rep(1,p-1)%*%t(tau1)))%*%rep(1,n0)*2/n0
 var_gradient<-as.vector(-var_gradient_part1-var_gradient_part2)

 var_gradient<-var_gradient/cc1
 return(var_gradient)
}


########################
# Two-stage Procedure
########################

# simple random sampling
estimation_simpleMC<-function(m){
 samples_f<-takesamples(1:m,data_all[[1]])
 Hydro<-samples_f[[1]]
 Nuclear<-samples_f[[2]]
 Temperature<-samples_f[[3]]
 ElectricDemand<-samples_f[[4]]
 GasDemand<-samples_f[[5]]
 results<-BlackBox(GasDemand, ElectricDemand, Temperature, Hydro, Nuclear)
 return(results)   
}

# fix mixture proportions
estimation_det<-function(beta_vec,m,alpha_vec,tiltflags){
 # beta_vec is the mixture proportions, m is the total sample size, alpha_vec is the tilted parameters and tiltflags is the tilted months of each proposal
 samples<-list(0)
 dens_samples_log<-matrix(0,8,m)
 m_all<-c(m-sum(floor(m*beta_vec[2:8])),floor(m*beta_vec[2:8]))
 
 # formal samples
 for(i in 1:8){
  if(m_all[i]>0) samples[[i]]<-takesamples(1:(m_all[i]),data_all[[i]])
  if(m_all[i]==0) samples[[i]]<-list(NULL,NULL,NULL,NULL,NULL)
 }
 samples_all<-list_cbind(samples)
 
 # joint densities  
 for(i in 1:8){
  dens_samples_log[i,]<-colSum(matrix(unlist(dens_tilt_all_log(samples_all,par_m1,par_m2,par_m3,alpha_vec[i]*tiltflags[,i])),ncol=15))
}

 # sample weights  
 ratio_q_f<-exp(dens_samples_log[2:8,]-rep(1,7)%*%t(dens_samples_log[1,]))
 weights_samples<-1/(beta_vec[1]+t(beta_vec[2:8])%*%ratio_q_f)
  
 # blackbox results  
 Hydro<-samples_all[[1]]
 Nuclear<-samples_all[[2]]
 Temperature<-samples_all[[3]]
 ElectricDemand<-samples_all[[4]]
 GasDemand<-samples_all[[5]]
 results<-BlackBox(GasDemand, ElectricDemand, Temperature, Hydro, Nuclear) 

 return(list(results,c(weights_samples)))
}

estimation_MLE<-function(beta_vec,m,alpha_vec,tiltflags){
 # beta_vec is the mixture proportions, m is the total sample size, alpha_vec is the tilted parameters and tiltflags is the tilted months of each proposal
 samples<-list(0)
 dens_samples_log<-matrix(0,8,m)
 m_all<-c(m-sum(floor(m*beta_vec[2:8])),floor(m*beta_vec[2:8]))
 
 # formal samples
 for(i in 1:8){
  if(m_all[i]>0) samples[[i]]<-takesamples(1:(m_all[i]),data_all[[i]])
  if(m_all[i]==0) samples[[i]]<-list(NULL,NULL,NULL,NULL,NULL)
 }
 samples_all<-list_cbind(samples)
 
 # joint densities  5.39s (reduced from 31.28s)
 for(i in 1:8){
  dens_samples_log[i,]<-colSum(matrix(unlist(dens_tilt_all_log(samples_all,par_m1,par_m2,par_m3,alpha_vec[i]*tiltflags[,i])),ncol=15))
}
 
  # MLE weights 
 ui_matrix<-as.matrix(bdiag(list(c(-1,1),c(-1,1),c(-1,1),c(-1,1),c(-1,1),c(-1,1),c(-1,1))))
 ci_vec<-rep(-1,14)
 dens_samples<-exp(dens_samples_log)
 g_dens<-t(dens_samples[2:8,]-rep(1,7)%*%t(dens_samples[1,]))
 q_alpha<-c(beta_vec%*%dens_samples)
 
 zeta_optim_results<-constrOptim(theta=rep(0,7),f=l_fun_neg,method="Nelder-Mead",ui=ui_matrix,ci=ci_vec,q_dens=dens_samples,g_dens=g_dens,alpha_vec=beta_vec) # 7.51s
 zeta_optim<-zeta_optim_results$par
 #zeta_optim<-solve(t(g_dens)%*%(g_dens/(q_alpha%*%t(rep(1,7)))))%*%(t(g_dens/(q_alpha%*%t(rep(1,7))))%*%rep(1,m))
 
 q_alpha_zeta<-q_alpha+g_dens%*%zeta_optim

 weights_sample_MLE<-dens_samples[1,]/q_alpha_zeta

 # blackbox results  
 Hydro<-samples_all[[1]]
 Nuclear<-samples_all[[2]]
 Temperature<-samples_all[[3]]
 ElectricDemand<-samples_all[[4]]
 GasDemand<-samples_all[[5]]
 results<-BlackBox(GasDemand, ElectricDemand, Temperature, Hydro, Nuclear) # 0.23s (reduced from 5.35s)

 return(list(results,c(weights_sample_MLE)))
}

# two-stage mixture proportions with several proposals
estimation_twostages<-function(n,n0,gamma_vec,alpha_vec,tiltflags){
 delta_lowbound_q1<-10^(-3)
 samples_pilot<-list(0)
 dens_samples_pilot_log<-matrix(0,8,n0)
 n0_all<-c(n0-sum(floor(n0*gamma_vec[2:8])),floor(n0*gamma_vec[2:8]))
 
 # pilot samples  
 for(i in 1:8) samples_pilot[[i]]<-takesamples(1:(n0_all[i]),data_all[[i]])
 samples_pilot_all<-list_cbind(samples_pilot)

 # joint densities 
 for(i in 1:8){
   dens_samples_pilot_log[i,]<-colSum(matrix(unlist(dens_tilt_all_log(samples_pilot_all,par_m1,par_m2,par_m3,alpha_vec[i]*tiltflags[,i])),ncol=15))
 }

 dens_samples_pilot<-exp(dens_samples_pilot_log)
 pi_dens_pilot<-dens_samples_pilot[1,]

 ui_optim<-rbind(rep(-1,p-1),diag(rep(1,p-1)))
 ci_optim<-c(-(1-delta_lowbound_q1),rep(0,7))
 
 # blackbox results
 Hydro_pilot<-samples_pilot_all[[1]]
 Nuclear_pilot<-samples_pilot_all[[2]]
 Temperature_pilot<-samples_pilot_all[[3]]
 ElectricDemand_pilot<-samples_pilot_all[[4]]
 GasDemand_pilot<-samples_pilot_all[[5]]
 results_pilot<-BlackBox(GasDemand_pilot, ElectricDemand_pilot, Temperature_pilot, Hydro_pilot, Nuclear_pilot) 
 h_integrand_pilot<-results_pilot[[1]] # Outage Cost

 q_gamma_pilot<-t(gamma_vec)%*%dens_samples_pilot
 temp_samples<-!((pi_dens_pilot==0)&(q_gamma_pilot==0))
 g_dens_pilot<-dens_samples_pilot[2:8,]-rep(1,7)%*%t(dens_samples_pilot[1,])
 mu_hat_pilot<-mean((h_integrand_pilot*pi_dens_pilot/q_gamma_pilot)[temp_samples])/mean((pi_dens_pilot/q_gamma_pilot)[temp_samples])

 temp_beta<-constrOptim(theta=gamma_vec[2:8],f=var_MLE_value,grad=var_MLE_gradient,method="BFGS",ui=ui_optim,ci=ci_optim,pi_dens=pi_dens_pilot,q_dens=dens_samples_pilot,h=h_integrand_pilot,mu_hat=mu_hat_pilot,q_gamma=q_gamma_pilot,g_dens=g_dens_pilot,outer.iterations = 1000)
 
 alpha_hat_excludefirst<-temp_beta$par
 alpha_hat<-c(1-sum(alpha_hat_excludefirst),alpha_hat_excludefirst)

 samples_formal<-list(0)
 dens_samples_log<-matrix(0,8,n)
 n1_all<-c(n-n0-sum(floor((n-n0)*alpha_hat[2:8])),floor((n-n0)*alpha_hat[2:8])); 
 
 # formal samples  
  for(i in 1:8){
  if(n1_all[i]>0) samples_formal[[i]]<-takesamples((n0_all[i]+1):(n0_all[i]+n1_all[i]),data_all[[i]])
  if(n1_all[i]==0) samples_formal[[i]]<-list(NULL,NULL,NULL,NULL,NULL)  
 }
 samples_formal_all<-list_cbind(samples_formal)
 samples_all<-list_cbind(list(samples_pilot_all,samples_formal_all))

 # joint densities 
 for(i in 1:8){
   dens_samples_log[i,]<-colSum(matrix(unlist(dens_tilt_all_log(samples_all,par_m1,par_m2,par_m3,alpha_vec[i]*tiltflags[,i])),ncol=15))
 }
 dens_samples<-exp(dens_samples_log)
 g_dens<-t(dens_samples[2:8,]-rep(1,7)%*%t(dens_samples[1,])) 
 alpha_tilde<-n0/n*gamma_vec[1]+(n-n0)/n*alpha_hat

 # MLE weights 
 ui_matrix<-as.matrix(bdiag(list(c(-1,1),c(-1,1),c(-1,1),c(-1,1),c(-1,1),c(-1,1),c(-1,1))))
 ci_vec<-rep(-1,14)
 zeta_optim_results<-constrOptim(theta=rep(0,7),f=l_fun_neg,method="Nelder-Mead",ui=ui_matrix,ci=ci_vec,q_dens=dens_samples,g_dens=g_dens,alpha_vec=alpha_tilde) 
 zeta_optim<-zeta_optim_results$par
  
 q_alpha<-c(t(alpha_tilde)%*%dens_samples)
 q_alpha_zeta<-q_alpha+g_dens%*%zeta_optim

 weights_sample_MLE<-dens_samples[1,]/q_alpha_zeta
 
 # blackbox results
 Hydro<-samples_all[[1]]
 Nuclear<-samples_all[[2]]
 Temperature<-samples_all[[3]]
 ElectricDemand<-samples_all[[4]]
 GasDemand<-samples_all[[5]]
 results<-BlackBox(GasDemand, ElectricDemand, Temperature, Hydro, Nuclear)

 return(list(results,weights_sample_MLE,alpha_hat))
}


# parameters of target distribution
par_m1<-rbind(c(5,500/5),c(100,0),c(54,5),c(1600,100),c(1600,100))
par_m2<-rbind(c(6,600/6),c(100,0),c(52,5),c(1650,100),c(1700,100))
par_m3<-rbind(c(7,600/7),c(100,0),c(55,5),c(1600,100),c(1600,100))

# tilted parameters
tiltflags<-matrix(c(0,0,0, 1,0,0, 0,1,0, 0,0,1, 1,1,0, 1,0,1, 0,1,1, 1,1,1), 3)
mean_balance_all<-c(0,-216,-66,-416,-282,-632,-482,-698)
alpha_vec<-rep(0,8)
data_all<-list(0)

 #calculate alpha
for(i in 2:8){
  alpha_vec[i]<-uniroot(oil_req_mean_diff,interval<-c(0,10),par_m1=par_m1,par_m2=par_m2,par_m3=par_m3,whichmonth=tiltflags[,i],mean_balance=mean_balance_all[i])$root
}

# simulate samples and weights
replic<-1000; n=4000; n0=400;
samples_det_all<-list(0)
samples_MINV_all<-list(0)
alpha_hat_all<-list(0)

####################################
# deterministic mixture weights
####################################
save.time<-proc.time() 
for(k in 1:replic){ 
 set.seed(200+k)
 # take samples
 data_all[[1]]<-sample_all(n,par_m1,par_m2,par_m3)
 for(i in 2:8) data_all[[i]]<-sample_tilt_all(n,par_m1,par_m2,par_m3,alpha_vec[i]*tiltflags[,i]) 

 alpha_paper<-c(0.5,0.5*c(0.007,0.056,0.001,0.472,0.036,0.127,0.301))
 samples_det_results<-estimation_det(alpha_paper,n,alpha_vec,tiltflags)
 samples_det_all[[k]]<-samples_det_results
}
proc.time()-save.time
 
########################
# MINV mixture weights
########################
save.time<-proc.time() 
for(k in 1:replic){ 
 set.seed(200+k)
 # take samples
 data_all[[1]]<-sample_all(n,par_m1,par_m2,par_m3)
 for(i in 2:8) data_all[[i]]<-sample_tilt_all(n,par_m1,par_m2,par_m3,alpha_vec[i]*tiltflags[,i]) 

 samples_MINV_results<-estimation_twostages(n,n0,1/rep(8,8),alpha_vec,tiltflags)
 samples_MINV_all[[k]]<-samples_MINV_results
}
proc.time()-save.time


# expectation results
results_det<-matrix(0,replic,7); aa1<-0
for(k in 1:replic){
 OutageCost_det<-sum(samples_det_all[[k]][[1]][[1]]*samples_det_all[[k]][[2]])/sum(samples_det_all[[k]][[2]])
 InventoryCost_det<-sum(samples_det_all[[k]][[1]][[2]]*samples_det_all[[k]][[2]])/sum(samples_det_all[[k]][[2]])
 TotalCost_det<-sum(samples_det_all[[k]][[1]][[3]]*samples_det_all[[k]][[2]])/sum(samples_det_all[[k]][[2]])
 OilInventory_det<-colSums(samples_det_all[[k]][[1]][[4]]*(samples_det_all[[k]][[2]]%*%t(rep(1,3))))/sum(samples_det_all[[k]][[2]])
 ShortageInd_det<-mean(samples_det_all[[k]][[1]][[5]]*samples_det_all[[k]][[2]])
 results_det[k,]<-c(OutageCost_det,InventoryCost_det,TotalCost_det,OilInventory_det,ShortageInd_det)
} 
colnames(results_det)<-list("OutageCost","InventoryCost","TotalCost","Invent1","Invent2","Invent3","ShortageProb")

results_MINV<-matrix(0,replic,7)
alpha_hat_MINV<-matrix(0,replic,8)
alpha_hat_var<-0; gamma_vec_var<-0
for(k in 1:replic){
 OutageCost_MINV<-sum(samples_MINV_all[[k]][[1]][[1]]*samples_MINV_all[[k]][[2]])/sum(samples_MINV_all[[k]][[2]])
 InventoryCost_MINV<-sum(samples_MINV_all[[k]][[1]][[2]]*samples_MINV_all[[k]][[2]])/sum(samples_MINV_all[[k]][[2]])
 TotalCost_MINV<-sum(samples_MINV_all[[k]][[1]][[3]]*samples_MINV_all[[k]][[2]])/sum(samples_MINV_all[[k]][[2]])
 OilInventory_MINV<-apply(samples_MINV_all[[k]][[1]][[4]]*(samples_MINV_all[[k]][[2]]%*%t(rep(1,3))),2,sum)/sum(samples_MINV_all[[k]][[2]])
 ShortageInd_MINV<-sum(samples_MINV_all[[k]][[1]][[5]]*samples_MINV_all[[k]][[2]])/sum(samples_MINV_all[[k]][[2]])
 results_MINV[k,]<-c(OutageCost_MINV,InventoryCost_MINV,TotalCost_MINV,OilInventory_MINV,ShortageInd_MINV)

 alpha_hat_MINV[k,]<-samples_MINV_all[[k]][[3]]
}
colnames(results_MINV)<-list("OutageCost","InventoryCost","TotalCost","Invent1","Invent2","Invent3","ShortageProb")
colnames(alpha_hat_MINV)<-list("Original(f)","Dec","Jan","Feb","Dec&Jan","Dec&Feb","Jan&Feb","Dec-Feb")

results_det_all<-results_det
results_MINV_all<-results_MINV
alpha_hat_MINV_all<-alpha_hat_MINV

results_mean<-rbind(apply(results_det_all,2,mean),apply(results_MINV_all,2,mean))
rownames(results_mean)<-list("det","MINV")

results_sd<-rbind(apply(results_det_all,2,sd),apply(results_MINV_all,2,sd))
rownames(results_sd)<-list("det","MINV")

results_mean
results_sd^2


#######################
# VaR GARCH example
######################

#######################
# SP500 data
#######################
logindex_hist<-read.csv2("E:\\Dropbox\\SP500.csv",sep=",",dec=".",header=T,colClasses=c("Date",NA))
length_hist<-dim(logindex_hist)[1]
logreturn_hist<-logindex_hist$VALUE[-1]-logindex_hist$VALUE[-length_hist]
y_hist<-100*logreturn_hist[2701:2900]

##############
# Functions
##############
rowProds2<-function(a) Reduce("*", as.data.frame(a))
colProds2<-function(a) Reduce("*", as.data.frame(t(a)))
targetdens<-function(x,y_hist,h_ini,logind,a_pars){ # a_pars is c(mu_a,sd_a) of a's prior
   Time<-length(y_hist)-1
   n<-dim(x)[1]; d_ahead<-dim(x)[2]-3
   y_future<-matrix(x[,d_ahead:1],ncol=d_ahead); pars<-matrix(x[,(d_ahead+1):(d_ahead+3)],ncol=3)
   y2_hist<-y_hist^2
   y2_future<-y_future^2
   nonzero_ind<-(1:n)[as.logical(rowProds2(pars[,2:3]>0)*(rowSums(pars[,2:3])<1))]
   n_nonzero<-length(nonzero_ind)
   alpha0<-exp(pars[nonzero_ind,1]); alpha1<-pars[nonzero_ind,2]; beta_par<-pars[nonzero_ind,3]
   targ_dens_log_temp<-rep(0,n_nonzero)  
   for(i in 1:(Time+d_ahead)){     
     if(i==1) H_current<-alpha0+alpha1*y_hist[1]^2+beta_par*h_ini
	 if(i>1) H_current<-alpha0+alpha1*Y_old+beta_par*H_old
	 if(i<=Time) Y_current<-y2_hist[i+1]
	 if(i>Time) Y_current<-y2_future[nonzero_ind,i-Time]
	 targ_dens_log_temp<-targ_dens_log_temp-.5*(Y_current/H_current+log(H_current))
	 H_old<-H_current
	 Y_old<-Y_current
    }
   priordens_a<-(pars[nonzero_ind,1]-a_pars[1])^2/(-2*a_pars[2]^2)
   targ_dens_log<-rep(-Inf,n)
   targ_dens_log[nonzero_ind]<-targ_dens_log_temp+priordens_a
   if(logind==T) return(targ_dens_log)
   if(logind==F) targ_dens<-exp(targ_dens_log)
   return(targ_dens)
}
log_neg_targetdens<-function(x,y_hist,h_ini,a_pars){
   Time<-length(y_hist)-1
   d_ahead<-length(x)-3
   pars<-x[(d_ahead+1):(d_ahead+3)]
   if(!(prod(pars[2:3]>0)*(sum(pars[2:3])<1))) return(Inf)
   y2_hist<-y_hist^2
   if(d_ahead>0){
     y_future<-x[d_ahead:1]
     y2_future<-y_future^2
     Y<-c(y2_hist[-1],y2_future)
   }
   if(d_ahead==0) Y<-y2_hist[-1]
   H<-rep(0,Time+d_ahead)
   for(i in 1:(Time+d_ahead)){
     if(i==1) H[1]<-exp(pars[1])+pars[2]*y_hist[1]^2+pars[3]*h_ini
	 if(i>1) H[i]<-exp(pars[1])+pars[2]*Y[i-1]+pars[3]*H[i-1]
    }
   priordens_a<-(pars[1]-a_pars[1])^2/(-2*a_pars[2]^2)
   targ_dens_log<-(-.5)*(t(Y/H+log(H))%*%rep(1,Time+d_ahead))+priordens_a
browser()
   return(c(-targ_dens_log))   
}
rnorm_trunc<-function(n,mu,std,truncat){ # sampling from truncated univariate normal
   x<-mu+std*qnorm(runif(n)*pnorm((truncat-mu)/std))
   return(x)
}
y_comp_pred<-function(n,y_future,sample_pars,H_T,d_ahead,truncVaR,is_sampling,comp_numb){
  alpha0<-exp(sample_pars[,1]); alpha1<-abs(sample_pars[,2]); beta_par<-abs(sample_pars[,3])
  cum_logreturn<-0
  if(is_sampling==T){
    y_future<-matrix(0,n,d_ahead)
   # Calculate H_T
	H_current<-H_T
    if(d_ahead>1){
	    for(k in 1:(d_ahead-1)){
            y_future[,k]<-rnorm(n,sd=sqrt(H_current))
			H_old<-H_current
			H_current<-alpha0+alpha1*y_future[,k]^2+beta_par*H_old
			infin_ind<-(1:n)[!is.finite(H_current)]
			if(length(infin_ind)>0) H_current[infin_ind]<-max(H_current[-infin_ind])
		}
	   if(d_ahead>2) cum_logreturn<-y_future[,1:(d_ahead-1)]%*%rep(1,d_ahead-1)
	   if(d_ahead==2) cum_logreturn<-y_future[,1]
	}	
    infin_ind<-(1:n)[!is.finite(H_current)]
    if(length(infin_ind)>0) H_current[infin_ind]<-max(H_current[-infin_ind])
	std_H<-sqrt(H_current)
	if(comp_numb==1) y_future[,d_ahead]<-rnorm(n,mean=0,sd=std_H)
	if(comp_numb==2) y_future[,d_ahead]<-rnorm(n,mean=-std_H,sd=std_H)
	if(comp_numb==3) y_future[,d_ahead]<-rnorm_trunc(n,mu=0,std=std_H,truncVaR-cum_logreturn)
	if(comp_numb==4) y_future[,d_ahead]<-rnorm_trunc(n,mu=-std_H,std=std_H,truncVaR-cum_logreturn)
    return(y_future[,d_ahead:1]) 
  }
  if(is_sampling==F){
   # Calculate H_T
	if(d_ahead==1) y_future<-matrix(y_future,ncol=1)
	n<-dim(y_future)[1]
	H_current<-H_T
    if(d_ahead>1){
	    for(k in 1:(d_ahead-1)){
			H_old<-H_current
			H_current<-alpha0+alpha1*y_future[,k]^2+beta_par*H_old
			infin_ind<-(1:n)[!is.finite(H_current)]
			if(length(infin_ind)>0) H_current[infin_ind]<-max(H_current[-infin_ind])
		}
	   if(d_ahead>2) cum_logreturn<-y_future[,1:(d_ahead-1)]%*%rep(1,d_ahead-1)
	   if(d_ahead==2) cum_logreturn<-y_future[,1]
	}
    infin_ind<-(1:n)[!is.finite(H_current)]
    if(length(infin_ind)>0) H_current[infin_ind]<-max(H_current[-infin_ind])
	std_H<-sqrt(H_current); 
	y_dens<-matrix(0,n,length(comp_numb))
	for(i in 1:length(comp_numb)){
		if(comp_numb[i]==1) y_dens[,i]<-dnorm(y_future[,d_ahead],0,std_H)
		if(comp_numb[i]==2) y_dens[,i]<-dnorm(y_future[,d_ahead],-std_H,std_H)
		if(comp_numb[i]==3){
			nc<-pnorm(truncVaR-cum_logreturn,0,std_H)
			y_dens[,i]<-dnorm(y_future[,d_ahead],0,std_H)/nc
		    y_dens[y_future[,d_ahead]>truncVaR-cum_logreturn,i]<-0
		}
		if(comp_numb[i]==4){
			nc<-pnorm(truncVaR-cum_logreturn,-std_H,std_H)
			y_dens[,i]<-dnorm(y_future[,d_ahead],-std_H,std_H)/nc
		    y_dens[y_future[,d_ahead]>truncVaR-cum_logreturn,i]<-0
		}	
	}
	return(y_dens)
  }
}
y_comp_pred_allcomp<-function(n,y_future,sample_pars,mixprop,H_T,d_ahead,truncVaR,is_sampling){
  n_comp<-c(n-sum(floor(mixprop[-1]*n)),floor(mixprop[-1]*n)); n_vec<-list(1:(n_comp[1]))
  alpha0<-exp(sample_pars[,1]); alpha1<-abs(sample_pars[,2]); beta_par<-abs(sample_pars[,3])
  for(i in 2:8){
    if(n_comp[i]>0) n_vec[[i]]<-(sum(n_comp[1:(i-1)])+1):sum(n_comp[1:i])
	if(n_comp[i]==0) n_vec[[i]]<-NULL
  }
  np<-sum(n_comp[c(1,3,5,7)]); nq<-sum(n_comp[c(2,4,6,8)])
  np_vec<-unlist(n_vec[c(1,3,5,7)]); nq_vec<-unlist(n_vec[c(2,4,6,8)])
  cum_logreturn<-rep(0,n)
  if(is_sampling==T){
    y_future<-matrix(0,n,d_ahead)
   # Calculate H_T
	H_current<-H_T
    if(d_ahead>1){
	    for(k in 1:(d_ahead-1)){
            y_future[,k]<-rnorm(n,sd=sqrt(H_current))
			H_old<-H_current
			H_current<-alpha0+alpha1*y_future[,k]^2+beta_par*H_old
			infin_ind<-(1:n)[!is.finite(H_current)]
			if(length(infin_ind)>0) H_current[infin_ind]<-max(H_current[-infin_ind])
		}
	   if(d_ahead>2) cum_logreturn<-y_future[,1:(d_ahead-1)]%*%rep(1,d_ahead-1)
	   if(d_ahead==2) cum_logreturn<-y_future[,1]
	}
    infin_ind<-(1:n)[!is.finite(H_current)]
    if(length(infin_ind)>0) H_current[infin_ind]<-max(H_current[-infin_ind])
	std_H<-sqrt(H_current)
   # Generate samples
	y_last_mup<-rep(0,n); y_last_muq<-(-std_H)
	n13<-n_comp[1]+n_comp[3]; n13_vec<-c(n_vec[[1]],n_vec[[3]])
	n24<-n_comp[2]+n_comp[4]; n24_vec<-c(n_vec[[2]],n_vec[[4]])
	n5<-n_comp[5]; if(n5>0) n5_vec<-n_vec[[5]]
	n6<-n_comp[6]; if(n6>0) n6_vec<-n_vec[[6]]
	n7<-n_comp[7]; if(n7>0) n7_vec<-n_vec[[7]]
	n8<-n_comp[8]; if(n8>0) n8_vec<-n_vec[[8]]	
    if(n13>0) y_future[n13_vec,d_ahead]<-rnorm(n13,mean=0,sd=std_H[n13_vec])
	if(n24>0) y_future[n24_vec,d_ahead]<-rnorm(n24,mean=y_last_muq[n24_vec],sd=std_H[n24_vec])
	if(n5>0) y_future[n5_vec,d_ahead]<-rnorm_trunc(n5,0,std_H[n5_vec],truncVaR[1]-cum_logreturn[n5_vec])
	if(n6>0) y_future[n6_vec,d_ahead]<-rnorm_trunc(n6,y_last_muq[n6_vec],std_H[n6_vec],truncVaR[1]-cum_logreturn[n6_vec])
    if(n7>0) y_future[n7_vec,d_ahead]<-rnorm_trunc(n7,0,std_H[n7_vec],truncVaR[2]-cum_logreturn[n7_vec])
	if(n8>0) y_future[n8_vec,d_ahead]<-rnorm_trunc(n8,y_last_muq[n8_vec],std_H[n8_vec],truncVaR[2]-cum_logreturn[n8_vec])
   # Calculate densities	
    y_dens<-matrix(0,n,8); y_last<-y_future[,d_ahead]
    y_last_bound1<-truncVaR[1]-cum_logreturn; y_last_bound2<-truncVaR[2]-cum_logreturn
	y_dens[,c(1,3)]<-dnorm(y_last,y_last_mup,std_H)
	y_dens[,c(2,4)]<-dnorm(y_last,y_last_muq,std_H)
	nc_pVaR1<-pnorm(y_last_bound1,y_last_mup,std_H)
	nc_qVaR1<-pnorm(y_last_bound1,y_last_muq,std_H)
	y_dens[,5]<-dnorm(y_last,y_last_mup,std_H)/nc_pVaR1
	y_dens[y_last>y_last_bound1,5]<-0
	y_dens[,6]<-dnorm(y_last,y_last_muq,std_H)/nc_qVaR1
	y_dens[y_last>y_last_bound1,6]<-0
	nc_pVaR2<-pnorm(y_last_bound2,y_last_mup,std_H)
	nc_qVaR2<-pnorm(y_last_bound2,y_last_muq,std_H)	
	y_dens[,7]<-dnorm(y_last,y_last_mup,std_H)/nc_pVaR2
	y_dens[y_last>y_last_bound2,7]<-0
	y_dens[,8]<-dnorm(y_last,y_last_muq,std_H)/nc_qVaR2
	y_dens[y_last>y_last_bound2,8]<-0
	return(list(y_future[,d_ahead:1],y_dens))
   }
  if(is_sampling==F){
   # Calculate H_T
	if(d_ahead==1) y_future<-matrix(y_future,ncol=1)
	n<-dim(y_future)[1]
	H_current<-H_T
    if(d_ahead>1){
	    for(k in 1:(d_ahead-1)){
			H_old<-H_current
			H_current<-alpha0+alpha1*y_future[,k]^2+beta_par*H_old
			infin_ind<-(1:n)[!is.finite(H_current)]
			if(length(infin_ind)>0) H_current[infin_ind]<-max(H_current[-infin_ind])
		}
       if(d_ahead>2) cum_logreturn<-y_future[,1:(d_ahead-1)]%*%rep(1,d_ahead-1)
	   if(d_ahead==2) cum_logreturn<-y_future[,1]
    }
    infin_ind<-(1:n)[!is.finite(H_current)]
    if(length(infin_ind)>0) H_current[infin_ind]<-max(H_current[-infin_ind])
	std_H<-sqrt(H_current)
   # Calculate densities	
 	y_last_mup<-rep(0,n); y_last_muq<-(-std_H)
    y_dens<-matrix(0,n,8); y_last<-y_future[,d_ahead]
    y_last_bound1<-truncVaR[1]-cum_logreturn; y_last_bound2<-truncVaR[2]-cum_logreturn
	y_dens[,c(1,3)]<-dnorm(y_last,y_last_mup,std_H)
	y_dens[,c(2,4)]<-dnorm(y_last,y_last_muq,std_H)
	nc_pVaR1<-pnorm(y_last_bound1,y_last_mup,std_H)
	nc_qVaR1<-pnorm(y_last_bound1,y_last_muq,std_H)
	y_dens[,5]<-dnorm(y_last,y_last_mup,std_H)/nc_pVaR1
	y_dens[y_last>y_last_bound1,5]<-0
	y_dens[,6]<-dnorm(y_last,y_last_muq,std_H)/nc_qVaR1
	y_dens[y_last>y_last_bound1,6]<-0
	nc_pVaR2<-pnorm(y_last_bound2,y_last_mup,std_H)
	nc_qVaR2<-pnorm(y_last_bound2,y_last_muq,std_H)	
	y_dens[,7]<-dnorm(y_last,y_last_mup,std_H)/nc_pVaR2
	y_dens[y_last>y_last_bound2,7]<-0
	y_dens[,8]<-dnorm(y_last,y_last_muq,std_H)/nc_qVaR2
	y_dens[y_last>y_last_bound2,8]<-0
	return(y_dens)	
	}    
  }
y_comp_pred_prop2<-function(n,y_future,sample_pars,H_T,d_ahead,truncVaR,is_sampling,comp_numb){
  alpha0<-exp(sample_pars[,1]); alpha1<-abs(sample_pars[,2]); beta_par<-abs(sample_pars[,3])
  cum_logreturn<-0
  if(is_sampling==T){
    y_future<-matrix(0,n,d_ahead)
   # Calculate H_T
	H_current<-H_T
    if(d_ahead>1){
	    for(k in 1:(d_ahead-1)){
            if(comp_numb==1||comp_numb==3) y_future[,k]<-rnorm(n,sd=sqrt(H_current))
            if(comp_numb==2||comp_numb==4) y_future[,k]<-rnorm(n,mean=-sqrt(H_current),sd=sqrt(H_current))
			H_old<-H_current
			H_current<-alpha0+alpha1*y_future[,k]^2+beta_par*H_old
			infin_ind<-(1:n)[!is.finite(H_current)]
			if(length(infin_ind)>0) H_current[infin_ind]<-max(H_current[-infin_ind])
		}
	   if(d_ahead>2) cum_logreturn<-y_future[,1:(d_ahead-1)]%*%rep(1,d_ahead-1)
	   if(d_ahead==2) cum_logreturn<-y_future[,1]
	}	
    infin_ind<-(1:n)[!is.finite(H_current)]
    if(length(infin_ind)>0) H_current[infin_ind]<-max(H_current[-infin_ind])
	std_H<-sqrt(H_current)
	if(comp_numb==1) y_future[,d_ahead]<-rnorm(n,mean=0,sd=std_H)
	if(comp_numb==2) y_future[,d_ahead]<-rnorm(n,mean=-std_H,sd=std_H)
	if(comp_numb==3) y_future[,d_ahead]<-rnorm_trunc(n,mu=0,std=std_H,truncVaR-cum_logreturn)
	if(comp_numb==4) y_future[,d_ahead]<-rnorm_trunc(n,mu=-std_H,std=std_H,truncVaR-cum_logreturn)
    return(y_future[,d_ahead:1]) 
  }
  if(is_sampling==F){
   # Calculate H_T
	if(d_ahead==1) y_future<-matrix(y_future,ncol=1)
	n<-dim(y_future)[1]
	H_current<-H_T
	y_dens_dminus1_13<-rep(1,n); y_dens_dminus1_24<-rep(1,n);
    if(d_ahead>1){
	    for(k in 1:(d_ahead-1)){
            if(length(intersect(c(1,3),comp_numb))>0) y_dens_dminus1_13<-y_dens_dminus1_13*dnorm(y_future[,k],sd=sqrt(H_current))
            if(length(intersect(c(2,4),comp_numb))>0) y_dens_dminus1_24<-y_dens_dminus1_24*dnorm(y_future[,k],mean=-sqrt(H_current),sd=sqrt(H_current))
			H_old<-H_current
			H_current<-alpha0+alpha1*y_future[,k]^2+beta_par*H_old
			infin_ind<-(1:n)[!is.finite(H_current)]
			if(length(infin_ind)>0) H_current[infin_ind]<-max(H_current[-infin_ind])
		}
	   if(d_ahead>2) cum_logreturn<-y_future[,1:(d_ahead-1)]%*%rep(1,d_ahead-1)
	   if(d_ahead==2) cum_logreturn<-y_future[,1]
	}
    infin_ind<-(1:n)[!is.finite(H_current)]
    if(length(infin_ind)>0) H_current[infin_ind]<-max(H_current[-infin_ind])
	std_H<-sqrt(H_current); 
	y_dens<-matrix(0,n,length(comp_numb))
	for(i in 1:length(comp_numb)){
		if(comp_numb[i]==1) y_dens[,i]<-dnorm(y_future[,d_ahead],0,std_H)*y_dens_dminus1_13
		if(comp_numb[i]==2) y_dens[,i]<-dnorm(y_future[,d_ahead],-std_H,std_H)*y_dens_dminus1_24
		if(comp_numb[i]==3){
			nc<-pnorm(truncVaR-cum_logreturn,0,std_H)
			y_dens[,i]<-dnorm(y_future[,d_ahead],0,std_H)/nc*y_dens_dminus1_13
		    y_dens[y_future[,d_ahead]>truncVaR-cum_logreturn,i]<-0
		}
		if(comp_numb[i]==4){
			nc<-pnorm(truncVaR-cum_logreturn,-std_H,std_H)
			y_dens[,i]<-dnorm(y_future[,d_ahead],-std_H,std_H)/nc*y_dens_dminus1_24
		    y_dens[y_future[,d_ahead]>truncVaR-cum_logreturn,i]<-0
		}	
	}
	return(y_dens)
  }
}
y_comp_pred_allcomp_prop2<-function(n,y_future,sample_pars,mixprop,H_T,d_ahead,truncVaR,is_sampling){
  n_comp<-c(n-sum(floor(mixprop[-1]*n)),floor(mixprop[-1]*n)); n_vec<-list(1:(n_comp[1]))
  alpha0<-exp(sample_pars[,1]); alpha1<-abs(sample_pars[,2]); beta_par<-abs(sample_pars[,3])
  for(i in 2:8){
    if(n_comp[i]>0) n_vec[[i]]<-(sum(n_comp[1:(i-1)])+1):sum(n_comp[1:i])
	if(n_comp[i]==0) n_vec[[i]]<-NULL
  }
  np<-sum(n_comp[c(1,3,5,7)]); nq<-sum(n_comp[c(2,4,6,8)])
  np_vec<-unlist(n_vec[c(1,3,5,7)]); nq_vec<-unlist(n_vec[c(2,4,6,8)])
  y_dens_dminus1_1357<-rep(1,n); y_dens_dminus1_2468<-rep(1,n);
  if(is_sampling==T){
    y_future<-matrix(0,n,d_ahead)
   # Calculate H_T
	H_current<-H_T
	cum_logreturn<-rep(0,n)
    if(d_ahead>1){
	    for(k in 1:(d_ahead-1)){
            if(np>0) y_future[np_vec,k]<-rnorm(np,sd=sqrt(H_current[np_vec]))	    
            if(nq>0) y_future[nq_vec,k]<-rnorm(nq,mean=-sqrt(H_current),sd=sqrt(H_current[nq_vec]))
            y_dens_dminus1_1357<-y_dens_dminus1_1357*dnorm(y_future[,k],sd=sqrt(H_current))
            y_dens_dminus1_2468<-y_dens_dminus1_2468*dnorm(y_future[,k],mean=-sqrt(H_current),sd=sqrt(H_current))
			H_old<-H_current
			H_current<-alpha0+alpha1*y_future[,k]^2+beta_par*H_old
			infin_ind<-(1:n)[!is.finite(H_current)]
			if(length(infin_ind)>0) H_current[infin_ind]<-max(H_current[-infin_ind])
		}
	   if(d_ahead>2) cum_logreturn<-y_future[,1:(d_ahead-1)]%*%rep(1,d_ahead-1)
	   if(d_ahead==2) cum_logreturn<-y_future[,1]
	}
    infin_ind<-(1:n)[!is.finite(H_current)]
    if(length(infin_ind)>0) H_current[infin_ind]<-max(H_current[-infin_ind])
	std_H<-sqrt(H_current)
   # Generate samples
	y_last_mup<-rep(0,n); y_last_muq<-(-std_H)
	n13<-n_comp[1]+n_comp[3]; n13_vec<-c(n_vec[[1]],n_vec[[3]])
	n24<-n_comp[2]+n_comp[4]; n24_vec<-c(n_vec[[2]],n_vec[[4]])
	n5<-n_comp[5]; if(n5>0) n5_vec<-n_vec[[5]]
	n6<-n_comp[6]; if(n6>0) n6_vec<-n_vec[[6]]
	n7<-n_comp[7]; if(n7>0) n7_vec<-n_vec[[7]]
	n8<-n_comp[8]; if(n8>0) n8_vec<-n_vec[[8]]	
    if(n13>0) y_future[n13_vec,d_ahead]<-rnorm(n13,mean=0,sd=std_H[n13_vec])
	if(n24>0) y_future[n24_vec,d_ahead]<-rnorm(n24,mean=y_last_muq[n24_vec],sd=std_H[n24_vec])
	if(n5>0) y_future[n5_vec,d_ahead]<-rnorm_trunc(n5,0,std_H[n5_vec],truncVaR[1]-cum_logreturn[n5_vec])
	if(n6>0) y_future[n6_vec,d_ahead]<-rnorm_trunc(n6,y_last_muq[n6_vec],std_H[n6_vec],truncVaR[1]-cum_logreturn[n6_vec])
    if(n7>0) y_future[n7_vec,d_ahead]<-rnorm_trunc(n7,0,std_H[n7_vec],truncVaR[2]-cum_logreturn[n7_vec])
	if(n8>0) y_future[n8_vec,d_ahead]<-rnorm_trunc(n8,y_last_muq[n8_vec],std_H[n8_vec],truncVaR[2]-cum_logreturn[n8_vec])
   # Calculate densities	
    y_dens<-matrix(0,n,8); y_last<-y_future[,d_ahead]
    y_last_bound1<-truncVaR[1]-cum_logreturn; y_last_bound2<-truncVaR[2]-cum_logreturn
	y_dens[,c(1,3)]<-dnorm(y_last,y_last_mup,std_H)*y_dens_dminus1_1357
	y_dens[,c(2,4)]<-dnorm(y_last,y_last_muq,std_H)*y_dens_dminus1_2468
	nc_pVaR1<-pnorm(y_last_bound1,y_last_mup,std_H)
	nc_qVaR1<-pnorm(y_last_bound1,y_last_muq,std_H)
	y_dens[,5]<-dnorm(y_last,y_last_mup,std_H)/nc_pVaR1*y_dens_dminus1_1357
	y_dens[y_last>y_last_bound1,5]<-0
	y_dens[,6]<-dnorm(y_last,y_last_muq,std_H)/nc_qVaR1*y_dens_dminus1_2468
	y_dens[y_last>y_last_bound1,6]<-0
	nc_pVaR2<-pnorm(y_last_bound2,y_last_mup,std_H)
	nc_qVaR2<-pnorm(y_last_bound2,y_last_muq,std_H)	
	y_dens[,7]<-dnorm(y_last,y_last_mup,std_H)/nc_pVaR2*y_dens_dminus1_1357
	y_dens[y_last>y_last_bound2,7]<-0
	y_dens[,8]<-dnorm(y_last,y_last_muq,std_H)/nc_qVaR2*y_dens_dminus1_2468
	y_dens[y_last>y_last_bound2,8]<-0
	return(list(y_future[,d_ahead:1],y_dens))
   }
  if(is_sampling==F){
   # Calculate H_T
	if(d_ahead==1) y_future<-matrix(y_future,ncol=1)
	n<-dim(y_future)[1]
	H_current<-H_T
    if(d_ahead>1){
	    for(k in 1:(d_ahead-1)){
            y_dens_dminus1_1357<-y_dens_dminus1_1357*dnorm(y_future[,k],sd=sqrt(H_current))
            y_dens_dminus1_2468<-y_dens_dminus1_2468*dnorm(y_future[,k],mean=-sqrt(H_current),sd=sqrt(H_current))
			H_old<-H_current
			H_current<-alpha0+alpha1*y_future[,k]^2+beta_par*H_old
			infin_ind<-(1:n)[!is.finite(H_current)]
			if(length(infin_ind)>0) H_current[infin_ind]<-max(H_current[-infin_ind])
		}
    }
    infin_ind<-(1:n)[!is.finite(H_current)]
    if(length(infin_ind)>0) H_current[infin_ind]<-max(H_current[-infin_ind])
	std_H<-sqrt(H_current)
   # Calculate densities	
 	y_last_mup<-rep(0,n); y_last_muq<-(-std_H)
    y_dens<-matrix(0,n,8); y_last<-y_future[,d_ahead]
    y_last_bound1<-truncVaR[1]-cum_logreturn; y_last_bound2<-truncVaR[2]-cum_logreturn
	y_dens[,c(1,3)]<-dnorm(y_last,y_last_mup,std_H)*y_dens_dminus1_1357
	y_dens[,c(2,4)]<-dnorm(y_last,y_last_muq,std_H)*y_dens_dminus1_2468
	nc_pVaR1<-pnorm(y_last_bound1,y_last_mup,std_H)
	nc_qVaR1<-pnorm(y_last_bound1,y_last_muq,std_H)
	y_dens[,5]<-dnorm(y_last,y_last_mup,std_H)/nc_pVaR1*y_dens_dminus1_1357
	y_dens[y_last>y_last_bound1,5]<-0
	y_dens[,6]<-dnorm(y_last,y_last_muq,std_H)/nc_qVaR1*y_dens_dminus1_2468
	y_dens[y_last>y_last_bound1,6]<-0
	nc_pVaR2<-pnorm(y_last_bound2,y_last_mup,std_H)
	nc_qVaR2<-pnorm(y_last_bound2,y_last_muq,std_H)	
	y_dens[,7]<-dnorm(y_last,y_last_mup,std_H)/nc_pVaR2*y_dens_dminus1_1357
	y_dens[y_last>y_last_bound2,7]<-0
	y_dens[,8]<-dnorm(y_last,y_last_muq,std_H)/nc_qVaR2*y_dens_dminus1_2468
	y_dens[y_last>y_last_bound2,8]<-0
	return(y_dens)	
	}    
  }
  
 times<-function(x,y) return(x*y) 
 numb_g<-7
 I_G1_1<-NULL
 for(i in 1:numb_g) I_G1_1<-rbind(I_G1_1,diag(1,numb_g))
 I_G1_2<-matrix(0,numb_g^2,numb_g)
 for(i in 1:numb_g) I_G1_2[((i-1)*numb_g+1):(i*numb_g),i]<-rep(1,numb_g)
 I_G2_1<-t(I_G1_1)
 I_G2_2<-t(I_G1_2)
 A1_index<-NULL; A1_re_index<-NULL # rearrange numb_g*numb_g k1*k2 matrices
 k1<-numb_g; k2<-numb_g
 for(i in 1:numb_g){
  for(j in 1:numb_g){
   A1_index<-rbind(A1_index,cbind(rep(1:k1,rep(k2,k1))+(i-1)*k1,rep(1:k2,k1)+(j-1)*k2))
   A1_re_index<-rbind(A1_re_index,cbind(rep(1:k1,rep(k2,k1))+(j-1)*k1,rep(1:k2,k1)+(i-1)*k2))
  }
 }  
 A2_index<-NULL; A2_re_index<-NULL # rearrange numb_g*numb_g k1*k2 matrices
 k1<-numb_g; k2<-1
 for(i in 1:numb_g){
  for(j in 1:numb_g){
   A2_index<-rbind(A2_index,cbind(rep(1:k1,rep(k2,k1))+(i-1)*k1,rep(1:k2,k1)+(j-1)*k2))
   A2_re_index<-rbind(A2_re_index,cbind(rep(1:k1,rep(k2,k1))+(j-1)*k1,rep(1:k2,k1)+(i-1)*k2))
  }
 }
blockdiag<-function(A,k){ # bdiag is costly, need modification
 l1<-dim(A)[1]; l2<-dim(A)[2];
 A_diag<-matrix(0,k*l1,k*l2)
 for(i in 1:k) A_diag[((i-1)*l1+1):(i*l1),((i-1)*l2+1):(i*l2)]<-A
 return(A_diag)
}
beta_reg_est<-function(alpha_vec_excludefirst,pi_dens,q_dens,h,mu_hat,q_gamma,g_dens){
 n0<-length(pi_dens)  
 alpha_vec<-c(1-sum(alpha_vec_excludefirst),alpha_vec_excludefirst)
 q_alpha<-t(alpha_vec)%*%q_dens
 
 temp<-!(pi_dens==0)
 pi_dens<-pi_dens[temp]
 q_gamma<-q_gamma[temp]
 g_dens<-t(g_dens[,temp])
 q_alpha<-q_alpha[temp] 
 h<-h[temp]

browser() 
 part1_beta<-t(g_dens*((1/q_gamma)%*%t(rep(1,7))))
 part2_beta<-g_dens*((1/q_alpha)%*%t(rep(1,7)))
 part3_beta<-solve(part1_beta%*%part2_beta)
 part4_beta<-t(g_dens*((((h-mu_hat)*pi_dens)/q_alpha/q_gamma)%*%t(rep(1,7))))
 part5_beta<-part4_beta%*%rep(1,n0-length(temp))
 beta_MLE<-part3_beta%*%part5_beta

 return(beta_MLE)
}

partial_var_MLE_value<-function(theta_excludefirst,pi_dens,q_dens,h,mu_hat,q_gamma,g_dens){ #q_dens is numb_comp*n, g_dens is numb_g*n
 if(sum(theta_excludefirst<0)>0||sum(theta_excludefirst)>0.999) return(Inf)
 numb_g<-dim(g_dens)[1]
 alpha_vec<-c(1-sum(theta_excludefirst),theta_excludefirst)
 q_alpha<-t(alpha_vec)%*%q_dens
 temp<-!(pi_dens==0)
 pi_dens<-pi_dens[temp]
 q_gamma<-q_gamma[temp]
 g_dens<-g_dens[,temp]
 q_alpha<-q_alpha[temp] 
 h<-h[temp]
 #
 cc1<-min(q_gamma)
 q_gamma_temp<-q_gamma
 q_gamma<-q_gamma/cc1
 cc2<-median(pi_dens)
 pi_dens_temp<-pi_dens
 pi_dens<-pi_dens/cc2
 #
 f<-(h-mu_hat)*pi_dens
 part1_beta<-g_dens/(rep(1,numb_g)%*%t(q_alpha))
 part2_beta<-t(g_dens/(rep(1,numb_g)%*%t(q_gamma)))
 B_inv<-solve(part1_beta%*%part2_beta)
 #
 part4_beta<-f/q_gamma
 part5_beta<-part1_beta%*%part4_beta
 beta_MLE<-B_inv%*%part5_beta
 #
 G<-g_dens
 var_MLE_val<-mean(((f-c(t(beta_MLE)%*%G))/q_alpha)*((f-c(t(beta_MLE)%*%G))/q_gamma)) 
 var_MLE_val<-var_MLE_val/cc1
 return(var_MLE_val)
}
var_MLE<-function(theta_excludefirst,pi_dens,q_dens,h_all,mu_hat_all,q_gamma,g_dens,fnscale){
   # h is n0*length(rhos) matrix with each column being the indicator function of each VaR; mu_hat_all includse all rho for VaR; q_dens is numb_comp*n0 matrix; g_dens is numb_g*n matrix
   numb_mu<-length(mu_hat_all)
   var_MLE_part<-0
   for(i in 1:numb_mu) var_MLE_part[i]<-partial_var_MLE_value(theta_excludefirst,pi_dens,q_dens,h_all[,i],mu_hat_all[i],q_gamma,g_dens)
   var_MLE_value<-sum(var_MLE_part)/numb_mu
   return(var_MLE_value/fnscale)
}
partial_var_MLE_value_beta<-function(theta_excludefirst_beta,pi_dens,q_dens,h,mu_hat,q_gamma,g_dens){ #q_dens is numb_comp*n, g_dens is numb_g*n
 theta_excludefirst<-theta_excludefirst_beta[1:(p-1)]
 beta_MLE<-theta_excludefirst_beta[-(1:(p-1))]
 if(sum(theta_excludefirst<0)>0||sum(theta_excludefirst)>0.999) return(Inf)
 numb_g<-dim(g_dens)[1]
 alpha_vec<-c(1-sum(theta_excludefirst),theta_excludefirst)
 q_alpha<-t(alpha_vec)%*%q_dens
 temp<-!(pi_dens==0)
 pi_dens<-pi_dens[temp]
 q_gamma<-q_gamma[temp]
 g_dens<-g_dens[,temp]
 q_alpha<-q_alpha[temp] 
 h<-h[temp]
 #
 cc1<-min(q_gamma)
 q_gamma_temp<-q_gamma
 q_gamma<-q_gamma/cc1
 cc2<-median(pi_dens)
 pi_dens_temp<-pi_dens
 pi_dens<-pi_dens/cc2
 #
 f<-(h-mu_hat)*pi_dens
 #
 G<-g_dens
 var_MLE_val<-mean(((f-c(t(beta_MLE)%*%G))/q_alpha)*((f-c(t(beta_MLE)%*%G))/q_gamma)) 
 var_MLE_val<-var_MLE_val/cc1
 return(var_MLE_val)
}
var_MLE_beta<-function(theta_excludefirst_beta,pi_dens,q_dens,h_all,mu_hat_all,q_gamma,g_dens,fnscale){
   # h is n0*length(rhos) matrix with each column being the indicator function of each VaR; mu_hat_all includse all rho for VaR; q_dens is numb_comp*n0 matrix; g_dens is numb_g*n matrix
   numb_mu<-length(mu_hat_all)
   var_MLE_part<-0
   #for(i in 1:numb_mu) var_MLE_part[i]<-partial_var_MLE_value(theta_excludefirst,pi_dens,q_dens,h_all[,i],mu_hat_all[i],q_gamma,g_dens)
   for(i in 1:numb_mu) var_MLE_part[i]<-partial_var_MLE_value_beta(theta_excludefirst_beta,pi_dens,q_dens,h_all[,i],mu_hat_all[i],q_gamma,g_dens)
   var_MLE_value<-sum(var_MLE_part)/numb_mu
   return(var_MLE_value/fnscale)
}

 numb_g<-7
 I_G1_1<-NULL
 for(i in 1:numb_g) I_G1_1<-rbind(I_G1_1,diag(1,numb_g))
 I_G1_2<-matrix(0,numb_g^2,numb_g)
 for(i in 1:numb_g) I_G1_2[((i-1)*numb_g+1):(i*numb_g),i]<-rep(1,numb_g)
 I_G2_1<-t(I_G1_1)
 I_G2_2<-t(I_G1_2)

partial_var_MLE_gradient<-function(theta_excludefirst,pi_dens,q_dens,h,mu_hat,q_gamma,g_dens){ # the form of g_dens has to be (q2-q1,q3-q1,...,q_p-q1)
 if(sum(theta_excludefirst<0)>0||sum(theta_excludefirst)>0.999) return(Inf)

 numb_g<-dim(g_dens)[1]
 I_G1_1<-NULL
 for(i in 1:numb_g) I_G1_1<-rbind(I_G1_1,diag(1,numb_g))
 I_G1_2<-matrix(0,numb_g^2,numb_g)
 for(i in 1:numb_g) I_G1_2[((i-1)*numb_g+1):(i*numb_g),i]<-rep(1,numb_g)
 I_G2_1<-t(I_G1_1)
 I_G2_2<-t(I_G1_2)

 alpha_vec<-c(1-sum(theta_excludefirst),theta_excludefirst)
 q_alpha<-t(alpha_vec)%*%q_dens
 #temp<-!((pi_dens==0)&(q_alpha==0))
 temp<-!(pi_dens==0)
 pi_dens<-pi_dens[temp]
 q_gamma<-q_gamma[temp]
 g_dens<-g_dens[,temp]
 q_alpha<-q_alpha[temp]
 h<-h[temp]
 n0<-length(pi_dens)
 #
 cc1<-min(q_gamma)
 q_gamma_temp<-q_gamma
 q_gamma<-q_gamma/cc1
 cc2<-median(pi_dens)
 pi_dens_temp<-pi_dens
 pi_dens<-pi_dens/cc2
 #
 f<-(h-mu_hat)*pi_dens
 part1_beta<-g_dens/(rep(1,numb_g)%*%t(q_alpha))
 part2_beta<-t(g_dens/(rep(1,numb_g)%*%t(q_gamma)))
 B_inv<-solve(part1_beta%*%part2_beta)
 #
 part4_beta<-f/q_gamma
 part5_beta<-part1_beta%*%part4_beta
 beta_MLE<-B_inv%*%part5_beta
 #
 G<-g_dens
 tau1<-(f-c(t(beta_MLE)%*%G))/q_alpha/q_gamma
 #
 q_alpha2_gamma<-1/q_alpha^2/q_gamma #could be Inf
 C<-part5_beta
 B_inv_C_diag<-blockdiag(B_inv%*%C,numb_g)
 G2<-(t(G)%*%I_G2_1)*(((q_alpha2_gamma%*%t(rep(1,numb_g)))*t(G))%*%I_G2_2)
 G3<-((f*q_alpha2_gamma)%*%t(rep(1,numb_g)))*t(G)
 #
 B_p<--G%*%G2
 C_p<--G%*%G3
 #
 beta_gradient<-t(-B_inv%*%B_p%*%B_inv_C_diag+B_inv%*%C_p)

 var_gradient_part1<-G%*%(tau1^2*q_gamma)/n0
 var_gradient_part2<-beta_gradient%*%(G*(rep(1,numb_g)%*%t(tau1)))%*%rep(1,n0)*2/n0
 var_gradient<-as.vector(-var_gradient_part1-var_gradient_part2)

 var_gradient<-var_gradient/cc1
 return(var_gradient)
}
var_MLE_gradient<-function(theta_excludefirst,pi_dens,q_dens,h_all,mu_hat_all,q_gamma,g_dens,fnscale){
   # h is n0*length(rhos) matrix with each column being the indicator function of each VaR; mu_hat_all includse all rho for VaR; q_dens is numb_comp*n0 matrix; g_dens is numb_g*n matrix
   numb_mu<-length(mu_hat_all)
   numb_comp<-8
   var_MLE_gradient_part<-matrix(0,numb_comp-1,numb_mu)
   for(i in 1:numb_mu) var_MLE_gradient_part[,i]<-partial_var_MLE_gradient(theta_excludefirst,pi_dens,q_dens,h_all[,i],mu_hat_all[i],q_gamma,g_dens)
   var_MLE_gradient_value<-rowSums(var_MLE_gradient_part)/numb_mu
   return(var_MLE_gradient_value/fnscale)
}
partial_var_MLE_gradient_beta<-function(theta_excludefirst_beta,pi_dens,q_dens,h,mu_hat,q_gamma,g_dens){ # the form of g_dens has to be (q2-q1,q3-q1,...,q_p-q1)
 theta_excludefirst<-theta_excludefirst_beta[1:(p-1)]
 beta_MLE<-theta_excludefirst_beta[-(1:(p-1))]
 if(sum(theta_excludefirst<0)>0||sum(theta_excludefirst)>0.999) return(Inf)
 numb_g<-dim(g_dens)[1]
 alpha_vec<-c(1-sum(theta_excludefirst),theta_excludefirst)
 q_alpha<-t(alpha_vec)%*%q_dens
 #temp<-!((pi_dens==0)&(q_alpha==0))
 temp<-!(pi_dens==0)
 pi_dens<-pi_dens[temp]
 q_gamma<-q_gamma[temp]
 g_dens<-g_dens[,temp]
 q_alpha<-q_alpha[temp]
 h<-h[temp]
 n0<-length(pi_dens)
 #
 cc1<-min(q_gamma)
 q_gamma_temp<-q_gamma
 q_gamma<-q_gamma/cc1
 cc2<-median(pi_dens)
 pi_dens_temp<-pi_dens
 pi_dens<-pi_dens/cc2
 #
 f<-(h-mu_hat)*pi_dens
 #
 G<-g_dens
 tau1<-(f-c(t(beta_MLE)%*%G))/q_alpha/q_gamma
#
 var_gradient<-c(-G%*%(tau1^2*q_gamma)/n0,-G%*%tau1*2/n0)
 var_gradient<-as.vector(var_gradient/cc1)
 return(var_gradient)
}
var_MLE_gradient_beta<-function(theta_excludefirst_beta,pi_dens,q_dens,h_all,mu_hat_all,q_gamma,g_dens,fnscale){
   # h is n0*length(rhos) matrix with each column being the indicator function of each VaR; mu_hat_all includse all rho for VaR; q_dens is numb_comp*n0 matrix; g_dens is numb_g*n matrix
   numb_mu<-length(mu_hat_all)
   numb_comp<-8
   var_MLE_gradient_part<-matrix(0,numb_comp-1,numb_mu)
   #for(i in 1:numb_mu) var_MLE_gradient_part[,i]<-partial_var_MLE_gradient(theta_excludefirst,pi_dens,q_dens,h_all[,i],mu_hat_all[i],q_gamma,g_dens)
   for(i in 1:numb_mu) var_MLE_gradient_part[,i]<-partial_var_MLE_gradient_beta(theta_excludefirst_beta,pi_dens,q_dens,h_all[,i],mu_hat_all[i],q_gamma,g_dens)
   var_MLE_gradient_value<-rowSums(var_MLE_gradient_part)/numb_mu
   return(var_MLE_gradient_value/fnscale)
}

partial_var_MLE_hessian<-function(theta_excludefirst,pi_dens,q_dens,h,mu_hat,q_gamma,g_dens){ #q_dens is p*n, g_dens is (p-1)*n
 if(sum(theta_excludefirst<0)>0||sum(theta_excludefirst)>0.999) return(list(value=Inf,gradient=Inf,hessian=Inf))
 
 numb_g<-dim(g_dens)[1]
 n0<-length(pi_dens)  
 alpha_vec<-c(1-sum(theta_excludefirst),theta_excludefirst)
 q_alpha<-t(alpha_vec)%*%q_dens
 temp<-!((pi_dens==0)&(q_alpha==0))
 pi_dens<-pi_dens[temp]
 q_gamma<-q_gamma[temp]
 g_dens<-g_dens[,temp]
 q_alpha<-q_alpha[temp]
 h<-h[temp]
 #
 cc1<-min(q_gamma)
 q_gamma_temp<-q_gamma
 q_gamma<-q_gamma/cc1
 cc2<-median(pi_dens)
 pi_dens_temp<-pi_dens
 pi_dens<-pi_dens/cc2
 #
 f<-(h-mu_hat)*pi_dens
 part1_beta<-g_dens*(rep(1,numb_g)%*%t(1/q_alpha))
 part2_beta<-t(g_dens*(rep(1,numb_g)%*%t(1/q_gamma)))
 B_inv<-solve(part1_beta%*%part2_beta)
 B_inv_diag<-blockdiag(B_inv,numb_g)
 #
 part4_beta<-f/q_gamma
 part5_beta<-part1_beta%*%part4_beta
 beta_MLE<-B_inv%*%part5_beta
 #
 G<-g_dens
 tau1<-(f-c(t(beta_MLE)%*%G))/q_alpha/q_gamma
 tau2<-tau1/q_alpha
 tau3<-tau1^2/q_alpha*q_gamma
 G_tau1_diag<-blockdiag(G%*%tau1,numb_g)
 #
 q_alpha1<-1/q_alpha
 q_alpha2_gamma<-1/q_alpha^2/q_gamma #could be Inf
 C<-part5_beta
 Cdiag<-blockdiag(C,numb_g)
 B_inv_C_diag<-blockdiag(B_inv%*%C,numb_g)
 G1<-(I_G1_1%*%(G*(rep(1,numb_g)%*%t(q_alpha1))))*(I_G1_2%*%G)
 G2<-(t(G)%*%I_G2_1)*(((q_alpha2_gamma%*%t(rep(1,numb_g)))*t(G))%*%I_G2_2)
 G3<-((f*q_alpha2_gamma)%*%t(rep(1,numb_g)))*t(G)
 G4<-((f*q_alpha1)%*%t(rep(1,numb_g)))*t(G)
 #
 B_p<--G%*%G2
 C_p<--G%*%G3
 #
 beta_gradient<-t(-B_inv%*%B_p%*%B_inv_C_diag+B_inv%*%C_p)
 #
 beta_hessian_sp1<-B_inv_diag%*%t(B_p)%*%B_inv%*%B_p%*%B_inv_diag
 beta_hessian_sp2<-beta_hessian_sp1
 beta_hessian_sp2[A1_index]<-beta_hessian_sp2[A1_re_index]
 beta_hessian_part1<-beta_hessian_sp1%*%Cdiag
 beta_hessian_part2<-beta_hessian_sp2%*%Cdiag
 beta_hessian_part3<-2*B_inv_diag%*%G1%*%G2%*%B_inv_C_diag
 beta_hessian_part4<-B_inv_diag%*%t(B_p)%*%B_inv%*%C_p
 beta_hessian_part5<-beta_hessian_part4
 beta_hessian_part5[A2_index]<-beta_hessian_part5[A2_re_index]
 beta_hessian_part6<-2*B_inv_diag%*%t(G2)%*%G4
 beta_hessian<-beta_hessian_part1+beta_hessian_part2-beta_hessian_part3-beta_hessian_part4-beta_hessian_part5+beta_hessian_part6

 var_hessian_part1<-(G*(rep(1,numb_g)%*%t(tau3)))%*%t(G)*2/n0
 var_hessian_part2<-((beta_gradient%*%G)*(rep(1,numb_g)%*%t(tau2)))%*%t(G)*2/n0
 var_hessian_part3<-t(var_hessian_part2)
 var_hessian_part4<-t(beta_hessian)%*%G_tau1_diag*2/n0
 Z<-beta_gradient%*%G
 var_hessian_part5<-(Z*(rep(1,numb_g)%*%t(1/q_alpha/q_gamma)))%*%t(Z)*2/n0
 var_hessian<-as.matrix(var_hessian_part1+var_hessian_part2+var_hessian_part3-var_hessian_part4-var_hessian_part5)
 
 var_hessian<-var_hessian/cc1
 
 return(var_hessian)
}
var_MLE_hessian<-function(theta_excludefirst,pi_dens,q_dens,h_all,mu_hat_all,q_gamma,g_dens,fnscale){
   # h is n0*length(rhos) matrix with each column being the indicator function of each VaR; mu_hat_all includse all rho for VaR; q_dens is numb_comp*n0 matrix; g_dens is numb_g*n matrix
   numb_mu<-length(mu_hat_all)
   numb_comp<-length(theta_excludefirst)+1
   var_MLE_hessian_part<-list(0)
   var_MLE_hessian_value<-0
   for(i in 1:numb_mu){
     var_MLE_hessian_part[[i]]<-partial_var_MLE_hessian(theta_excludefirst,pi_dens,q_dens,h_all[,i],mu_hat_all[i],q_gamma,g_dens)
	 var_MLE_hessian_value<-var_MLE_hessian_value+var_MLE_hessian_part[[i]]
   }
   var_MLE_hessian_value<-var_MLE_hessian_value/(numb_mu*fnscale)
   return(var_MLE_hessian_value)
}

loglik_neg_trust<-function(zeta,h_prop,g_vec){
  # dens_all is n*p matrix, g_dens is n*(p-1) matrix
  num_g<-dim(g_vec)[2]
  q_alpha_zeta<-h_prop+g_vec%*%zeta
  if(sum(q_alpha_zeta<0)>0) return(list(value=Inf,gradient=Inf,hessian=Inf))
  l_funval<-sum(log(q_alpha_zeta))
  temp<-g_vec/(q_alpha_zeta%*%t(rep(1,num_g)))
  l_funval_der<-colSums(temp)
  l_funval_hessian<--crossprod(temp)
  return(list(value=-l_funval,gradient=-l_funval_der,hessian=-l_funval_hessian))
 }
MLE_mixture_p<-function(h_prop,g){
   numb_g<-dim(g)[2]
   minimizing_results<-trust(loglik_neg_trust,parinit=rep(0,numb_g),rinit=1,rmax=10,h_prop=h_prop,g_vec=g,iterlim = 10)
   zeta_MLE<-minimizing_results$argument
   return(zeta_MLE)
}

AdMit<-function(dimension,y_hist,h_ini,a_pars,n_t,df_t,max_step){ # The "Nelder-Mead" is costly, try to give the gradient function of AdMit_sample_weights and use "trust" or "BFGS", need modification
   Time<-length(y_hist)-1; d_ahead<-0
   #inistate<-c(0,0.8,0.1);

   inistate<-c(0,0.184526,0.715474);
   #inistate<-c(-3.087324,0,0.184526,0.715474);
   #inistate<-c(2.3644340,5.5226237,0.1665309,0.2907468,0.6092532);
   #inistate<-c(-6.3516406,-5.4911529,-1.8401861,-2.6684418,2.0706780,1.5721214,0.3286961,0.5713039) 
   ui_matrix<-matrix(0,d_ahead+5,d_ahead+3)
   ui_matrix[d_ahead+1,(d_ahead+2):(d_ahead+3)]<-rep(-1,2);
   ui_matrix[(d_ahead+2):(d_ahead+3),(d_ahead+2)]<-c(-1,1);
   ui_matrix[(d_ahead+4):(d_ahead+5),(d_ahead+3)]<-c(-1,1)
   ci_vec<-c(rep(-1,d_ahead),-1,rep(c(-1,0),2)) 
	step1_results<-constrOptim(theta=inistate,f=log_neg_targetdens,method="Nelder-Mead",ui=ui_matrix,ci=ci_vec,y_hist=y_hist,h_ini=h_ini,a_pars=a_pars)
   mu_AdMit<-(step1_results$par)[d_ahead+1:3]
   Sigma_AdMit_results<-solve(numericHessian(log_neg_targetdens,t0=step1_results$par,y_hist=y_hist,h_ini=h_ini,a_pars=a_pars))
   Sigma_AdMit<-Sigma_AdMit_results[d_ahead+1:3,d_ahead+1:3] 
   #Sigma_AdMit<-8*Sigma_AdMit
   Sigma_AdMit[1,1]<-4*Sigma_AdMit[1,1] 
   Sigma_AdMit[1,2:3]<-2*Sigma_AdMit[1,2:3]
   Sigma_AdMit[2:3,1]<-2*Sigma_AdMit[2:3,1] 
   return(list(mu_AdMit,Sigma_AdMit)) 
}

VaR_ISest<-function(x,rho_VaR,samp_weights){ # Estimate VaR
   n<-length(x)
   order_x<-order(x)
   x_sort<-x[order_x]
   normalized_samp_weights_sort<-samp_weights[order_x]/sum(samp_weights)
   percent_est<-cumsum(normalized_samp_weights_sort)
   temp<-percent_est-rho_VaR
   temp_ind<-(1:n)[temp<0]
   j<-max(temp_ind)
   VaR<-x_sort[j]+(x_sort[j+1]-x_sort[j])/(percent_est[j+1]-percent_est[j])*(rho_VaR-percent_est[j])
   return(VaR)
}
VaR_ARCH_stage1<-function(rhos,d_ahead,nt,y_hist,h_ini,a_pars,df_t,is_stage1){
   # mu_comp is dimension*numb_q1, y_future are in front of the pars with the last y_t+p in the first
   Time<-length(y_hist)-1 
   dimension<-d_ahead+3
   pars_prop<-AdMit(dimension,y_hist,h_ini,a_pars,nt,df_t,1)
   mu_comp<-pars_prop[[1]]
   Sigma_comp<-pars_prop[[2]]
   numb_q1<-2
   numb_comp<-numb_q1*(length(rhos)+2) 
   n0_all<-rep(nt*numb_q1,length(rhos)+2)
   n0<-sum(n0_all)
   VaR_hat<-0; VaR_hat_adj<-0; qtrunc_dens<-matrix(0,n0,length(rhos))
   comp_dens<-matrix(0,n0,numb_comp)
   target_dens<-rep(0,n0)
   samples_1stage<-matrix(0,n0,d_ahead+3)
   h_all<-matrix(0,nt*numb_q1*(length(rhos)+2),length(rhos))
   delta_lowbound_qt<-10^(-3)

   # Generate samples of parameters and calculate parameter part of target dens
   samples_1stage_pars_norm<-rmvnorm(sum(n0_all[c(1,3,4)]),mu_comp,Sigma_comp)
   samples_1stage_pars_t<-cbind(rmvt(sum(n0_all[2]),delta=mu_comp[1],sigma=matrix(Sigma_comp[1,1]),df=df_t,type = "shifted"),rmvt(sum(n0_all[2]),delta=mu_comp[2],sigma=matrix(Sigma_comp[2,2]),df=df_t,type = "shifted"),rmvt(sum(n0_all[2]),delta=mu_comp[3],sigma=matrix(Sigma_comp[3,3]),df=df_t,type = "shifted"))
   samples_1stage_pars<-rbind(samples_1stage_pars_norm[1:sum(n0_all[1]),],samples_1stage_pars_t,samples_1stage_pars_norm[(sum(n0_all[1])+1):sum(n0_all[c(1,3,4)]),])
   samples_1stage[,(d_ahead+1):(d_ahead+3)]<-samples_1stage_pars
   samples_1stage_pars_normdens<-dmvnorm(samples_1stage_pars,mu_comp,Sigma_comp)
   samples_1stage_pars_tdens<-dt((samples_1stage_pars[,1]-mu_comp[1])/sqrt(Sigma_comp[1,1]),df=df_t)*dt((samples_1stage_pars[,2]-mu_comp[2])/sqrt(Sigma_comp[2,2]),df=df_t)*dt((samples_1stage_pars[,3]-mu_comp[3])/sqrt(Sigma_comp[3,3]),df=df_t)/sqrt(Sigma_comp[1,1]*Sigma_comp[2,2]*Sigma_comp[3,3])   
   
   pars_targ_dens_1stage<-exp((samples_1stage[,d_ahead+1]-a_pars[1])^2/(-2*a_pars[2]^2))   
   
   # Calculate h_T and calculate h_1:T part of target dens 
   y2_hist<-y_hist^2
   nonzero_ind<-(1:n0)[as.logical(rowProds2(samples_1stage_pars[,2:3]>0)*(rowSums(samples_1stage_pars[,2:3])<1))]
   n0_nonzero<-length(nonzero_ind)
   alpha0<-exp(samples_1stage_pars[,1]); alpha1<-abs(samples_1stage_pars[,2]); beta_par<-abs(samples_1stage_pars[,3])
   partial_targ_dens_log_tmp<-rep(0,n0)  
   for(i in 1:Time){     
     if(i==1) H_current<-alpha0+alpha1*y_hist[1]^2+beta_par*h_ini
	 if(i>1) H_current<-alpha0+alpha1*Y_old+beta_par*H_old
	 Y_current<-y2_hist[i+1]
	 partial_targ_dens_log_tmp<-partial_targ_dens_log_tmp-.5*(Y_current/H_current+log(H_current))
	 H_old<-H_current
	 Y_old<-Y_current
    }
   H_T<-alpha0+alpha1*Y_old+beta_par*H_old
   infin_ind<-(1:n0)[!is.finite(H_T)]
   if(length(infin_ind)>0) H_T[infin_ind]<-max(H_T[-infin_ind])
  
   partial_targ_dens_log_1stage<-rep(-Inf,n0)
   partial_targ_dens_log_1stage[nonzero_ind]<-partial_targ_dens_log_tmp[nonzero_ind]
   partial_targ_dens_1stage<-exp(partial_targ_dens_log_1stage)

   # Generate y_future and Calculate densities
   n_comp_vec<-matrix(0,nt,numb_comp)
   for(i in 1:numb_comp) n_comp_vec[,i]<-((i-1)*nt+1):(i*nt)
   norm_t_vec<-c(n_comp_vec[,1:4])
   truncnorm_vec<-list(c(n_comp_vec[,5:6]))
   truncnorm_vec[[2]]<-c(n_comp_vec[,7:8])
   norm_t_trunc1_vec<-c(norm_t_vec,truncnorm_vec[[1]])
   
   samples_1stage[n_comp_vec[,1],1:d_ahead]<-y_comp_pred(nt,NULL,samples_1stage_pars[n_comp_vec[,1],],H_T[n_comp_vec[,1]],d_ahead,NULL,T,1)   
   samples_1stage[n_comp_vec[,2],1:d_ahead]<-y_comp_pred(nt,NULL,samples_1stage_pars[n_comp_vec[,2],],H_T[n_comp_vec[,2]],d_ahead,NULL,T,2)
   samples_1stage[n_comp_vec[,3],1:d_ahead]<-y_comp_pred(nt,NULL,samples_1stage_pars[n_comp_vec[,3],],H_T[n_comp_vec[,3]],d_ahead,NULL,T,1)
   samples_1stage[n_comp_vec[,4],1:d_ahead]<-y_comp_pred(nt,NULL,samples_1stage_pars[n_comp_vec[,4],],H_T[n_comp_vec[,4]],d_ahead,NULL,T,2)
   comp_dens_tmp<-y_comp_pred(NULL,samples_1stage[norm_t_vec,d_ahead:1],samples_1stage_pars[norm_t_vec,],H_T[norm_t_vec],d_ahead,NULL,F,1:2)
   comp_dens[norm_t_vec,1:4]<-cbind(comp_dens_tmp,comp_dens_tmp)*cbind(samples_1stage_pars_normdens[norm_t_vec],samples_1stage_pars_normdens[norm_t_vec],samples_1stage_pars_tdens[norm_t_vec],samples_1stage_pars_tdens[norm_t_vec])
   target_dens[norm_t_vec]<-comp_dens_tmp[,1]*partial_targ_dens_1stage[norm_t_vec]*pars_targ_dens_1stage[norm_t_vec]   
   samples_weights_tmp<-4*target_dens[norm_t_vec]/(comp_dens[norm_t_vec,1:4]%*%rep(1,4))
   zero_ind<-(1:(4*nt))[!is.finite(samples_weights_tmp)]
   if(length(zero_ind)>0) samples_weights_tmp[zero_ind]<-0
   
   if(d_ahead>1) cum_logreturn<-c(samples_1stage[norm_t_vec,1:d_ahead]%*%rep(1,d_ahead))
   if(d_ahead==1) cum_logreturn<-samples_1stage[norm_t_vec,1]
   VaR_hat[1]<-VaR_ISest(cum_logreturn,rhos[1],samples_weights_tmp)
   VaR_hat_adj[1]<-.8*VaR_hat[1]
   samples_1stage[n_comp_vec[,5],1:d_ahead]<-y_comp_pred(nt,NULL,samples_1stage_pars[n_comp_vec[,5],],H_T[n_comp_vec[,5]],d_ahead,VaR_hat_adj[1],T,3)	 
   samples_1stage[n_comp_vec[,6],1:d_ahead]<-y_comp_pred(nt,NULL,samples_1stage_pars[n_comp_vec[,6],],H_T[n_comp_vec[,6]],d_ahead,VaR_hat_adj[1],T,4)
   comp_dens_tmp<-y_comp_pred(NULL,samples_1stage[truncnorm_vec[[1]],d_ahead:1],samples_1stage_pars[truncnorm_vec[[1]],],H_T[truncnorm_vec[[1]]],d_ahead,NULL,F,1:2)
   comp_dens[truncnorm_vec[[1]],1:4]<-cbind(comp_dens_tmp,comp_dens_tmp)*cbind(samples_1stage_pars_normdens[truncnorm_vec[[1]]],samples_1stage_pars_normdens[truncnorm_vec[[1]]],samples_1stage_pars_tdens[truncnorm_vec[[1]]],samples_1stage_pars_tdens[truncnorm_vec[[1]]])
   target_dens[truncnorm_vec[[1]]]<-comp_dens_tmp[,1]*partial_targ_dens_1stage[truncnorm_vec[[1]]]*pars_targ_dens_1stage[truncnorm_vec[[1]]]    
   comp_dens_tmp<-y_comp_pred(NULL,samples_1stage[norm_t_trunc1_vec,d_ahead:1],samples_1stage_pars[norm_t_trunc1_vec,],H_T[norm_t_trunc1_vec],d_ahead,VaR_hat_adj[1],F,3:4)
   comp_dens[norm_t_trunc1_vec,5:6]<-comp_dens_tmp*cbind(samples_1stage_pars_normdens[norm_t_trunc1_vec],samples_1stage_pars_normdens[norm_t_trunc1_vec])
   samples_weights_tmp<-6*target_dens[norm_t_trunc1_vec]/(comp_dens[norm_t_trunc1_vec,1:6]%*%rep(1,6))
   zero_ind<-(1:(6*nt))[!is.finite(samples_weights_tmp)]
   if(length(zero_ind)>0) samples_weights_tmp[zero_ind]<-0
      
   if(d_ahead>1) cum_logreturn<-c(samples_1stage[norm_t_trunc1_vec,1:d_ahead]%*%rep(1,d_ahead))
   if(d_ahead==1) cum_logreturn<-samples_1stage[norm_t_trunc1_vec,1]
   VaR_hat[2]<-VaR_ISest(cum_logreturn,rhos[2],samples_weights_tmp)
   VaR_hat_adj[2]<-.8*VaR_hat[2]
   samples_1stage[n_comp_vec[,7],1:d_ahead]<-y_comp_pred(nt,NULL,samples_1stage_pars[n_comp_vec[,7],],H_T[n_comp_vec[,7]],d_ahead,VaR_hat_adj[2],T,3)	 
   samples_1stage[n_comp_vec[,8],1:d_ahead]<-y_comp_pred(nt,NULL,samples_1stage_pars[n_comp_vec[,8],],H_T[n_comp_vec[,8]],d_ahead,VaR_hat_adj[2],T,4)
   comp_dens_tmp<-y_comp_pred(NULL,samples_1stage[truncnorm_vec[[2]],d_ahead:1],samples_1stage_pars[truncnorm_vec[[2]],],H_T[truncnorm_vec[[2]]],d_ahead,VaR_hat_adj[1],F,1:4)
   comp_dens[truncnorm_vec[[2]],1:6]<-cbind(comp_dens_tmp[,1:2],comp_dens_tmp)*cbind(samples_1stage_pars_normdens[truncnorm_vec[[2]]],samples_1stage_pars_normdens[truncnorm_vec[[2]]],samples_1stage_pars_tdens[truncnorm_vec[[2]]],samples_1stage_pars_tdens[truncnorm_vec[[2]]],samples_1stage_pars_normdens[truncnorm_vec[[2]]],samples_1stage_pars_normdens[truncnorm_vec[[2]]])
   target_dens[truncnorm_vec[[2]]]<-comp_dens_tmp[,1]*partial_targ_dens_1stage[truncnorm_vec[[2]]]*pars_targ_dens_1stage[truncnorm_vec[[2]]]    
   comp_dens_tmp<-y_comp_pred(NULL,samples_1stage[,d_ahead:1],samples_1stage_pars,H_T,d_ahead,VaR_hat_adj[2],F,3:4)
   comp_dens[,7:8]<-comp_dens_tmp*cbind(samples_1stage_pars_normdens,samples_1stage_pars_normdens)
   
   g_dens<-t(comp_dens[,-1]-comp_dens[,1]%*%t(rep(1,numb_comp-1)))   
   if(is_stage1==T){
     ui_optim<-rbind(rep(-1,numb_comp-1),diag(rep(1,numb_comp-1)))
     ci_optim<-c(-1,rep(0,numb_q1-1),rep(delta_lowbound_qt,numb_q1),rep(0,numb_q1*length(rhos)))
     if(d_ahead>1) cum_logreturn<-c(samples_1stage[,1:d_ahead]%*%rep(1,d_ahead))
     if(d_ahead==1) cum_logreturn<-samples_1stage[,1]
     for(i in 1:length(rhos)) h_all[,i]<-(cum_logreturn<=VaR_hat[i])*1
     gamma_vec<-rep(1,numb_comp)/numb_comp
     gamma_vec_excludefirst<-gamma_vec[-1]
     q_gamma<-comp_dens%*%gamma_vec
     var_scale<-var_MLE(gamma_vec_excludefirst,target_dens,t(comp_dens),h_all,rhos,q_gamma,g_dens,1)
      
	theta_hat_results<-constrOptim(theta=gamma_vec_excludefirst,f=var_MLE,grad=var_MLE_gradient,method="BFGS",ui=ui_optim,ci=ci_optim,pi_dens=target_dens,q_dens=t(comp_dens),h_all=h_all,mu_hat_all=rhos,q_gamma=q_gamma,g_dens=g_dens,fnscale=var_scale,control=list(reltol=1e-05))
     theta_excludefirst<-theta_hat_results$par
     theta_hat<-c(1-sum(theta_excludefirst),theta_excludefirst)
	 alpha_sd=0
   }
   if(is_stage1==F){
     theta_hat<-rep(1,numb_comp)/numb_comp
	 alpha_sd=0
	}
	return(list(theta_hat,VaR_hat,mu_comp,Sigma_comp,comp_dens,target_dens,g_dens,samples_1stage,n0,alpha_sd))
} 
VaR_ARCH_twostage<-function(n,d_ahead,rhos,nt,y_hist,h_ini,a_pars,df_t){
   stage1_results<-VaR_ARCH_stage1(rhos,d_ahead,nt,y_hist,h_ini,a_pars,df_t,T)
   Time<-length(y_hist)-1
   theta_hat<-stage1_results[[1]]
   VaR_hat_stage1<-stage1_results[[2]]
   VaR_hat_adj<-.8*VaR_hat_stage1 
   mu_comp<-stage1_results[[3]]
   Sigma_comp<-stage1_results[[4]]
   comp_dens_stage1<-stage1_results[[5]]
   target_dens_stage1<-stage1_results[[6]]
   g_dens_stage1<-t(stage1_results[[7]])
   samples_stage1<-stage1_results[[8]]
   n0<-stage1_results[[9]]
   numb_q1<-2
   dimension<-d_ahead+3
   numb_comp<-numb_q1*(length(rhos)+2)
   gamma_vec<-rep(1,numb_comp)/numb_comp
   comp_ind_vec<-matrix(0,length(rhos)+2,numb_q1)
   for(i in 1:(length(rhos)+2)) comp_ind_vec[i,]<-((i-1)*numb_q1+1):(i*numb_q1)
   VaR_hat<-0
   pars_hat<-rep(0,3)
   
   # Stage2 sampling
   samples_stage2<-matrix(0,n-n0,dimension)
   comp_dens_stage2<-matrix(0,n-n0,numb_comp)
   # Determine the sample sizes needed and corresponding mixture proportions for each mixture of numb_q1 components
   n_comp<-0; theta_hat_comp<-matrix(0,length(rhos)+2,numb_q1)
   for(i in 1:(length(rhos)+2)){
	 n_comp[i]<-floor((n-n0)*sum(theta_hat[comp_ind_vec[i,]]))
	 theta_hat_comp[i,]<-theta_hat[comp_ind_vec[i,]]
   }
   n_comp<-c(n-n0-sum(n_comp[-1]),n_comp[-1])

   # Generate samples of parameters and calculate parameter part of target dens
   samples_stage2_pars_norm<-rmvnorm(sum(n_comp[c(1,3,4)]),mu_comp,Sigma_comp)
   samples_stage2_pars_t<-cbind(rmvt(sum(n_comp[2]),delta=mu_comp[1],sigma=matrix(Sigma_comp[1,1]),df=df_t,type = "shifted"),rmvt(sum(n_comp[2]),delta=mu_comp[2],sigma=matrix(Sigma_comp[2,2]),df=df_t,type = "shifted"),rmvt(sum(n_comp[2]),delta=mu_comp[3],sigma=matrix(Sigma_comp[3,3]),df=df_t,type = "shifted"))
   samples_stage2_pars<-rbind(samples_stage2_pars_norm[1:(n_comp[1]),],samples_stage2_pars_t,samples_stage2_pars_norm[(sum(n_comp[1])+1):sum(n_comp[c(1,3,4)]),])
   samples_stage2[,(d_ahead+1):(d_ahead+3)]<-samples_stage2_pars
   samples_stage2_pars_normdens<-dmvnorm(samples_stage2_pars,mu_comp,Sigma_comp)
   samples_stage2_pars_tdens<-dt((samples_stage2_pars[,1]-mu_comp[1])/sqrt(Sigma_comp[1,1]),df=df_t)*dt((samples_stage2_pars[,2]-mu_comp[2])/sqrt(Sigma_comp[2,2]),df=df_t)*dt((samples_stage2_pars[,3]-mu_comp[3])/sqrt(Sigma_comp[3,3]),df=df_t)/sqrt(Sigma_comp[1,1]*Sigma_comp[2,2]*Sigma_comp[3,3])
   pars_targ_dens<-exp((samples_stage2[,d_ahead+1]-a_pars[1])^2/(-2*a_pars[2]^2))   

   # Calculate h_T and calculate h_1:T part of target dens 
   y2_hist<-y_hist^2
   nonzero_ind<-(1:(n-n0))[as.logical(rowProds2(samples_stage2_pars[,2:3]>0)*(rowSums(samples_stage2_pars[,2:3])<1))]
   n_nonzero<-length(nonzero_ind)
   alpha0<-exp(samples_stage2_pars[,1]); alpha1<-abs(samples_stage2_pars[,2]); beta_par<-abs(samples_stage2_pars[,3])
   partial_targ_dens_log_tmp<-rep(0,n-n0)  
   for(i in 1:Time){     
     if(i==1) H_current<-alpha0+alpha1*y_hist[1]^2+beta_par*h_ini
	 if(i>1) H_current<-alpha0+alpha1*Y_old+beta_par*H_old
	 Y_current<-y2_hist[i+1]
	 partial_targ_dens_log_tmp<-partial_targ_dens_log_tmp-.5*(Y_current/H_current+log(H_current))
	 H_old<-H_current
	 Y_old<-Y_current
    }
   H_T<-alpha0+alpha1*Y_old+beta_par*H_old
   infin_ind<-(1:(n-n0))[!is.finite(H_T)]
   if(length(infin_ind)>0) H_T[infin_ind]<-max(H_T[-infin_ind])
   
   partial_targ_dens_log<-rep(-Inf,n-n0)
   partial_targ_dens_log[nonzero_ind]<-partial_targ_dens_log_tmp[nonzero_ind]
   partial_targ_dens<-exp(partial_targ_dens_log)
   
   # Generate y_future and Calculate densities 
   sampling_results<-y_comp_pred_allcomp(n-n0,NULL,samples_stage2_pars,theta_hat,H_T,d_ahead,VaR_hat_adj,T)
   samples_stage2[,1:d_ahead]<-sampling_results[[1]]
   comp_dens_stage2<-sampling_results[[2]]*cbind(samples_stage2_pars_normdens,samples_stage2_pars_normdens,samples_stage2_pars_tdens,samples_stage2_pars_tdens,samples_stage2_pars_normdens,samples_stage2_pars_normdens,samples_stage2_pars_normdens,samples_stage2_pars_normdens)
   
   # Calculate the MLE weights
   g_dens_stage2<-comp_dens_stage2[,-1]-comp_dens_stage2[,1]%*%t(rep(1,numb_comp-1))   
   comp_dens<-rbind(comp_dens_stage1,comp_dens_stage2)
   g_dens<-rbind(g_dens_stage1,g_dens_stage2)
   theta_tilde<-n0/n*gamma_vec+(n-n0)/n*theta_hat
   q_theta_tilde<-comp_dens%*%theta_tilde
   NA_ind<-(1:n)[!is.finite(q_theta_tilde)]
   if(length(NA_ind)>0) q_theta_tilde[NA_ind]<-0
   zero_ind<-(1:n)[q_theta_tilde==0]
   if(length(zero_ind)>0){
     q_theta_tilde_adj<-q_theta_tilde[-zero_ind]
     g_dens_adj<-g_dens[-zero_ind,] 
   }
   if(length(zero_ind)==0){
     q_theta_tilde_adj<-q_theta_tilde
     g_dens_adj<-g_dens
   }
   zeta_MLE<-MLE_mixture_p(q_theta_tilde_adj,g_dens_adj) 
    
   # Calculate the samples weights
   target_dens_stage2<-sampling_results[[2]][,1]*partial_targ_dens*pars_targ_dens
   target_dens<-c(target_dens_stage1,target_dens_stage2)
   if(length(zero_ind)>0) target_dens_adj<-target_dens[-zero_ind]
   if(length(zero_ind)==0) target_dens_adj<-target_dens 
   weights_sample_MLE<-target_dens_adj/(q_theta_tilde_adj+g_dens_adj%*%zeta_MLE)
   
   # Point estimate
   samples_all<-rbind(samples_stage1,samples_stage2)
   if(d_ahead>1) cum_logreturn<-c(samples_all[,1:d_ahead]%*%rep(1,d_ahead))
   if(d_ahead==1) cum_logreturn<-samples_all[,1]
   if(length(zero_ind)>0){
      for(i in 1:length(rhos)) VaR_hat[i]<-VaR_ISest(cum_logreturn[-zero_ind],rhos[i],weights_sample_MLE)
 	  for(k in 1:3) pars_hat[k]<-sum(samples_all[-zero_ind,d_ahead+k]*weights_sample_MLE)/sum(weights_sample_MLE)     
   }
   if(length(zero_ind)==0){
      for(i in 1:length(rhos)) VaR_hat[i]<-VaR_ISest(cum_logreturn,rhos[i],weights_sample_MLE)
      for(k in 1:3) pars_hat[k]<-sum(samples_all[,d_ahead+k]*weights_sample_MLE)/sum(weights_sample_MLE) 
   }
   return(list(VaR_hat,theta_hat,pars_hat,zeta_MLE,mu_comp,Sigma_comp,VaR_hat_stage1))
}
VaR_ARCH_onestage<-function(n,d_ahead,rhos,nt,y_hist,h_ini,a_pars,df_t){
   stage1_results<-VaR_ARCH_stage1(rhos,d_ahead,nt,y_hist,h_ini,a_pars,df_t,F)
   Time<-length(y_hist)-1
   theta_hat<-stage1_results[[1]]
   VaR_hat_stage1<-stage1_results[[2]]
   VaR_hat_adj<-.8*VaR_hat_stage1
   mu_comp<-stage1_results[[3]]
   Sigma_comp<-stage1_results[[4]]
   comp_dens_stage1<-stage1_results[[5]]
   target_dens_stage1<-stage1_results[[6]]
   g_dens_stage1<-t(stage1_results[[7]])
   samples_stage1<-stage1_results[[8]]
   n0<-stage1_results[[9]]
   numb_q1<-2
   dimension<-d_ahead+3
   numb_comp<-numb_q1*(length(rhos)+2)
   gamma_vec<-rep(1,numb_comp)/numb_comp
   comp_ind_vec<-matrix(0,length(rhos)+2,numb_q1)
   for(i in 1:(length(rhos)+2)) comp_ind_vec[i,]<-((i-1)*numb_q1+1):(i*numb_q1)
   VaR_hat<-0
   pars_hat<-rep(0,3)	

   # Stage2 sampling
   samples_stage2<-matrix(0,n-n0,dimension)
   comp_dens_stage2<-matrix(0,n-n0,numb_comp)
   # Determine the sample sizes needed and corresponding mixture proportions for each mixture of numb_q1 components
   n_comp<-0; theta_hat_comp<-matrix(0,length(rhos)+2,numb_q1)
   for(i in 1:(length(rhos)+2)){
	 n_comp[i]<-floor((n-n0)*sum(theta_hat[comp_ind_vec[i,]]))
	 theta_hat_comp[i,]<-theta_hat[comp_ind_vec[i,]]
   }
   n_comp<-c(n-n0-sum(n_comp[-1]),n_comp[-1])

   # Generate samples of parameters and calculate parameter part of target dens
   samples_stage2_pars_norm<-rmvnorm(sum(n_comp[c(1,3,4)]),mu_comp,Sigma_comp)
   samples_stage2_pars_t<-cbind(rmvt(sum(n_comp[2]),delta=mu_comp[1],sigma=matrix(Sigma_comp[1,1]),df=df_t,type = "shifted"),rmvt(sum(n_comp[2]),delta=mu_comp[2],sigma=matrix(Sigma_comp[2,2]),df=df_t,type = "shifted"),rmvt(sum(n_comp[2]),delta=mu_comp[3],sigma=matrix(Sigma_comp[3,3]),df=df_t,type = "shifted"))
   samples_stage2_pars<-rbind(samples_stage2_pars_norm[1:(n_comp[1]),],samples_stage2_pars_t,samples_stage2_pars_norm[(sum(n_comp[1])+1):sum(n_comp[c(1,3,4)]),])
   samples_stage2[,(d_ahead+1):(d_ahead+3)]<-samples_stage2_pars
   samples_stage2_pars_normdens<-dmvnorm(samples_stage2_pars,mu_comp,Sigma_comp)
   samples_stage2_pars_tdens<-dt((samples_stage2_pars[,1]-mu_comp[1])/sqrt(Sigma_comp[1,1]),df=df_t)*dt((samples_stage2_pars[,2]-mu_comp[2])/sqrt(Sigma_comp[2,2]),df=df_t)*dt((samples_stage2_pars[,3]-mu_comp[3])/sqrt(Sigma_comp[3,3]),df=df_t)/sqrt(Sigma_comp[1,1]*Sigma_comp[2,2]*Sigma_comp[3,3])
   pars_targ_dens<-exp((samples_stage2[,d_ahead+1]-a_pars[1])^2/(-2*a_pars[2]^2))   

   # Calculate h_T and calculate h_1:T part of target dens 
   y2_hist<-y_hist^2
   nonzero_ind<-(1:(n-n0))[as.logical(rowProds2(samples_stage2_pars[,2:3]>0)*(rowSums(samples_stage2_pars[,2:3])<1))]
   n_nonzero<-length(nonzero_ind)
   alpha0<-exp(samples_stage2_pars[,1]); alpha1<-abs(samples_stage2_pars[,2]); beta_par<-abs(samples_stage2_pars[,3])
   partial_targ_dens_log_tmp<-rep(0,n-n0)  
   for(i in 1:Time){     
     if(i==1) H_current<-alpha0+alpha1*y_hist[1]^2+beta_par*h_ini
	 if(i>1) H_current<-alpha0+alpha1*Y_old+beta_par*H_old
	 Y_current<-y2_hist[i+1]
	 partial_targ_dens_log_tmp<-partial_targ_dens_log_tmp-.5*(Y_current/H_current+log(H_current))
	 H_old<-H_current
	 Y_old<-Y_current
    }
   H_T<-alpha0+alpha1*Y_old+beta_par*H_old
   infin_ind<-(1:(n-n0))[!is.finite(H_T)]
   if(length(infin_ind)>0) H_T[infin_ind]<-max(H_T[-infin_ind])
   
   partial_targ_dens_log<-rep(-Inf,n-n0)
   partial_targ_dens_log[nonzero_ind]<-partial_targ_dens_log_tmp[nonzero_ind]
   partial_targ_dens<-exp(partial_targ_dens_log)
   
   # Generate y_future and Calculate densities 
   sampling_results<-y_comp_pred_allcomp(n-n0,NULL,samples_stage2_pars,theta_hat,H_T,d_ahead,VaR_hat_adj,T)
   samples_stage2[,1:d_ahead]<-sampling_results[[1]]
   comp_dens_stage2<-sampling_results[[2]]*cbind(samples_stage2_pars_normdens,samples_stage2_pars_normdens,samples_stage2_pars_tdens,samples_stage2_pars_tdens,samples_stage2_pars_normdens,samples_stage2_pars_normdens,samples_stage2_pars_normdens,samples_stage2_pars_normdens)
   
   # Calculate the MLE weights
   g_dens_stage2<-comp_dens_stage2[,-1]-comp_dens_stage2[,1]%*%t(rep(1,numb_comp-1))   
   comp_dens<-rbind(comp_dens_stage1,comp_dens_stage2)
   g_dens<-rbind(g_dens_stage1,g_dens_stage2)
   theta_tilde<-n0/n*gamma_vec+(n-n0)/n*theta_hat
   q_theta_tilde<-comp_dens%*%theta_tilde
   NA_ind<-(1:n)[!is.finite(q_theta_tilde)]
   if(length(NA_ind)>0) q_theta_tilde[NA_ind]<-0
   zero_ind<-(1:n)[q_theta_tilde==0]
   if(length(zero_ind)>0){
     q_theta_tilde_adj<-q_theta_tilde[-zero_ind]
     g_dens_adj<-g_dens[-zero_ind,] 
   }
   if(length(zero_ind)==0){
     q_theta_tilde_adj<-q_theta_tilde
     g_dens_adj<-g_dens
   }
   zeta_MLE<-MLE_mixture_p(q_theta_tilde_adj,g_dens_adj) 
    
   # Calculate the samples weights
   target_dens_stage2<-sampling_results[[2]][,1]*partial_targ_dens*pars_targ_dens  
   target_dens<-c(target_dens_stage1,target_dens_stage2)
   if(length(zero_ind)>0) target_dens_adj<-target_dens[-zero_ind]
   if(length(zero_ind)==0) target_dens_adj<-target_dens 
   weights_sample_MLE<-target_dens_adj/(q_theta_tilde_adj+g_dens_adj%*%zeta_MLE)
   
   # Point estimate
   samples_all<-rbind(samples_stage1,samples_stage2)
   if(d_ahead>1) cum_logreturn<-c(samples_all[,1:d_ahead]%*%rep(1,d_ahead))  
   if(d_ahead==1) cum_logreturn<-samples_all[,1]
   if(length(zero_ind)>0){
      for(i in 1:length(rhos)) VaR_hat[i]<-VaR_ISest(cum_logreturn[-zero_ind],rhos[i],weights_sample_MLE)
 	  for(k in 1:3) pars_hat[k]<-sum(samples_all[-zero_ind,d_ahead+k]*weights_sample_MLE)/sum(weights_sample_MLE)     
   }
   if(length(zero_ind)==0){
      for(i in 1:length(rhos)) VaR_hat[i]<-VaR_ISest(cum_logreturn,rhos[i],weights_sample_MLE)
      for(k in 1:3) pars_hat[k]<-sum(samples_all[,d_ahead+k]*weights_sample_MLE)/sum(weights_sample_MLE) 
   }
 #browser()  
   return(list(VaR_hat,theta_hat,pars_hat,zeta_MLE,mu_comp,Sigma_comp,VaR_hat_stage1))
}

##############################
# One, two and five days VaR
##############################
nt<-1e+4; n<-4e+6; a_pars<-c(-1,2); h_ini<-sd(y_hist)
rhos<-c(.05,.01); MLE_twostage_results<-list(0)
{d_ahead<-1; replic<-10
# two stage
Rprof("E:\\Dropbox\\Print\\runtime.out")
x11();plot(0,0,xlim=c(0,replic),ylim=c(0,replic))
for(i in 1:replic) {set.seed(200+i);MLE_twostage_results[[i]]<-VaR_ARCH_twostage(n,d_ahead,rhos,nt,y_hist,h_ini,a_pars,df_t=1);points(i,i,pch=20)} 
Rprof(NULL)
summaryRprof("E:\\Dropbox\\Print\\runtime.out")
# one stage
for(i in 1:replic) {set.seed(200+i);MLE_onestage_results[[i]]<-VaR_ARCH_onestage(n,d_ahead,rhos,nt,y_hist,h_ini,a_pars,df_t=1)}
}
save.image("1ahead_GARCH_twostage")
save.image("1ahead_GARCH_onestage")

{d_ahead<-2; replic<-300
# two stage
x11();plot(0,0,xlim=c(0,replic),ylim=c(0,replic))
for(i in 1:replic) {set.seed(200+i);MLE_twostage_results[[i]]<-VaR_ARCH_twostage(n,d_ahead,rhos,nt,y_hist,h_ini,a_pars,df_t=1);points(i,i,pch=20)}
# one stage
for(i in 1:replic) {set.seed(200+i);MLE_onestage_results[[i]]<-VaR_ARCH_onestage(n,d_ahead,rhos,nt,y_hist,h_ini,a_pars,df_t=1)}
}
save.image("2ahead_GARCH_twostage")
save.image("2ahead_GARCH_onestage")

{d_ahead<-5; replic<-300
# two stage
x11();plot(0,0,xlim=c(0,replic),ylim=c(0,replic))
for(i in 1:replic) {set.seed(200+i);MLE_twostage_results[[i]]<-VaR_ARCH_twostage(n,d_ahead,rhos,nt,y_hist,h_ini,a_pars,df_t=1);points(i,i,pch=20)}
# one stage 
for(i in 1:replic) {set.seed(200+i);MLE_onestage_results[[i]]<-VaR_ARCH_onestage(n,d_ahead,rhos,nt,y_hist,h_ini,a_pars,df_t=1)}
}
save.image("5ahead_GARCH_twostage")
save.image("5ahead_GARCH_onestage")

VaR_twostage<-list(0); VaR_onestage<-list(0)
load("1ahead_GARCH_twostage")
VaR_twostage[[1]]<-MLE_twostage_results
load("1ahead_GARCH_onestage")
VaR_onestage[[1]]<-MLE_onestage_results
load("2ahead_GARCH_twostage")
VaR_twostage[[2]]<-MLE_twostage_results
load("2ahead_GARCH_onestage")
VaR_onestage[[2]]<-MLE_onestage_results
load("5ahead_GARCH_twostage")
VaR_twostage[[3]]<-MLE_twostage_results
load("5ahead_GARCH_onestage")
VaR_onestage[[3]]<-MLE_onestage_results


replic<-40; results<-list(0)
VaR_results_twostage<-list(matrix(0,replic,2),matrix(0,replic,2),matrix(0,replic,2))
VaR_results_onestage<-list(matrix(0,replic,2),matrix(0,replic,2),matrix(0,replic,2))
theta_hat_all_twostage<-list(NULL,NULL,NULL)
VaR_stage1<-matrix(0,replic,2)
for(k in 1:3){
  for(i in 1:replic){
    VaR_results_twostage[[k]][i,1]<-VaR_twostage[[k]][[i]][[1]][1]
    VaR_results_twostage[[k]][i,2]<-VaR_twostage[[k]][[i]][[1]][2]
    theta_hat_all_twostage[[k]]<-rbind(theta_hat_all_twostage[[k]],VaR_twostage[[k]][[i]][[2]])
    #VaR_results_onestage[[k]][i,1]<-VaR_onestage[[k]][[i]][[1]][1]
    #VaR_results_onestage[[k]][i,2]<-VaR_onestage[[k]][[i]][[1]][2]
  }
  results[[k]]<-rbind(apply(VaR_results_twostage[[k]],2,mean),apply(VaR_results_onestage[[k]],2,mean),apply(VaR_results_twostage[[k]],2,var),apply(VaR_results_onestage[[k]],2,var))
  colnames(results[[k]])<-list("VaR1","VaR2")
  rownames(results[[k]])<-list("mean_twostage","mean_onestage","var_twostage","var_onestage")
}
names(results)<-list("1day","2days","5days")
results
  