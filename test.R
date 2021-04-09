library(MASS)
library(Matrix)
library(numDeriv)
library(mvtnorm)
library(msm)
library(maxLik)
library(weights)
library(trust)

logindex_hist<-read.csv2("SP500.csv",sep=",",dec=".",header=T,colClasses=c("Date",NA))
length_hist<-dim(logindex_hist)[1]
logreturn_hist<-logindex_hist$VALUE[-1]-logindex_hist$VALUE[-length_hist]
y_hist<-100*logreturn_hist[2701:2900]

log_neg_targetdens<-function(x,y_hist,h_ini,a_pars){
    Time<-length(y_hist)-1 #?# 只有199个y作为history，1个作为y0
    d_ahead<-length(x)-3
    pars<-x[(d_ahead+1):(d_ahead+3)]
    if(!(prod(pars[2:3]>0)*(sum(pars[2:3])<1))) return(Inf)
    y2_hist<-y_hist^2
    if(d_ahead>0){
        y_future<-x[d_ahead:1] #?# 用全joint分布，future不影响优化吗
        y2_future<-y_future^2
        Y<-c(y2_hist[-1],y2_future)
    }
    if(d_ahead==0) Y<-y2_hist[-1]
    H<-rep(0,Time+d_ahead)
    for(i in 1:(Time+d_ahead)){
        if(i==1) H[1]<-exp(pars[1])+pars[2]*y_hist[1]^2+pars[3]*h_ini
        if(i>1) H[i]<-exp(pars[1])+pars[2]*Y[i-1]+pars[3]*H[i-1]
    }
    priordens_a<-(pars[1]-a_pars[1])^2/(-2*a_pars[2]^2) #?# 不是normalized
    targ_dens_log<-(-.5)*(t(Y/H+log(H))%*%rep(1,Time+d_ahead))+priordens_a
    return(c(-targ_dens_log))   
}

nt<-1e+4; n<-4e+6; a_pars<-c(-1,2); h_ini<-sd(y_hist)
d_ahead<-0
Time<-length(y_hist)-1 

inistate<-c(0,0.184526,0.715474);
#inistate<-c(-3.087324,0,0.184526,0.715474);
#inistate<-c(2.3644340,5.5226237,0.1665309,0.2907468,0.6092532);
#inistate<-c(-6.3516406,-5.4911529,-1.8401861,-2.6684418,2.0706780,1.5721214,0.3286961,0.5713039) 
ui_matrix<-matrix(0,d_ahead+5,d_ahead+3)
ui_matrix[d_ahead+1,(d_ahead+2):(d_ahead+3)]<-rep(-1,2);
ui_matrix[(d_ahead+2):(d_ahead+3),(d_ahead+2)]<-c(-1,1);
ui_matrix[(d_ahead+4):(d_ahead+5),(d_ahead+3)]<-c(-1,1) #?# 前几行行全是零有什么用
ci_vec<-c(rep(-1,d_ahead),-1,rep(c(-1,0),2)) 
step1_results<-constrOptim(theta=inistate,f=log_neg_targetdens,method="Nelder-Mead",ui=ui_matrix,ci=ci_vec,y_hist=y_hist,h_ini=h_ini,a_pars=a_pars)
mu_AdMit<-(step1_results$par)[d_ahead+1:3]
Sigma_AdMit_results<-solve(numericHessian(log_neg_targetdens,t0=step1_results$par,y_hist=y_hist,h_ini=h_ini,a_pars=a_pars))
Sigma_AdMit<-Sigma_AdMit_results[d_ahead+1:3,d_ahead+1:3]
#Sigma_AdMit<-8*Sigma_AdMit
Sigma_AdMit[1,1]<-4*Sigma_AdMit[1,1] #?# 这里是用来加大方差的吗
Sigma_AdMit[1,2:3]<-2*Sigma_AdMit[1,2:3]
Sigma_AdMit[2:3,1]<-2*Sigma_AdMit[2:3,1] 

