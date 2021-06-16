library(ggplot2)
library(reshape2)

graphics.off()
rm(list=ls())
cat('\f')

toy<-read.csv("toy.csv")
data<-scale(toy[1:3])
d<-dist(data,method="euclidean")^2
fit1<-hclust(d,method="ward.D")

ggplot(mapping=aes(x=980:length(fit1$height),y=fit1$height[980:length(fit1$height)]))+
    geom_line()+
    geom_point()+
    labs(x="stage",y="height")

par(mar=c(1,4,1,1))
plot(fit1,sub="",xlab="",main="")

num=2
cluster<-cutree(fit1,k=num)
centers<-aggregate(x=data,by=list(cluster=cluster),FUN=mean)

fit2<-kmeans(x=data,centers=centers[-1],algorithm="MacQueen")

tb<-fit2$centers
tb<-data.frame(cbind(tb,cluster=1:num))
tbm<-melt(tb,id.vars="cluster")
tbm$cluster<-factor(tbm$cluster)
ggplot(tbm,aes(x=variable,y=value,group=cluster,colour=cluster))+
    geom_line(aes(linetype=cluster))+
    geom_point(aes(shape=cluster))+
    geom_hline(yintercept=0)+
    labs(x=NULL,y="mean")

table(toy$tail,fit2$cluster)

