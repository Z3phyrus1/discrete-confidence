# ------------------------------------------------------------------ #
# This script does the following (not necessarily in this order)
# 1) calculate group-level estimates of goodness of fit (AIC)
# 2) perform Bayesian model comparison (calculate exceedance probabilities)
# 3) plot the results
# 4) plot the average confidence functions, based on estimated parameters
# ------------------------------------------------------------------ #

rm(list=ls())
setwd("D:/xiangmu/scripts/R")
source("./R/code_all_models.R")

# ------------------------------------------------------------------ #
# plotting settings
sub_2_col <- rgb(80,80,255,maxColorValue=255)
bayes_col <- rgb(255,80,80,maxColorValue=255)
opti_col <- "black"
library(ggplot2)
library(viridis)
library(mlisi)
library(gridExtra)
library(bmsR)

nice_theme <- theme_bw() +
  theme(text=element_text(family="Helvetica",size=9),
        panel.border=element_blank(),
        strip.background = element_rect(fill="white",color="white",size=0),
        strip.text=element_text(size=rel(0.8)),
        panel.grid.major.x=element_blank(),
        panel.grid.major.y=element_blank(),
        panel.grid.minor=element_blank(),
        axis.line.x=element_line(size=.4),
        axis.line.y=element_line(size=.4),
        axis.text.x=element_text(size=7,color="black"),
        axis.text.y=element_text(size=7,color="black"),
        axis.line=element_line(size=.4),
        axis.ticks=element_line(color="black"))

# ------------------------------------------------------------------ #
# load
dfit <- read.table("D:/xiangmu/data/behavior/idsess/fit_par.txt", header=TRUE)

# seq 分组（后续绘图和 model_freq 分面用）
dfit$seq_group <- ifelse(dfit$seq == 1, "seq1", "seq2_3")

# ------------------------------------------------------------------ #
# group-level AIC
dfit_ag <- aggregate(AIC ~ model, dfit, sum)
dfit_ag$AIC_diff <- dfit_ag$AIC - min(dfit_ag$AIC)
print(dfit_ag)

# individual AIC differences (matrix)
dfit_ag_i <- aggregate(AIC ~ model + id, dfit, sum)
print(tapply(dfit_ag_i$AIC, list(dfit_ag_i$model, dfit_ag_i$id), mean) -
        matrix(rep(tapply(dfit_ag_i$AIC, list(dfit_ag_i$model, dfit_ag_i$id), mean)[1,],
                   4), 4, length(unique(dfit_ag_i$id)), byrow=TRUE))

# ------------------------------------------------------#
# plot confidence functions (as in Fig 3)
par_s2  <- c(median(dfit$w1[dfit$model=="sub2"], na.rm=TRUE),
             median(dfit$theta2[dfit$model=="sub2"], na.rm=TRUE))
par_m   <- median(dfit$m[dfit$model=="metanoise"], na.rm=TRUE)
par_fixb<- median(dfit$fixed_shift, na.rm=TRUE)

evidence   <- seq(0,2,length.out=500)
conf_opti  <- pnorm(evidence)
conf_m     <- pnorm(evidence, sd=par_m)
conf_s2    <- ifelse(evidence < par_s2[1], 0.5, (1-erf(par_s2[2]/sqrt(2)))/2)
conf_fixb  <- (1-erf(par_fixb/sqrt(2)))/2
d_cf <- data.frame(evidence, conf_opti, conf_s2, conf_fixb, conf_m)

bayes_col <- rgb(255,80,80,maxColorValue=255)
meta_col  <- rgb(255,150,0,maxColorValue=255)

pdf("D:/xiangmu/figures/behavior_idsess/conf_fixb.pdf", width=1.6, height=1.6)
ggplot(d_cf)+
  geom_line(aes(x=evidence,y=conf_opti),size=1,color=bayes_col)+
  nice_theme+
  labs(x=expression(paste("evidence [",sigma,"]")),y="confidence")+
  geom_line(aes(x=evidence,y=conf_fixb),size=1,color="dark grey")+
  annotate("text",x=1,y=pnorm(2.5),label="ideal Bayesian",color=bayes_col,size=2.5)+
  annotate("text",x=1.5,y=conf_fixb[1]-0.03,label="fixed-bias",color="dark grey",size=2.5)
dev.off()

pdf("D:/xiangmu/figures/behavior_idsess/conf_bayes.pdf", width=1.6, height=1.6)
ggplot(d_cf)+
  geom_line(aes(x=evidence,y=conf_opti),size=1,color=bayes_col)+
  nice_theme+
  labs(x=expression(paste("evidence [",sigma,"]")),y="confidence")+
  geom_line(aes(x=evidence,y=conf_m),size=1,color=meta_col)+
  annotate("text",x=1,y=pnorm(2.5),label="ideal Bayesian",color=bayes_col,size=2.5)+
  annotate("text",x=1.35,y=0.545,label="biased-Bayesian",color=meta_col,size=2.5)
dev.off()

pdf("D:/xiangmu/figures/behavior_idsess/conf_discrete.pdf", width=1.6, height=1.6)
ggplot(d_cf)+
  geom_line(aes(x=evidence,y=conf_opti),size=1,color=bayes_col)+
  nice_theme+
  labs(x=expression(paste("evidence [",sigma,"]")),y="confidence")+
  geom_line(aes(x=evidence,y=conf_s2),size=1,color="blue")+
  annotate("text",x=1.58,y=0.87,label="ideal",color=bayes_col,size=2.5)+
  annotate("text",x=1.58,y=0.82,label="Bayesian",color=bayes_col,size=2.5)+
  annotate("text",x=1.2,y=0.55,label="discrete",color="blue",size=2.5)
dev.off()

# ------------------------------------------------------#
# average noise over-estimation bias
M <- dfit$m[!is.na(dfit$m)]
print(mean(M))
print(bootMeanCI(M))

# ------------------------------------------------------#
# statistical tests on AIC values (fixed-effects model comparison)
m_aic <- aov(AIC ~ model + Error(id), dfit)
library(sjstats)
print(effectsize::eta_squared(m_aic, partial=TRUE))
print(summary(m_aic))
print(t.test(dfit$AIC[dfit$model=="metanoise"], dfit$AIC[dfit$model=="sub2"], paired=TRUE))
print(t.test(dfit$AIC[dfit$model=="bayes"], dfit$AIC[dfit$model=="sub2"], paired=TRUE))
print(t.test(dfit$AIC[dfit$model=="bayes"], dfit$AIC[dfit$model=="sub1"], paired=TRUE))
print(t.test(dfit$AIC[dfit$model=="sub1"], dfit$AIC[dfit$model=="sub2"], paired=TRUE))
print(t.test(dfit$AIC[dfit$model=="sub1"], dfit$AIC[dfit$model=="metanoise"], paired=TRUE))

# ------------------------------------------------------#
# plot pooled AIC values
d_plot_i <- {}
for(i in unique(dfit$id)){
  AIC_i <- with(dfit[dfit$id==i,], tapply(AIC, model, mean))
  AIC_diff_i <- AIC_i - AIC_i[which(names(AIC_i)=="sub2")]
  seq_i <- unique(dfit$seq[dfit$id==i])
  d_i <- data.frame(AIC=AIC_i,
                    AIC_diff=AIC_diff_i,
                    model=c("Bayesian","biased-Bayesian","fixed bias","discrete"),
                    id=i,
                    cl=factor(c(1,2,3,4)),
                    seq=seq_i[1])
  d_plot_i <- rbind(d_plot_i, d_i)
}

bootSumSE <- function(v, nsim=1000,...){
  bootfoo <- function(v,i) sum(v[i], na.rm=TRUE, ...)
  bootRes <- boot::boot(v, bootfoo, nsim)
  sd(bootRes$t, na.rm=TRUE)
}

d_plot <- aggregate(AIC_diff ~ model, d_plot_i, sum)
d_plot$cl <- factor(c(1,3,2,4))
d_plot$se <- aggregate(AIC_diff ~ model, d_plot_i, bootSumSE)$AIC_diff
print(aggregate(AIC_diff ~ model, d_plot_i, function(x){bootFooCI(x, foo="sum", nsim=10000)}))
d_plot$model <- reorder(d_plot$model, d_plot$AIC_diff)
d_plot_i$se <- NA

bayes_col <- rgb(255,80,80,maxColorValue=255)
meta_col  <- rgb(255,150,0,maxColorValue=255)

pdf("D:/xiangmu/figures/behavior_idsess/model_comp.pdf", height=4.8, width=1.3)
ggplot(d_plot, aes(x=model, y=AIC_diff, ymin=AIC_diff-se, ymax=AIC_diff+se, fill=model)) +
  geom_col(width=0.7) +
  geom_hline(yintercept=0, lty=2) +
  geom_errorbar(width=0) +
  nice_theme +
  scale_fill_manual(values=c(sub_2_col, meta_col, "dark grey", bayes_col), guide=FALSE) +
  labs(x="", y="summed AIC difference") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(breaks=seq(0,10000,100))
dev.off()

# ------------------------------------------------------#
# Bayesian (random-effects) model selection — 按 seq_group 分开
d_xp_all <- data.frame()
for (sg in unique(dfit$seq_group)) {
  d_sub <- subset(dfit, seq_group == sg)
  m <- tapply(d_sub$relModLik, list(d_sub$id, d_sub$model), mean)
  m <- m[, 1:4, drop = FALSE]
  m <- m / repmat(t(t(apply(m, 1, sum))), 1, 4)
  m <- log(m)
  m <- na.omit(m)
  if (nrow(m) == 0) next
  
  bms0 <- VB_bms(m, n_samples = 1e+06)
  tmp <- data.frame(
    xp    = bms0$pxp,
    model = c("Bayesian","biased-Bayesian","fixed-bias","discrete"),
    cl    = factor(c(4,3,2,1)),
    freq  = bms0$r,
    seq_group = sg
  )
  tmp$model <- reorder(tmp$model, c(4,3,2,1))
  d_xp_all <- rbind(d_xp_all, tmp)
}

pdf("D:/xiangmu/figures/behavior_idsess/model_freq.pdf", height=2.5, width=3.0)
ggplot(d_xp_all, aes(x=model, y=freq, fill=cl)) +
  geom_hline(yintercept=0, lty=2, size=0.2) +
  geom_bar(stat="identity", width=0.7) +
  nice_theme +
  scale_fill_manual(values=c(sub_2_col,"dark grey", meta_col, bayes_col), guide=FALSE) +
  labs(x="", y="p(model)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  facet_wrap(~ seq_group) +
  coord_cartesian(ylim=c(0,1))
dev.off()

# ------------------------------------------------------#
# Individual AIC by seq grouping (facet: seq==1 vs seq==2/3)
dfit$seq_group <- factor(dfit$seq_group, levels=c("seq1","seq2_3"))
d_plot_i <- {}
for(i in unique(dfit$id)){
  AIC_i <- with(dfit[dfit$id==i,], tapply(AIC, model, mean))[1:4]
  AIC_diff_i <- AIC_i - AIC_i[which(names(AIC_i)=="sub2")]
  seq_i <- unique(dfit$seq_group[dfit$id==i])
  d_i <- data.frame(AIC=AIC_i,
                    AIC_diff=AIC_diff_i,
                    model=c("Bayesian","biased-Bayesian","fixed bias","discrete"),
                    id=i,
                    cl=factor(c(1,2,3,4)),
                    seq_group=seq_i[1])
  d_plot_i <- rbind(d_plot_i, d_i)
}
d_plot_i$seq_group <- factor(d_plot_i$seq_group, levels=c("seq1","seq2_3"))

make_panel <- function(data_model, col_pos, col_neg, ylab_txt="AIC difference"){
  data_model$col_d <- ifelse(data_model$AIC_diff >= 0, col_pos, col_neg)
  data_model$id_ordered <- reorder(data_model$id, data_model$AIC_diff)
  ggplot(data_model, aes(x=id_ordered, y=AIC_diff, fill=col_d)) +
    geom_bar(stat="identity") +
    geom_text(aes(label=id, y=ifelse(AIC_diff>=0, AIC_diff+5, AIC_diff-5)),
              size=2, angle=90, hjust=ifelse(data_model$AIC_diff>=0, 0, 1)) +
    scale_fill_identity() +
    nice_theme +
    scale_x_discrete(labels=NULL, name=" ") +
    labs(y=ylab_txt) +
    coord_cartesian(ylim=c(-ceiling(max(abs(data_model$AIC_diff)))/2.5,
                           ceiling(max(abs(data_model$AIC_diff)))*1.2)) +
    facet_grid(. ~ seq_group)
}

p0 <- make_panel(subset(d_plot_i, model=="Bayesian"), sub_2_col, bayes_col, "AIC difference")
p1 <- make_panel(subset(d_plot_i, model=="biased-Bayesian"), sub_2_col, meta_col, " ")
p2 <- make_panel(subset(d_plot_i, model=="fixed bias"), sub_2_col, "dark grey", " ")

pdf("D:/xiangmu/figures/behavior_idsess/AIC_iindividual.pdf", height=2.5, width=8.5)
grid.arrange(p0, p1, p2, ncol=3)
dev.off()