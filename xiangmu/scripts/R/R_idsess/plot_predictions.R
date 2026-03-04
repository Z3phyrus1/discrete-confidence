# ------------------------------------------------------------------ #
# This script plots data together with model predictions
# (split by correctness of the first response), and further
# splits/facets by seq_group (seq==1 vs seq==2/3).
# ------------------------------------------------------------------ #

rm(list=ls())
setwd("D:/xiangmu/scripts/R")
source("./R/code_all_models.R")

# ------------------------------------------------------#
# plotting settings
sub_2_col <- rgb(80,80,255,maxColorValue=255)
bayes_col <- rgb(255,80,80,maxColorValue=255)
opti_col <- "black"
library(ggplot2)
library(viridis)
library(mlisi)
library(dplyr)
library(ggpubr)

nice_theme <- theme_bw()+
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

# ------------------------------------------------------#
# helpers
m_ci <- function(v,alpha=0.05){
  bootMeanSE(v) # bootstrapped SEM
}

prep_data <- function(df, cut_points){
  # binning
  df$bin_s1 <- cut(df$abs1, breaks=cut_points)
  df$x_s1   <- cut(df$abs1, breaks=cut_points)
  
  # aggregate model preds per id then across ids, keeping seq_group
  d_m_1 <- df %>%
    group_by(seq_group, x_s1, id) %>%
    summarise(across(c(p_r2_discrete, p_r2_bayesian, p_r2_fixed,
                       abs1, p_r2_naive, p_r2_optimal), mean),
              .groups="drop")
  d_m <- d_m_1 %>%
    group_by(seq_group, x_s1) %>%
    summarise(across(c(p_r2_discrete, p_r2_bayesian, p_r2_fixed,
                       abs1, p_r2_naive, p_r2_optimal), mean),
              .groups="drop")
  
  d_m <- d_m %>%
    left_join(
      d_m_1 %>% group_by(seq_group, x_s1) %>% summarise(
        p_r2_discrete_se = m_ci(p_r2_discrete),
        p_r2_bayesian_se = m_ci(p_r2_bayesian),
        p_r2_fixed_se    = m_ci(p_r2_fixed),
        p_r2_naive_se    = m_ci(p_r2_naive),
        p_r2_optimal_se  = m_ci(p_r2_optimal),
        .groups="drop"),
      by=c("seq_group","x_s1")
    )
  
  # observed data
  dag  <- df %>%
    group_by(seq_group, bin_s1, id) %>%
    summarise(abs1=mean(abs1), r2=mean(r2), .groups="drop")
  dag2 <- dag %>%
    group_by(seq_group, bin_s1) %>%
    summarise(s1=mean(abs1), r2=mean(r2),
              r2_se=bootMeanSE(r2), .groups="drop")
  
  list(d_m=d_m, dag2=dag2)
}

make_plot <- function(d_m, dag2, curve_col, fill_col, yval, yse, title_txt,
                      ylab_txt="p(choose 'right' option)\n difference from baseline",
                      show_xlab=TRUE, y_max, add_opt=TRUE, color_points="black",
                      x_coord_max=2.45, x_breaks=c(0,0.5,1,1.5,2), x_labels=c("0","0.5","1","1.5","2")){
  ggplot(d_m, aes(x=abs1, y=.data[[yval]], ymin=.data[[yval]]-.data[[yse]], ymax=.data[[yval]]+.data[[yse]])) +
    nice_theme +
    geom_hline(yintercept=0, color="dark grey") +
    geom_ribbon(alpha=0.3, size=0, fill=fill_col) +
    geom_line(size=0.6, color=curve_col) +
    geom_point(data=dag2, aes(x=s1, y=r2), inherit.aes=FALSE, pch=21, color=color_points) +
    geom_line(data=dag2, aes(x=s1, y=r2), inherit.aes=FALSE, size=0.4, color=color_points) +
    geom_errorbar(data=dag2, aes(x=s1, y=r2, ymin=r2-r2_se, ymax=r2+r2_se),
                  inherit.aes=FALSE, width=0, color=color_points, size=0.4) +
    {if(add_opt) geom_line(aes(y=p_r2_optimal), size=0.4, color="red", lty=2) else NULL} +
    coord_cartesian(xlim=c(0,x_coord_max), ylim=c(0,y_max)) +
    ggtitle(title_txt) +
    labs(x = if (show_xlab) expression(paste("|",S[1],"| [",sigma,"]")) else NULL,
         y = ylab_txt) +
    theme(plot.title = element_text(size = 8)) +
    scale_x_continuous(breaks=x_breaks, labels=x_labels) +
    facet_wrap(~seq_group)
}

# ------------------------------------------------------#
# GENERAL PLOT PARAMETERS
x_coord_max <- 2.45
y_coord_max_c1 <- 0.25
y_coord_max_c0 <- 0.30
fig_height <- 1.8
fig_width  <- 1.5

# ------------------------------------------------------#
# correct first decision (acc1==1)
d <- read.table("D:/xiangmu/data/behavior/idsess/data_wide_wmPred.txt", sep=";", header=TRUE)
d <- d[d$acc1==1,]
d$seq_group <- ifelse(d$seq == 1, "seq1", "seq2_3")

# transform as differences from baseline
d$p_r2_discrete <- d$p_r2_discrete - d$p_r2_naive
d$p_r2_bayesian <- d$p_r2_bayesian - d$p_r2_naive
d$p_r2_optimal  <- d$p_r2_optimal  - d$p_r2_naive
d$p_r2_fixed    <- d$p_r2_fixed    - d$p_r2_naive
d$r2            <- d$r2            - d$p_r2_naive

cut_points_c1 <- c(0, 0.5, 1, 1.5, 2, 20)
prep_c1 <- prep_data(d, cut_points_c1)
d_m_c1  <- prep_c1$d_m
dag2_c1 <- prep_c1$dag2

plo_bayes_c1 <- make_plot(d_m_c1, dag2_c1, curve_col=bayes_col, fill_col=bayes_col,
                          yval="p_r2_bayesian", yse="p_r2_bayesian_se",
                          title_txt="After correct\nfirst decision",
                          ylab_txt="p(choose 'right' option)\n difference from baseline",
                          show_xlab=TRUE, y_max=y_coord_max_c1, add_opt=TRUE, color_points="black",
                          x_coord_max=x_coord_max)
plo_discr_c1 <- make_plot(d_m_c1, dag2_c1, curve_col=sub_2_col, fill_col=sub_2_col,
                          yval="p_r2_discrete", yse="p_r2_discrete_se",
                          title_txt=" ", ylab_txt=" ",
                          show_xlab=TRUE, y_max=y_coord_max_c1, add_opt=TRUE, color_points="black",
                          x_coord_max=x_coord_max)
plo_fixed_c1 <- make_plot(d_m_c1, dag2_c1, curve_col="dark grey", fill_col="dark grey",
                          yval="p_r2_fixed", yse="p_r2_fixed_se",
                          title_txt=" ", ylab_txt=" ",
                          show_xlab=TRUE, y_max=y_coord_max_c1, add_opt=TRUE, color_points="black",
                          x_coord_max=x_coord_max)

# ------------------------------------------------------#
# wrong first decision (acc1==0)
d0 <- read.table("D:/xiangmu/data/behavior/idsess/data_wide_wmPred.txt", sep=";", header=TRUE)
d0 <- d0[d0$acc1==0,]
d0$seq_group <- ifelse(d0$seq == 1, "seq1", "seq2_3")

d0$p_r2_discrete <- d0$p_r2_discrete - d0$p_r2_naive
d0$p_r2_bayesian <- d0$p_r2_bayesian - d0$p_r2_naive
d0$p_r2_optimal  <- d0$p_r2_optimal  - d0$p_r2_naive
d0$p_r2_fixed    <- d0$p_r2_fixed    - d0$p_r2_naive
d0$r2            <- d0$r2            - d0$p_r2_naive

cut_points_c0 <- c(0, 0.5, 1, 1.5, 20)
prep_c0 <- prep_data(d0, cut_points_c0)
d_m_c0  <- prep_c0$d_m
dag2_c0 <- prep_c0$dag2

plo_bayes_c0 <- make_plot(d_m_c0, dag2_c0, curve_col=bayes_col, fill_col=bayes_col,
                          yval="p_r2_bayesian", yse="p_r2_bayesian_se",
                          title_txt="After wrong\nfirst decision",
                          ylab_txt=" ", show_xlab=TRUE, y_max=y_coord_max_c0,
                          add_opt=TRUE, color_points="black", x_coord_max=x_coord_max)
plo_discr_c0 <- make_plot(d_m_c0, dag2_c0, curve_col=sub_2_col, fill_col=sub_2_col,
                          yval="p_r2_discrete", yse="p_r2_discrete_se",
                          title_txt=" ", ylab_txt=" ",
                          show_xlab=TRUE, y_max=y_coord_max_c0,
                          add_opt=TRUE, color_points="black", x_coord_max=x_coord_max)
plo_fixed_c0 <- make_plot(d_m_c0, dag2_c0, curve_col="dark grey", fill_col="dark grey",
                          yval="p_r2_fixed", yse="p_r2_fixed_se",
                          title_txt="After wrong\nfirst decision", ylab_txt=" ",
                          show_xlab=TRUE, y_max=y_coord_max_c0,
                          add_opt=TRUE, color_points="black", x_coord_max=x_coord_max)

# ------------------------------------------------------#
# arrange and save (faceted plots inside each panel by seq_group)
pdf("D:/xiangmu/figures/behavior_idsess/pred_all.pdf", width=fig_width*2, height=fig_height*3)
ggarrange(
  plo_bayes_c0, plo_bayes_c1,
  plo_discr_c0, plo_discr_c1,
  plo_fixed_c0, plo_fixed_c1,
  ncol = 2, nrow = 3
)
dev.off()