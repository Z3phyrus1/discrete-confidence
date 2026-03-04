# ------------------------------------------------------------------ #
# This script does the following:
# 1) estimate internal noise, using data from control/training task
# 2) transform stimuli in internal noise units (correcting for bias)
# 3) remove sjs that have large bias & poor performance (2 in total)
# Matteo Lisi, 2019
# ------------------------------------------------------------------ #

# clear workspace
rm(list=ls())

# set directory
setwd("D:/11.13/nhb_data_code")

# load everything (noise estimate made based also on control task)
d <- read.table("D:/11.13/nhb_data_code/data/data_src.txt",header=T,sep=";")
tapply(d$decision,list(d$decision,d$task),length)

# this select only decisions with **equal prior probabilities**
# (both subjectively and objectively)
# i.e. during control task or first decision in dual-decision task
d$keep <- ifelse(d$decision==1 | d$dual==0, 1, 0)
# tapply(d$keep, list(d$task, d$decision,d$dual), sum) # sanity check

# exclude the rest
d <- d[d$keep==1,]

# load functions
source("D:/11.13/nhb_data_code/R/noise_estimation_functions.R")

# ------------------------------------------------------------------ #
# do estimation 
d_noise <- {}
for(i in unique(d$id)){
  cat("\n",i)
  FT <- estimateNoise(d[d$id==i,])
  d_line <- data.frame(t(FT))
  d_line$id <- i
  d_noise <- rbind(d_noise, d_line)
  write.table(d_noise, file="D:/11.13/nhb_data_code/data/noise_estimates.txt")
}

# display noise estimates
data.frame(d_noise$id, round(d_noise[,c(1,4)],digits=2))
# d_noise <- read.table("D:/11.13/nhb_data_code/data/noise_estimates.txt", header=T)

# ------------------------------------------------------------------ #
# now convert the stimuli in units of internal noise
d <- read.table("D:/11.13/nhb_data_code/data/data_src.txt",header=T,sep=";")
d$s <- NA
d$sign_s <- NA
for(i in unique(d$id)){
  d_line <- d[which(d$id==i),]
  
  # convert stimuli to noise sd (first bias is subtracted)
  d$s[d$id==i] <- ifelse(d$task[d$id==i]=="motion",
                         (d$signed_coherence[d$id==i]  - d_line$bias_m) / d_line$sigma_m,
                         (d$signed_mu[d$id==i]  - d_line$bias_o) / d_line$sigma_o)
  d$sign_s[d$id==i] <- sign(d$s[d$id==i])
}

# 要保留的id
id_to_keep <- c(101, 102, 103,  14,  15,  17,  18,  19,  21,  22,  23,  50,  61,
                 62,  63,  65,  68,  69,  71,  72,  73,  74,  75,  76,  78,  80,
                 82,  83,  84,  85,  86,  88,  89,  90,  92,  93,  95,  97,  98,
                 99)

# 仅保留指定id的数据
d <- d[d$id %in% id_to_keep, ]

# ------------------------------------------------------------------ #
# save dataset noise units
write.table(d, file="D:/11.13/nhb_data_code/data/data_nu.txt", quote=F, row.names=F, sep=";")

