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
setwd("D:/xiangmu/scripts/R")

# load everything (noise estimate made based also on control task)
d <- read.table("D:/xiangmu/data/behavior/id/data.csv",header=T,sep=",")# D:/xiangmu/data/behavior/id/data.csv
tapply(d$decision,list(d$decision,d$task),length)

# this select only decisions with **equal prior probabilities**
# (both subjectively and objectively)
# i.e. during control task or 430 decision in dual-decision task
d$keep <- ifelse(d$decision==1 | d$dual==0, 1, 0)#（修改了0变成1[改回来了]）
tapply(d$keep, list(d$task, d$decision,d$dual), sum) # sanity check

# exclude the rest
d <- d[d$keep==1,]
d$acc <- as.numeric(d$acc)
# load functions
source("./R/noise_estimation_functions.R")

# ------------------------------------------------------------------ #
# do estimation 
d_noise <- {}
for(i in unique(d$id)){
  cat("/n",i)
  FT <- estimateNoise(d[d$id==i,])
  
  d_line <- data.frame(t(FT))
  d_line$id <- i
  
  d_noise <- rbind(d_noise, d_line)
  write.table(d_noise, file="D:/xiangmu/data/behavior/id/noise_estimates.txt")#D:/xiangmu/data/behavior/id/noise_estimates.txt
}

# display noise estimates
data.frame(d_noise$id, round(d_noise[,c(1,4)],digits=2))
# d_noise <- read.table("D:/Lisi/data_34/noise_estimates.txt", header=T)

# ------------------------------------------------------------------ #
# now convert the stimuli in units of internal noise
# ------------------------------------------------------------------ #
# now convert the stimuli in units of internal noise
d <- read.table("D:/xiangmu/data/behavior/id/data.csv",header=T,sep=",")#D:/xiangmu/data/behavior/id/data.csv
d$s <- NA
d$sign_s <- NA
for(i in unique(d$id)){
  d_line <- d_noise[which(d_noise$id==i),]
  
  # convert stimuli to noise sd (430 bias is subtracted)
  d$s[d$id==i] <- ifelse(d$task[d$id==i]=="motion",
                         (d$signed_coherence[d$id==i]  - d_line$bias_m) / d_line$sigma_m,
                         (d$signed_mu[d$id==i]  - d_line$bias_o) / d_line$sigma_o)
  d$sign_s[d$id==i] <- sign(d$s[d$id==i])
}

# ------------------------------------------------------------------ #
# check participant biases
tapply(d_noise$bias_o, d_noise$id, mean)  # high bias: "sj400"
tapply(d_noise$bias_m, d_noise$id, mean)  # high bias: "sj1500"


# "large" in the sense that it is more than 2SD from group mean
excl_bias_o <- d_noise$id[abs((d_noise$bias_o-mean(d_noise$bias_o))/sd(d_noise$bias_o))>2.5]
excl_bias_m <- d_noise$id[abs((d_noise$bias_m-mean(d_noise$bias_m))/sd(d_noise$bias_m))>2.5]
if (!(is.integer(excl_bias_o) && length(excl_bias_o) == 0)){
  d <- d[d$id!=excl_bias_o,]
}
if (!(is.integer(excl_bias_m) && length(excl_bias_m) == 0)){
  d <- d[d$id!=excl_bias_m,]
}



# ------------------------------------------------------------------ #
# save dataset noise units
write.table(d, file="D:/xiangmu/data/behavior/id/data_nu.txt", quote=F, row.names=F, sep=";")## ------------------------------------------------------------------ #D:/xiangmu/data/behavior/id/data_nu.txt
# # This script does the following:
# # 1) estimate internal noise, using data from control/training task
# # 2) transform stimuli in internal noise units (correcting for bias)
# # 3) remove sjs that have large bias & poor performance (2 in total)
# # Matteo Lisi, 2019
# # ------------------------------------------------------------------ #
# 
# # clear workspace
# rm(list=ls())
# 
# # set directory
# setwd("D:/Lisi")
# 
# # load everything (noise estimate made based also on control task)
# d <- read.table("D:/reysy/EEG/data/data_1.csv",header=T,sep=",")#D:/xiangmu/data/behavior/id/data.csv
# tapply(d$decision,list(d$decision,d$task),length)
# 
# # this select only decisions with **equal prior probabilities**
# # (both subjectively and objectively)
# # i.e. during control task or 430 decision in dual-decision task
# d$keep <- ifelse(d$decision==1 | d$dual==0, 1, 0)#（修改了0变成1[改回来了]）
# tapply(d$keep, list(d$task, d$decision,d$dual), sum) # sanity check
# 
# # exclude the rest
# d <- d[d$keep==1,]
# d$acc <- as.numeric(d$acc)
# # load functions
# source("./R/noise_estimation_functions.R")
# 
# # ------------------------------------------------------------------ #
# # do estimation 
# d_noise <- {}
# for(i in unique(d$id)){
#   cat("/n",i)
#   FT <- estimateNoise(d[d$id==i,])
#   
#   d_line <- data.frame(t(FT))
#   d_line$id <- i
#   
#   d_noise <- rbind(d_noise, d_line)
#   write.table(d_noise, file="D:/reysy/EEG/data/noise_estimates.txt")#D:/xiangmu/data/behavior/id/noise_estimates.txt
# }
# 
# # display noise estimates
# data.frame(d_noise$id, round(d_noise[,c(1,4)],digits=2))
# # d_noise <- read.table("D:/Lisi/data_34/noise_estimates.txt", header=T)
# 
# # ------------------------------------------------------------------ #
# # now convert the stimuli in units of internal noise
# # ------------------------------------------------------------------ #
# # now convert the stimuli in units of internal noise
# d <- read.table("D:/reysy/EEG/data/data.csv",header=T,sep=",")#D:/xiangmu/data/behavior/id/data.csv
# d$s <- NA
# d$sign_s <- NA
# for(i in unique(d$id)){
#   d_line <- d_noise[which(d_noise$id==i),]
#   
#   # convert stimuli to noise sd (430 bias is subtracted)
#   d$s[d$id==i] <- ifelse(d$task[d$id==i]=="motion",
#                          (d$signed_coherence[d$id==i]  - d_line$bias_m) / d_line$sigma_m,
#                          (d$signed_mu[d$id==i]  - d_line$bias_o) / d_line$sigma_o)
#   d$sign_s[d$id==i] <- sign(d$s[d$id==i])
# }
# 
# # ------------------------------------------------------------------ #
# # check participant biases
# tapply(d_noise$bias_o, d_noise$id, mean)  # high bias: "sj400"
# tapply(d_noise$bias_m, d_noise$id, mean)  # high bias: "sj1500"
# 
# 
# # "large" in the sense that it is more than 2SD from group mean
# excl_bias_o <- d_noise$id[abs((d_noise$bias_o-mean(d_noise$bias_o))/sd(d_noise$bias_o))>2.5]
# excl_bias_m <- d_noise$id[abs((d_noise$bias_m-mean(d_noise$bias_m))/sd(d_noise$bias_m))>2.5]
# # 
# # # ------------------------------------------------------------------ #
# # # 可视化偏差分布和剔除标准
# # # ------------------------------------------------------------------ #
# # 
# # # 计算统计数据
# # mean_bias_o <- mean(d_noise$bias_o)
# # sd_bias_o <- sd(d_noise$bias_o)
# # mean_bias_m <- mean(d_noise$bias_m)
# # sd_bias_m <- sd(d_noise$bias_m)
# # 
# # # 计算剔除阈值
# # threshold_low_o <- mean_bias_o - 2.5 * sd_bias_o
# # threshold_high_o <- mean_bias_o + 2.5 * sd_bias_o
# # threshold_low_m <- mean_bias_m - 2.5 * sd_bias_m
# # threshold_high_m <- mean_bias_m + 2.5 * sd_bias_m
# # 
# # # 确定x轴范围以确保阈值线可见
# # x_range_o <- range(d_noise$bias_o, threshold_low_o, threshold_high_o)
# # x_range_m <- range(d_noise$bias_m, threshold_low_m, threshold_high_m)
# # 
# # # 设置图形参数
# # par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))
# # 
# # # 1. Motion Bias分布
# # hist(d_noise$bias_o, 
# #      main = "Motion Bias Distribution", 
# #      xlab = "Motion Bias", 
# #      col = "skyblue",
# #      border = "black",
# #      breaks = 15,
# #      xlim = x_range_o)  # 使用计算出的x轴范围
# # abline(v = mean_bias_o, col = "blue", lwd = 2, lty = 2)
# # abline(v = c(threshold_low_o, threshold_high_o), 
# #        col = "red", lwd = 2, lty = 2)
# # legend("topright", 
# #        legend = c(paste("Mean:", round(mean_bias_o, 3)),
# #                   paste("SD:", round(sd_bias_o, 3)),
# #                   paste("Excluded:", length(excl_bias_o)),
# #                   paste("Thresholds:", 
# #                         round(threshold_low_o, 3), "to", 
# #                         round(threshold_high_o, 3))),
# #        col = c("blue", "black", "red", "red"), 
# #        lty = c(2, NA, 2, 2), cex = 0.8)
# # 
# # # 2. Motion Bias Z分数分布
# # z_scores_o <- (d_noise$bias_o - mean_bias_o) / sd_bias_o
# # hist(z_scores_o,
# #      main = "Motion Bias Z-Scores",
# #      xlab = "Z-Score",
# #      col = "lightblue",
# #      border = "black",
# #      breaks = 15,
# #      xlim = range(z_scores_o, -2.5, 2.5))  # 确保包含阈值
# # abline(v = c(-2.5, 2.5), col = "red", lwd = 2, lty = 2)
# # # 标记异常值
# # if(length(excl_bias_o) > 0) {
# #   excl_z_scores_o <- z_scores_o[d_noise$id %in% excl_bias_o]
# #   rug(excl_z_scores_o, col = "red", lwd = 2, ticksize = 0.05)
# # }
# # legend("topright",
# #        legend = c(paste("Threshold: ±2.5"),
# #                   paste("Excluded:", length(excl_bias_o))),
# #        col = "red", lty = 2, cex = 0.8)
# # 
# # # 3. Motion Bias箱线图
# # boxplot(d_noise$bias_o,
# #         main = "Motion Bias Boxplot",
# #         ylab = "Motion Bias",
# #         col = "skyblue",
# #         ylim = x_range_o)  # 使用计算出的y轴范围
# # points(rep(1, length(d_noise$bias_o)), d_noise$bias_o, 
# #        pch = 16, col = ifelse(d_noise$id %in% excl_bias_o, "red", "gray"),
# #        cex = 0.8)
# # abline(h = c(threshold_low_o, threshold_high_o), 
# #        col = "red", lwd = 2, lty = 2)
# # abline(h = mean_bias_o, col = "blue", lwd = 2, lty = 2)
# # 
# # # 4. Orientation Bias分布
# # hist(d_noise$bias_m,
# #      main = "Orientation Bias Distribution",
# #      xlab = "Orientation Bias",
# #      col = "lightgreen",
# #      border = "black",
# #      breaks = 15,
# #      xlim = x_range_m)  # 使用计算出的x轴范围
# # abline(v = mean_bias_m, col = "blue", lwd = 2, lty = 2)
# # abline(v = c(threshold_low_m, threshold_high_m), 
# #        col = "red", lwd = 2, lty = 2)
# # legend("topright",
# #        legend = c(paste("Mean:", round(mean_bias_m, 3)),
# #                   paste("SD:", round(sd_bias_m, 3)),
# #                   paste("Excluded:", length(excl_bias_m)),
# #                   paste("Thresholds:", 
# #                         round(threshold_low_m, 3), "to", 
# #                         round(threshold_high_m, 3))),
# #        col = c("blue", "black", "red", "red"), 
# #        lty = c(2, NA, 2, 2), cex = 0.8)
# # 
# # # 5. Orientation Bias Z分数分布
# # z_scores_m <- (d_noise$bias_m - mean_bias_m) / sd_bias_m
# # hist(z_scores_m,
# #      main = "Orientation Bias Z-Scores",
# #      xlab = "Z-Score",
# #      col = "lightgreen",
# #      border = "black",
# #      breaks = 15,
# #      xlim = range(z_scores_m, -2.5, 2.5))  # 确保包含阈值
# # abline(v = c(-2.5, 2.5), col = "red", lwd = 2, lty = 2)
# # # 标记异常值
# # if(length(excl_bias_m) > 0) {
# #   excl_z_scores_m <- z_scores_m[d_noise$id %in% excl_bias_m]
# #   rug(excl_z_scores_m, col = "red", lwd = 2, ticksize = 0.05)
# # }
# # legend("topright",
# #        legend = c(paste("Threshold: ±2.5"),
# #                   paste("Excluded:", length(excl_bias_m))),
# #        col = "red", lty = 2, cex = 0.8)
# # 
# # # 6. 两个偏差的散点图
# # plot(d_noise$bias_o, d_noise$bias_m,
# #      main = "Motion vs Orientation Bias",
# #      xlab = "Motion Bias",
# #      ylab = "Orientation Bias",
# #      pch = 16,
# #      col = ifelse((d_noise$id %in% excl_bias_o) | (d_noise$id %in% excl_bias_m), "red", "blue"),
# #      xlim = x_range_o,  # 使用计算出的x轴范围
# #      ylim = x_range_m)  # 使用计算出的y轴范围
# # abline(v = c(threshold_low_o, threshold_high_o), 
# #        col = "red", lty = 2)
# # abline(h = c(threshold_low_m, threshold_high_m), 
# #        col = "red", lty = 2)
# # # 标记被剔除的点
# # if(length(c(excl_bias_o, excl_bias_m)) > 0) {
# #   excl_data <- d_noise[(d_noise$id %in% excl_bias_o) | (d_noise$id %in% excl_bias_m), ]
# #   text(excl_data$bias_o, excl_data$bias_m, 
# #        labels = excl_data$id, pos = 3, cex = 0.7, col = "red")
# # }
# # legend("topleft",
# #        legend = c(paste("Total excluded:", length(unique(c(excl_bias_o, excl_bias_m)))),
# #                   paste("Red dots: excluded")),
# #        col = c("black", "red"), cex = 0.8)
# # 
# # # 重置图形布局
# # par(mfrow = c(1, 1))
# # 
# # # 打印统计摘要
# # cat("\n" + rep("=", 50) + "\n")
# # cat("           BIAS EXCLUSION SUMMARY\n")
# # cat(rep("=", 50) + "\n\n")
# # 
# # cat("Motion Bias:\n")
# # cat("  Participants:", length(d_noise$id), "\n")
# # cat("  Mean:", round(mean_bias_o, 3), "\n")
# # cat("  SD:", round(sd_bias_o, 3), "\n")
# # cat("  Threshold (±2.5SD):", round(threshold_low_o, 3), "to", 
# #     round(threshold_high_o, 3), "\n")
# # cat("  Excluded participants (", length(excl_bias_o), "):", 
# #     if(length(excl_bias_o) > 0) paste(excl_bias_o, collapse = ", ") else "None", "\n\n")
# # 
# # cat("Orientation Bias:\n")
# # cat("  Participants:", length(d_noise$id), "\n")
# # cat("  Mean:", round(mean_bias_m, 3), "\n")
# # cat("  SD:", round(sd_bias_m, 3), "\n")
# # cat("  Threshold (±2.5SD):", round(threshold_low_m, 3), "to", 
# #     round(threshold_high_m, 3), "\n")
# # cat("  Excluded participants (", length(excl_bias_m), "):", 
# #     if(length(excl_bias_m) > 0) paste(excl_bias_m, collapse = ", ") else "None", "\n\n")
# # 
# # cat("Total unique participants excluded:", 
# #     length(unique(c(excl_bias_o, excl_bias_m))), "\n")
# # cat("Remaining participants after exclusion:", 
# #     length(unique(d_noise$id)) - length(unique(c(excl_bias_o, excl_bias_m))), "\n")
# # cat(rep("=", 50) + "\n")
# # 
# # # 保存图表到文件
# # png("bias_exclusion_visualization.png", width = 1200, height = 800)
# # par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))
# # 
# # # 重新绘制所有图表以保存到文件
# # # 1. Motion Bias分布
# # hist(d_noise$bias_o, 
# #      main = "Motion Bias Distribution", 
# #      xlab = "Motion Bias", 
# #      col = "skyblue",
# #      border = "black",
# #      breaks = 15,
# #      xlim = x_range_o)
# # abline(v = mean_bias_o, col = "blue", lwd = 2, lty = 2)
# # abline(v = c(threshold_low_o, threshold_high_o), 
# #        col = "red", lwd = 2, lty = 2)
# # 
# # # 2. Motion Bias Z分数分布
# # hist(z_scores_o,
# #      main = "Motion Bias Z-Scores",
# #      xlab = "Z-Score",
# #      col = "lightblue",
# #      border = "black",
# #      breaks = 15,
# #      xlim = range(z_scores_o, -2.5, 2.5))
# # abline(v = c(-2.5, 2.5), col = "red", lwd = 2, lty = 2)
# # if(length(excl_bias_o) > 0) {
# #   rug(excl_z_scores_o, col = "red", lwd = 2, ticksize = 0.05)
# # }
# # 
# # # 3. Motion Bias箱线图
# # boxplot(d_noise$bias_o,
# #         main = "Motion Bias Boxplot",
# #         ylab = "Motion Bias",
# #         col = "skyblue",
# #         ylim = x_range_o)
# # points(rep(1, length(d_noise$bias_o)), d_noise$bias_o, 
# #        pch = 16, col = ifelse(d_noise$id %in% excl_bias_o, "red", "gray"),
# #        cex = 0.8)
# # abline(h = c(threshold_low_o, threshold_high_o), 
# #        col = "red", lwd = 2, lty = 2)
# # abline(h = mean_bias_o, col = "blue", lwd = 2, lty = 2)
# # 
# # # 4. Orientation Bias分布
# # hist(d_noise$bias_m,
# #      main = "Orientation Bias Distribution",
# #      xlab = "Orientation Bias",
# #      col = "lightgreen",
# #      border = "black",
# #      breaks = 15,
# #      xlim = x_range_m)
# # abline(v = mean_bias_m, col = "blue", lwd = 2, lty = 2)
# # abline(v = c(threshold_low_m, threshold_high_m), 
# #        col = "red", lwd = 2, lty = 2)
# # 
# # # 5. Orientation Bias Z分数分布
# # hist(z_scores_m,
# #      main = "Orientation Bias Z-Scores",
# #      xlab = "Z-Score",
# #      col = "lightgreen",
# #      border = "black",
# #      breaks = 15,
# #      xlim = range(z_scores_m, -2.5, 2.5))
# # abline(v = c(-2.5, 2.5), col = "red", lwd = 2, lty = 2)
# # if(length(excl_bias_m) > 0) {
# #   rug(excl_z_scores_m, col = "red", lwd = 2, ticksize = 0.05)
# # }
# # 
# # # 6. 两个偏差的散点图
# # plot(d_noise$bias_o, d_noise$bias_m,
# #      main = "Motion vs Orientation Bias",
# #      xlab = "Motion Bias",
# #      ylab = "Orientation Bias",
# #      pch = 16,
# #      col = ifelse((d_noise$id %in% excl_bias_o) | (d_noise$id %in% excl_bias_m), "red", "blue"),
# #      xlim = x_range_o,
# #      ylim = x_range_m)
# # abline(v = c(threshold_low_o, threshold_high_o), 
# #        col = "red", lty = 2)
# # abline(h = c(threshold_low_m, threshold_high_m), 
# #        col = "red", lty = 2)
# # 
# # dev.off()
# # 
# # cat("\nVisualization saved as 'bias_exclusion_visualization.png'\n")
# # 
# # # ------------------------------------------------------------------ #
# if (!(is.integer(excl_bias_o) && length(excl_bias_o) == 0)){
#   d <- d[d$id!=excl_bias_o,]
# }
# if (!(is.integer(excl_bias_m) && length(excl_bias_m) == 0)){
#   d <- d[d$id!=excl_bias_m,]
# }
# # 
# # 
# # 
# # # ------------------------------------------------------------------ #
# # save dataset noise units
# write.table(d, file="D:/reysy/EEG/data/data_nu.txt", quote=F, row.names=F, sep=";")#D:/xiangmu/data/behavior/id/data_nu.txt
# 
# 
# 
