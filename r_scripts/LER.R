#!/usr/bin/env Rscript

suppressMessages(library(argparse))
suppressMessages(library(minpack.lm))

parser <- ArgumentParser()
parser$add_argument('-i', '--input', dest = 'input')
args <- parser$parse_args()

data <- read.csv(args$input)
data <- subset(data, leaf_area_index != "NA")
data$pheno_date <- as.Date(data$pheno_date, "%d/%m/%Y")
data$year <- format(data$pheno_date, "%Y")

init_L <- max(data$leaf_area_index)
init_k = 0.001
init_midpoint = max(data$degree_days)/2.0

tryCatch({
    sig_fit <- nlsLM(leaf_area_index ~ L / (1 + 
                     exp(-k * (degree_days - midpoint))),
                     data = data,
                     start = list(L = init_L,
                                  k = init_k, 
                                  midpoint = init_midpoint))

    # report the results 
    coeff <- summary(sig_fit)$coefficients
    cat(sprintf("L:%f\n", coeff[1]))
    cat(sprintf("k:%f\n", coeff[2]))
    cat(sprintf("x_mid:%f\n", coeff[3]))

}, error = function(cond){
    data$degree_days[data$degree_days == 0] <- 0.0001
    fit <- lm(leaf_area_index ~ log(degree_days), data = data)

    # report the results
    cat(sprintf("intercept:%f\n", fit$coefficients[1]))
    cat(sprintf("x_b:%f\n", fit$coefficients[2]))
})
