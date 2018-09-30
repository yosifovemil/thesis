#!/usr/bin/env Rscript

#This script uses nls (non linear least squares) to find value for k for Chris Davey's model

require(argparse)
parser <- ArgumentParser()
parser$add_argument('-i', '--input', required = TRUE)

args <- parser$parse_args()

data <- read.csv(args$input)
data$pheno_date <- as.Date(data$pheno_date, "%d/%m/%Y")
data$year <- format(data$pheno_date, "Y")
data <- aggregate(cbind(Transmission, leaf_area_index) ~ year * pheno_date * genotype, data, FUN = mean)
fit <- nls(Transmission ~ exp(-k * leaf_area_index), data = data, start = list(k = 1))
summary(fit)$coefficients[1]
