#!/usr/bin/env Rscript

suppressMessages(library(argparse))

parser <- ArgumentParser()
parser$add_argument('-i', '--input', required = TRUE)

args <- parser$parse_args()

data <- read.csv(args$input)

data$pheno_date <- as.Date(data$pheno_date, "%d/%m/%Y")
data <- subset(data, dry_weight != "NA")
fit <- lm(dry_weight ~ PAR + 0, data = data)

print(summary(fit)$coefficients[1])
