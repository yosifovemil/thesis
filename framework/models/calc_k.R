#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
fname <- args[1]
genotype <- args[2]

data <- read.csv(fname)
data$Date <- as.Date(data$Date, "%Y-%m-%d %H:%M:%S")

geno_sub <- subset(data, Genotype == genotype)
years <- levels(factor(geno_sub$Year))
print(years)
year_subs <- lapply(years, function(year) subset(geno_sub, Year == year))
fits <- lapply(year_subs, function(year_sub) nls(Transmission ~ exp(-k * LAI), 
												data = year_sub, 
												start = list(k = 1)))


print(fits)
