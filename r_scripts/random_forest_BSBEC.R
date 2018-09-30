#!/usr/bin/env Rscript

require(neuralnet)
require(randomForest)
require(ggplot2)

lm_eqn <- function(df){
    m <- lm(dry_weight ~ rf, df);
    eq <- substitute(~~italic(r)^2~"="~r2, 
         list(a = format(coef(m)[1], digits = 2), 
              b = format(coef(m)[2], digits = 2), 
             r2 = format(summary(m)$r.squared, digits = 3)))
    as.character(as.expression(eq));                 
}

unscale_by <- function(data, example){
	center = attr(example, "scaled:center")
	scale = attr(example, "scaled:scale")
	return(data * scale + center)
}

scale_by <- function(data, example){
	center = attr(example, "scaled:center")
	scale = attr(example, "scaled:scale")
	return(scale(data, center = center, scale = scale))
}

fix_format <- function(data){
	data$pheno_date <- as.Date(data$pheno_date, "%d/%m/%Y")
	data$genotype <- factor(data$genotype)
	data$species <- factor(data$species)
	data$EuroPheno.stock_id <- factor(data$EuroPheno.stock_id)
	return(data)
}

xyplot <- function(fname, data, prediction){
	p <- ggplot(data, aes_string(x = prediction, y = "dry_weight", colour = "genotype")) + 
			geom_point() + geom_text(x = -2, y = 2, label = lm_eqn(data), parse = TRUE, colour = "black") + 
			xlim(-2.5, 2.5) + ylim(-2.5, 2.5)
	ggsave(fname, p)
}

do_nn = FALSE
do_rf = TRUE

data <- fix_format(read.csv("data.csv"))
yield <- fix_format(read.csv("yield.csv"))
validation <- fix_format(read.csv("validation.csv"))
validation_yield <- fix_format(read.csv("validation_yield.csv"))

data$predicted_yield <- data$dry_weight
data$dry_weight <- NULL

data$predicted_yield <- scale(data$predicted_yield)
data$stem_count <- scale(data$stem_count)
data$leaf_area <- scale(data$leaf_area)
data$canopy_height <- scale(data$canopy_height)

#scale the validation dataset
validation$stem_count <- scale_by(validation$stem_count, data$stem_count)
validation$leaf_area <- scale_by(validation$leaf_area, data$leaf_area)
validation$canopy_height <- scale_by(validation$canopy_height, data$canopy_height)

if (do_rf){
	rf_model <- randomForest(predicted_yield ~ stem_count + canopy_height + leaf_area + genotype, 
							data = data, ntree = 1000)
	data$rf <- predict(rf_model, data)
	data$rf <- unscale_by(data$rf, data$predicted_yield)
}

if (do_nn){
	nn_model <- neuralnet(predicted_yield ~ canopy_height + stem_count + leaf_area, data = data, 
							hidden = 5, threshold = 1.0, stepmax = 200000, lifesign = 'full')
	data$nn <- compute(nn_model, data[, 3:5])$net.result
	data$nn <- unscale_by(data$nn, data$predicted_yield)
}


if (do_rf){
	validation$rf <- predict(rf_model, validation)
	validation$rf <- unscale_by(validation$rf, data$predicted_yield)
}

if (do_nn){
	validation$nn <- compute(nn_model, validation[, 2:4])$net.result
	validation$nn  <- unscale_by(validation$nn , data$predicted_yield)
}

data$predicted_yield <- unscale_by(data$predicted_yield, data$predicted_yield)
merg <- merge(data, yield, by = c('EuroPheno.stock_id', 'species', 'genotype', 'pheno_date'))

validation_m  <- merge(validation, validation_yield, 
				by = c('EuroPheno.stock_id', 'species', 'genotype', 'pheno_date'))

xyplot("random_forest_training.png", merg, "rf")
xyplot("random_forest_validation.png", validation_m, "rf")
dev.off()
