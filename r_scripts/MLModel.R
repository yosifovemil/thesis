#!/usr/bin/env Rscript
suppressMessages(library(randomForest))
suppressMessages(library(nnet))
suppressMessages(library(caret))
suppressMessages(library(mgcv))
suppressMessages(library(kknn))
suppressMessages(library(e1071))
suppressMessages(library(gbm))
suppressMessages(library(neuralnet))
suppressMessages(library(LiblineaR))
suppressMessages(library(plyr))

fix_transmission <- function(data, all_data, predict_transmission){
	if (predict_transmission == "predict"){
		tr <- subset(all_data, Transmission != 999)
		tr$dry_weight <- NULL
		model <- randomForest(Transmission ~ ., data = tr)
		data$Transmission[data$Transmission == 999] <- predict(model, data[data$Transmission == 999,])
	} else {
		data$Transmission[data$Transmission == 999] <- 0
	}

	return(data)
}

normalise_frame <- function(data_frame, reference){
    if (unique(data_frame$pseudo_rep_no) == 0){
        data_frame = subset(data_frame, select = -c(pseudo_rep_no))
    }

	for (name in colnames(data_frame)){
        data_frame[[name]] <- as.numeric(data_frame[[name]])
        if (missing(reference)){
            data_frame[[name]] <- normalise(data_frame[[name]])
        } else {
            data_frame[[name]] <- normalise(data_frame[[name]], 
                                            reference[[name]])
        }
	} 

	return(data_frame) 
}

unnormalise_frame <- function(data_frame, reference){
	numerics = c("dry_weight", "Transmission", "canopy_height", "stem_count", "PAR",
					"degree_days", "rainfall_mm")
	for (name in numerics){
        if (name %in% colnames(data_frame)){
            if (missing(reference)){
                data_frame[[name]] <- unnormalise(data_frame[[name]])
            } else {
                data_frame[[name]] <- unnormalise(data_frame[[name]],
                                                  reference[[name]])
            }
        }
	} 

	return(data_frame) 
}

normalise <- function(vect, reference){
	if (missing(reference)){
		x_max = max(vect)
		x_min = min(vect)
	} else {
		x_max = attr(reference, "x_max")
		x_min = attr(reference, "x_min")
	}

	vect = (vect - x_min) / (x_max - x_min)
	attr(vect, "x_max") <- x_max
	attr(vect, "x_min") <- x_min
	return(vect)
}

unnormalise <- function(vect, reference){
	if (missing(reference)){ 
		reference = vect
	}

	x_max = attr(reference, "x_max")
	x_min = attr(reference, "x_min")
	vect = (x_max - x_min) * vect + x_min
	return(vect)
	
}

sample_data <- function(train_data){
	size = nrow(train_data)
	output <- train_data[sample(nrow(train_data), size, replace = TRUE), ]
    if ("genotype" %in% colnames(output)){
        levels(output$genotype) <- c("EMI-11", "Giganteus", "Goliath", "Sac-5")
    }
	return(output)
}

prep_data <- function(fname, train_data){
	data <- read.csv(fname)
	data$stem_id <- NULL
	data$time_taken <- NULL
	data$who <- NULL
	data$measurement_no <- NULL
    data$EuroPheno.stock_id <- NULL

    if ('row' %in% colnames(data)) {
        rc_factors <- 1:8
        data$row <- factor(data$row, levels = rc_factors)
        data$col <- factor(data$col, levels = rc_factors)
    }

    if ("genotype" %in% colnames(data)){
        data$genotype <- factor(data$genotype)
        levels(data$genotype) <- c("EMI-11", "Giganteus", "Goliath", "Sac-5")
    }

    if ('field_flowering_score' %in% colnames(data)){
        data$field_flowering_score <- as.numeric(data$field_flowering_score)
    }

    if ("doy" %in% colnames(data)){
        data$doy <- as.numeric(data$doy)
    }

    if ("year" %in% colnames(data)){
        data <- subset(data, year %in% c(2011, 2015, 2016))
        data$year <- factor(data$year)
        levels(data$year) <- c(2011, 2015, 2016)
    }

	data$pheno_date <- NULL
    data$stem_diam_half_height <- NULL
    data$base_diameter_1 <- NULL
    data$base_diameter_2 <- NULL

	sub <- subset(data, !is.na(dry_weight))
	return(na.omit(sub))
}

bag_me <- function(train_data, test_data, train_model, predict_function, n){
	if (missing(n)) {
		n = 100
	}
	#train n versions of the model (100 by default)
	models <- lapply(1:n, train_model)

	predictions <- sapply(models, predict_function)
	test_data$predicted <- apply(predictions, 1, mean)	

	return(test_data)
}

random_forest_scenario <- function(train_data, test_data){
	rf <- randomForest(dry_weight ~ ., data = train_data)
	test_data$predicted <- predict(rf, test_data, OOB=TRUE)

	return(test_data)
}	

random_forest_scenario_caret <- function(train_data, test_data){
	rf <- train(dry_weight ~ ., data = train_data, method='rf')
	test_data$predicted <- predict(rf, test_data)

	return(test_data)
}	

lm_scenario_caret <- function(train_data, test_data){
	#train_data <- normalise_frame(train_data)
	#test_data <- normalise_frame(test_data, train_data)

	fit <- train(dry_weight ~ ., data = train_data, method = 'lm')
    if ("genotype" %in% colnames(train_data)){
        fit$xlevels[["genotype"]] <- c("EMI-11", "Giganteus", "Goliath", "Sac-5")
    }

    if ("year" %in% colnames(train_data)){
        fit$xlevels[["year"]] <- c(2011, 2015, 2016)
    }

	test_data$predicted <- predict(fit, test_data)

	#test_data$predicted <- unnormalise(test_data$predicted, train_data$dry_weight)
	#test_data <- unnormalise_frame(test_data)
	#train_data <- unnormalise_frame(train_data)

	return(test_data)
}

lm_bagging_scenario_caret <- function(train_data, test_data){

	train_model <- function(i){
		train_sample <- sample_data(train_data)
		fit <- train(dry_weight ~ ., data = train_sample, method='lm')
        if ("genotype" %in% colnames(train_sample)){
            fit$xlevels[["genotype"]] <- c("EMI-11", "Giganteus", "Goliath", "Sac-5")
        }

        if ("year" %in% colnames(train_sample)){
            fit$xlevels[["year"]] <- c(2011, 2015, 2016)
        }

		return(fit)
	}

	predict_function <- function(model){
		return(predict(model, test_data))
	}

	return(bag_me(train_data, test_data, train_model, predict_function))
}

lm_scenario <- function(train_data, test_data){
	#train_data <- normalise_frame(train_data)
	#test_data <- normalise_frame(test_data, train_data)

	fit <- lm(dry_weight ~ ., data = train_data)
    if ("genotype" %in% colnames(train_data)){
        fit$xlevels[["genotype"]] <- c("EMI-11", "Giganteus", "Goliath", "Sac-5")
    }

    if ("year" %in% colnames(train_data)){
        fit$xlevels[["year"]] <- c(2011, 2015, 2016)
    }

	test_data$predicted <- predict(fit, test_data)

	#test_data$predicted <- unnormalise(test_data$predicted, train_data$dry_weight)
	#test_data <- unnormalise_frame(test_data)
	#train_data <- unnormalise_frame(train_data)

	return(test_data)
}


lm_bagging_scenario <- function(train_data, test_data){

	train_model <- function(i){
		train_sample <- sample_data(train_data)
		fit <- lm(dry_weight ~ ., data = train_sample)
        if ("genotype" %in% colnames(train_sample)){
            fit$xlevels[["genotype"]] <- c("EMI-11", "Giganteus", "Goliath", "Sac-5")
        }

        if ("year" %in% colnames(train_sample)){
            fit$xlevels[["year"]] <- c(2011, 2015, 2016)
        }

		return(fit)
	}

	predict_function <- function(model){
		return(predict(model, test_data))
	}

	return(bag_me(train_data, test_data, train_model, predict_function))
}

ann_scenario_neuralnet <- function(train_data, test_data){
    trd <- normalise_frame(train_data)
    ted <- normalise_frame(test_data, trd)
    capture.output(fit <- train(dry_weight ~ ., data = trd, method='neuralnet'))

    ted$predicted <- predict(fit, ted)
    test_data$predicted <- unnormalise(ted$predicted, ted$dry_weight)
    return(test_data)
}

ann_bagging_scenario_neuralnet <- function(train_data, test_data){
    trd <- normalise_frame(train_data)
    ted <- normalise_frame(test_data, trd)

    train_model <- function(i){
        train_sample <- sample_data(trd)
        capture.output(fit <- train(dry_weight ~ ., data = train_sample,
                                    method='neuralnet'))
        return(fit)
    }

    predict_function <- function(model){
        return(predict(model, ted))
    }

    ted <- bag_me(trd, ted, train_model, predict_function, n = 10)

    test_data$predicted <- unnormalise(ted$predicted, ted$dry_weight)
    return(test_data)
}

ann_scenario <- function(train_data, test_data){
	train_data$dry_weight <- normalise(train_data$dry_weight)
	test_data$dry_weight <- normalise(test_data$dry_weight, train_data$dry_weight)

	capture.output(fit <- train(dry_weight ~ ., data = train_data, method = 'nnet'))
	test_data$predicted <- predict(fit, test_data)

	test_data$dry_weight <- unnormalise(test_data$dry_weight)
	test_data$predicted <- unnormalise(test_data$predicted, test_data$dry_weight)

	return(test_data)
}

ann_bagging_scenario <- function(train_data, test_data){
	train_data$dry_weight <- normalise(train_data$dry_weight)
	test_data$dry_weight <- normalise(test_data$dry_weight, train_data$dry_weight)
	
	train_model <- function(i){
		train_sample <- sample_data(train_data)
		capture.output(fit <- train(dry_weight ~ ., data = train_sample, method = 'nnet'))
		return(fit)
	}

	predict_function <- function(model){
		return(predict(model, test_data))
	}

	test_data <- bag_me(train_data, test_data, train_model, predict_function, n = 10)
	test_data$dry_weight <- unnormalise(test_data$dry_weight)
	test_data$predicted <- unnormalise(test_data$predicted, test_data$dry_weight)

	return(test_data)
	
}

glm_scenario <- function(train_data, test_data){
	fit <- glm(dry_weight ~ .,, data = train_data)
    if ("year" %in% colnames(train_data)){
        fit$xlevels[["year"]] <- c(2011, 2015, 2016)
    }
	test_data$predicted <- predict(fit, test_data)
	
	return(test_data)
}

knn_scenario <- function(train_data, test_data){
	fit <- kknn(dry_weight ~ .,, train = train_data, test = test_data)
	test_data$predicted <- predict(fit)
	
	return(test_data)
}

knn_bagging_scenario <- function(train_data, test_data){
	train_model <- function(i){
		train_sample <- sample_data(train_data)
		fit <- kknn(dry_weight ~ ., train = train_sample, test = test_data)
		return(fit)
	}

	predict_function <- function(model){
		return(predict(model))
	}

	return(bag_me(train_data, test_data, train_model, predict_function))
}

knn_scenario_caret <- function(train_data, test_data){
	fit <- train(dry_weight ~ ., data = train_data, method='kknn')
	test_data$predicted <- predict(fit, test_data)
	
	return(test_data)
}

knn_bagging_scenario_caret <- function(train_data, test_data){
	train_model <- function(i){
		train_sample <- sample_data(train_data)
		fit <- train(dry_weight ~ ., data = train_sample, method='kknn')
		return(fit)
	}

	predict_function <- function(model){
		return(predict(model, test_data))
	}

	return(bag_me(train_data, test_data, train_model, predict_function))
}

svm_scenario_caret <- function(train_data, test_data){

    capture.output(fit <- train(dry_weight ~ ., data = train_data,
                                method='svmLinear3'))

	test_data$predicted <- predict(fit, test_data)

	return(test_data)
}

svm_bagging_scenario_caret <- function(train_data, test_data){
	train_model <- function(i){
		train_sample <- sample_data(train_data)
        capture.output(fit <- train(dry_weight ~ ., data = train_sample,
                                    method='svmLinear3'))
		return(fit)
	}

	predict_function <- function(model){
		return(predict(model, test_data))
	}

	return(bag_me(train_data, test_data, train_model, predict_function, 5))
}

svm_scenario <- function(train_data, test_data){
	#use tune to get the "optimum" svm parameters
	#"optimum" in quotes because I suspect this will not work
	tuneResult <- tune(svm, dry_weight ~ ., data = train_data, 
					ranges = list(epsilon = seq(0, 0.2, 0.01), cost = 2^(2:9)))
	fit <- tuneResult$best.model

	test_data$predicted <- predict(fit, test_data)

	return(test_data)
}

svm_bagging_scenario <- function(train_data, test_data){
	train_model <- function(i){
		train_sample <- sample_data(train_data)
		tuneResult <- tune(svm, dry_weight ~ ., data = train_sample, 
						ranges = list(epsilon = seq(0, 0.2, 0.01), cost = 2^(2:9)))
		fit <- tuneResult$best.model
		return(fit)
	}

	predict_function <- function(model){
		return(predict(model, test_data))
	}

	return(bag_me(train_data, test_data, train_model, predict_function, 5))
}

gbm_scenario_caret <- function(train_data, test_data){
	capture.output(fit <- train(dry_weight ~ ., data = train_data,
                                method='gbm'))
	test_data$predicted <- predict(fit, test_data)
	return(test_data)
}

gbm_bagging_scenario_caret <- function(train_data, test_data){

	train_model <- function(i){
		train_sample <- sample_data(train_data)
		capture.output(fit <- train(dry_weight ~ ., data = train_data,
                                    method='gbm'))
		return(fit)
	}

	predict_function <- function(model){
		return(predict(model, test_data))
	}

	return(bag_me(train_data, test_data, train_model, predict_function))
}

gbm_scenario <- function(train_data, test_data){
	capture.output(fit <- gbm(dry_weight ~ ., data = train_data, n.trees = 1000))
	test_data$predicted <- predict(fit, test_data, n.trees = 1000)

	return(test_data)
}

gbm_bagging_scenario <- function(train_data, test_data){

	train_model <- function(i){
		train_sample <- sample_data(train_data)
		capture.output(fit <- gbm(dry_weight ~ ., data = train_sample, n.trees = 1000))
		return(fit)
	}

	predict_function <- function(model){
		return(predict(model, test_data, n.trees = 1000))
	}

	return(bag_me(train_data, test_data, train_model, predict_function))
}

combine_scenario <- function(train_data, test_data, fns, reps){
    results <- list()
    counter = 1
    for (i in rep(1:length(fns))){
        for (j in rep(1:reps[[i]])){
            train_sample <- sample_data(train_data)
            new_data <- suppressWarnings(fns[[i]](train_sample, test_data))
            results[[counter]] <- new_data$predicted
            counter = counter + 1
        }
    }

    d <- data.frame(results)
    m <- rowMeans(d)
    test_data$predicted <- m
    return(test_data)
}

prep_output <- function(test){
    cols <- c("dry_weight", "predicted")
    test <- test[,cols]

    return(test)
}

test_model <- function(train_file, test_file, model, reps){
	#Load data
	train_data <- prep_data(train_file)
	test_data <- prep_data(test_file, train_data)
	predict_transmission = "no" #TODO

	#fix transmission by either filling to 0 or predicting
    if ("Transmission" %in% colnames(train_data)){
        all_data <- rbind(train_data, test_data)
        train_data <- fix_transmission(train_data, all_data, predict_transmission)
        test_data <- fix_transmission(test_data, all_data, predict_transmission)
    }

	train_data$leaf_area <- NULL
	test_data$leaf_area <- NULL

	if (model == "lm"){
		fn <- lm_scenario_caret
	} else if (model == "lm_bag"){
		fn <- lm_bagging_scenario_caret
	} else if (model == "rf"){
		fn <- random_forest_scenario_caret
	} else if (model == "ann"){
        fn <- ann_scenario
	} else if (model == "ann_bag"){
		fn <- ann_bagging_scenario
	} else if (model == "glm"){
		fn <- glm_scenario
	} else if (model == "knn"){
		fn <- knn_scenario_caret
	} else if (model == "knn_bag"){
		fn <- knn_bagging_scenario_caret
	} else if (model == "svm"){
		fn <- svm_scenario_caret
	} else if (model == "svm_bag"){
		fn <- svm_bagging_scenario_caret
	} else if (model == "gbm"){
		fn <- gbm_scenario_caret
	} else if (model == "gbm_bag"){
		fn <- gbm_bagging_scenario_caret
	} else if (model == "combine"){
        fns <- c(random_forest_scenario, svm_scenario, knn_bagging_scenario)
        test <- combine_scenario(train_data, test_data, fns, reps)
        test <- prep_output(test)
        return(test)
    }

	test <- suppressWarnings(fn(train_data, test_data))
    test <- prep_output(test)
    return(test)
} 
