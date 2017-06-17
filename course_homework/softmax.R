require('nnet')

na.zero <- function (x) {
	x[is.na(x)] <- 0
	x
}


softmax.function <- function (x, temp) {
	values = sapply(x, function(this.x) {
										exp(this.x/temp) / sum(exp(exp(x / temp)))})
	values
}


softmax <- function(k.arms, reward.probs, t.step, epsilon) {
	results <- data.frame(t.step=c(0),
												regret=c(0),
												cum.regret=c(0),
												reward=c(0),
												chosen.arm=c(0))
	status <- list(expected.reward=rep(0, k.arms))
	for (i in 1:t.step) {
		choice <- sample(1:k.arms, 1, prob=softmax.function(status$expected.reward, temp=0.5))
		reward <- rbinom(1, 1, reward.probs[choice])
		regret <- as.numeric(reward)
		cum.regret <- regret + results[i,'cum.regret']
		results <- rbind(results, data.frame(t.step=c(i),
																				 regret=c(regret),
																				 cum.regret=c(cum.regret),
																				 reward=c(reward),
																				 chosen.arm=c(choice)))
		status$expected.reward[choice] <- na.zero(mean(results[results$chosen.arm == choice,]$reward))
	}
	results
}


if (getOption('run.main', default=TRUE)) {
	ε <- 0.1
	k <- 2
	print(softmax(k, c(rep(0.5-ε,k-1),0.5), 100))
}
