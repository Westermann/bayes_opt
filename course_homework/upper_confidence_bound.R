require('nnet')

na.zero <- function (x) {
  x[is.na(x)] <- 0
  x
}


decision.values <- function (x, chosen.arms) {
  values <- rep(0, length(x))
  for (i in 1:length(x)){
    n <- sum(chosen.arms == i)
    t <- length(chosen.arms) - 1
    values[i] <- x[i] + na.zero(suppressWarnings(x[i] + sqrt(2 * log(t) / n)))
  }
  values
}


ucb1 <- function(k.arms, reward.probs, t.step, epsilon) {
  results <- data.frame(t.step=c(0),
                        regret=c(0),
                        cum.regret=c(0),
                        reward=c(0),
                        chosen.arm=c(0))
  status <- list(expected.reward=rep(0, k.arms))
  for (i in 1:t.step) {
    choice <- which.is.max(decision.values(status$expected.reward, results$chosen.arm))
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
  print(ucb1(k, c(rep(0.5-ε,k-1),0.5), 100))
}
