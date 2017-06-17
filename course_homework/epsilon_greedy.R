na.zero <- function (x) {
  x[is.na(x)] <- 0
  return(x)
}


epsilon.greedy <- function(k.arms, reward.probs, t.step, epsilon) {
  results <- data.frame(t.step=c(0),
                        regret=c(0),
                        cum.regret=c(0),
                        reward=c(0),
                        chosen.arm=c(0))
  status <- list(expected.reward=rep(0.5, k.arms))
  for (i in 1:t.step) {
    if (runif(1, 0, 1) > epsilon) {
      choice <- sample(1:k.arms,1) 
    } else {
      choice <- which.max(status$expected.reward)
    }
    reward <- runif(1, 0, 1) < reward.probs[choice]
    print(reward)
    regret <- as.numeric(reward)
    cum.regret <- regret + results[i,'cum.regret']
    status$expected.reward[choice] <- na.zero(
      suppressWarnings(
        mean(results[results$chosen.arm == choice,]$reward)))
    results <- rbind(results, data.frame(t.step=c(i),
                                         regret=c(regret),
                                         cum.regret=c(cum.regret),
                                         reward=c(reward),
                                         chosen.arm=c(choice)))
  }
  return(results)
}


if (getOption('run.main', default=TRUE)) {
  ε <- 0.1
  k <- 10
  print(epsilon.greedy(k, c(rep(0.5-ε,k-1),0.5), 10, epsilon=0.1))
}
