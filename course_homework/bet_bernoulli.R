thompson <- function(k.arms, rewards.prob, t.step, epsilon){
  P <- numeric(k.arms)
  Q <- numeric(k.arms)
  N <- numeric(k.arms)
  S <- numeric(k.arms)
  F <- numeric(k.arms)
  R <- numeric(t.step)
  Pr <- rbeta(k.arms, S + 1, F + 1, ncp = 0)

  a1 <- sample(1:k.arms,1, replace=TRUE)
  P[a1] <- 1

  for (t in 1:t.step){
    a <- which.max(Pr) 
    if (P[a]==1){
      r <- rbinom(1,1,0.5)
    }
    else{
      r <- rbinom(1,1,0.5-epsilon)
    }
    N[a] <- N[a] + 1
    Q[a] <- Q[a] + (r - Q[a])/N[a]
    S[a] <- S[a] + r
    F[a] <- F[a] + 1 -r
    R[t] <- sum(S)
    Pr[a] <- rbeta(1, S[a] + 1, F[a] + 1, ncp = 0)
  }
  R
}
