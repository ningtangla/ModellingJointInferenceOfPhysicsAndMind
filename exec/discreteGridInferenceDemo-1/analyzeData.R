chase = as.data.frame(read.csv("chasingInference.csv"))
noneZeroData = chase[chase$posterior != 0, ]
View(noneZeroData)

write.csv(noneZeroData, "nonezeroChasing.csv")

