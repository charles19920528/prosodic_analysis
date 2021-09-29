# set work directory to the source file location before running the code.

library(qvalue)
p_values <- read.csv("data/p_values.csv", header=TRUE)
p_values = p_values[order(p_values[, 3]), ] 

result = qvalue(p=p_values[, 3], fdr.level = 0.1)
p_values[result$significant, ]
