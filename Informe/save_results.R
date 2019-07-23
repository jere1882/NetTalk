method_names <- c("NN","NN Syl","RF","SVM","GP","ANN 20","ANN 50")

tr_size <- c(200,1000,5000)
err_NN  <- c(49.12,48.63,49.12)
err_NNS <- c(32.40,23.48,21.30)
err_RF  <- c(33.72,22.14,14.27)
err_SVM <- c(35.14,24.66,16.63)
err_GP  <- c(36.03,26.09,NA)
err_ANN_20 <- c(33.76,24.63,19.30)
err_ANN_50 <- c(30.47,19.81,13.80)

plot(tr_size,err_NN, type="b", ylim=c(10,70),col="black",lty="dashed", xlab="Number of words in training dataset",
     ylab="Error in test (%)",main="Test error for different classificators")
lines(tr_size,err_NNS,type="b",col="blue")
lines(tr_size,err_RF,type="b",col="green")
lines(tr_size,err_SVM,type="b",col="gold")
lines(tr_size,err_GP,type="b",col="pink")
lines(tr_size,err_ANN_20,type="b",col="red")
lines(tr_size,err_ANN_50,type="b",col="deepskyblue")

legend("topright",legend=method_names,lty=c("dashed",rep("solid",6)),col=c("black","blue","green","gold","pink","red","deepskyblue"))

