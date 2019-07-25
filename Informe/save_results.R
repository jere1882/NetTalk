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

################ ONLY ANNS ###############

err_ANN_1   <- c(82.94,78.96,79.22)
err_ANN_10  <- c(37.36,33.21,30.75)
err_ANN_20  <- c(33.76,24.63,19.30)
err_ANN_50  <- c(30.47,19.81,13.80)
err_ANN_100 <- c(30.03,19.42,13.22)

data <- cbind(err_ANN_1,err_ANN_10,err_ANN_20, err_ANN_50, err_ANN_100)
neurons <- c(1,10,20,50,100)

plot(neurons,data[1,], type="b", ylim=c(10,85),col="red", xlab="Number of hidden neurons",
     ylab="Error in test (%)",main="Test error for ANNs")
lines(neurons,data[2,],type="b",col="blue")
lines(neurons,data[3,],type="b",col="green")
legend("topright",legend=c("200 words in training","1000 words in training","5000 words in training"),lty=rep("solid",3),col=c("red","blue","green"))

##################### LIMITE DE ANNS #####################
avisa que modificaste el codigo para q haga esto bibentry(
  
errors_150 <- matrix(c(0.073086,	0.127730,	0.131792,
0.058791,	0.127328,	0.132246,
0.051456,	0.128737,	0.132573,
0.046329,	0.129844,	0.134417,
0.042437,	0.130146,	0.135017,
0.040379,	0.130750,	0.135734,
0.038059,	0.131756,	0.136670,
0.036786,	0.133266,	0.138023,
0.035289,	0.133870,	0.138786,
0.034279,	0.135078,	0.139975),ncol=3,byrow=TRUE)*100

epocas <- c(500,1000,1500,2000,2500,3000,3500,4000,4500,5000)

plot(epocas,errors_150[,1], type="b", ylim=c(0,20),col="red", xlab="Number of epochs",
     ylab="Error (%)",main="Evolution of error for an ANN with 150 hidden neurons")
lines(epocas,errors_150[,2],type="b",col="blue")
lines(epocas,errors_150[,3],type="b",col="green")
legend("topright",legend=c("Error in training","Error in validation","Error in test"),lty=rep("solid",3),col=c("red","blue","green"))




