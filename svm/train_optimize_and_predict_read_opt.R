library("kernlab")

englishAlphabet <- strsplit("abcdefghijklmnopqrstuvwxyz","")[[1]]
project_path      <-  "~/Desktop/NetTalk/svm" #getwd() 
art_features_path <- paste(project_path,"/../encodedDs/articulatoryFeatures/phonems.csv",sep="") 



trainSize<-5000

cat("Iterating with train size",trainSize,"\n")
inputIndexes <- 1:(length(englishAlphabet)*7)
trainDs <- read.table(file = paste(project_path,"/../encodedDs/datasets/",toString(trainSize),"/nettalk.data",sep=""))
testDs <- read.table(file = paste(project_path,"/../encodedDs/datasets/",toString(trainSize),"/nettalk.test",sep=""))

articulatoryFeatures <- read.table(file=art_features_path, header=FALSE, sep=",")

outputIndexes <- (length(englishAlphabet)*7+1):(length(englishAlphabet)*7+length(articulatoryFeatures))


prediction <- matrix(NA,nrow=dim(testDs)[1],ncol=dim(testDs)[2])
prediction[,inputIndexes] <- as.matrix(testDs[,inputIndexes])

opt.param <- read.table(file = paste(project_path,"/predictions/200/optimized_parameters",sep=""))

best_c <- as.integer(opt.param[1,])
best_s <- as.integer(opt.param[2,])

for (idx in outputIndexes){
  
  response_train <- trainDs[,idx]
  temp_train <- as.matrix(trainDs[,inputIndexes])
  
  temp_test  <- testDs[,inputIndexes]

  cat ("Optimized c is ",best_c[idx-length(inputIndexes)]," and optimized s is ",best_s[idx-length(inputIndexes)],"\n")
  mod <-ksvm(response_train~.,
             cbind(temp_train,response_train),
             kernel="rbfdot",
             type="C-svc",
             C=2**best_c[idx-length(inputIndexes)],
             kpar=list(sigma=2**best_s[idx-length(inputIndexes)]),
             scaled=FALSE) 
  
  prediction[,idx] <- predict(mod,temp_test)
}

write.table(prediction, file = paste(project_path,"/predictions/",toString(trainSize),"/nettalk.predict",sep=""), row.names = FALSE, col.names = FALSE)

