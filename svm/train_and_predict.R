library("kernlab")

englishAlphabet <- strsplit("abcdefghijklmnopqrstuvwxyz","")[[1]]
project_path      <- "~/Desktop/NetTalk/svm" #getwd() 
art_features_path <- paste(project_path,"/../encodedDs/articulatoryFeatures/phonems.csv",sep="") 

for (trainSize in c(1000)) {
  cat("Iterating with train size",trainSize,"\n")
  inputIndexes <- 1:(length(englishAlphabet)*7)
  trainDs <- read.table(file = paste(project_path,"/../encodedDs/datasets/",toString(trainSize),"/nettalk.data",sep=""))
  testDs <- read.table(file = paste(project_path,"/../encodedDs/datasets/",toString(trainSize),"/nettalk.test",sep=""))
  articulatoryFeatures <- read.table(file=art_features_path, header=FALSE, sep=",")
  
  outputIndexes <- (length(englishAlphabet)*7+1):(length(englishAlphabet)*7+length(articulatoryFeatures))
  
  
  prediction <- matrix(NA,nrow=dim(testDs)[1],ncol=dim(testDs)[2])
  prediction[,inputIndexes] <- as.matrix(testDs[,inputIndexes])
  
  for (idx in outputIndexes){
    cat(idx-length(inputIndexes),"\n")
    response_train <- (trainDs[,idx])
    temp_train <- as.matrix(trainDs[,inputIndexes])
    
    temp_test  <- testDs[,inputIndexes]
    
    
    # Should optimize nu...
    mod <-  ksvm(x=temp_train,
                 y=response_train,
                 kernel="rbfdot",
                 type="C-svc",
                 scaled=FALSE)
    
    prediction[,idx-length(inputIndexes)] <- predict(mod,temp_test)

  }
  
  write.table(prediction, file = paste(project_path,"/predictions/",toString(trainSize),"/nettalk.predict",sep=""), row.names = FALSE, col.names = FALSE)
  
}