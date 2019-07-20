library("kernlab")

englishAlphabet <- strsplit("abcdefghijklmnopqrstuvwxyz","")[[1]]
project_path      <- "~/Desktop/NetTalk/gp" #getwd() 
art_features_path <- paste(project_path,"/../encodedDs/articulatoryFeatures/phonems.csv",sep="") 

articulatoryFeatures <- read.table(file=art_features_path, header=FALSE, sep=",")
articulatoryFeatures[is.na(articulatoryFeatures)] <- 0
articulatoryFeatures [which(phonemicAlphabet=='-'),] <- rep(0,dim(articulatoryFeatures)[2])
articulatoryFeatures <- as.matrix(articulatoryFeatures)

nFeatures <- dim(articulatoryFeatures)[2]


for (trainSize in c(1000)) { 
  
  cat("Iterating with train size",trainSize,"\n")
  inputIndexes <- 1:(length(englishAlphabet)*7)
  trainDs <- read.table(file = paste(project_path,"/../encodedDs/datasets/",toString(trainSize),"/nettalk.data",sep=""))
  testDs <- read.table(file = paste(project_path,"/../encodedDs/datasets/",toString(trainSize),"/nettalk.test",sep=""))
  outputIndexes <- (length(englishAlphabet)*7+1):(length(englishAlphabet)*7+nFeatures)
  
  
  prediction <- matrix(NA,nrow=dim(testDs)[1],ncol=dim(testDs)[2])
  prediction[,inputIndexes] <- as.matrix(testDs[,inputIndexes])
  
  for (idx in outputIndexes){
    
    response_train <- trainDs[,idx]
    temp_train <- as.matrix(trainDs[,inputIndexes])
    
    temp_test  <- testDs[,inputIndexes]

    cat("Generating predictions for articulatory feature ",idx-length(inputIndexes),"\n")
    
    mod <-gausspr(x=temp_train,
                  y=response_train,
               kernel="rbfdot",
               type="regression",
               scaled=FALSE) 

    prediction[,idx] <- as.integer(lapply(predict(mod,temp_test), function(x) if (x>0.5) return(1) else return(0)))
    
  }
  
  write.table(prediction, file = paste(project_path,"/predictions/",toString(trainSize),"/nettalk.predict",sep=""), row.names = FALSE, col.names = FALSE)
  
}
