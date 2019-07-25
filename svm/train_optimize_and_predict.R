library("kernlab")

englishAlphabet <- strsplit("abcdefghijklmnopqrstuvwxyz","")[[1]]
project_path      <-  getwd() 
art_features_path <- paste(project_path,"/../encodedDs/articulatoryFeatures/phonems.csv",sep="") 

trainSize<-1000

cat("Iterating with train size",trainSize,"\n")
inputIndexes <- 1:(length(englishAlphabet)*7)
trainDs <- read.table(file = paste(project_path,"/../encodedDs/datasets/",toString(trainSize),"/nettalk.data",sep=""))
testDs <- read.table(file = paste(project_path,"/../encodedDs/datasets/",toString(trainSize),"/nettalk.test",sep=""))
valid_indexes <- sample(1:(dim(testDs)[1]),5000)

validDs <- testDs[valid_indexes,]
testDs  <- testDs[-valid_indexes,]

articulatoryFeatures <- read.table(file=art_features_path, header=FALSE, sep=",")

outputIndexes <- (length(englishAlphabet)*7+1):(length(englishAlphabet)*7+length(articulatoryFeatures))


prediction <- matrix(NA,nrow=dim(testDs)[1],ncol=dim(testDs)[2])
prediction[,inputIndexes] <- as.matrix(testDs[,inputIndexes])

best_c <- rep(NA,length(outputIndexes))
best_s <- rep(NA,length(outputIndexes))

cp_values <- -5:15
s_values  <- -15:3

for (idx in outputIndexes){
  
  response_train <- trainDs[,idx]
  temp_train <- as.matrix(trainDs[,inputIndexes])
  
  response_valid <- validDs[,idx]
  temp_valid <- validDs[,inputIndexes]
  
  temp_test  <- testDs[,inputIndexes]
  respose_test <- testDs[,idx]
  
  cat("Generating predictions for articulatory feature ",idx-length(inputIndexes),"\n")
  
  error <- matrix(nrow=length(cp_values), ncol=length(s_values))
  
  for (cpix in 1:length(cp_values)){
    cp <- cp_values[cpix]
    for (six in 1:length(s_values)) {
      cat(cpix,six,"\n")
      s <- s_values[six]
      mod <-ksvm(response_train~.,
                 cbind(temp_train,response_train),
                 kernel="rbfdot",
                 type="C-svc",
                 C=2**cp,
                 kpar=list(sigma=2**s),
                 scaled=FALSE) 
                   
      error[[cpix,six]] <- sum(abs(response_valid - predict(mod,temp_valid)))
    }
  }
  
  idx_best = which(error==min(error), arr.ind=TRUE)
  best_c[idx-length(inputIndexes)] <- cp_values[idx_best[1]]
  best_s[idx-length(inputIndexes)] <- s_values[idx_best[2]]
  cat ("Optimized c is ",best_c," and optimized s is ",best_s,"\n")
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
write.table(rbind(best_c,best_s), file = paste(project_path,"/predictions/",toString(trainSize),"/optimized_parameters",sep=""), row.names = FALSE, col.names = FALSE)
write.table(valid_indexes, file = paste(project_path,"/predictions/",toString(trainSize),"/validIdx",sep=""), row.names = FALSE, col.names = FALSE)

