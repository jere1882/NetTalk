library(e1071)

englishAlphabet <- strsplit("abcdefghijklmnopqrstuvwxyz","")[[1]]
phonemicAlphabet <-  strsplit("abcdefghiklmnoprstuvwxyzACDEGIJKLMNOQRSTUWXYZ@!#*^-","")[[1]]
nPhonems <- length(phonemicAlphabet)

project_path      <- getwd() 
art_features_path <- paste(project_path,"/../encodedDs/articulatoryFeatures/phonems.csv",sep="") 


trainDs <- read.table(file = paste(project_path,"/../encodedDs/datasets/5000/nettalk.data",sep=""))
testDs <- read.table(file = paste(project_path,"/../encodedDs/datasets/5000/nettalk.test",sep=""))
articulatoryFeatures <- read.table(file=art_features_path, header=FALSE, sep=",")
outputIndexes <- (length(englishAlphabet)*7+1):(length(englishAlphabet)*7+length(articulatoryFeatures))


just_data <- rbind(trainDs,testDs)[,-outputIndexes]
  
answer.kmeans <- kmeans(just_data,nPhonems,10)$cluster

save.image(file=paste(project_path,"/kmeans.RData",sep=""))
