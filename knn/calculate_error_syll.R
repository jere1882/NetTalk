library(class)

englishAlphabet <- strsplit("abcdefghijklmnopqrstuvwxyz","")[[1]]
project_path      <-  getwd() 
art_features_path <- paste(project_path,"/../encodedDs/articulatoryFeatures/phonems.csv",sep="") 

center_inputs <- (52+1):(104+26)

trainSize<-5000

cat("Iterating with train size",trainSize,"\n")
inputIndexes <- 1:(length(englishAlphabet)*7)
trainDs <- read.table(file = paste(project_path,"/../encodedDs/datasets/",toString(trainSize),"/nettalk.data",sep=""))
testDs <- read.table(file = paste(project_path,"/../encodedDs/datasets/",toString(trainSize),"/nettalk.test",sep=""))

articulatoryFeatures <- read.table(file=art_features_path, header=FALSE, sep=",")

outputIndexes <- (length(englishAlphabet)*7+1):(length(englishAlphabet)*7+length(articulatoryFeatures))


###########################################################################
################ Set parameters of the dataset and encoding ###############

contextSize     <- 3  
inputSize       <- 2*contextSize+1  # i.e. window size

englishAlphabet <- strsplit("abcdefghijklmnopqrstuvwxyz","")[[1]]
nLetters        <- length(englishAlphabet)


######

closestPhonem <- function(vec){
  maxHits    <- 0
  maxHitsIdx <- 0
  for (i in 1:nPhonems){
    hits <- sum(vec == articulatoryFeatures[i,])
    if (hits > maxHits){
      maxHits    <- hits
      maxHitsIdx <- i
    }
  }
  return(maxHitsIdx);
}


decodeLetter <- function(code){
  if (sum(code)==0)
    return('-')
  else{
    idx <- which(code==1)
    return(englishAlphabet[idx])
  }
}

decodeRow <- function(vec){
  for(i in 1:7){
    letter <- vec[((i-1)*26+1):(i*26)]
    if (i!=4) cat(decodeLetter(letter),"\n")
    if (i==4) cat(decodeLetter(letter)," <- \n")
  }
} 



decodePhonem <- function(v){
  for (i in 1:length(phonemicAlphabet)){
    if(sum(v!=articulatoryFeatures[i,]) == 0)
      return (phonemicAlphabet[i])
  }
  return('?')
}


getPhonemRow <- function(row){
  return(decodePhonem(row[-(1:(inputSize*nLetters))]))
}

closestPhonemRow <- function(row){
  return(closestPhonem(row[-(1:(inputSize*nLetters))]))
}

testIdx <- function(i){
  
  center_bits <- testEncoded[i,center_inputs]
  actualPh <- getPhonemRow(testEncoded[i,])
  
  min_idx  <- which.min(rowSums(abs(sweep(as.matrix(trainDs[,center_inputs]),2,as.matrix(center_bits)))))
  
  #cat("Input is:\n")
  #decodeRow(testEncoded[i,])
  
  
  #w <- closestPhonemRow(  trainDs[min_idx,]   )  
  #predicPh <- phonemicAlphabet[w]
  #cat("actual Ph: ",actualPh," Its NN phoenm ",predicPh,"\n")
  #decodeRow(trainDs[min_idx,])
  #return (actualPh==predicPh)
  return(sum(trainDs[min_idx,-(1:(inputSize*nLetters))]!=testEncoded[i,-(1:(inputSize*nLetters))] )==0)
  
}

lenTest <- dim(testEncoded)[1]


calculateError <- function(){
  hits <- 0
  for (i in 1:lenTest){
    cat(i,"\n");
    if(testIdx(i)){ hits <- hits+1 }
  }
  return(hits);
}

