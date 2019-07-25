
englishAlphabet <- strsplit("abcdefghijklmnopqrstuvwxyz","")[[1]]
project_path      <- getwd()    
art_features_path <- paste(project_path,"/../encodedDs/articulatoryFeatures/phonems.csv",sep="") 
trainSize <- 1000

testEncoded <- read.table(file = paste(project_path,"/../encodedDs/datasets/",toString(trainSize),"/nettalk.test",sep=""))
prediction  <- read.csv(file=paste(project_path,"/predictions/",toString(trainSize),"/nettalk.predict",sep=""), header=FALSE, sep=" ")
valid_idexes <- as.matrix(read.csv(file = paste(project_path,"/predictions/",toString(trainSize),"/validIdx",sep=""),header = FALSE))
  
testEncoded <- testEncoded[-valid_idexes,]
  
### parameters for the encoding of the phonetics 

phonemicAlphabet <-  strsplit("abcdefghiklmnoprstuvwxyzACDEGIJKLMNOQRSTUWXYZ@!#*^-","")[[1]]
nPhonems <- length(phonemicAlphabet)
### Note: All words including the foreign phonem '+' have been manually removed from the raw dataset

# Read the rmatrix that contains the encoding for every phonem in binary articulatory features
articulatoryFeatures <- read.table(file=art_features_path, header=FALSE, sep=",")
articulatoryFeatures[is.na(articulatoryFeatures)] <- 0
articulatoryFeatures [which(phonemicAlphabet=='-'),] <- rep(0,dim(articulatoryFeatures)[2])
articulatoryFeatures <- as.matrix(articulatoryFeatures)

nFeatures <- dim(articulatoryFeatures)[2]


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
  #decodeRow(testEncoded[i,])
  actualPh <- getPhonemRow(testEncoded[i,])
  #cat(actualPh,"<- real \n")
  w <- closestPhonemRow(  prediction[i,]   )  
  predicPh <- phonemicAlphabet[w]
  #cat(predicPh,"<- predicted \n")
  return (actualPh==predicPh)
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

n <- calculateError()

