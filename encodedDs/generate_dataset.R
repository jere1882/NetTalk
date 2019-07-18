## !! IMPORTANT !!  Run this script from NetTalk/encodedDs/

### THIS SCRIPT GENERATES A NUMERICAL DATASET, UNDERSTANDABLE BY MACHINE 
### LEARNING ALGORITHMS, ENCODING THE RAW NETTALK DATASET  

project_path      <- getwd()    #"/home/jeremiasrodriguez/Desktop/NetTalk/encodedDs"
raw_dataset_path  <- paste(project_path,"/../rawDataset/nettalk.data",sep="")
art_features_path <- paste(project_path,"/articulatoryFeatures/phonems.csv",sep="") 

###########################################################################
################ Set parameters of the dataset and encoding ###############

contextSize     <- 3  
inputSize       <- 2*contextSize+1  # i.e. window size

englishAlphabet <- strsplit("abcdefghijklmnopqrstuvwxyz","")[[1]]
nLetters        <- length(englishAlphabet)

# Read the raw dataset
rawDataset <- read.csv(file=raw_dataset_path, header=FALSE, sep="\t")
names(rawDataset) <- c("english","phonetics","structure","foreign")

###########################################################################
#######  Let's start encoding the written English representation    #######

getContext <- function(word,len,i){
  ans <- substr(word,i-3,i+3)
  if (i <= 3){
    prefx <- paste(rep('-',4-i),collapse="")
    ans   <- paste(prefx,ans,sep="")
  }
  if (len- i < 3) {
    sufix <- paste(rep('-',3-(len-i)),collapse="")
    ans   <- paste(ans,sufix,sep="")
  }
  return(ans)
}

# > splitWord("pumpkin")
# [,1] [,2] [,3] [,4] [,5] [,6] [,7]
# [1,] "-"  "-"  "-"  "p"  "u"  "m"  "p" 
# [2,] "-"  "-"  "p"  "u"  "m"  "p"  "k" 
# [3,] "-"  "p"  "u"  "m"  "p"  "k"  "i" 
# [4,] "p"  "u"  "m"  "p"  "k"  "i"  "n" 
# [5,] "u"  "m"  "p"  "k"  "i"  "n"  "-" 
# [6,] "m"  "p"  "k"  "i"  "n"  "-"  "-" 
# [7,] "p"  "k"  "i"  "n"  "-"  "-"  "-" 

# Splits a word, creating a row for each (letter+its context)
splitWord <- function(word){ 
  len      <- nchar(word)
  ans      <- c()
  for (i in 1:len){
    newRow <- getContext(word,len,i)
    ans <-  rbind(ans,newRow)
  }
  
  ans <- strsplit(ans,"")   # SPlit strings in vectors of letters
  ans <- do.call(rbind,ans) # Format it as a matrix of characters
  
  return(ans)
}

# encodes a letter in a simple binary representation
# e.g. encodeLetter(a) = 1 0 0 0 0 0 0 0 0 0 (..) 0 0 0
encodeLetter <- function(letter){
  nLetter <- which(englishAlphabet==letter)
  ans     <- rep(0,nLetters)
  ans[nLetter] <- 1
  return(ans)
}

decodeLetter <- function(code){
  if (sum(code)==0)
    return('-')
  else{
    idx <- which(code==1)
    return(englishAlphabet[idx])
  }
}

# splits the word and then encodes each letter
encodeWord   <-function(word){
  rows       <- splitWord(word)
  ans        <- matrix(0,ncol = nLetters * inputSize,nrow = 0)
  for (line in 1:nchar(word)){
    currentRow <- rows[line,]
    encodedRow <- unlist(lapply(currentRow,encodeLetter))
    ans        <- rbind(ans,encodedRow)
    
  }
  
  return(ans)
}

############################ Just for testing  ############################
decodeRow <- function(vec){
  for(i in 1:7){
    letter <- vec[((i-1)*26+1):(i*26)]
    if (i!=4) cat(decodeLetter(letter),"\n")
    if (i==4) cat(decodeLetter(letter)," <- \n")
  }
} 

# encodeDecode("probame")
encodeDecode <- function(word){
  ans <- encodeWord(word)
  for (i in 1:(nchar(word))){
    decodeRow(ans[i,])
    cat('\n')
  }
}

###########################################################################
##############   Let's encode the phonetic representation    ##############


### parameters for the encoding of the phonetics 

phonemicAlphabet <-  strsplit("abcdefghiklmnoprstuvwxyzACDEGIJKLMNOQRSTUWXYZ@!#*^-","")[[1]]
### Note: All words including the foreign phonem '+' have been manually removed from the raw dataset

# Read the rmatrix that contains the encoding for every phonem in binary articulatory features
articulatoryFeatures <- read.table(file=art_features_path, header=FALSE, sep=",")
articulatoryFeatures[is.na(articulatoryFeatures)] <- 0
articulatoryFeatures [which(phonemicAlphabet=='-'),] <- rep(0,dim(articulatoryFeatures)[2])
articulatoryFeatures <- as.matrix(articulatoryFeatures)

nFeatures <- dim(articulatoryFeatures)[2]

######

encodePhonem <- function(f){
  nPhon <- which(phonemicAlphabet==f)
  return(articulatoryFeatures[nPhon,])
}

decodePhonem <- function(v){
  for (i in 1:length(phonemicAlphabet)){
    if(sum(v!=articulatoryFeatures[i,]) == 0)
      return(phonemicAlphabet[i])
  }
  return('?')
}

# encodes a phonetic word like "dImoni@k" (from demoniac) into a 1-0 matrix
# each row corresponds to one of the phomens, and encodes that phonem using
# +- 16 articulatory features such as high, tensed, voiced, unvoiced, dental

encodePhoneticWord <- function(word){
  phonems <- strsplit(word,"")[[1]]
  ans <- matrix(0,nrow=length(phonems),ncol=nFeatures)
  for (i in 1:length(phonems)){
    ans[i,] <- encodePhonem(phonems[i])
  }
  return(ans)
}

#### debud 
getPhonemRow <- function(row){
  return(decodePhonem(row[-(1:(inputSize*nLetters))]))
}

###########################################################################
###########################################################################

# Now, we can translate a whole dataset
generateDataset <- function(dataset){
  
  # Generate the left submatrix corresponding to the features
  englishWords <- as.matrix(dataset$english)
  
  encodedWords <- matrix(0,ncol=inputSize*nLetters,nrow=0)
  
  for (w in englishWords){
    cat(w,"\n")
    encodedWords <- rbind(encodedWords,encodeWord(w))
  }
  
  # Generate the submatrix corresponding to the target outputs
  phoneticTranscriptions <- as.matrix(dataset$phonetics)
  
  encodedPhonetics <- matrix(0,ncol=nFeatures,nrow=0)
  
  for (p in phoneticTranscriptions){
    cat(p,"\n")
    encodedPhonetics <- rbind(encodedPhonetics,encodePhoneticWord(p))
  }
  
  # join them in the final dataset
  return(cbind(encodedWords,encodedPhonetics))
}


## Testing purposes
testEncoding <- function(){
  i<- sample(1:19987,5)
  cat("Testing encoding of rows",i,"\n")
  
  rows <- rawDataset[i,]
  ed   <- generateDataset(rows)
  
  lastidx <- 1
  for (idx in i){
    row <- rawDataset[idx,]
    print("Let's check word: ")
    print(row$english)
    nChars <- nchar(toString(row$english))
    for (n in lastidx:(lastidx +nChars -1)) {
      nrow <- ed[n,]
      # LETS DECODE EACH ROW OF THIS WORD
      cat("\nrow ",n," ground truth ",toString(row$english)," ph:",toString(row$phonetics)," \n")
      decodeRow(nrow)
      cat("Phonems is: ", getPhonemRow(nrow),"\n") 
    }
    lastidx <- lastidx + nChars
  }
  
}
###########################################################################
###########################################################################

## PLAN: Repeat this code for 

for(trainSize in c(200,1000,5000)){
trainIdx <- sample(1:(dim(rawDataset)[1]),trainSize)
trainRaw <- rawDataset[trainIdx,]
testRaw  <- rawDataset[-trainIdx,]

trainEncoded <- generateDataset(trainRaw)
testEncoded  <- generateDataset(testRaw)

write.table(trainEncoded, file = paste(project_path,"/datasets/",toString(trainSize),"/nettalk.data",sep=""), row.names = FALSE, col.names = FALSE)
write.table(testEncoded, file = paste(project_path,"/datasets/",toString(trainSize),"/nettalk.test",sep=""), row.names = FALSE, col.names = FALSE)
write.table(trainIdx,file = paste(project_path,"/datasets/",toString(trainSize),"/trainIdx",sep=""), row.names = FALSE, col.names = FALSE)
}

############################################################################
############################################################################



