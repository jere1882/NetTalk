### THIS SCRIPTS ENCODES THE RAW NETTALK DATASET INTO A NUMERICAL DATASET
### UNDERSTANDABLE BY MACHINE LEARNING ALGORITMS


# Parameters 
contextSize     <- 3  # Window size = 2 x contextSize +1
englishAlphabet <- strsplit("abcdefghijklmnopqrstuvwxyz","")[[1]]
nLetters        <- length(englishAlphabet)

# Read the raw dataset
rawDataset <- read.csv(file="/home/jeremias/Desktop/netTalk/rawDataset/nettalk.data", header=FALSE, sep="\t")
names(rawDataset) <- c("english","phonetics","structure","foreign")

english   <- rawDataset$english
phonetics <- rawDataset$phonetics


# Let's start encoding the written English representation

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

# Word == String
# splitWord :: Word -> Matrix(Char)  
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

encodeLetter <- function(letter){
  nLetter <- which(englishAlphabet==letter)
  ans     <- rep(0,nLetters)
  ans[nLetter] <- 1
  return(ans)
}

encodeWord   <-function(word){
  rows        <- splitWord(word)
  unformatted <- c(apply(rows,c(1,2),encodeLetter))
  formatted   <- matrix(unformatted,ncol= nLetters*(1+2*contextSize),byrow=TRUE)
  return(formatted)
}





