### THIS SCRIPTS ENCODES THE RAW NETTALK DATASET INTO A NUMERICAL DATASET
### UNDERSTANDABLE BY MACHINE LEARNING ALGORITMS


# Parameters of the dataset
contextSize <- 3  # Window size = 2 x contextSize +1

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

# word :: String
splitWord <- function(word){ 
  len      <- nchar(word)
  ans      <- c()
  for (i in 1:len){
    newRow <- getContext(word,len,i)
    ans <-  rbind(ans,newRow)
  }
  return(ans)
}

englishAlphabet <- "abcdefghijklmnopqrstuvwxyz"

