INCLUDES=-I.
CC=g++
CFLAGS=-g -Wall $(INCLUDES)
LINKARGS=-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui
LIBS=-lopencv_core -lopencv_imgproc -lopencv_imgcodecs   -lopencv_highgui
DEPS=support.h

all: extractSingleLetters-threaded extractSingleLetters-sequential

extractSingleLetters-threaded: extractSingleLetters-threaded.cpp support.o
	$(CC) -pthread $(CFLAGS) $^ -o $@ $(LIBS) 

extractSingleLetters-sequential: extractSingleLetters-sequential.cpp support.o
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS)

%.o: %.cpp $(DEPS)
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -Rf extractSingleLetters-threaded extractSingleLetters-sequential support.o