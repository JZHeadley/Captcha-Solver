INCLUDES=-I.
CC=g++
CFLAGS=-g -Wall $(INCLUDES)
LINKARGS=-lm -lopencv_core -lopencv_imgproc -lopencv_imgcodecs 
LIBS=-lm -lopencv_core -lopencv_imgproc -lopencv_imgcodecs  
DEPS=support.h

all: imageProcessing-threaded imageProcessing-sequential

imageProcessing-threaded: ImageProcessing-threaded.cpp support.o
	$(CC) -pthread $(CFLAGS) $^ -o $@ $(LIBS) 

imageProcessing-sequential: ImageProcessing-sequential.cpp support.o
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS)

%.o: %.cpp $(DEPS)
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -Rf imageProcessing-threaded imageProcessing-sequential support.o