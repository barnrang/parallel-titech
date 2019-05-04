CC = gcc
CFLAGS = -g -O2 -fopenmp -std=c99
LDFLAGS = -fopenmp
LIBS = 

APP = sort
OBJS = $(APP).o

APP_P = sort_omp
OBJS_P = $(APP_P).o

MERGE = merge
MERGE_OBJS = $(MERGE).o

MERGE_P = merge_omp
MERGE_P_OBJS = $(MERGE_P).o

OE = odd_even
OE_OBJS = $(OE).o

OE_P = odd_even_omp
OE_P_OBJS = $(OE_P).o

all: $(APP) $(APP_P) $(MERGE) $(MERGE_P) $(OE_P) $(OE)

$(OE): $(OE_OBJS)
	$(CC) $^ $(LIBS) -o $@ $(LDFLAGS)

$(OE_P): $(OE_P_OBJS)
	$(CC) $^ $(LIBS) -o $@ $(LDFLAGS)

$(MERGE_P): $(MERGE_P_OBJS)
	$(CC) $^ $(LIBS) -o $@ $(LDFLAGS)

$(MERGE): $(MERGE_OBJS)
	$(CC) $^ $(LIBS) -o $@ $(LDFLAGS)

$(APP_P): $(OBJS_P)
	$(CC) $^ $(LIBS) -o $@ $(LDFLAGS)

$(APP): $(OBJS)
	$(CC) $^ $(LIBS) -o $@ $(LDFLAGS)

%.o : %.c
	$(CC) $(CFLAGS) -c $< -o $*.o

clean:
	rm -f *.o
	rm -f *~
	rm -f $(APP)
