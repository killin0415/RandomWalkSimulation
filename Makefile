CXX      = g++-15
CXXFLAGS = -O2 -std=c++17 -fopenmp -Wall
LDFLAGS  = -fopenmp -lm
TARGET   = rw_sim

PYTHON   = python3
DATADIR  = data
FIGDIR   = figures

DIMS     = 1 2 3 4
STEPS    = 100 1000 10000 100000 1000000
WALKS    = 1000

.PHONY: all build run plot report clean help

all: build run plot

help:
	@echo "Targets:"
	@echo "  build  - Compile the simulation"
	@echo "  run    - Run all simulations (D=1..4, n=100..1000000)"
	@echo "  plot   - Generate all figures"
	@echo "  report - Compile LaTeX report"
	@echo "  clean  - Remove generated files"
	@echo "  all    - build + run + plot"

build: $(TARGET)

$(TARGET): main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

run: $(TARGET)
	@mkdir -p $(DATADIR)
	./$(TARGET) -w $(WALKS) -o $(DATADIR)

run-quick: $(TARGET)
	@mkdir -p $(DATADIR)
	./$(TARGET) -w 10 -n 100 -d 1 -o $(DATADIR)

plot:
	@mkdir -p $(FIGDIR)
	$(PYTHON) figure.py --datadir $(DATADIR) --figdir $(FIGDIR)

report:
	pdflatex main.tex
	pdflatex main.tex

clean:
	rm -f $(TARGET)
	rm -rf $(DATADIR) $(FIGDIR)
	rm -f main.pdf main.aux main.log main.out main.toc main.synctex.gz
