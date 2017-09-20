
all:
EXTRA_CXXFLAGS = -Wno-sign-compare

KALDI_SRC = $(KALDI_ROOT)/src

include $(KALDI_SRC)/kaldi.mk

BINFILES = 


OBJFILES = 


ADDLIBS = $(KALDI_SRC)/lm/kaldi-lm.a $(KALDI_SRC)/decoder/kaldi-decoder.a $(KALDI_SRC)/lat/kaldi-lat.a \
          $(KALDI_SRC)/hmm/kaldi-hmm.a $(KALDI_SRC)/transform/kaldi-transform.a $(KALDI_SRC)/gmm/kaldi-gmm.a \
          $(KALDI_SRC)/tree/kaldi-tree.a  $(KALDI_SRC)/util/kaldi-util.a $(KALDI_SRC)/matrix/kaldi-matrix.a \
          $(KALDI_SRC)/base/kaldi-base.a

TESTFILES =

PYLIB = $(shell python-config --libs)
PYINC = $(shell python-config --includes)
NPINC = -I$(shell python -c 'import numpy; print(numpy.get_include())')

PYLIBS = kaldi_io_internal.so

#include $(KALDI_SRC)/makefiles/default_rules.mk

%.so: %.cpp
		g++ -shared -o $@ -Wall -fPIC -I$(KALDI_SRC) $(PYINC) $(NPINC) $(CXXFLAGS) $< $(ADDLIBS) $(LDFLAGS) -L$(PYLIB) $(LOADLIBES) $(LDLIBS) -lpython2.7 -lboost_python -lboost_system 

clean:
	-rm -f *.o *.a *.so $(TESTFILES) $(BINFILES) $(TESTOUTPUTS) tmp* *.tmp

depend:
	-$(CXX) -I$(KALDI_SRC) $(PYINC) $(NPINC) -M $(CXXFLAGS) *.cpp > .depend.mk  

test:

all: $(PYLIBS)
