SRCDIR = kaldi-python

ifndef KALDI_ROOT
$(error please set KALDI_ROOT to point ot the base of the kaldi installation)
endif

.PHONY: all

all:
	$(MAKE) -C $(SRCDIR) depend
	$(MAKE) -C $(SRCDIR)

clean:
	$(MAKE) -C $(SRCDIR) clean
