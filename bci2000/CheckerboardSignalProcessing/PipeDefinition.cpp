////////////////////////////////////////////////////////////////////////////////
// CheckerboardSignalProcessing: the signal-processing module for the Thai checkerboard speller.
// One filter: RiemannP300Classifier, which epochs, band-passes and scores every flash itself
// (see RiemannP300Classifier.h), so no stock BCI2000 filter touches the signal before it.
////////////////////////////////////////////////////////////////////////////////
#include "RiemannP300Classifier.h"

Filter(RiemannP300Classifier, 2.A);
