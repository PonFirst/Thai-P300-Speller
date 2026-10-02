###########################################################################
## CheckerboardSignalProcessing: BCI2000 signal-processing module that scores
## each checkerboard flash with the exported xDAWN/tangent-space/LDA model.
## See RiemannP300Classifier.h.
###########################################################################

bci2000_include(EIGEN)
bci2000_add_signal_processing_module(
  CheckerboardSignalProcessing
  PipeDefinition.cpp
  RiemannP300Classifier.cpp
)
