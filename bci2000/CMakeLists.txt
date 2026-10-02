###########################################################################
## Thai P300 speller: this project's BCI2000 modules.
##
## BCI2000 builds src/custom when it exists, so link this folder there once:
##   ln -s "<repo>/bci2000" ~/BCI2000/trunk/src/custom
## then configure and build from ~/BCI2000/trunk/build as usual. See bci2000/README.md.
###########################################################################

add_subdirectory(UnicornSerialSource)
add_subdirectory(CheckerboardSpeller)
add_subdirectory(CheckerboardSignalProcessing)
