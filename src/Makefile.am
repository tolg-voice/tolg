AUTOMAKE_OPTIONS = subdir-objects
AM_CXXFLAGS = -std=c++11 -Wall -Wextra -pedantic

noinst_LIBRARIES = libgslwrap.a
libgslwrap_a_SOURCES = gslwrap/matrix_float.h gslwrap/matrix_double.h gslwrap/matrix_int.h \
gslwrap/vector_double.h  gslwrap/vector_float.h  gslwrap/vector_int.h \
gslwrap/permutation.h \
gslwrap/histogram.h \
gslwrap/matrix_vector_operators.h \
gslwrap/min_fminimizer.h gslwrap/multimin_fdfminimizer.h \
gslwrap/random_generator.h \
gslwrap/random_number_distribution.h \
gslwrap/matrix_float.cc gslwrap/matrix_double.cc gslwrap/matrix_int.cc \
gslwrap/vector_float.cc gslwrap/vector_double.cc gslwrap/vector_int.cc \
gslwrap/random_generator.cc 

bin_PROGRAMS = Analysis Synthesis LsfPostFilter

Analysis_SOURCES = glott/Analysis.cpp Filters.h \
					glott/definitions.h glott/definitions.cpp \
					glott/ReadConfig.h glott/ReadConfig.cpp \
					glott/FileIo.h glott/FileIo.cpp \
					glott/InverseFiltering.h glott/InverseFiltering.cpp \
					glott/ComplexVector.h glott/ComplexVector.cpp \
					glott/SpFunctions.h glott/SpFunctions.cpp \
					glott/QmfFunctions.h glott/QmfFunctions.cpp \
					glott/PitchEstimation.h glott/PitchEstimation.cpp \
					glott/AnalysisFunctions.h glott/AnalysisFunctions.cpp \
					glott/Utils.h glott/Utils.cpp

Analysis_LDADD = libgslwrap.a libcore.a libepoch_tracker.a libwave.a

Synthesis_SOURCES = glott/Synthesis.cpp Filters.h \
					glott/definitions.h glott/definitions.cpp \
					glott/ReadConfig.h glott/ReadConfig.cpp \
					glott/FileIo.h glott/FileIo.cpp \
					glott/InverseFiltering.h glott/InverseFiltering.cpp \
					glott/ComplexVector.h glott/ComplexVector.cpp \
					glott/SpFunctions.h glott/SpFunctions.cpp \
					glott/DnnClass.h glott/DnnClass.cpp \
					glott/SynthesisFunctions.h glott/SynthesisFunctions.cpp \
					glott/Utils.h glott/Utils.cpp

Synthesis_LDADD = libgslwrap.a

LsfPostFilter_SOURCES = glott/LsfPostFilterMain.cpp Filters.h \
					glott/definitions.h glott/definitions.cpp \
					glott/ReadConfig.h glott/ReadConfig.cpp \
					glott/FileIo.h glott/FileIo.cpp \
					glott/InverseFiltering.h glott/InverseFiltering.cpp \
					glott/ComplexVector.h glott/ComplexVector.cpp \
					glott/SpFunctions.h glott/SpFunctions.cpp \
					glott/DnnClass.h glott/DnnClass.cpp \
					glott/SynthesisFunctions.h glott/SynthesisFunctions.cpp \
					glott/Utils.h glott/Utils.cpp

LsfPostFilter_LDADD = libgslwrap.a
