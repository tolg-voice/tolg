// Copyright 2016-2018 Lauri Juvela and Manu Airaksinen
// LF modelling extraction code Copyright: Phonetics and Speech Laboratory, Trinity College Dublin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//  <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
//               GlottDNN Speech Parameter Extractor including LF modelling Rd extraction
//  <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
//
//  This program reads a speech file and extracts speech
//  parameters using glottal inverse filtering.
//
//  This program has been written in Aalto University,
//  Department of Signal Processing and Acoustics, Espoo, Finland
//
//  This program uses some code from the original GlottHMM vocoder program
//  written by Tuomo Raitio, now re-factored and re-written in C++
//
//  Authors: Lauri Juvela, Manu Airaksinen
//  Acknowledgements: Tuomo Raitio, Paavo Alku
//  File Analysis.cpp
//  Version: 1.0

// This program referred the Matlab code from the original @ Voice_Analysis_Toolkit https://github.com/jckane/Voice_Analysis_Toolkit
// written by John Kane (Phonetics and Speech Laboratory, Trinity College Dublin) in Matlab, now re-factored and re-written in C++
// Author: Xiao Zhang (Phonetics and Speech Laboratory, Trinity College Dublin)  zhangx16@tcd.ie



/***********************************************/
/*                 INCLUDE                     */
/***********************************************/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cstring>

#include <iostream>
#include <iomanip>

#include <vector>
#include <gslwrap/vector_double.h>

#include "definitions.h"
#include "Filters.h"
#include "FileIo.h"
#include "ReadConfig.h"
#include "SpFunctions.h"
#include "AnalysisFunctions.h"

#include "Utils.h"


#include <gslwrap/random_generator.h>
#include <gslwrap/random_number_distribution.h>
#include <gsl/gsl_statistics_double.h>

#include "mex.h"
#include "math.h"
#include <algorithm>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_vector_int.h>
#include <gsl/gsl_filter.h>



int main(int argc, char *argv[]) {

   if (CheckCommandLineAnalysis(argc) == EXIT_FAILURE) {
      return EXIT_FAILURE;
   }

   const char *wav_filename = argv[1];
   const char *default_config_filename = argv[2];
   const char *user_config_filename = argv[3];

   /* Declare configuration parameter struct */
   Param params;

   /* Read configuration file */
   if (ReadConfig(default_config_filename, true, &params) == EXIT_FAILURE)
      return EXIT_FAILURE;
   if (argc > 3) {
      if (ReadConfig(user_config_filename, false, &params) == EXIT_FAILURE)
         return EXIT_FAILURE;
   }

   /* Read sound file and allocate data */
   AnalysisData data;

   if(ReadWavFile(wav_filename, &(data.signal), &params) == EXIT_FAILURE)
      return EXIT_FAILURE;

   data.AllocateData(params);

   /* High-pass filter signal to eliminate low frequency "rumble" */
   HighPassFiltering(params, &(data.signal));

   if(!params.use_external_f0 || !params.use_external_gci || (params.signal_polarity == POLARITY_DETECT))
      GetIaifResidual(params, data.signal, (&data.source_signal_iaif));

   /* Read or estimate signal polarity */
   PolarityDetection(params, &(data.signal), &(data.source_signal_iaif));

   /* Read or estimate fundamental frequency (F0)  */
   if(GetF0(params, data.signal, data.source_signal_iaif, &(data.fundf)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   /* Read or estimate glottal closure instants (GCIs)*/
   GetGci(params, data.signal, data.source_signal_iaif, data.fundf, &(data.gci_inds));

   /* Estimate frame log-energy (Gain) */
   GetGain(params, data.fundf, data.signal, &(data.frame_energy));

   /* Spectral analysis for vocal tract transfer function*/
   if(params.qmf_subband_analysis) {
      SpectralAnalysisQmf(params, data, &(data.poly_vocal_tract));
   } else {
      SpectralAnalysis(params, data, &(data.poly_vocal_tract));
   }

   /* Smooth vocal tract estimates in LSF domain */
   Poly2Lsf(data.poly_vocal_tract, &data.lsf_vocal_tract);
   MedianFilter(5, &data.lsf_vocal_tract);
   MovingAverageFilter(3, &data.lsf_vocal_tract);
   Lsf2Poly(data.lsf_vocal_tract, &data.poly_vocal_tract);

   /* Perform glottal inverse filtering with the estimated VT AR polynomials */
   InverseFilter(params, data, &(data.poly_glot), &(data.source_signal));

   /* Re-estimate GCIs on the residual */
   //if(GetGci(params, data.signal, data.source_signal, data.fundf, &(data.gci_inds)) == EXIT_FAILURE)
   //   return EXIT_FAILURE;

   /* Extract pitch synchronous (excitation) waveforms at each frame */
   if (params.use_waveforms_directly) {
      GetPulses(params, data.signal, data.gci_inds, data.fundf, &(data.excitation_pulses));
   } else {
      GetPulses(params, data.source_signal, data.gci_inds, data.fundf, &(data.excitation_pulses));
   }

   HnrAnalysis(params, data.source_signal, data.fundf, &(data.hnr_glot));

   /* Convert vocal tract AR polynomials to LSF */
   Poly2Lsf(data.poly_vocal_tract, &(data.lsf_vocal_tract));

   /* Convert glottal source AR polynomials to LSF */
   Poly2Lsf(data.poly_glot, &(data.lsf_glot));

   /* start to do the Rd param extraction */
   GetRd(params, data.source_signal, data.gci_inds, &(data.Rd_opt));

    std::cout << "********************* Rd_opt params *********************" << data.gci_inds << std::endl;

   /* Write analyzed features to files */
   data.SaveData(params);


    return EXIT_SUCCESS;

}

/***********/
/*   EOF   */
/***********/

