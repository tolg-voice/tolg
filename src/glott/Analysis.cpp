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

#include <memory>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "./reaper/core/file_resource.h"
#include "./reaper/core/track.h"
#include "./reaper/epoch_tracker/epoch_tracker.h"
#include "./reaper/wave/wave.h"

//void Rd2R(double Rd, double EE, double F0, double& Ra, double& Rk, double& Rg) {
//    Ra = (-1 + (4.8 * Rd)) / 100;
//    Rk = (22.4 + (11.8 * Rd)) / 100;
//    double EI = (M_PI * Rk * EE) / 2;
//    double UP = (Rd * EE) / (10 * F0);
//    Rg = EI / (F0 * UP * M_PI);
//}

Track* MakeEpochOutput(EpochTracker &et, float unvoiced_pm_interval) {
    std::vector<float> times;
    std::vector<int16_t> voicing;
    et.GetFilledEpochs(unvoiced_pm_interval, &times, &voicing);
    Track* pm_track = new Track;
    pm_track->resize(times.size());
    for (int32_t i = 0; i < times.size(); ++i) {
        pm_track->t(i) = times[i];
        pm_track->set_v(i, voicing[i]);
    }
    return pm_track;
}

Track* MakeF0Output(EpochTracker &et, float resample_interval, Track** cor) {
    std::vector<float> f0;
    std::vector<float> corr;
    if (!et.ResampleAndReturnResults(resample_interval, &f0, &corr)) {
        return NULL;
    }

    Track* f0_track = new Track;
    Track* cor_track = new Track;
    f0_track->resize(f0.size());
    cor_track->resize(corr.size());
    for (int32_t i = 0; i < f0.size(); ++i) {
        float t = resample_interval * i;
        f0_track->t(i) = t;
        cor_track->t(i) = t;
        f0_track->set_v(i, (f0[i] > 0.0) ? true : false);
        cor_track->set_v(i, (f0[i] > 0.0) ? true : false);
        f0_track->a(i) = (f0[i] > 0.0) ? f0[i] : -1.0;
        cor_track->a(i) = corr[i];
    }
    *cor = cor_track;
    return f0_track;
}

bool ComputeEpochsAndF0(EpochTracker &et, float unvoiced_pulse_interval,
                        float external_frame_interval,
                        Track** pm, Track** f0, Track** corr) {
    if (!et.ComputeFeatures()) {
        return false;
    }
    bool tr_result = et.TrackEpochs();
    et.WriteDiagnostics("");  // Try to save them here, even after tracking failure.
    if (!tr_result) {
        fprintf(stderr, "Problems in TrackEpochs");
        return false;
    }

    // create pm and f0 objects, these need to be freed in calling client.
    *pm = MakeEpochOutput(et, unvoiced_pulse_interval);
    *f0 = MakeF0Output(et, external_frame_interval, corr);
    return true;
}



void PrintTrack(const Track& track) {
    for (int i = 0; i < track.num_frames(); ++i) {
//        std::cout << "Time: " << track.t(i) << " ";
//        std::cout << "Voicing: " << track.v(i) << std::endl;
//        std::cout << "F0: " << track.a(i) << std::endl;

        if (track.v(i) == 1) {
            double GCI_val = track.t(i) * 16000.0;
            std::cout << GCI_val << std::endl;
        }
    }
}



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



//    std::cout << "********************* cost params *********************" << data.gci_inds << std::endl;


    int opt = 0;
//    std::string filename;
    std::string f0_output;
    std::string pm_output;
    std::string corr_output;
    bool do_hilbert_transform = kDoHilbertTransform;
    bool do_high_pass = kDoHighpass;
    float external_frame_interval = kExternalFrameInterval;
    float max_f0 = kMaxF0Search;
    float min_f0 = kMinF0Search;
    float inter_pulse = kUnvoicedPulseInterval;
    float unvoiced_cost = kUnvoicedCost;
    bool ascii = false;
    std::string debug_output;

    std::string filename(wav_filename);

//    filename = "./slt_arctic_a0004.wav";
    f0_output = "./arctic_a0001.f0";
    pm_output = "./arctic_a0001.pm";
    // Load input.
    Wave wav;
    if (!wav.Load(filename)) {
        fprintf(stderr, "Failed to load waveform '%s'\n", filename.c_str());
        return 1;
    }

    EpochTracker et;
    et.set_unvoiced_cost(unvoiced_cost);
    int16_t* wave_datap = const_cast<int16_t *>(wav.data()->data());
    int32_t n_samples = wav.num_samples();
    float sample_rate = wav.sample_rate();
    if (!et.Init(wave_datap, n_samples, sample_rate,
                 min_f0, max_f0, do_high_pass, do_hilbert_transform)) {
        return 1;
    }
    if (!debug_output.empty()) {
        et.set_debug_name(debug_output);
    }
    // Compute f0 and pitchmarks.
    Track *f0 = NULL;
    Track *pm = NULL;
    Track *corr = NULL;
    if (!ComputeEpochsAndF0(et, inter_pulse, external_frame_interval, &pm, &f0, &corr)) {
        fprintf(stderr, "Failed to compute epochs\n");
        return 1;
    }

    // Print f0 track
//    if (f0 != nullptr) {
//        PrintTrack(*f0);
//    } else {
//        std::cerr << "F0 track is null" << std::endl;
//    }

    std::vector<double> F0_Reaper;
    if (f0 != nullptr) {
        const Track& track = *f0;
        for (int i = 0; i < track.num_frames(); ++i) {
            if (track.a(i) != -1) {
                double F0_val = track.a(i) ;
//                std::cout << F0_val << std::endl;
                F0_Reaper.push_back(F0_val); // Insert GCI_val into GCI_Reaper vector
            }
        }
    } else {
        std::cerr << "F0 track is null" << std::endl;
    }


    // Convert GCI_Reaper to gsl::vector
    data.F0_Reaper_gsl.resize(F0_Reaper.size());
    for (size_t i = 0; i < F0_Reaper.size(); ++i) {
        data.F0_Reaper_gsl[i] = F0_Reaper[i];
    }



    std::vector<double> GCI_Reaper;
    if (f0 != nullptr) {
        const Track& track = *f0;
        for (int i = 0; i < track.num_frames(); ++i) {
            if (track.v(i) == 1) {
                int GCI_val = track.t(i) * 16000.0;
//                std::cout << GCI_val << std::endl;
                GCI_Reaper.push_back(GCI_val); // Insert GCI_val into GCI_Reaper vector
            }
        }
    } else {
        std::cerr << "GCI track is null" << std::endl;
    }


    // Convert GCI_Reaper to gsl::vector
    data.GCI_Reaper_gsl.resize(GCI_Reaper.size());
    for (size_t i = 0; i < GCI_Reaper.size(); ++i) {
        data.GCI_Reaper_gsl[i] = GCI_Reaper[i];
    }
//    gsl::vector GCI_Reaper;
//
//    if (f0 != nullptr) {
//        const Track& track = *f0;
//        GCI_Reaper.resize(track.num_frames()); // Resize GCI_Reaper vector
//
////        int index = 0;
//        for (int i = 0; i < track.num_frames(); ++i) {
//            if (track.v(i) == 1) {
//                int GCI_val = track.t(i) * 16000;
//                GCI_Reaper[i] = GCI_val; // Assign GCI_val to element at index
////                ++index;
//            }
//        }
//    } else {
//        std::cerr << "F0 track is null" << std::endl;
//    }


    /* start to do the Rd param extraction */
   GetRd(params, data.source_signal, data.GCI_Reaper_gsl, &(data.Rd_opt_temp), &(data.EE));

    data.Rd_opt.resize(data.fundf.size());
    data.Rd_opt.set_zero();

    for (int i = 0; i < data.Rd_opt_temp.size(); ++i) {
        data.Rd_opt[i] = data.Rd_opt_temp[i];
    }



   data.Ra.resize(data.Rd_opt_temp.size());
   data.Rk.resize(data.Rd_opt_temp.size());
   data.Rg.resize(data.Rd_opt_temp.size());




    //void Rd2R(double Rd, double EE, double F0, double& Ra, double& Rk, double& Rg) {
//    Ra = (-1 + (4.8 * Rd)) / 100;
//    Rk = (22.4 + (11.8 * Rd)) / 100;
//    double EI = (M_PI * Rk * EE) / 2;
//    double UP = (Rd * EE) / (10 * F0);
//    Rg = EI / (F0 * UP * M_PI);
//}
    double  Ra_cur;
    double Rk_cur;
    double Rg_cur;


    for (size_t i = 0; i < data.Rd_opt_temp.size(); ++i) {
        Rd2R(data.Rd_opt_temp(i), data.EE(i), data.F0_Reaper_gsl(i), Ra_cur, Rk_cur, Rg_cur);
        data.Ra[i] = Ra_cur;
        data.Rk[i] = Rk_cur;
        data.Rg[i] = Rg_cur;

//        Ra.push_back(Ra_cur); // Insert GCI_val into GCI_Reaper vector

    }

//    std::cout << "********************* cost params *********************" << data.Ra << std::endl;
//    std::cout << "********************* cost params *********************" << data.Rk << std::endl;
//    std::cout << "********************* cost params *********************" << data.Rg << std::endl;
//
//    std::cout << "********************* cost params *********************" << data.EE << std::endl;
//    std::cout << "********************* cost params *********************" << data.GCI_Reaper_gsl << std::endl;
    std::cout << "********************* cost params *********************" << data.Rd_opt.size() << std::endl;
    std::cout << "********************* cost params *********************" << data.fundf.size() << std::endl;

//    std::cout << "********************* cost params *********************" << data.GCI_Reaper_gsl << std::endl;
//    std::cout << "********************* cost params *********************" << data.F0_Reaper << std::endl;

    /* Write analyzed features to files */
   data.SaveData(params);


   return EXIT_SUCCESS;

}

/***********/
/*   EOF   */
/***********/

