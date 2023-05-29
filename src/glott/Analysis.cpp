// Copyright 2016-2018 Lauri Juvela and Manu Airaksinen
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

//  <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
//               GlottDNN Speech Parameter Extractor
//  <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
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
//
//  File Analysis.cpp
//  Version: 1.0


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



const double pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679;

/* summation function */
double sum(double a[], int size)
{
    double summed=0;
    int i;
    for (i=0;i<size;i++)
    {
        summed=summed+a[i];
    }
    return summed;
}

void lfSource(double &alpha, double &epsi, double Tc, double fs, double Tp, double Te, double Ta, double EE) {

    // Initialize
    double TolFun = 0.0000001;
    int MaxIter = 100;
    int count = 1;
    double Tb = Tc - Te;
    double omega_g = M_PI / Tp;
    double eps0, change = 1.0, f_eps, f_eps_prime;

    // Solve epsilon using Newton-Raphson method
    eps0 = 1 / Ta;
    while (count <= MaxIter && std::fabs(change) > TolFun) {
        f_eps = (eps0 * Ta - 1.0 + std::exp(-eps0 * Tb));
        f_eps_prime = (Ta - Tb * std::exp(-eps0 * Tb));
        change = f_eps / f_eps_prime;
        eps0 = eps0 - change;
        eps0 = eps0 - (eps0 * Ta - 1 + std::exp(-eps0 * Tb)) / (Ta - Tb * std::exp(-eps0 * Tb));
        count++;
    }
    epsi = eps0;

    // Solve alpha - Do Newton-Raphson iterations
    double a0 = 0.0;
    double change_alpha = 1.0;
    double E0 = -EE / (std::exp(a0 * Te) * std::sin(omega_g * Te));
    double A2, f_a, f_a_prime, e0, part1, part2, part3, partAtan, part4, A1;

    A2 = (-EE / ((epsi * epsi) * Ta)) * (1 - std::exp(-epsi * Tb) * (1 + epsi * Tb));

    int count_alpha = 1; // Add declaration and initialization of count_alpha here

    while (count_alpha <= MaxIter && std::fabs(change_alpha) > TolFun) {
        part1 = std::sqrt((a0 * a0) + (omega_g * omega_g));
        partAtan = 2 * std::atan((std::sqrt((a0 * a0) + (omega_g * omega_g)) - a0) / omega_g);
        part2 = std::sin(omega_g * Te - partAtan);
        part3 = omega_g * std::exp(-a0 * Te) - ((A2 / EE) * ((a0 * a0) + (omega_g * omega_g)) * std::sin(omega_g * Te));
        part4 = (std::sin(omega_g * Te) * (1 - 2 * a0 * A2 / EE) - omega_g * Te * std::exp(-a0 * Te));
        a0 = a0 - ((part1 * part2) + part3) / part4;

        part1 = std::sqrt((a0 * a0) + (omega_g * omega_g));
        partAtan = 2 * std::atan((std::sqrt((a0 * a0) + (omega_g * omega_g)) - a0) / omega_g);
        part2 = std::sin(omega_g * Te - partAtan);
        part3 = omega_g * std::exp(-a0 * Te) - ((A2 / EE) * ((a0 * a0) + (omega_g * omega_g)) * std::sin(omega_g * Te));
        part4 = (std::sin(omega_g * Te) * (1 - 2 * a0 * A2 / EE) - omega_g * Te * std::exp(-a0 * Te));

        a0 = a0 - ((part1 * part2) + part3) / part4;

        count_alpha++;
    }

    alpha = a0;

}


/*******************************************************************/
/*                          MAIN                                   */
/*******************************************************************/
void Rd2R(double Rd, double EE, double F0, double& Ra, double& Rk, double& Rg) {
    Ra = (-1 + (4.8 * Rd)) / 100;
    Rk = (22.4 + (11.8 * Rd)) / 100;
    double EI = (M_PI * Rk * EE) / 2;
    double UP = (Rd * EE) / (10 * F0);
    Rg = EI / (F0 * UP * M_PI);
}


//gsl::vector lf_cont(double F0, double fs, double Ra, double Rk, double Rg, double EE) {
//    double F0min = 20;
//    double F0max = 500;
//
//    // Set LF model parameters
//    double T0 = 1 / F0;
//    double Ta = Ra * T0;
//    double Te = ((1 + Rk) / (2 * Rg)) * T0;
//    double Tp = Te / (Rk + 1);
//    double Tb = ((1 - (Rk + 1) / (2 * Rg)) * 1 / F0);
//    double Tc = Tb + Te;
//
//    if (F0 < F0min || F0 > F0max) {
//        return gsl::vector(); // Return empty vector for invalid F0
//    } else {
//        // Solve area balance using Newton-Raphson method
//        double alpha, epsi;
//
//        lfSource(alpha, epsi, Tc, fs, Tp, Te, Ta, EE);
//
//        double omega = M_PI / Tp;
//        double E0 = -(std::abs(EE)) / (std::exp(alpha * Te) * std::sin(omega * Te));
//
//        // Generate open phase and closed phase and combine
//        double dt = 1 / fs;
//
//        size_t T1_size = static_cast<size_t>(std::round(Te / dt));
//
//        size_t T2_size = static_cast<size_t>(std::round((Tc - Te) / dt));
//
//        gsl::vector T1(T1_size);
//        gsl::vector T2(T2_size);
//
//        for (size_t i = 0; i < T1_size; i++) {
//            double t = dt * i;
//            T1[i] = E0 * std::exp(alpha * t) * std::sin(omega * t);
//        }
////Todo
//
//        for (size_t i = 0; i < T2_size; i++) {
//            double t = (T1_size * dt) + dt * i;
//            T2[i] = (-EE / (epsi * Ta)) * (std::exp(-epsi * (t - Te)) - std::exp(-epsi * Tb));
//        }
//
//        gsl::vector g_LF(T1_size + T2_size);
//        for (size_t i = 0; i < T1_size; i++) {
//            g_LF[i] = T1[i];
//        }
//        for (size_t i = 0; i < T2_size; i++) {
//            g_LF[T1_size + i] = T2[i];
//        }
//
//
//        return g_LF;
//
//    }
//}


gsl::vector lf_cont(double F0, double fs, double Ra, double Rk, double Rg, double EE) {
    const double F0min = 20.0;
    const double F0max = 500.0;

    // Set LF model parameters
    double T0 = 1.0 / F0;
    double Ta = Ra * T0;
    double Te = ((1.0 + Rk) / (2.0 * Rg)) * T0;
    double Tp = Te / (Rk + 1.0);
    double Tb = ((1.0 - (Rk + 1.0) / (2.0 * Rg)) * 1.0 / F0);
    double Tc = Tb + Te;

    if (F0 < F0min || F0 > F0max) {
        return gsl::vector(); // Return empty vector for invalid F0
    } else {
        // Solve area balance using Newton-Raphson method
        double alpha, epsi;

        lfSource(alpha, epsi, Tc, fs, Tp, Te, Ta, EE);

        double omega = M_PI / Tp;
        double E0 = -(std::abs(EE)) / (std::exp(alpha * Te) * std::sin(omega * Te));

        // Generate open phase and closed phase and combine
        double dt = 1.0 / fs;

        size_t T1_size = static_cast<size_t>(std::round(Te / dt));
        size_t T2_size = static_cast<size_t>(std::round((Tc - Te) / dt));

        // Ensure T1_size and T2_size are positive
        T1_size = std::max(T1_size, static_cast<size_t>(1));
        T2_size = std::max(T2_size, static_cast<size_t>(1));

        gsl::vector T1(T1_size);
        gsl::vector T2(T2_size);

        for (size_t i = 0; i < T1_size; i++) {
            double t = dt * i;
            T1[i] = E0 * std::exp(alpha * t) * std::sin(omega * t);
        }

        for (size_t i = 0; i < T2_size; i++) {
            double t = (T1_size * dt) + dt * i;
            T2[i] = (-EE / (epsi * Ta)) * (std::exp(-epsi * (t - Te)) - std::exp(-epsi * Tb));
        }

        gsl::vector g_LF(T1_size + T2_size);
        for (size_t i = 0; i < T1_size; i++) {
            g_LF[i] = T1[i];
        }
        for (size_t i = 0; i < T2_size; i++) {
            g_LF[T1_size + i] = T2[i];
        }

        return g_LF;
    }
}


//std::cout << "alpha " << alpha  << std::endl;

gsl::vector makePulseCentGCI(const gsl::vector pulse, int winLen, int start, int finish) {
    size_t pulseLen = pulse.size();

    // Find the index of the minimum value in pulse
    double minVal = pulse(0);
    size_t idx = 0;

    for (size_t i = 1; i < pulse.size(); ++i) {
        if (pulse(i) < minVal) {
            minVal = pulse(i);
            idx = i;
        }
    }

    size_t group_idx = idx + pulseLen;

    size_t pulseGroupLen = pulseLen * 3;



    gsl::vector pulseGroup(pulse.size() * 3);  // Create a vector to store pulseGroup

    // Repeat pulse three times
    for (size_t i = 0; i < pulseGroupLen; i += pulseLen) {
        for (size_t j = 0; j < pulseLen; ++j) {
            pulseGroup[i + j] = pulse[j];
        }
    }

    if (start == -1 && finish == -1) {
        if (winLen % 2 != 0) {
            start = group_idx - std::ceil(winLen / 2.0);
        } else {
            start = group_idx - winLen / 2;
        }
        finish = group_idx + std::floor(winLen / 2.0);
    } else {
        start = group_idx - start;
        finish = group_idx + finish;
    }

    if (finish > pulseGroupLen || start < 0) {
        return gsl::vector(); // Return empty vector if start or finish indices are out of range
    } else {
        gsl::vector LFgroup = pulseGroup.subvector(start, finish - start + 1); // Extract the desired segment of pulseGroup
        return LFgroup;
    }
}




//double computeCorrelation(const gsl::vector& vector1, const gsl::vector& vector2) {
//    size_t size = vector1.size();
//    double mean1 = gsl_stats_mean(&vector1[0], 1, size);
//    double mean2 = gsl_stats_mean(&vector2[0], 1, size);
//    double stdDev1 = gsl_stats_sd(&vector1[0], 1, size);
//    double stdDev2 = gsl_stats_sd(&vector2[0], 1, size);
//
//    double correlation = 0.0;
//    for (size_t i = 0; i < size; ++i) {
//        double deviation1 = vector1[i] - mean1;
//        double deviation2 = vector2[i] - mean2;
//        correlation += (deviation1 / stdDev1) * (deviation2 / stdDev2);
//    }
//    correlation /= size;
//
//    return correlation;
//}
//
//


double computeCorrelation(const gsl::vector& X, const gsl::vector& Y)
{
    double sum_X = 0.0, sum_Y = 0.0, sum_XY = 0.0;
    double squareSum_X = 0.0, squareSum_Y = 0.0;
    int n = X.size();

    for (int i = 0; i < n; i++)
    {
        // Sum of elements of vector X.
        sum_X += X[i];

        // Sum of elements of vector Y.
        sum_Y += Y[i];

        // Sum of X[i] * Y[i].
        sum_XY += X[i] * Y[i];

        // Sum of squares of vector elements.
        squareSum_X += X[i] * X[i];
        squareSum_Y += Y[i] * Y[i];
    }

    // Use the formula for calculating the correlation coefficient.
    double corr = (n * sum_XY - sum_X * sum_Y)
                  / sqrt((n * squareSum_X - sum_X * sum_X)
                         * (n * squareSum_Y - sum_Y * sum_Y));

    return corr;
}


size_t nextPowerOf2(size_t n) {
    size_t power = 1;
    while (power < n) {
        power <<= 1;
    }
    return power;
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
   if(GetGci(params, data.signal, data.source_signal_iaif, data.fundf, &(data.gci_inds)) == EXIT_FAILURE)
      return EXIT_FAILURE;

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

   /* Write analyzed features to files */
   data.SaveData(params);

   /* Finish */
//   std::cout << "Finished analysis." << std::endl << std::endl;


    LfData lf_data;

    double time_wgt = 0.1;
    double freq_wgt = 0.3;
    double trans_wgt = 0.3;

    // EE=zeros(1,length(GCI));
    lf_data.EE.resize(data.gci_inds.size());
    lf_data.EE.set_zero();


    // Rd_set=[0.3:0.17:2];
    //    double start = 0.3;
    //    double step = 0.17;
    //    double end = 2.0;
    int size = static_cast<int>((2.0 - 0.3) / 0.17) + 2;

    lf_data.Rd_set.resize(size);
    for (int i = 0; i < size; i++) {
        double value = 0.3 + i * 0.17;
        lf_data.Rd_set[i] = value;
    }

    // pulseNum=2;
    double pulseNum = 2;

    // Dynamic programming settings
    // nframe=length(GCI);
    double nframe = data.gci_inds.size();


    // ncands = 5; Number of candidate LF model configurations to consider
    double ncands = 5;

    // Rd_n=zeros(nframe,ncands);
    lf_data.Rd_n = gsl::matrix(nframe, ncands);
    // cost=zeros(nframe,ncands);      % cumulative cost
    lf_data.cost = gsl::matrix(nframe, ncands);
    // prev=zeros(nframe,ncands);      % traceback pointer
    lf_data.prev = gsl::matrix(nframe, ncands);


    // Do processing - exhaustive search and dynamic programming

    // for n=1:length(GCI)
    for (int n = 0; n < data.gci_inds.size(); ++n) {
        double pulseLen;
        std::cout << "n " << n  << std::endl;

        if (n == 0)
        {
            pulseLen = round((data.gci_inds[n + 1] - data.gci_inds[n]) * pulseNum);
            lf_data.F0_cur = params.fs / (round(data.gci_inds[n + 1] - data.gci_inds[n]));
        }
        else
        {
            pulseLen = round((data.gci_inds[n] - data.gci_inds[n - 1]) * pulseNum);
            lf_data.F0_cur = params.fs / (round(data.gci_inds[n] - data.gci_inds[n - 1]));
        }

        // pulseLen=abs(pulseLen);
        pulseLen = std::abs(pulseLen);

//        if GCI(n)-round(pulseLen/2) > 0
//            start=GCI(n)-round(pulseLen/2);
//        else start=1;
//        end

        int start;
        int finish;

        if (data.gci_inds[n] - round(pulseLen / 2) > 0) {
            start = data.gci_inds[n] - round(pulseLen / 2);
        } else {
            start = 1;
        }
//        if GCI(n)+round(pulseLen/2) <= length(glot)
//        finish = GCI(n)+round(pulseLen/2);
//        else finish = length(glot);
//        end
        if (data.gci_inds[n] + round(pulseLen / 2) <= data.source_signal.size())
        {
            finish = data.gci_inds[n] + round(pulseLen / 2);
        }
        else
        {
            finish = data.source_signal.size();
        }

//        glot_seg=glot(start:finish);
//        glot_seg=glot_seg(:);
        int segment_length = finish - start + 1;


        lf_data.glot_seg.resize(segment_length);

        for (int i = start; i <= finish; ++i)
        {
            lf_data.glot_seg[i - start] = data.source_signal[i];
        }
//        std::cout << "lf_data.glot_seg" << lf_data.glot_seg  << std::endl;
//
//        glot_seg_spec=20*log10(abs(fft(glot_seg)));
        size_t fft_len = lf_data.glot_seg.size();
        ComplexVector glot_seg_spec(fft_len);
        // Perform FFT on glot_seg
        FFTRadix2(lf_data.glot_seg, fft_len, &glot_seg_spec);

        lf_data.glot_seg_spec = glot_seg_spec.getAbs();

        for (size_t i = 0; i < lf_data.glot_seg_spec.size(); i++) {
            lf_data.glot_seg_spec(i) = 20 * log10(lf_data.glot_seg_spec(i));
//                std::cout << lf_data.glot_seg_fft_mag(i) << std::endl;
//                lf_data.glot_seg_log_mag(i) = val;  // Min log-power = -60dB
        }

        //   freq=linspace(0,fs,length(glot_seg));
        lf_data.freq.resize(lf_data.glot_seg.size());
//            double start = 0.0;
//            double stop = params.fs;
//            double step = (stop - start) / (lf_data.glot_seg.size() - 1);

        for (size_t i = 0; i < lf_data.glot_seg.size(); i++) {
            lf_data.freq(i) = 0.0 + i * ((params.fs - 0.0) / (lf_data.glot_seg.size() - 1));
        }

        // err_mat=zeros(1,length(Rd_set));
        lf_data.err_mat.resize(lf_data.Rd_set.size());
        lf_data.err_mat.set_zero();

        // err_mat_time=zeros(1,length(Rd_set));
        lf_data.err_mat_time.resize(lf_data.Rd_set.size());
        lf_data.err_mat_time.set_zero();

        // EE(n)=abs(min(glot_seg));
        double min_value = lf_data.glot_seg.min();

        double abs_min_value = std::abs(min_value);
        lf_data.EE(n) = abs_min_value;

// Randomly generate the length for lf_data.pulse


        // for m=1:length(Rd_set)
        for (int m = 0; m < lf_data.Rd_set.size(); ++m) {

            Rd2R(lf_data.Rd_set(m), lf_data.EE(n), lf_data.F0_cur, lf_data.Ra_cur, lf_data.Rk_cur, lf_data.Rg_cur);



//            lf_data.pulse = lf_cont(lf_data.F0_cur, params.fs, lf_data.Ra_cur, lf_data.Rk_cur, lf_data.Rg_cur, lf_data.EE(n));


                const double F0min = 20.0;
                const double F0max = 500.0;

                double T0 = 1.0 / lf_data.F0_cur;
                double Ta = lf_data.Ra_cur * T0;
                double Te = ((1.0 + lf_data.Rk_cur) / (2.0 * lf_data.Rg_cur)) * T0;
                double Tp = Te / (lf_data.Rk_cur + 1.0);
                double Tb = ((1.0 - (lf_data.Rk_cur + 1.0) / (2.0 * lf_data.Rg_cur)) * 1.0 / lf_data.F0_cur);
                double Tc = Tb + Te;

                if (lf_data.F0_cur < F0min || lf_data.F0_cur > F0max) {
                    break; // Return empty vector for invalid F0
                } else {
                    // Solve area balance using Newton-Raphson method
                    double alpha, epsi;

                    lfSource(alpha, epsi, Tc, params.fs, Tp, Te, Ta, lf_data.EE(n));

                    double omega = M_PI / Tp;
                    double E0 = -(std::abs(lf_data.EE(n))) / (std::exp(alpha * Te) * std::sin(omega * Te));

                    // Generate open phase and closed phase and combine
                    double dt = 1.0 / params.fs;

                    size_t T1_size = static_cast<size_t>(std::round(Te / dt));
                    size_t T2_size = static_cast<size_t>(std::round((Tc - Te) / dt));

                    // Ensure T1_size and T2_size are positive
                    T1_size = std::max(T1_size, static_cast<size_t>(1));
                    T2_size = std::max(T2_size, static_cast<size_t>(1));

                    gsl::vector T1(T1_size);
                    gsl::vector T2(T2_size);

                    for (size_t i = 0; i < T1_size; i++) {
                        double t = dt * i;
                        T1[i] = E0 * std::exp(alpha * t) * std::sin(omega * t);
                    }

                    for (size_t i = 0; i < T2_size; i++) {
                        double t = (T1_size * dt) + dt * i;
                        T2[i] = (-lf_data.EE(n) / (epsi * Ta)) * (std::exp(-epsi * (t - Te)) - std::exp(-epsi * Tb));
                    }

                    gsl::vector g_LF(T1_size + T2_size);
                    for (size_t i = 0; i < T1_size; i++) {
                        g_LF[i] = T1[i];
                    }

                    for (size_t i = 0; i < T2_size; i++) {
                        g_LF[T1_size + i] = T2[i];
                    }
                    lf_data.pulse = g_LF;

                }


                lf_data.LFgroup = makePulseCentGCI(lf_data.pulse, pulseLen, data.gci_inds(n)-start, finish-data.gci_inds(n));



                //  glot_seg_spec=20*log10(abs(fft(glot_seg)));
                size_t fft_len = lf_data.LFgroup.size() ;

                ComplexVector LFgroup_win_spec(fft_len);
                // Perform FFT on glot_seg
                FFTRadix2(lf_data.LFgroup, fft_len, &LFgroup_win_spec);
                lf_data.LFgroup_win_spec = LFgroup_win_spec.getAbs();

                for (size_t i = 0; i < lf_data.LFgroup_win_spec.size(); i++) {
                    lf_data.LFgroup_win_spec(i) = 20 * log10(lf_data.LFgroup_win_spec(i));
                }


                lf_data.LFgroup_win = lf_data.LFgroup;
    //            double cor_time = gsl_stats_correlation(&lf_data.glot_seg[0], 1, &lf_data.LFgroup_win[0], 1, lf_data.glot_seg.size());
                lf_data.cor_time = computeCorrelation(lf_data.glot_seg, lf_data.LFgroup_win);
                lf_data.cor_time = std::abs(lf_data.cor_time);
                lf_data.err_time = 1 - lf_data.cor_time;

                lf_data.err_mat_time[m] = lf_data.err_time;

                //            % Frequency domain error function
                //            cor_freq = corrcoef(glot_seg_spec(freq<MVF),LFgroup_win_spec(freq<MVF));
                //            cor_freq=abs(cor_freq(2));
                //            err_freq=1-cor_freq;


                lf_data.cor_freq = computeCorrelation(lf_data.glot_seg_spec, lf_data.LFgroup_win_spec);
                lf_data.cor_freq = std::abs(lf_data.cor_freq);
                lf_data.err_freq = 1 - lf_data.cor_freq;


                //            % Combined error with weights
                //            err_mat(m)=(err_time*time_wgt)+(err_freq*freq_wgt);

                // Combined error with weights
                lf_data.err_mat[m] = (lf_data.err_time * time_wgt) + (lf_data.err_freq * freq_wgt);

                // Copy err_mat to a new vector for sorting
                lf_data.err_mat_sort = lf_data.err_mat;

            }

            //        % Find best ncands (local costs and Rd values)
            //        [err_mat_sort,err_mat_sortIdx]=sort(err_mat);

            std::vector<double> err_mat_sort_std(lf_data.err_mat_sort.size());  // Create a std::vector to store sorted elements
            // Copy elements from gsl::vector to std::vector
            for (size_t i = 0; i < lf_data.err_mat_sort.size(); ++i) {
                err_mat_sort_std[i] = lf_data.err_mat_sort[i];
            }

            // Sort std::vector in ascending order
            std::sort(err_mat_sort_std.begin(), err_mat_sort_std.end());

            // Copy sorted elements back to gsl::vector
            for (size_t i = 0; i < lf_data.err_mat_sort.size(); ++i) {
                lf_data.err_mat_sort[i] = err_mat_sort_std[i];
            }


            lf_data.err_mat_sortIdx.resize(lf_data.err_mat_sort.size());
            // Obtain the sorted indices
            for (size_t i = 0; i < lf_data.err_mat_sort.size(); ++i) {
                for (size_t j = 0; j < lf_data.err_mat_sort.size(); ++j) {
                    if (lf_data.err_mat_sort[i] == lf_data.err_mat[j]) {
                        lf_data.err_mat_sortIdx[i] = j;
                        break;
                    }
                }
            }


            //        Rd_n(n,1:ncands)=Rd_set(err_mat_sortIdx(1:ncands));
            lf_data.Rd_set_err_mat_sortIdx = lf_data.err_mat_sortIdx.subvector(1, ncands);

            lf_data.Rd_set_err_mat_sortVal.resize(lf_data.Rd_set_err_mat_sortIdx.size());

            for (size_t i = 0; i < ncands; i++)
            {
                int index = lf_data.Rd_set_err_mat_sortIdx[i];
                lf_data.Rd_set_err_mat_sortVal(i) = lf_data.Rd_set[index];
                lf_data.Rd_n(n, i) = lf_data.Rd_set_err_mat_sortVal(i);
            }

            // exh_err_n=err_mat_sort(1:ncands);
            lf_data.exh_err_n = lf_data.err_mat_sort.subvector(1, ncands);


            for (size_t i = 0; i < ncands; i++)
            {
                lf_data.cost(n, i) = lf_data.exh_err_n(i);
            }




//            std::cout << "lf_data.exh_err_n"<< lf_data.LFgroup_win_spec.size() << std::endl;
            std::cout << "lf_data.exh_err_n"<< lf_data.err_mat_sort.size() << std::endl;



//        for (int c = 0; c < lf_data.Rd_set_err_mat_sortIdx.size(); c++)
//        {
//            int index = lf_data.err_mat_sortIdx[c];
//
//            std::cout << "sss"<<data.gci_inds.size() << std::endl;
//
//        }



//        for (int c = 0; c < ncands; c++)
//        {
//            int index = lf_data.err_mat_sortIdx[c];
//
//            std::cout << "sss"<<data.gci_inds.size() << std::endl;
//
//        }





//                gsl::vector_subvector_view clippedView = gsl::vector_subvector(lf_data.err_mat_sortIdx, 1, 5);
//
//        std::cout << lf_data.Rd_set_err_mat_sortIdx << std::endl;
//
//                lf_data.Rd_set_err_mat_sortIdx = lf_data.err_mat_sortIdx[1:ncands]
//         Nested loops to assign values from Rd_set to lf_data.Rd_n
//            lf_data.Rd_n(n, c) = lf_data.Rd_set[index];


//            lf_data.Rd_n.row(n).set_all(lf_data.Rd_set[index]);









    }











    //   Glottal Source signal is: "data.source_signal" >> glot
    //   GCI signal is: data.gci_inds  >> GCI

    //    std::vector<double> EE(data.gci_inds.size(), 0.0);
//    std::vector<double> Rd_set;
//    double start = 0.3;
//    double step = 0.17;
//    double end = 2.0;
//
//    for (double value = start; value <= end; value += step) {
//        Rd_set.push_back(value);
//    }
//    double pulseNum = 2;
//
//    // Dynamic programming settings
//    double nframe = data.gci_inds.size();
//    double ncands = 5;
//
//
//    std::vector<std::vector<double>> Rd_n(nframe, std::vector<double>(ncands, 0.0));
//    std::vector<std::vector<double>> cost(nframe, std::vector<double>(ncands, 0.0));
//    std::vector<std::vector<double>> prev(nframe, std::vector<int>(ncands, 0));





    return EXIT_SUCCESS;

}

/***********/
/*   EOF   */
/***********/

