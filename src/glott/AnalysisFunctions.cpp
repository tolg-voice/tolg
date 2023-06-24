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

#include <gslwrap/vector_double.h>
#include <gslwrap/vector_int.h>
#include <cmath>
#include <cfloat>

#include "definitions.h"
#include "SpFunctions.h"
#include "FileIo.h"
#include "InverseFiltering.h"
#include "PitchEstimation.h"
#include "AnalysisFunctions.h"
#include "Utils.h"
#include "Filters.h"
#include "QmfFunctions.h"



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

//#include "medfilt1.h"

// Add the Reaper
//#include "core/file_resource.h"
//#include "core/track.h"
//#include "epoch_tracker/epoch_tracker.h"
//#include "wave/wave.h"

int PULSE_NOT_FOUND = -1;












int PolarityDetection(const Param &params, gsl::vector *signal,
                      gsl::vector *source_signal_iaif) {
    switch (params.signal_polarity) {
        case POLARITY_DEFAULT:
            return EXIT_SUCCESS;

        case POLARITY_INVERT:
            std::cout << " -- Inverting polarity (SIGNAL_POLARITY = \"INVERT\")"
                      << std::endl;
            (*signal) *= (double)-1.0;
            return EXIT_SUCCESS;

        case POLARITY_DETECT:
            std::cout << "Using automatic polarity detection ...";

            if (Skewness(*source_signal_iaif) > 0) {
                std::cout << "... Detected negative polarity. Inverting signal."
                          << std::endl;
                (*signal) *= (double)-1.0;
                (*source_signal_iaif) *= (double)-1.0;
            } else {
                std::cout << "... Detected positive polarity." << std::endl;
            }
            return EXIT_SUCCESS;
    }
    return EXIT_FAILURE;
}

/**
 * Get the F0 vector if the analyzed signal.
 * input: params, signal
 * output: fundf: Obtained F0 vector.
 *
 */
int GetF0(const Param &params, const gsl::vector &signal,
          const gsl::vector &source_signal_iaif, gsl::vector *fundf) {
    std::cout << "F0 analysis ";

    if (params.use_external_f0) {
        std::cout << "using external F0 file: " << params.external_f0_filename
                  << " ...";
        gsl::vector fundf_ext;
        if (ReadGslVector(params.external_f0_filename.c_str(), params.data_type,
                          &fundf_ext) == EXIT_FAILURE)
            return EXIT_FAILURE;

        if (fundf_ext.size() != (size_t)params.number_of_frames) {
            std::cout << "Warning: External F0 file length differs from number of "
                         "frames. Interpolating external "
                         "F0 length to match number of frames.  External F0 length: "
                      << fundf_ext.size()
                      << ", Number of frames: " << params.number_of_frames
                      << std::endl;
            InterpolateNearest(fundf_ext, params.number_of_frames, fundf);
        } else {
            fundf->copy(fundf_ext);
        }
    } else {
        *fundf = gsl::vector(params.number_of_frames);
        gsl::vector signal_frame = gsl::vector(params.frame_length);
        // gsl::vector glottal_frame = gsl::vector(2*params.frame_length); // Longer
        // frame
        gsl::vector glottal_frame =
                gsl::vector(params.frame_length_long);  // Longer frame
        int frame_index;
        double ff;
        gsl::matrix fundf_candidates(params.number_of_frames,
                                     NUMBER_OF_F0_CANDIDATES);
        gsl::vector candidates_vec(NUMBER_OF_F0_CANDIDATES);
        for (frame_index = 0; frame_index < params.number_of_frames;
             frame_index++) {
            GetFrame(signal, frame_index, params.frame_shift, &signal_frame, NULL);
            GetFrame(source_signal_iaif, frame_index, params.frame_shift,
                     &glottal_frame, NULL);

            FundamentalFrequency(params, glottal_frame, signal_frame, &ff,
                                 &candidates_vec);
            (*fundf)(frame_index) = ff;

            fundf_candidates.set_row_vec(frame_index, candidates_vec);
        }

        /* Copy original F0 */
        gsl::vector fundf_orig(*fundf);

        /* Process */
        MedianFilter(3, fundf);
        FillF0Gaps(fundf);
        FundfPostProcessing(params, fundf_orig, fundf_candidates, fundf);
        MedianFilter(3, fundf);
        FillF0Gaps(fundf);
        FundfPostProcessing(params, fundf_orig, fundf_candidates, fundf);
        MedianFilter(3, fundf);
    }
    std::cout << " done." << std::endl;
    return EXIT_SUCCESS;
}

/**
 * Get the glottal closure instants (GCIs) of the analyzed signal.
 * input: params, signal
 * output: gci_signal: Sparse signal-length representation of gcis as ones and otherwise zeros
 * gci_inds I think so
 */



// This code should be modified for the later days cuz the low efficicency, which read the RemoveDuplicatedGciIndices
// This is very low efficiency!!!!!!!!!! TODO
void RemoveDuplicateGciIndices(gsl::vector_int *gci_inds) {
    gsl::vector_int temp(gci_inds->size());
    size_t temp_index = 0;

    for(size_t i = 0; i < gci_inds->size(); ++i) {
        bool is_duplicate = false;
        for (size_t j = 0; j < temp_index; ++j) {
            if ((*gci_inds)(i) == temp(j)) {
                is_duplicate = true;
                break;
            }
        }

        if (!is_duplicate) {
            temp(temp_index) = (*gci_inds)(i);
            temp_index++;
        }
    }

    // Resize gci_inds to the correct size
    *gci_inds = gsl::vector_int(temp_index);

    // Copy unique values from temp to gci_inds
    for(size_t i = 0; i < temp_index; ++i) {
        (*gci_inds)(i) = temp(i);
    }
}



int GetGci(const Param &params, const gsl::vector &signal, const gsl::vector &source_signal_iaif, const gsl::vector &fundf, gsl::vector_int *gci_inds) {
    if(params.use_external_gci) {
        std::cout << "Reading GCI information from external file: " << params.external_gci_filename << " ...";
        gsl::vector gcis;
        if(ReadGslVector(params.external_gci_filename.c_str(), params.data_type, &gcis) == EXIT_FAILURE)
            return EXIT_FAILURE;
        *gci_inds = gsl::vector_int(gcis.size());
        size_t i;
        for (i=0; i<gci_inds->size();i++) {
            (*gci_inds)(i) = (int)round(gcis(i) * params.fs);
        }
    } else {
        std::cout << "GCI estimation using the SEDREAMS algorithm ...";

        gsl::vector mean_based_signal(signal.size(),true);

        MeanBasedSignal(signal, params.fs, getMeanF0(fundf), &mean_based_signal);

//      MovingAverageFilter(3,&mean_based_signal); // remove small fluctuation

        SedreamsGciDetection(source_signal_iaif,mean_based_signal,gci_inds);

    }

    RemoveDuplicateGciIndices(gci_inds);

    std::cout << " done." << std::endl;
    return EXIT_SUCCESS;
}

int GetGain(const Param &params, const gsl::vector &fundf,
            const gsl::vector &signal, gsl::vector *gain_ptr) {
    // double E_REF = 0.00001;
    gsl::vector frame = gsl::vector(params.frame_length);
    gsl::vector unvoiced_frame = gsl::vector(params.frame_length_unvoiced);
    gsl::vector gain = gsl::vector(params.number_of_frames);
    ComplexVector frame_fft;
    size_t NFFT = 4096;  // Long FFT
    double MIN_LOG_POWER = -100.0;
    gsl::vector fft_mag(NFFT / 2 + 1);
    // int min_uv_frequency = rint((double)NFFT/(double)(params.fs)*000.0);

    int frame_index;

    frame.set_all(1.0);
    ApplyWindowingFunction(params.default_windowing_function, &frame);
    double frame_energy_compensation = sqrt(frame.size() / getSquareSum(frame));
    double frame_energy;
    bool frame_is_voiced;
    for (frame_index = 0; frame_index < params.number_of_frames; frame_index++) {
        frame_is_voiced = fundf(frame_index) > 0.0;
        if (frame_is_voiced) {
            GetFrame(signal, frame_index, params.frame_shift, &frame, NULL);
            ApplyWindowingFunction(params.default_windowing_function, &frame);
            frame_energy = getEnergy(frame);
            if (frame_energy == 0.0) frame_energy = +DBL_MIN;

            frame_energy *= frame_energy_compensation;  //(8.0/3.0);// Compensate
            // windowing gain loss
            gain(frame_index) = FrameEnergy2LogEnergy(frame_energy, frame.size());

        } else {
            GetFrame(signal, frame_index, params.frame_shift, &unvoiced_frame, NULL);
            ApplyWindowingFunction(params.default_windowing_function,
                                   &unvoiced_frame);
            frame_energy = getEnergy(unvoiced_frame);

            if (frame_energy == 0.0) frame_energy = +DBL_MIN;

            frame_energy *=
                    frame_energy_compensation;  // Compensate windowing gain loss
            gain(frame_index) =
                    FrameEnergy2LogEnergy(frame_energy, unvoiced_frame.size());
        }

        /* Clip gain at lower bound (prevent very low values for zero frames) */
        if (gain(frame_index) < MIN_LOG_POWER) gain(frame_index) = MIN_LOG_POWER;
    }
    *gain_ptr = gain;
    return EXIT_SUCCESS;
}

/**
 *
 *
 */
int SpectralAnalysis(const Param &params, const AnalysisData &data,
                     gsl::matrix *poly_vocal_tract) {
    gsl::vector frame(params.frame_length);
    gsl::vector unvoiced_frame(params.frame_length_unvoiced, true);
    gsl::vector pre_frame(params.lpc_order_vt * 2, true);
    gsl::vector lp_weight(params.frame_length + params.lpc_order_vt * 3, true);
    gsl::vector A(params.lpc_order_vt + 1, true);
    gsl::vector G(params.lpc_order_glot_iaif, true);
    gsl::vector B(1);
    B(0) = 1.0;
    // gsl::vector lip_radiation(2);lip_radiation(0) = 1.0; lip_radiation(1) =
    // 0.99;
    gsl::vector frame_pre_emph(params.frame_length);
    gsl::vector frame_full;  // frame + preframe
    gsl::vector residual(params.frame_length);

    if (params.use_external_lsf_vt == false) {
        std::cout << "Spectral analysis ...";
        /* Do analysis frame-wise */
        size_t frame_index;
        for (frame_index = 0; frame_index < (size_t)params.number_of_frames;
             frame_index++) {
            // GetPitchSynchFrame(data.signal, frame_index, params.frame_shift,
            // &frame, &pre_frame);
            /** Voiced analysis **/
            if (data.fundf(frame_index) != 0) {
                if (params.use_pitch_synchronous_analysis)
                    GetPitchSynchFrame(params, data.signal, data.gci_inds, frame_index,
                                       params.frame_shift, data.fundf(frame_index),
                                       &frame, &pre_frame);
                else
                    GetFrame(data.signal, frame_index, params.frame_shift, &frame,
                             &pre_frame);

                /* Estimate Weighted Linear Prediction weight */
                GetLpWeight(params, params.lp_weighting_function, data.gci_inds, frame,
                            frame_index, &lp_weight);
                /* Pre-emphasis and windowing */
                Filter(std::vector<double>{1.0, -params.gif_pre_emphasis_coefficient},
                       B, frame, &frame_pre_emph);
                ApplyWindowingFunction(params.default_windowing_function,
                                       &frame_pre_emph);
                /* First-loop envelope */
                ArAnalysis(params.lpc_order_vt, params.warping_lambda_vt,
                           params.lp_weighting_function, lp_weight, frame_pre_emph, &A);
                /* Second-loop envelope (if IAIF is used) */

                if (params.use_iterative_gif) {
                    ConcatenateFrames(pre_frame, frame, &frame_full);
                    if (params.warping_lambda_vt != 0.0) {
                        Filter(A, B, frame_full, &residual);
                    } else {
                        WFilter(A, B, frame_full, params.warping_lambda_vt, &residual);
                    }
                    ApplyWindowingFunction(params.default_windowing_function, &residual);
                    ArAnalysis(params.lpc_order_glot_iaif, 0.0, NONE, lp_weight, residual,
                               &G);
                    Filter(G, B, frame, &frame_pre_emph);  // Iterated pre-emphasis
                    ApplyWindowingFunction(params.default_windowing_function,
                                           &frame_pre_emph);
                    ArAnalysis(params.lpc_order_vt, params.warping_lambda_vt,
                               params.lp_weighting_function, lp_weight, frame_pre_emph,
                               &A);
                }
                /** Unvoiced analysis **/
            } else {
                GetFrame(data.signal, frame_index, params.frame_shift, &unvoiced_frame,
                         &pre_frame);
                if (params.unvoiced_pre_emphasis_coefficient > 0.0) {
                    Filter(
                            std::vector<double>{
                                    1.0, -1.0 * params.unvoiced_pre_emphasis_coefficient},
                            std::vector<double>{1.0}, unvoiced_frame, &unvoiced_frame);
                }
                ApplyWindowingFunction(params.default_windowing_function,
                                       &unvoiced_frame);
                ArAnalysis(params.lpc_order_vt, params.warping_lambda_vt, NONE,
                           lp_weight, unvoiced_frame, &A);
            }
            poly_vocal_tract->set_col_vec(frame_index, A);
        }
    } else {
        std::cout << "Using external vocal tract LSFs ... ";
        /* Read external vocal tract filter LSFs*/
        gsl::matrix external_lsf;
        ReadGslMatrix(params.external_lsf_vt_filename, params.data_type,
                      params.lpc_order_vt, &external_lsf);
        if (external_lsf.size2() < poly_vocal_tract->size2()) {
            std::cerr << "Warning: external LSF is missing "
                      << poly_vocal_tract->size2() - external_lsf.size2()
                      << " frames, zero-padding at the end" << std::endl;
        }
        gsl::vector a(params.lpc_order_vt + 1);
        for (size_t i = 0; i < poly_vocal_tract->size2(); i++) {
            if (i < external_lsf.size2()) {
                /* Convert external lsf to filter polynomial */
                Lsf2Poly(external_lsf.get_col_vec(i), &a);
            } else {
                /* Pad missing frames with a flat filter */
                a.set_all(0.0);
                a(0) = 1.0;
            }
            poly_vocal_tract->set_col_vec(i, a);
        }
    }

    std::cout << " done." << std::endl;
    return EXIT_SUCCESS;
}

int SpectralAnalysisQmf(const Param &params, const AnalysisData &data,
                        gsl::matrix *poly_vocal_tract) {
    gsl::vector frame(params.frame_length);
    gsl::vector frame_pre_emph(params.frame_length);
    gsl::vector pre_frame(params.lpc_order_vt, true);
    gsl::vector frame_qmf1(frame.size() / 2);  // Downsampled low-band frame
    gsl::vector frame_qmf2(frame.size() / 2);  // Downsampled high-band frame
    gsl::vector lp_weight_downsampled(frame_qmf1.size() +
                                      params.lpc_order_vt_qmf1);
    gsl::vector B(1);
    B(0) = 1.0;

    gsl::vector H0 =
            StdVector2GslVector(kCUTOFF05PI);  // Load hard-coded low-pass filter
    gsl::vector H1 = Qmf::GetMatchingFilter(H0);

    gsl::vector lp_weight(params.frame_length + params.lpc_order_vt, true);
    gsl::vector A(params.lpc_order_vt + 1, true);
    gsl::vector A_qmf1(params.lpc_order_vt_qmf1 + 1, true);
    gsl::vector A_qmf2(params.lpc_order_vt_qmf2 + 1, true);
    // gsl::vector lsf_qmf1(params.lpc_order_vt_qmf1,true);
    // gsl::vector lsf_qmf2(params.lpc_order_vt_qmf2,true);
    // gsl::vector gain_qmf(params.number_of_frames);
    double gain_qmf, e1, e2;

    gsl::vector lip_radiation(2);
    lip_radiation(0) = 1.0;
    lip_radiation(1) = -params.gif_pre_emphasis_coefficient;

    // gsl::vector frame_full; // frame + preframe
    // gsl::vector residual_full; // residual with preframe

    std::cout << "QMF sub-band-based spectral analysis ...";

    size_t frame_index;
    for (frame_index = 0; frame_index < (size_t)params.number_of_frames;
         frame_index++) {
        GetFrame(data.signal, frame_index, params.frame_shift, &frame, &pre_frame);

        /** Voiced analysis (Low-band = QCP, High-band = LPC) **/
        if (data.fundf(frame_index) != 0) {
            /* Pre-emphasis */
            Filter(lip_radiation, B, frame, &frame_pre_emph);
            Qmf::GetSubBands(frame_pre_emph, H0, H1, &frame_qmf1, &frame_qmf2);
            /* Gain differences between frame_qmf1 and frame_qmf2: */

            e1 = getEnergy(frame_qmf1);
            e2 = getEnergy(frame_qmf2);
            if (e1 == 0.0) e1 += DBL_MIN;
            if (e2 == 0.0) e2 += DBL_MIN;
            gain_qmf = 20 * log10(e2 / e1);

            /** Low-band analysis **/
            GetLpWeight(params, params.lp_weighting_function, data.gci_inds, frame,
                        frame_index, &lp_weight);
            Qmf::Decimate(lp_weight, 2, &lp_weight_downsampled);

            ApplyWindowingFunction(params.default_windowing_function, &frame_qmf1);
            ArAnalysis(params.lpc_order_vt_qmf1, 0.0, params.lp_weighting_function,
                       lp_weight_downsampled, frame_qmf1, &A_qmf1);

            /** High-band analysis **/
            // ApplyWindowingFunction(params.default_windowing_function,&frame_qmf2);
            ArAnalysis(params.lpc_order_vt_qmf2, 0.0, NONE, lp_weight_downsampled,
                       frame_qmf2, &A_qmf2);

            Qmf::CombinePoly(A_qmf1, A_qmf2, gain_qmf, (int)frame_qmf1.size(), &A);
            /** Unvoiced analysis (Low-band = LPC, High-band = LPC, no pre-emphasis)
             * **/
        } else {
            // Qmf::GetSubBands(frame, H0, H1, &frame_qmf1, &frame_qmf2);

            // e1 = getEnergy(frame_qmf1);
            // e2 = getEnergy(frame_qmf2);
            // if(e1 == 0.0)
            //   e1 += DBL_MIN;
            // if(e2 == 0.0)
            //   e2 += DBL_MIN;
            // gain_qmf = 20*log10(e2/e1);

            /** Low-band analysis **/
            // ApplyWindowingFunction(params.default_windowing_function,&frame_qmf1);
            // ArAnalysis(params.lpc_order_vt_qmf1,0.0,NONE, lp_weight_downsampled,
            // frame_qmf2, &A_qmf1);

            /** High-band analysis **/
            // ApplyWindowingFunction(params.default_windowing_function,&frame_qmf2);
            // ArAnalysis(params.lpc_order_vt_qmf2,0.0,NONE, lp_weight_downsampled,
            // frame_qmf2, &A_qmf2);
            ApplyWindowingFunction(params.default_windowing_function, &frame);
            ArAnalysis(params.lpc_order_vt, 0.0, NONE, lp_weight_downsampled, frame,
                       &A);
        }

        poly_vocal_tract->set_col_vec(frame_index, A);
        // Poly2Lsf(A_qmf1,&lsf_qmf1);
        // Poly2Lsf(A_qmf2,&lsf_qmf2);
        // lsf_qmf1->set_col_vec(frame_index,lsf_qmf1);
        // lsf_qmf2->set_col_vec(frame_index,lsf_qmf2);
    }
    return EXIT_SUCCESS;
}

int InverseFilter(const Param &params, const AnalysisData &data,
                  gsl::matrix *poly_glot, gsl::vector *source_signal) {
    size_t frame_index;
    gsl::vector frame(params.frame_length, true);
    gsl::vector pre_frame(2 * params.lpc_order_vt, true);
    gsl::vector frame_full(frame.size() + pre_frame.size());  // Pre-frame + frame
    gsl::vector frame_residual(params.frame_length);
    gsl::vector a_glot(params.lpc_order_glot + 1);
    gsl::vector b(1);
    b(0) = 1.0;

    // for linear frequency scale inverse filtering
    gsl::vector a_lin(params.lpc_order_vt + 1);
    gsl::vector a_lin_high_order(3 * params.lpc_order_vt +
                                 1);  // arbitrary high order
    size_t NFFT = 4096;
    gsl::vector impulse(params.frame_length);
    gsl::vector imp_response(params.frame_length);
    gsl::vector pre_frame_high_order(3 * a_lin_high_order.size());
    gsl::vector frame_full_high_order(frame.size() + pre_frame_high_order.size());

    for (frame_index = 0; frame_index < (size_t)params.number_of_frames;
         frame_index++) {
        if (params.use_pitch_synchronous_analysis) {
            GetPitchSynchFrame(params, data.signal, data.gci_inds, frame_index,
                               params.frame_shift, data.fundf(frame_index), &frame,
                               &pre_frame);
            frame_residual.resize(frame.size());
        } else {
            GetFrame(data.signal, frame_index, params.frame_shift, &frame,
                     &pre_frame);
            GetFrame(data.signal, frame_index, params.frame_shift, &frame,
                     &pre_frame_high_order);
        }

        ConcatenateFrames(pre_frame, frame, &frame_full);
        ConcatenateFrames(pre_frame_high_order, frame, &frame_full_high_order);

        if (params.warping_lambda_vt == 0.0) {
            Filter(data.poly_vocal_tract.get_col_vec(frame_index), b, frame_full,
                   &frame_residual);
        } else {
            gsl::vector a_warp(data.poly_vocal_tract.get_col_vec(frame_index));
            // get warped filter linear frequency response via impulse response
            imp_response.set_zero();
            impulse.set_zero();
            // give pre-frame (only affects phase, not filter fit)
            impulse(a_lin_high_order.size()) = 1.0;
            // get inverse filter impulse response
            WFilter(a_warp, b, impulse, params.warping_lambda_vt, &imp_response);
            // Do high-order LP fit on the inverse filter (FIR polynomial)
            StabilizePoly(NFFT, imp_response, &a_lin_high_order);
            // Linear filtering
            Filter(a_lin_high_order, b, frame_full_high_order, &frame_residual);
        }

        double ola_gain =
                (double)params.frame_length / ((double)params.frame_shift * 2.0);
        // Scale by frame energy, TODO: remove?
        frame_residual *= LogEnergy2FrameEnergy(data.frame_energy(frame_index),
                                                frame_residual.size()) /
                          getEnergy(frame_residual) / ola_gain;
        ApplyWindowingFunction(params.default_windowing_function, &frame_residual);

        LPC(frame_residual, params.lpc_order_glot, &a_glot);
        size_t i;
        for (i = 0; i < a_glot.size(); i++) {
            if (gsl_isnan((a_glot)(i))) {
                (a_glot)(i) = (0.0);
            }
        }
        poly_glot->set_col_vec(frame_index, a_glot);

        OverlapAdd(frame_residual, frame_index * params.frame_shift,
                   source_signal);  // center index = frame_index*params.frame_shift
    }

    return EXIT_SUCCESS;
}

int Find_nearest_pulse_index(const int &sample_index,
                             const gsl::vector &gci_inds, const Param &params,
                             const double &f0) {
    int j;
    // int i,k;
    int pulse_index = -1;  // Return value initialization

    if (!gci_inds.is_set())
        return PULSE_NOT_FOUND;

    int dist, min_dist, ppos;
    min_dist = INT_MAX;
    /* Find the shortest distance between sample index and gcis */
    for (j = 1; j < (int)gci_inds.size() - 1; j++) {
        ppos = gci_inds(j);
        dist = abs(sample_index - ppos);
        if (dist > min_dist) {
            break;
        }
        min_dist = dist;
        pulse_index = j;
    }

    /* Return the closest GCI if unvoiced */
    if (f0 == 0) return pulse_index;

    double pulselen, targetlen;
    targetlen = 2.0 * params.fs / f0;
    pulselen = round(gci_inds(pulse_index + 1) - gci_inds(pulse_index - 1)) + 1;

    int new_pulse_index;
    int prev_index = pulse_index - 1;
    int next_index = pulse_index + 1;
    int prev_gci, next_gci;

    double max_relative_len_diff = params.max_pulse_len_diff;
    double relative_len_diff = (fabs(pulselen - targetlen) / targetlen);

    /* Choose next closest while pulse length deviates too much from f0 */
    while (relative_len_diff > max_relative_len_diff) {
        /* Prevent illegal reads*/
        if (prev_index < 0) prev_index = 0;
        if (next_index > (int)gci_inds.size() - 1) next_index = gci_inds.size() - 1;

        prev_gci = gci_inds(prev_index);
        next_gci = gci_inds(next_index);

        /* choose closest below or above, increment for next iteration */
        if (abs(sample_index - next_gci) < abs(sample_index - prev_gci)) {
            new_pulse_index = next_index;
            next_index++;
        } else {
            new_pulse_index = prev_index;
            prev_index++;
        }

        /* break if out of range */
        if (new_pulse_index - 1 < 0 ||
            new_pulse_index + 1 > (int)gci_inds.size() - 1) {
            break;
        } else {
            pulse_index = new_pulse_index;
        }

        /* if pulse center gets too far from sample index, relax constraint and
         * start over */
        if (fabs(sample_index - gci_inds(pulse_index)) > 1.0 * targetlen) {
            max_relative_len_diff += 0.02;  // increase by 5 percent
            //std::cout << "could not find pulse in F0 range, relaxing constraint" << std::endl;
            if (max_relative_len_diff > 3.0) {
                break;
            }
            pulse_index = j;
            prev_index = pulse_index - 1;
            next_index = pulse_index + 1;
        }

        /* break if out of range */
        if (new_pulse_index - 1 < 0 ||
            new_pulse_index + 1 > (int)gci_inds.size() - 1) {
            break;
        } else {
            pulse_index = new_pulse_index;
        }

        /* calculate new pulse length */
        pulselen = round(gci_inds(pulse_index + 1) - gci_inds(pulse_index - 1)) + 1;

        relative_len_diff = (fabs(pulselen - targetlen) / targetlen);
    }

    if (relative_len_diff > 3.0 || pulselen < 3) {
        return PULSE_NOT_FOUND;
    } else {
        return pulse_index;
    }

}

gsl::vector replace_nan(const gsl::vector& x) {
    gsl::vector replaced_pulse;
    replaced_pulse.resize(x.size());

    for (int32_t i = 0; i < x.size(); ++i) {
        if (std::isnan(x[i])) {
            replaced_pulse[i] = 0.0;
        } else {
            replaced_pulse[i] = x[i];  // Assign the non-NaN value
        }
    }

    return replaced_pulse;
}

void GetPulses(const Param &params, const gsl::vector &source_signal,
               const gsl::vector_int &gci_inds, gsl::vector &fundf,
               gsl::matrix *pulses_mat) {
    if (params.extract_pulses_as_features == false) return;

    std::cout << "Extracting excitation pulses ";

    size_t frame_index;
    for (frame_index = 0; frame_index < (size_t)params.number_of_frames;
         frame_index++) {
        size_t sample_index = frame_index * params.frame_shift;
        int pulse_index = Find_nearest_pulse_index(sample_index, gci_inds,
                                                   params, fundf(frame_index));

        gsl::vector paf_pulse(params.paf_pulse_length, true);
        gsl::vector pulse;

        int center_index;
        /* Use frame center directly for unvoiced */
        if (fundf(frame_index) == 0.0 || pulse_index == PULSE_NOT_FOUND) {
            center_index = sample_index;
        } else {
            center_index = gci_inds(pulse_index);

            /* Check that pulse center index is reasonably
             * close to frame center index
             */
            int THRESH = 100 * params.frame_length;
            if (abs(center_index - (int)sample_index) > THRESH) {
                std::cerr
                        << "Warning: no suitable pulse in range,"
                        << "treating frame as unvoiced"
                        << std::endl;
                std::cerr
                        << "Frame: " << frame_index
                        << ", distance: " << abs(center_index - (int)sample_index)
                        << std::endl;
                center_index = sample_index;
            }
        }

        int i;
        size_t j;

        /* No interpolation, window with selected window */
        if (params.paf_analysis_window != RECT) {
            /* Apply pitch-synchronous analysis window to pulse */

            size_t T;
            if (fundf(frame_index) != 0.0) {
                /* Voiced: use two pitch periods () */
                T = round(2.0 * (double)params.fs / fundf(frame_index));
                if (T > paf_pulse.size()) T = paf_pulse.size();
            } else {
                /* Unvoiced: use all available space */
                T = paf_pulse.size();
            }

            pulse = gsl::vector(T);
            for (j = 0; j < T; j++) {
                i = center_index - round(pulse.size() / 2.0) + j;
                if (i >= 0 && i < (int)source_signal.size())
                    pulse(j) = source_signal(i);
            }
            ApplyWindowingFunction(params.paf_analysis_window, &pulse);

            for (j = 0; j < pulse.size(); j++) {
                paf_pulse(
                        (round(paf_pulse.size() / 2.0) - round(pulse.size() / 2.0)) + j) =
                        pulse(j);
            }
        } else {
            /* params.paf_analysis_window == RECT */
            /* No windowing, just copy to paf_pulse */
            for (j = 0; j < paf_pulse.size(); j++) {
                i = center_index - round(paf_pulse.size() / 2.0) + j;
                if (i >= 0 && i < (int)source_signal.size())
                    paf_pulse(j) = source_signal(i);
            }
        }

        /* Normalize energy */
        if (params.use_paf_energy_normalization) {
            paf_pulse /= getEnergy(paf_pulse);
        }

        paf_pulse = replace_nan(paf_pulse);

//        std::cout << "********************* cost params *********************" << paf_pulse << std::endl;
        /* Save to matrix */
        pulses_mat->set_col_vec(frame_index, paf_pulse);
    }
    std::cout << "done." << std::endl;
}

void HighPassFiltering(const Param &params, gsl::vector *signal) {
    if (!params.use_highpass_filtering) return;

    std::cout
            << "High-pass filtering input signal with a cutoff frequency of 50Hz."
            << std::endl;

    gsl::vector signal_cpy(signal->size());
    signal_cpy.copy(*signal);

    if (params.fs < 40000) {
        Filter(k16HPCUTOFF50HZ, std::vector<double>{1}, signal_cpy, signal);
        signal_cpy.copy(*signal);
        signal_cpy.reverse();
        Filter(k16HPCUTOFF50HZ, std::vector<double>{1}, signal_cpy, signal);
        (*signal).reverse();
    } else {
        Filter(k44HPCUTOFF50HZ, std::vector<double>{1}, signal_cpy, signal);
        signal_cpy.copy(*signal);
        signal_cpy.reverse();
        Filter(k16HPCUTOFF50HZ, std::vector<double>{1}, signal_cpy, signal);
        (*signal).reverse();
    }
}

void GetIaifResidual(const Param &params, const gsl::vector &signal,
                     gsl::vector *residual) {
    gsl::vector frame(params.frame_length, true);
    gsl::vector frame_residual(params.frame_length, true);
    gsl::vector frame_pre_emph(params.frame_length, true);
    gsl::vector pre_frame(params.lpc_order_vt, true);
    gsl::vector frame_full(params.lpc_order_vt + params.frame_length, true);
    gsl::vector A(params.lpc_order_vt + 1, true);
    gsl::vector B(1);
    B(0) = 1.0;
    gsl::vector G(params.lpc_order_glot_iaif + 1, true);
    gsl::vector weight_fn;

    if (!residual->is_set()) *residual = gsl::vector(signal.size());

    size_t frame_index;
    for (frame_index = 0; frame_index < (size_t)params.number_of_frames;
         frame_index++) {
        GetFrame(signal, frame_index, params.frame_shift, &frame, &pre_frame);

        /* Pre-emphasis and windowing */
        Filter(std::vector<double>{1.0, -params.gif_pre_emphasis_coefficient}, B,
               frame, &frame_pre_emph);
        ApplyWindowingFunction(params.default_windowing_function, &frame_pre_emph);

        ArAnalysis(params.lpc_order_vt, 0.0, NONE, weight_fn, frame_pre_emph, &A);
        ConcatenateFrames(pre_frame, frame, &frame_full);

        Filter(A, B, frame_full, &frame_residual);

        ApplyWindowingFunction(params.default_windowing_function, &frame_residual);
        ArAnalysis(params.lpc_order_glot_iaif, 0.0, NONE, weight_fn, frame_residual,
                   &G);

        Filter(G, B, frame, &frame_pre_emph);  // Iterated pre-emphasis
        ApplyWindowingFunction(params.default_windowing_function, &frame_pre_emph);

        ArAnalysis(params.lpc_order_vt, 0.0, NONE, weight_fn, frame_pre_emph, &A);

        Filter(A, B, frame_full, &frame_residual);

        /* Set energy of residual equal to energy of frame */
        double ola_gain =
                (double)params.frame_length / ((double)params.frame_shift * 2.0);
        frame_residual *= getEnergy(frame) / getEnergy(frame_residual) / ola_gain;

        ApplyWindowingFunction(HANN, &frame_residual);

        OverlapAdd(frame_residual, frame_index * params.frame_shift, residual);
    }
}

void HnrAnalysis(const Param &params, const gsl::vector &source_signal,
                 const gsl::vector &fundf, gsl::matrix *hnr_glott) {
    std::cout << "HNR Analysis ...";

    /* Variables */
    int hnr_channels = params.hnr_order;
    gsl::vector frame(params.frame_length_long);
    ComplexVector frame_fft;
    size_t NFFT = 4096;  // Long FFT
    double MIN_LOG_POWER = -60.0;
    gsl::vector fft_mag(NFFT / 2 + 1);

    gsl::vector_int harmonic_index;
    gsl::vector hnr_values;

    gsl::vector harmonic_values;  // experimental, ljuvela
    gsl::vector upper_env_values;
    gsl::vector lower_env_values;
    gsl::vector fft_lower_env(NFFT / 2 + 1, true);
    gsl::vector fft_upper_env(NFFT / 2 + 1);

    double kbd_alpha = 2.3;
    gsl::vector kbd_window =
            getKaiserBesselDerivedWindow(frame.size(), kbd_alpha);

    /* Linearly spaced frequency axis */
    gsl::vector_int x_interp = LinspaceInt(0, 1, fft_mag.size() - 1);

    gsl::vector hnr_interp(fft_mag.size());
    gsl::vector hnr_erb(hnr_channels);

    size_t frame_index, i;
    double val;
    for (frame_index = 0; frame_index < (size_t)params.number_of_frames;
         frame_index++) {
        GetFrame(source_signal, frame_index, params.frame_shift, &frame, NULL);
        // ApplyWindowingFunction(params.default_windowing_function, &frame);
        frame *= kbd_window;
        FFTRadix2(frame, NFFT, &frame_fft);
        fft_mag = frame_fft.getAbs();
        for (i = 0; i < fft_mag.size(); i++) {
            val =
                    20 *
                    log10(fft_mag(i));  // save to temp to prevent evaluation twice in max
            fft_mag(i) = GSL_MAX(val, MIN_LOG_POWER);  // Min log-power = -60dB
        }

        if (fundf(frame_index) > 0) {
            UpperLowerEnvelope(fft_mag, fundf(frame_index), params.fs, &fft_upper_env,
                               &fft_lower_env);
        } else {
            /* Define the upper envelope as the maxima around pseudo-period of 100Hz
             */
            UpperLowerEnvelope(fft_mag, 100.0, params.fs, &fft_upper_env,
                               &fft_lower_env);
        }

        /* HNR as upper-lower envelope difference */
        for (i = 0; i < hnr_interp.size(); i++)
            hnr_interp(i) = fft_lower_env(i) - fft_upper_env(i);

        /* Convert to erb-bands */
        Linear2Erb(hnr_interp, params.fs, &hnr_erb);
        hnr_glott->set_col_vec(frame_index, hnr_erb);
    }
    std::cout << " done." << std::endl;
}

int GetPitchSynchFrame(const Param &params, const gsl::vector &signal,
                       const gsl::vector_int &gci_inds, const int &frame_index,
                       const int &frame_shift, const double &f0,
                       gsl::vector *frame, gsl::vector *pre_frame) {
    int i, ind;
    size_t T0;
    if (f0 == 0.0) T0 = (size_t)frame_shift;
        // T0 = (size_t)params.frame_length;
    else
        T0 = (size_t)rint(params.fs / f0);

    (*frame) = gsl::vector(2 * T0, true);
    int center_index = (int)frame_index * frame_shift;
    int pulse_index =
            (int)Find_nearest_pulse_index(center_index, gci_inds, params, f0);
    if (abs(center_index - pulse_index) <= frame_shift)
        center_index = pulse_index;

    // Get samples to frame
    if (frame != NULL) {
        for (i = 0; i < (int)frame->size(); i++) {
            ind = center_index - ((int)frame->size()) / 2 +
                  i;  // SPTK compatible, ljuvela
            if (ind >= 0 && ind < (int)signal.size()) {
                (*frame)(i) = signal(ind);
            }
        }
    } else {
        return EXIT_FAILURE;
    }

    // Get pre-frame samples for smooth filtering
    if (pre_frame) {
        for (i = 0; i < (int)pre_frame->size(); i++) {
            ind = center_index - (int)frame->size() / 2 + i -
                  pre_frame->size();  // SPTK compatible, ljuvela
            if (ind >= 0 && ind < (int)signal.size()) (*pre_frame)(i) = signal(ind);
        }
    }

    return EXIT_SUCCESS;
}







const double pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679;

// Smooth function
gsl::vector smooth(const gsl::vector& input, int windowSize)
{
    gsl::vector output(input.size());
    int halfWindowSize = windowSize / 2;

    for (int i = 0; i < input.size(); i++)
    {
        double sum = 0.0;
        int count = 0;

        for (int j = i - halfWindowSize; j <= i + halfWindowSize; j++)
        {
            if (j >= 0 && j < input.size())
            {
                sum += input[j];
                count++;
            }
        }

        output[i] = sum / count;
    }

    return output;
}


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


void Rd2R(double Rd, double EE, double F0, double& Ra, double& Rk, double& Rg) {
    Ra = (-1 + (4.8 * Rd)) / 100;
    Rk = (22.4 + (11.8 * Rd)) / 100;
    double EI = (M_PI * Rk * EE) / 2;
    double UP = (Rd * EE) / (10 * F0);
    Rg = EI / (F0 * UP * M_PI);
}

void lf_cont(double F0, double fs, double Ra, double Rk, double Rg, double EE, gsl::vector& g_LF) {
    const double F0min = 20.0;
    const double F0max = 500.0;

    // Set LF model parameters
    double T0 = 1.0 / F0;
    double Ta = Ra * T0;
    double Te = ((1.0 + Rk) / (2.0 * Rg)) * T0;
    double Tp = Te / (Rk + 1.0);
    double Tb = ((1.0 - (Rk + 1.0) / (2.0 * Rg)) * 1.0 / F0);
    double Tc = Tb + Te;

//    if (F0 < F0min || F0 > F0max) {
//        // Handle invalid F0 value
//        // For example, you can clear the input vector:
//        g_LF.resize(0);
//    } else {
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

    g_LF.resize(T1_size + T2_size);

    for (size_t i = 0; i < T1_size; i++) {
        double t = dt * i;
        g_LF[i] = E0 * std::exp(alpha * t) * std::sin(omega * t);
    }

    for (size_t i = 0; i < T2_size; i++) {
        double t = (T1_size * dt) + dt * i;
        g_LF[T1_size + i] = (-EE / (epsi * Ta)) * (std::exp(-epsi * (t - Te)) - std::exp(-epsi * Tb));
    }
//    }
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



double computeCorrelation(const gsl::vector X, const gsl::vector Y)
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



gsl::matrix computeCorrelationMatrix(const gsl::vector& X, const gsl::vector& Y)
{
    int n = X.size();
    gsl::matrix corrMatrix(1, n); // Create a matrix to store correlation scores

    double sum_X = 0.0, sum_Y = 0.0, sum_XY = 0.0;
    double squareSum_X = 0.0, squareSum_Y = 0.0;

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

    // Compute the correlation coefficient for each element
    for (int i = 0; i < n; i++)
    {
        double corr = (n * X[i] * Y[i] - sum_X * sum_Y)
                      / sqrt((n * X[i] * X[i] - sum_X * sum_X)
                             * (n * Y[i] * Y[i] - sum_Y * sum_Y));

        corrMatrix(0, i) = corr;
    }

    return corrMatrix;
}



//std::vector<double> medfilt1(const std::vector<double>& input, int windowSize) {
//    std::vector<double> output(input.size());
//    int halfWindowSize = windowSize / 2;
//
//    for (int i = 0; i < input.size(); i++) {
//        std::vector<double> window;
//
//        for (int j = std::max(0, i - halfWindowSize); j <= std::min(i + halfWindowSize, static_cast<int>(input.size()) - 1); j++) {
//            window.push_back(input[j]);
//        }
//
//        std::sort(window.begin(), window.end());
//
//        if (window.size() % 2 == 0) {
//            output[i] = (window[window.size() / 2 - 1] + window[window.size() / 2]) / 2.0;
//        } else {
//            output[i] = window[window.size() / 2];
//        }
//    }
//
//    return output;
//}

std::vector<double> medfilt1(const std::vector<double>& input, int windowSize) {
    std::vector<double> output(input.size());
    int halfWindowSize = windowSize / 2;

    for (int j = 0; j < input.size(); j++) {
        std::vector<double> DataAFMid(windowSize);

        // Build the median array
        for (int k = 0; k < windowSize; k++) {
            if (((j - (windowSize - 1) / 2) + k < 0) || ((j - (windowSize - 1) / 2) + k) > (input.size() - 1))
                DataAFMid[k] = 0;
            else
                DataAFMid[k] = input[(j - (windowSize - 1) / 2) + k];
        }

        // Sort the median array in ascending order
        std::sort(DataAFMid.begin(), DataAFMid.end());

        // Assign the middle value of the sorted array to the output
        output[j] = DataAFMid[windowSize / 2];
    }

    return output;
}


// Medfilt1 function
//gsl::vector medfilt1(const gsl::vector& input, int windowSize)
//{
//    gsl::vector output(input.size());
//    int halfWindowSize = windowSize / 2;
//
//    for (int i = 0; i < input.size(); i++)
//    {
//        std::vector<double> window;
//
//        for (int j = i - halfWindowSize; j <= i + halfWindowSize; j++)
//        {
//            if (j >= 0 && j < input.size())
//                window.push_back(input[j]);
//        }
//
//        std::sort(window.begin(), window.end());
//        output[i] = window[windowSize / 2];
//    }
//
//    return output;
//}
//







//std::vector<double> smooth(const std::vector<double>& input, int windowSize) {
//    std::vector<double> output(input.size());
//    int halfWindowSize = (windowSize - 1) / 2;
//
//    if (input.size() > windowSize) {
//        for (int i = 0; i < input.size(); i++) {
//            if (i - halfWindowSize <= 0) { // First segment
//                output[i] = 0;
//                double sum = 0.0;
//                int count = 0;
//                for (int j = 0; j <= 2 * i; j++) {
//                    sum += input[j];
//                    count++;
//                }
//                output[i] = sum / (count * 2 + 1);
//            } else if (i - halfWindowSize > 0 && input.size() - 1 - i > halfWindowSize) { // Second segment
//                output[i] = 0;
//                double sum = 0.0;
//                for (int j = i - halfWindowSize; j <= i + halfWindowSize; j++) {
//                    sum += input[j];
//                }
//                output[i] = sum / windowSize;
//            } else { // Third segment
//                output[i] = 0;
//                double sum = 0.0;
//                int count = 0;
//                for (int j = 2 * i - (input.size() - 1); j <= input.size() - 1; j++) {
//                    sum += input[j];
//                    count++;
//                }
//                output[i] = sum / (2 * (input.size() - 1 - i) + 1);
//            }
//        }
//    }
//
//    return output;
//}
//std::vector<double> smooth(const std::vector<double>& input, int windowSize) {
//    int size = input.size();
//    std::vector<double> Z(size, 0.0);
//    int b = (windowSize - 1) / 2;
//
//    for (int i = 0; i < size; i++) {
//        if (i <= b) {  // First segment
//            Z[i] = 0.0;
//            double sum = 0.0;
//            for (int j = 0; j <= 2 * i; j++) {
//                sum += input[j];
//            }
//            Z[i] = sum / ((i * 2) + 1);
//        } else if (i >= size - b) {  // Third segment
//            Z[i] = 0.0;
//            double sum = 0.0;
//            for (int j = 2 * i - size + 1; j < size; j++) {
//                sum += input[j];
//            }
//            Z[i] = sum / ((size - i - 1) * 2 + 1);
//        } else {  // Second segment
//            Z[i] = 0.0;
//            double sum = 0.0;
//            for (int j = i - b; j <= i + b; j++) {
//                sum += input[j];
//            }
//            Z[i] = sum / windowSize;
//        }
//    }
//
//    return Z;
//}

//
//#include <vector>
//
//std::vector<double> smooth(const std::vector<double>& input, int windowSize) {
//    std::vector<double> output(input.size());
//
//    int start = 0;
//    int end = input.size() - 1;
//    int b = (windowSize - 1) / 2;
//
//    if ((end - start + 1) > windowSize) {
//        for (int i = start; i <= end; i++) {
//            if ((i - start) <= b) {
//                output[i] = 0;
//                int j = start;
//                for (; j <= 2 * i - start; j++) {
//                    output[i] += input[j];
//                }
//                output[i] /= ((i - start) * 2 + 1);
//            } else if (((i - start) > b) && ((end - i) > b)) {
//                output[i] = 0;
//                for (int j = i - b; j <= i + b; j++) {
//                    output[i] += input[j];
//                }
//                output[i] /= windowSize;
//            } else {
//                output[i] = 0;
//                for (int j = 2 * i - end; j <= end; j++) {
//                    output[i] += input[j];
//                }
//                output[i] /= (2 * (end - i) + 1);
//            }
//        }
//    }
//
//    return output;
//}

std::vector<double> smooth(const std::vector<double>& input, int windowSize) {
    int dataSize = input.size();
    std::vector<double> output(dataSize);

    int b = (windowSize - 1) / 2;
    for (int i = 0; i < dataSize; i++) {
        if (i <= b) { // First segment
            output[i] = 0;
            for (int j = 0; j <= 2 * i; j++) {
                output[i] += input[j];
            }
            output[i] /= ((i + 1) * 2);
        }
        else if (i >= dataSize - b) { // Third segment
            output[i] = 0;
            for (int j = 2 * i - dataSize + 1; j < dataSize; j++) {
                output[i] += input[j];
            }
            output[i] /= ((dataSize - i) * 2);
        }
        else { // Second segment
            output[i] = 0;
            for (int j = i - b; j <= i + b; j++) {
                output[i] += input[j];
            }
            output[i] /= windowSize;
        }
    }

    return output;
}

gsl_matrix * RepMatHorizAlloc(gsl_vector * v, size_t k) {
    gsl_matrix *mat = gsl_matrix_alloc(k, v->size);
    for (size_t i = 0; i < k; ++i) {
        gsl_matrix_set_row(mat, i, v);
    }
    return mat;
}

gsl_matrix * RepMatVertAlloc(gsl_vector * v, size_t k) {
    gsl_matrix *mat = gsl_matrix_alloc(v->size, k);
    for (size_t i = 0; i < k; ++i) {
        gsl_matrix_set_col(mat, i, v);
    }
    return mat;
}




double GetRd(const Param &params, const gsl::vector &source_signal,
             const gsl::vector_int &gci_inds, gsl::vector *Rd_opt, gsl::vector *EE) {

//    if (params.use_external_f0) {
//    std::cout << "using external F0 file: " << params.external_f0_filename
//              << " ...";

    *Rd_opt = gsl::vector(gci_inds.size());

    struct LfData {
    public:
        gsl::vector Rd;
        gsl::vector EE;
        gsl::vector Ra;
        gsl::vector Rk;
        gsl::vector Rg;
        double Ra_cur;
        double Rk_cur;
        double Rg_cur;
        double F0_cur;
        double pulseLen;
        double cor_time;
        double err_time;
        double cor_freq;
        double err_freq;

        gsl::vector Rd_set;
        gsl::matrix Rd_n;
        gsl::matrix cost;
        gsl::matrix prev;
        gsl::vector glot_seg;
        gsl::vector glot_seg_spec;
        gsl::vector glot_seg_fft_mag;
        gsl::vector glot_seg_log_mag;
        gsl::vector freq;
        gsl::vector err_mat;
        gsl::vector err_mat_sort;
        gsl::vector err_mat_time;
        gsl::vector_int err_mat_sortIdx;
        gsl::vector_int Rd_set_err_mat_sortIdx;
        gsl::vector Rd_set_err_mat_sortVal;

        gsl::vector pulse;
        gsl::vector g_LF;

        gsl::vector LFgroup;
        gsl::vector LFgroup_win_spec;
        gsl::vector LFgroup_win;
        gsl::vector_int best;
        gsl::vector Rd_opt;

        gsl::vector temp;

        gsl::vector exh_err_n;
        gsl::vector LFpulse_cur;

        gsl::vector LFpulse_prev;
        gsl::matrix costm;
        double Ra_try, Rk_try, Rg_try;
        double Ra_prev, Rk_prev, Rg_prev;

    };

    LfData lf_data;

    /******************************** Initial settings *********************************************************************/

    // Dynamic programming weights
    double time_wgt = 0.1;
    double freq_wgt = 0.3;
    double trans_wgt = 0.3;


    // EE=zeros(1,length(GCI));
    lf_data.EE.resize(gci_inds.size());
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
    int nframe = gci_inds.size();


    // ncands = 5; Number of candidate LF model configurations to consider
    int ncands = 5;

    // Rd_n=zeros(nframe,ncands);
    lf_data.Rd_n = gsl::matrix(nframe, ncands);
    // cost=zeros(nframe,ncands);      % cumulative cost
    lf_data.cost = gsl::matrix(nframe, ncands);
    // prev=zeros(nframe,ncands);      % traceback pointer
    lf_data.prev = gsl::matrix(nframe, ncands);

    gsl_matrix_int* prev = gsl_matrix_int_alloc(nframe, ncands);



    /******************************** Do processing - exhaustive search and dynamic programming ***************************/
    // for n=1:length(GCI)
    for (int n = 0; n < gci_inds.size(); ++n) {
        double pulseLen;
        /************************************ get framing information *********************************************************/

        if (n == 0)
        {
            pulseLen = round((gci_inds[n + 1] - gci_inds[n]) * pulseNum);
            lf_data.F0_cur = params.fs / (round(gci_inds[n + 1] - gci_inds[n]));
        }
        else
        {
            pulseLen = round((gci_inds[n] - gci_inds[n - 1]) * pulseNum);
            lf_data.F0_cur = params.fs / (round(gci_inds[n] - gci_inds[n - 1]));
        }

        // pulseLen=abs(pulseLen);
        pulseLen = std::abs(pulseLen);

        //        if GCI(n)-round(pulseLen/2) > 0
        //            start=GCI(n)-round(pulseLen/2);
        //        else start=1;
        //        end
        int start;
        int finish;

        if (gci_inds[n] - round(pulseLen / 2) > 0) {
            start = gci_inds[n] - round(pulseLen / 2);
        } else {
            start = 0;
        }


        //        if GCI(n)+round(pulseLen/2) <= length(glot)
        //        finish = GCI(n)+round(pulseLen/2);
        //        else finish = length(glot);
        //        end
        if (gci_inds[n] + round(pulseLen / 2) <= source_signal.size())
        {
            finish = gci_inds[n] + round(pulseLen / 2);
        }
        else
        {
            finish = source_signal.size() - 1;
        }


        //        glot_seg=glot(start:finish);
        //        glot_seg=glot_seg(:);s

        gsl::vector glot_seg(source_signal.subvector(start, finish - start + 1));
        lf_data.glot_seg = glot_seg;



        //  glot_seg_spec=20*log10(abs(fft(glot_seg)));
        ComplexVector glot_seg_spec;
        FFTRadix2(lf_data.glot_seg, &glot_seg_spec);

        lf_data.glot_seg_spec = glot_seg_spec.getAbs();
        for (size_t i = 0; i < lf_data.glot_seg_spec.size(); i++) {
            lf_data.glot_seg_spec(i) = 20 * log10(lf_data.glot_seg_spec(i));
        }



        //   freq=linspace(0,fs,length(glot_seg));
        lf_data.freq.resize(lf_data.glot_seg.size());

        //  double start = 0.0;
        //  double stop = params.fs;
        //  double step = (stop - start) / (lf_data.glot_seg.size() - 1);

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


        /****************************************** exhaustive search *********************************************************/

        // for m=1:length(Rd_set)
        for (int m = 0; m < lf_data.Rd_set.size(); ++m) {
            //         [Ra_cur,Rk_cur,Rg_cur] = Rd2R(Rd_set(m),EE(n),F0_cur);
            Rd2R(lf_data.Rd_set(m), lf_data.EE(n), lf_data.F0_cur, lf_data.Ra_cur, lf_data.Rk_cur, lf_data.Rg_cur);


            //          pulse = lf_cont(F0_cur,fs,Ra_cur,Rk_cur,Rg_cur,EE(n));
            lf_cont(lf_data.F0_cur, params.fs, lf_data.Ra_cur, lf_data.Rk_cur, lf_data.Rg_cur, lf_data.EE(n), lf_data.pulse);

            // LFgroup = makePulseCentGCI(pulse,pulseLen,GCI(n)-start,finish-GCI(n));
            lf_data.LFgroup = makePulseCentGCI(lf_data.pulse, pulseLen, gci_inds(n)-start, finish-gci_inds(n));

            // LFgroup_win=LFgroup(:);
            lf_data.LFgroup_win = lf_data.LFgroup;


            //  glot_seg_spec=20*log10(abs(fft(glot_seg)));
            ComplexVector LFgroup_win_spec;
            FFTRadix2(lf_data.LFgroup, &LFgroup_win_spec);

            lf_data.LFgroup_win_spec = LFgroup_win_spec.getAbs();
            // Print the values of temp`
            for (size_t i = 0; i < lf_data.LFgroup.size(); i++) {
                lf_data.LFgroup_win_spec(i) = 20 * log10(lf_data.LFgroup_win_spec(i));
            }



            /******************************** Time domain error function **********************************************************/
            //                    cor_time = corrcoef(glot_seg,LFgroup_win);
            //                    cor_time=abs(cor_time(2));
            //                    err_time=1-cor_time;
            //                    err_mat_time(m)=err_time;


            lf_data.cor_time = computeCorrelation(lf_data.glot_seg, lf_data.LFgroup_win);

            lf_data.cor_time = std::abs(lf_data.cor_time);
            lf_data.err_time = 1 - lf_data.cor_time;
            lf_data.err_mat_time[m] = lf_data.err_time;



            /******************************* Frequency domain error function ******************************************************/
            //            % Frequency domain error function
            //            cor_freq = corrcoef(glot_seg_spec(freq<MVF),LFgroup_win_spec(freq<MVF));
            //            cor_freq=abs(cor_freq(2));
            //            err_freq=1-cor_freq;


            lf_data.cor_freq = computeCorrelation(lf_data.glot_seg_spec, lf_data.LFgroup_win_spec);
            lf_data.cor_freq = std::abs(lf_data.cor_freq);
            lf_data.err_freq = 1 - lf_data.cor_freq;


            /******************************** Combined error with weights *********************************************************/
//          err_mat(m)=(err_time*time_wgt)+(err_freq*freq_wgt);

            lf_data.err_mat[m] = (lf_data.err_time * time_wgt) + (lf_data.err_freq * freq_wgt);


        }

        /******************************** Find best ncands (local costs and Rd values) ****************************************/
//          [err_mat_sort,err_mat_sortIdx]=sort(err_mat);
//          Rd_n(n,1:ncands)=Rd_set(err_mat_sortIdx(1:ncands));


        // Copy err_mat to a new vector for sorting
        lf_data.err_mat_sort = lf_data.err_mat;

        // Convert gsl vector "err_mat_sort_std" into std::vector & Sort std::vector in ascending order
        std::vector<double> err_mat_sort_std(lf_data.err_mat_sort.size());
        for (size_t i = 0; i < lf_data.err_mat_sort.size(); ++i) {
            err_mat_sort_std[i] = lf_data.err_mat_sort[i];
        }
        std::sort(err_mat_sort_std.begin(), err_mat_sort_std.end());

        // Copy sorted elements back to gsl::vector
        for (size_t i = 0; i < lf_data.err_mat_sort.size(); ++i) {
            lf_data.err_mat_sort[i] = err_mat_sort_std[i];
        }

        // Create a new vector called "err_mat_sortIdx"
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



        //  Rd_n(n,1:ncands)=Rd_set(err_mat_sortIdx(1:ncands));

        // 1. Get the err_mat_sortIdx(1:ncands) like the index value of the vectors
        lf_data.Rd_set_err_mat_sortIdx = lf_data.err_mat_sortIdx.subvector(1, ncands);


        // 2. Use the ID vectors to tract the values to replace "Rd_set(err_mat_sortIdx(1:ncands))"
        lf_data.Rd_set_err_mat_sortVal.resize(lf_data.Rd_set_err_mat_sortIdx.size());

        for (size_t i = 0; i < ncands; i++)
        {
            int index = lf_data.Rd_set_err_mat_sortIdx[i];
            lf_data.Rd_set_err_mat_sortVal(i) = lf_data.Rd_set[index];
            lf_data.Rd_n(n, i) = lf_data.Rd_set_err_mat_sortVal(i);

        }




        // exh_err_n=err_mat_sort(1:ncands);
        lf_data.exh_err_n = lf_data.err_mat_sort.subvector(1, ncands);


        // cost(n,1:ncands) = exh_err_n(:)';
        for (size_t i = 0; i < ncands; i++)
        {
            lf_data.cost(n, i) = lf_data.exh_err_n(i);
        }



/******************************** Find optimum Rd value (dynamic programming) ****************************************/
        if (n > 1) {

            gsl::matrix costm(ncands, ncands); // transition cost matrix: rows (previous), cols (current)
            costm.set_all(0); // Initialize costm to all zeros


            for (int c = 0; c < ncands; ++c) {

/***************************************　Transitions TO states in current frame　**************************************/

                // Transitions TO states in current frame
                Rd2R(lf_data.Rd_n(n, c), lf_data.EE(n), lf_data.F0_cur, lf_data.Ra_try, lf_data.Rk_try, lf_data.Rg_try);

//                std::cout << lf_data.Rg_try << std::endl;

                lf_cont(lf_data.F0_cur, params.fs, lf_data.Ra_try, lf_data.Rk_try, lf_data.Rg_try, lf_data.EE(n), lf_data.LFpulse_cur);


                for (int p = 0; p < ncands; ++p) {

                    // Transitions FROM states in previous frame
                    // [Ra_prev,Rk_prev,Rg_prev] = Rd2R(Rd_n(n-1,p),EE(n),F0_cur);

                    Rd2R(lf_data.Rd_n(n-1,p), lf_data.EE(n), lf_data.F0_cur, lf_data.Ra_prev, lf_data.Rk_prev, lf_data.Rg_prev);

                    // LFpulse_prev = lf_cont(F0_cur,fs,Ra_prev,Rk_prev,Rg_prev,EE(n));
                    lf_cont(lf_data.F0_cur, params.fs, lf_data.Ra_prev, lf_data.Rk_prev, lf_data.Rg_cur, lf_data.EE(n), lf_data.LFpulse_prev);


                    if (std::isnan( lf_data.LFpulse_cur(0)) || std::isnan( lf_data.LFpulse_prev(0))) {
                        costm(p, c) = 0;
                    } else {
                        double cor_cur = computeCorrelation( lf_data.LFpulse_cur,  lf_data.LFpulse_prev);
                        costm(p, c) = (1 - std::abs(cor_cur)) * trans_wgt; // transition cost
                    }


                    //           costm=costm+repmat(cost(n-1,1:ncands)',1,ncands);  % add in cumulative costs
                    //           [costi,previ]=min(costm,[],1);
                    //           cost(n,1:ncands)=cost(n,1:ncands)+costi;
                    //           prev(n,1:ncands)=previ;


                    std::vector<double> costi(ncands);
                    std::vector<int> previ(ncands);

                    for (int j = 0; j < ncands; j++) {
                        for (int i = 0; i < costm.get_rows(); i++) {
                            costi[i] = costm(i, j);
                        }
                        // Find the index of the minimum value in costi
                        double minVal = costi[0];
                        size_t idx = 0;
                        for (size_t i = 1; i < costi.size(); ++i) {
                            if (costi[i] < minVal) {
                                minVal = costi[i];
                                idx = i;
                            }
                        }
                        previ[j] = idx;
                        lf_data.cost(n, j) += costi[previ[j]];


                    }



                    // Update prev matrix
                    for (int j = 0; j < ncands; j++) {
                        lf_data.prev(n, j) = previ[j];
                    }


                }


            }
        }



        // gsl::vector_int idx_values(n);  // Declare a gsl::vector_int to store the idx values
/************************************** Do traceback ******************************************************************/
        //        best=zeros(n,1);
        //        [~,best(n)]=min(cost(n,1:ncands));
        lf_data.best.resize(n+1);  // Declare a gsl::vector_int to store the idx values
        lf_data.best.set_zero(); // Declare a gsl::vector_int to store the idx values

        for (size_t i = 0; i < n; ++i) {
            // Find the index of the minimum value in the subset of cost matrix
            double minVal = lf_data.cost(i, 0);
            size_t idx = 0;

            for (size_t j = 1; j < ncands; ++j) {
                if (lf_data.cost(i, j) < minVal) {
                    minVal = lf_data.cost(i, j);
                    idx = j;
                }
            }

            lf_data.best(i) = static_cast<int>(idx);  // Store the idx value in the gsl::vector_int
        }

        //        for i=n:-1:2
        //          best(i-1)=prev(i,best(i));
        //        end


        for (int i = n; i >= 2; i--) {
            lf_data.best(i - 2) = lf_data.prev(i, lf_data.best(i - 1));
        }

    }


    //    Rd_opt=zeros(1,nframe);
    lf_data.Rd_opt.resize(nframe);
    lf_data.Rd_opt.set_zero(); // Declare a gsl::vector_int to store the idx values


    //    for n=1:nframe
    //    Rd_opt(n) = Rd_n(n,best(n));
    //    end
    for (int n = 0; n < nframe; n++) {
        lf_data.Rd_opt[n] = lf_data.Rd_n(n, lf_data.best[n]);
    }


    // Set the integer values for lf_data.prev
    for (size_t i = 0; i < nframe; ++i) {
        for (size_t j = 0; j < ncands; ++j) {
            int value = static_cast<int>(lf_data.prev(i, j));
            gsl_matrix_int_set(prev, i, j, value);
        }
    }

    // Printing the integer values of lf_data.prev
    for (size_t i = 0; i < nframe; ++i) {
        for (size_t j = 0; j < ncands; ++j) {
        }
    }



    //    medfilt1(lf_data.Rd_opt, 11);
    std::vector<double> input(lf_data.Rd_opt.size());
    for (size_t i = 0; i < lf_data.Rd_opt.size(); i++) {
        input[i] = lf_data.Rd_opt[i];
    }
    std::vector<double> medfilt1_result = medfilt1(input, 11);
    for (size_t i = 0; i < lf_data.Rd_opt.size(); i++) {
        lf_data.Rd_opt[i] = medfilt1_result[i];
    }


//    smooth(lf_data.Rd_opt, 5);
    std::vector<double> smooth_out = smooth(medfilt1_result, 5);
    for (size_t i = 0; i < lf_data.Rd_opt.size(); i++) {
        lf_data.Rd_opt[i] = smooth_out[i];
    }


    //    Rd_opt = smooth(medfilt1(Rd_opt,11),5)*.5;
    for (size_t i = 0; i < lf_data.Rd_opt.size(); i++) {
        lf_data.Rd_opt[i] *= 0.5;

    }

    *Rd_opt = gsl::vector(gci_inds.size());


    for (size_t i = 0; i < lf_data.EE.size(); i++) {
        (*Rd_opt)(i) = lf_data.Rd_opt[i];

    }

    *EE = gsl::vector(gci_inds.size());


    for (size_t i = 0; i < lf_data.EE.size(); i++) {
        (*EE)(i) = lf_data.EE[i];

    }


//        *Rd_opt(0) = lf_data.Rd_opt;
//    std::cout << "********************* EE *********************" << lf_data.EE.size() << std::endl;
//    std::cout << "********************* EE *********************" << lf_data.Rd_opt.size() << std::endl;

    std::cout << "LF Rd analysis done.\n";

    return EXIT_SUCCESS;
}


void ParameterSmoothing(const Param &params, AnalysisData *data) {
    if (params.lsf_vt_smooth_len > 2)
        MovingAverageFilter(params.lsf_vt_smooth_len, &(data->lsf_vocal_tract));

    if (params.lsf_glot_smooth_len > 2)
        MovingAverageFilter(params.lsf_glot_smooth_len, &(data->lsf_glot));

    if (params.gain_smooth_len > 2) {
        // MedianFilter(5, &(data->frame_energy));
        MovingAverageFilter(params.gain_smooth_len, &(data->frame_energy));
    }

    if (params.hnr_smooth_len > 2)
        MovingAverageFilter(params.hnr_smooth_len, &(data->hnr_glot));
}




void GenerateUnvoicedSignal(const Param &params, const SynthesisData &data,
                            gsl::vector *signal) {
    /* When using pulses-as-features for unvoiced, unvoiced part is filtered as
     * voiced */
    /*
    if ((params.use_paf_unvoiced_synthesis &&
         params.excitation_method == PULSES_AS_FEATURES_EXCITATION) ||
        params.use_external_excitation)
    { return; }
    */

    if (params.use_paf_unvoiced_synthesis &&
        params.excitation_method == PULSES_AS_FEATURES_EXCITATION) {
        //std::cout << "skipping unvoiced excitation generation" << std::endl;
        return;
    }

    //std::cout << "generating unvoiced" << std::endl;

    gsl::vector uv_signal((*signal).size(), true);
    gsl::vector noise_vec(params.frame_length_unvoiced);
    gsl::random_generator rand_gen;
    gsl::gaussian_random random_gauss_gen(rand_gen);

    gsl::vector A(params.lpc_order_vt + 1, true);
    gsl::vector A_tilt(params.lpc_order_glot + 1, true);

    ComplexVector noise_vec_fft;

    ComplexVector tilt_fft;
    size_t NFFT = 4096;  // Long FFT
    ComplexVector vt_fft(NFFT / 2 + 1);
    gsl::vector fft_mag(NFFT / 2 + 1);
    size_t i;

    // for de-warping filters
    gsl::vector impulse(params.frame_length);
    gsl::vector imp_response(params.frame_length);
    // gsl::vector impulse(NFFT);
    // gsl::vector imp_response(NFFT);
    gsl::vector b(1);
    b(0) = 1.0;

    /* Define analysis and synthesis window */
    //double kbd_alpha = 2.3;
    //gsl::vector kbd_window =
    //    getKaiserBesselDerivedWindow(noise_vec.size(), kbd_alpha);

    size_t frame_index;
    for (frame_index = 0; frame_index < params.number_of_frames; frame_index++) {
        if (data.fundf(frame_index) == 0) {

            if (params.use_generic_envelope) {
                for (i = 0; i < vt_fft.getSize(); i++) {
                    vt_fft.setReal(i, data.spectrum(i, frame_index));
                    vt_fft.setImag(i, 0.0);
                }
                // Spectrum2MinPhase(&vt_fft);
            } else {
                Lsf2Poly(data.lsf_vocal_tract.get_col_vec(frame_index), &A);
                if (params.warping_lambda_vt == 0.0) {
                    FFTRadix2(A, NFFT, &vt_fft);
                } else {
                    /* get warped filter linear frequency response via impulse response */
                    imp_response.set_zero();
                    impulse.set_zero();
                    impulse(0) = 1.0;
                    /* get inverse filter impulse response */
                    WFilter(A, b, impulse, params.warping_lambda_vt, &imp_response);
                    FFTRadix2(imp_response, NFFT, &vt_fft);
                }
            }

            if (params.use_external_excitation) {
                GetFrame(data.excitation_signal, frame_index,
                         rint(params.frame_shift / params.speed_scale), &noise_vec, NULL);
            } else {
                for (i = 0; i < noise_vec.size(); i++) {
                    noise_vec(i) = random_gauss_gen.get();
                }
            }


            /* Cancel pre-emphasis if needed */
            if (params.unvoiced_pre_emphasis_coefficient > 0.0) {
                gsl::vector noise_vec_copy(noise_vec);
                Filter(std::vector<double>{1.0},
                       std::vector<double>{
                               1.0, -1.0 * params.unvoiced_pre_emphasis_coefficient},
                       noise_vec_copy, &noise_vec);
            }

            ApplyWindowingFunction(COSINE, &noise_vec);

            FFTRadix2(noise_vec, NFFT, &noise_vec_fft);
            Lsf2Poly(data.lsf_glot.get_col_vec(frame_index), &A_tilt);
            FFTRadix2(A_tilt, NFFT, &tilt_fft);

            // Randomize phase
            double mag;
            double ang;
            for (i = 0; i < noise_vec_fft.getSize(); i++) {
                if (params.use_generic_envelope) {
                    mag = noise_vec_fft.getAbs(i) * vt_fft.getAbs(i);
                } else if (!params.use_spectral_matching) {
                    /* Only use vocal tract synthesis filter */
                    mag = noise_vec_fft.getAbs(i) *
                          GSL_MIN(1.0 / (vt_fft.getAbs(i)), 10000);
                } else {
                    /* Use both vocal tract and excitation LP envelope synthesis filters */
                    mag = noise_vec_fft.getAbs(i) *
                          GSL_MIN(1.0 / (vt_fft.getAbs(i)), 10000) *
                          GSL_MIN(1.0 / tilt_fft.getAbs(i), 10000);
                }
                ang = noise_vec_fft.getAng(i);

                noise_vec_fft.setReal(i, mag * cos(double(ang)));
                noise_vec_fft.setImag(i, mag * sin(double(ang)));
            }
            double e_target;
            e_target = LogEnergy2FrameEnergy(data.frame_energy(frame_index),
                                             noise_vec.size());

            IFFTRadix2(noise_vec_fft, &noise_vec);

            ApplyWindowingFunction(COSINE, &noise_vec);
            noise_vec *= params.noise_gain_unvoiced * e_target /
                         getEnergy(noise_vec) / sqrt(2.0);

            /* Normalize overlap-add window */
            noise_vec /= 0.5 * (double)noise_vec.size() / (double)params.frame_shift;
            OverlapAdd(noise_vec,
                       frame_index * rint(params.frame_shift / params.speed_scale),
                       &uv_signal);
        }
    }
    (*signal) += uv_signal;
}

