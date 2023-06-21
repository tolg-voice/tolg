/*
Copyright 2015 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <memory>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "core/file_resource.h"
#include "core/track.h"
#include "epoch_tracker/epoch_tracker.h"
#include "wave/wave.h"



#include <iostream>


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
        std::cout << "Voicing: " << track.a(i) << std::endl;

//        if (track.v(i) == 1) {
//            double timeInSeconds = track.t(i) * 16000.0;
//            std::cout << timeInSeconds << std::endl;
//        }
    }
}



int main(int argc, char* argv[]) {
    int opt = 0;
    std::string filename;
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


    filename = "./slt_arctic_a0002.wav";
    f0_output = "./slt_arctic_a0002.f0";
    pm_output = "./slt_arctic_a0002.pm";
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
    if (f0 != nullptr) {
        PrintTrack(*f0);
    } else {
        std::cerr << "F0 track is null" << std::endl;
    }



//    // Print correlation track
//    if (corr != nullptr) {
//        PrintTrack(*corr);
//    } else {
//        std::cerr << "Correlation track is null" << std::endl;
//    }



//    if (!pm_output.empty() && !pm->Save(pm_output, ascii)) {
////        std::cout << pm_output << std::endl;
//        fprintf(stderr, "Failed to save pitchmarks to '%s'\n", pm_output.c_str());
//        return 1;
//    }



//    std::string f0sring = f0->ToString();
//    std::cout << f0sring << std::endl;

//    if (!corr_output.empty() && !corr->Save(corr_output, ascii)) {
//        delete corr;
//        fprintf(stderr, "Failed to save correlations to '%s'\n", corr_output.c_str());
//        return 1;
//    }
    delete f0;
    delete pm;
    delete corr;
    return 0;
}