// streaming_acf_mvp.cpp
// -----------------------------------------------------------------------------
// Author:  Novichkov Gleb, PhD.
// Date:    20260616
// License: MIT License
// -----------------------------------------------------------------------------
// Description:
//
// MVP for the streaming multi-rate autocorrelation (ACF) system.
//
// Build:  g++ -std=c++17 -O2 -pthread streaming_acf_mvp.cpp -o acf_mvp
// Run:    ./acf_mvp [gen_freq_hz] [duration_seconds] [noise_sigma]
// Example:./acf_mvp 100 1.5 0.0
//
// What it implements (mapping to the specification):
//   * Sample stream at Fs = 2 kHz. A mock generator PUBLISHES samples;
//     the processor SUBSCRIBES to them (Observer / pub-sub pattern).
//   * 500 frequency channels: f = 1..500 Hz, 1 Hz step.
//   * Per channel f: integration time T = 1/f  ->  window N = Fs/f samples,
//     quarter-period lag tau = T/4  ->  k = N/4 samples.
//   * ACF_f = sum over the window of S[n] * S[n-k]   (single lag, per the
//     formula in the spec; it is the discrete form of the integral).
//   * Output: a 500 (freq) x 2000 (time) matrix. The time axis is a CIRCULAR
//     buffer addressed by a head pointer, so the "cyclic shift / rows move
//     down" is pure pointer arithmetic -- never a memcpy of the matrix.
//   * Input history is kept in a RING BUFFER (as instructed), sized to the
//     longest window (1 Hz -> 2000 samples).
//
// Datatype note: the spec asks for a 64-bit result range. We use double
// (64-bit float) for samples and accumulation -- the natural CPU fit and it
// sidesteps INT64 overflow bookkeeping for an MVP.
//
// Detector note: the lag is a quarter of the channel PERIOD, which makes each
// channel a quadrature-style detector, not a plain "peak at the input tone".
// Its response to an input tone f_in is roughly proportional to
// cos((pi/2) * f_in / f), so the f_in/2 channel responds strongly while the
// f_in channel sits near zero. The console readout reflects that structured
// response rather than a single clean peak -- this is faithful to the spec.
// -----------------------------------------------------------------------------

#include "streaming_acf_mvp.h"

// --------------------------------- Demo ---------------------------------------
int main(int argc, char** argv) {
    const double kFs = 2000.0;  // 2 kHz stream
    const int kFmin = 1, kFmax = 500, kFstep = 1;
    const std::size_t kTimeAxis = 2000;  // 1 s of history at 2 kHz

    double toneHz = (argc > 1) ? std::stod(argv[1]) : 100.0;
    double seconds = (argc > 2) ? std::stod(argv[2]) : 1.5;
    double noise = (argc > 3) ? std::stod(argv[3]) : 0.0;
    const auto numSamples = static_cast<std::uint64_t>(kFs * seconds);

    std::cout << "Streaming multi-rate ACF -- MVP\n"
              << "  Fs            : " << kFs << " Hz\n"
              << "  channels      : " << kFmin << ".." << kFmax
              << " Hz (step " << kFstep << ")\n"
              << "  matrix        : " << (kFmax - kFmin) / kFstep + 1
              << " x " << kTimeAxis << " (freq x time, circular)\n"
              << "  generator tone: " << toneHz << " Hz, noise sigma " << noise << "\n"
              << "  duration      : " << seconds << " s ("
              << numSamples << " samples)\n\n";

    AcfProcessor processor(kFs, kFmin, kFmax, kFstep, kTimeAxis);
    SignalGenerator generator(kFs, toneHz, 1.0, noise);
    generator.subscribe(&processor);  // <-- subscribe to the published stream

    // Run the generator on its own thread to model a live stream.
    const auto t0 = std::chrono::steady_clock::now();
    std::thread producer([&] { generator.run(numSamples, /*paceRealtime=*/true); });
    producer.join();
    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();

    const std::uint64_t cols = processor.columnsProduced();
    std::cout << "Processed " << numSamples << " samples -> " << cols
              << " columns in " << elapsed << " s ("
              << (elapsed > 0 ? cols / elapsed : 0) << " cols/s, "
              << "real-time budget is " << kFs << " cols/s)\n\n";

    // Read the latest matrix column and report a few channels + top responders.
    const double* latest = processor.matrix().latestColumn();
    const std::size_t nf = processor.numFreq();

    std::cout << "Latest column, selected channels (ACF value):\n";
    for (int f : {1, 2, 4, 5, 50, 99, 100, 101, 200, 500}) {
        if (f >= kFmin && f <= kFmax) {
            const std::size_t ch = static_cast<std::size_t>((f - kFmin) / kFstep);
            std::cout << "  f = " << f << " Hz : " << latest[ch] << "\n";
        }
    }

    std::vector<std::size_t> order(nf);
    for (std::size_t i = 0; i < nf; ++i) order[i] = i;
    std::partial_sort(order.begin(), order.begin() + 5, order.end(),
                      [&](std::size_t a, std::size_t b) {
                          return std::fabs(latest[a]) > std::fabs(latest[b]);
                      });
    std::cout << "\nTop 5 channels by |ACF| (strongest detector response):\n";
    for (int r = 0; r < 5; ++r) {
        const std::size_t ch = order[r];
        std::cout << "  f = " << processor.freqOf(ch) << " Hz : "
                  << latest[ch] << "\n";
    }
    return 0;
}
