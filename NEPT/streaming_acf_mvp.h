// streaming_acf_mvp.h
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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

// ----------------------------- Ring buffer -----------------------------------
// Fixed-capacity circular buffer of the most recent samples.
// at(ago) returns the sample 'ago' steps back from newest (0 == newest).
template <typename T>
class RingBuffer {
public:
    explicit RingBuffer(std::size_t capacity)
        : buf_(capacity, T{}), cap_(capacity) {}

    void push(const T& v) {
        buf_[next_] = v;
        next_ = (next_ + 1) % cap_;
        if (count_ < cap_) ++count_;
    }

    // ago = 0 -> newest sample; ago = count()-1 -> oldest available sample.
    T at(std::size_t ago) const {
        // newest index is (next_ - 1); step back by 'ago'.
        std::size_t idx = (next_ + cap_ - 1 - (ago % cap_)) % cap_;
        return buf_[idx];
    }

    std::size_t size() const { return count_; }
    std::size_t capacity() const { return cap_; }

private:
    std::vector<T> buf_;
    std::size_t cap_;
    std::size_t next_ = 0;   // index of next write
    std::size_t count_ = 0;  // number of valid samples so far
};

// --------------------------- Pub/sub interface --------------------------------
struct ISampleSubscriber {
    virtual ~ISampleSubscriber() = default;
    virtual void onSample(std::uint64_t index, double value) = 0;
};

// ------------------------- Signal generator (publisher) -----------------------
class SignalGenerator {
public:
    SignalGenerator(double sampleRateHz, double toneHz,
                    double amplitude = 1.0, double noiseSigma = 0.0)
        : fs_(sampleRateHz), tone_(toneHz), amp_(amplitude),
          noise_(noiseSigma), rng_(12345), gauss_(0.0, 1.0) {}

    void subscribe(ISampleSubscriber* sub) { subs_.push_back(sub); }

    // Generate 'numSamples' and publish each to all subscribers.
    // paceRealtime = true sleeps to approximate the true 2 kHz cadence;
    // false (default) runs as fast as possible with logical timestamps.
    void run(std::uint64_t numSamples, bool paceRealtime = false) {
        using clock = std::chrono::steady_clock;
        const auto t0 = clock::now();
        const double dt = 1.0 / fs_;
        for (std::uint64_t n = 0; n < numSamples; ++n) {
            const double t = static_cast<double>(n) * dt;
            double v = amp_ * std::sin(2.0 * M_PI * tone_ * t);
            if (noise_ > 0.0) v += noise_ * gauss_(rng_);
            for (auto* s : subs_) s->onSample(n, v);

            if (paceRealtime) {
                const auto target = t0 + std::chrono::duration_cast<clock::duration>(
                                             std::chrono::duration<double>((n + 1) * dt));
                std::this_thread::sleep_until(target);
            }
        }
    }

private:
    double fs_, tone_, amp_, noise_;
    std::mt19937 rng_;
    std::normal_distribution<double> gauss_;
    std::vector<ISampleSubscriber*> subs_;
};

// --------------------------- Output ACF matrix --------------------------------
// 500 (freq) x 2000 (time). Stored column-major (each time-slice contiguous).
// The time axis is circular: writeColumn advances a head pointer; oldest column
// is overwritten. No data is ever shifted in memory.
class AcfMatrix {
public:
    AcfMatrix(std::size_t numFreq, std::size_t numTime)
        : rows_(numFreq), cols_(numTime), data_(numFreq * numTime, 0.0) {}

    void writeColumn(const std::vector<double>& col) {
        std::copy(col.begin(), col.end(), data_.begin() + head_ * rows_);
        head_ = (head_ + 1) % cols_;
        if (filled_ < cols_) ++filled_;
    }

    // Pointer to the most recently written column (the "top" of the waterfall).
    const double* latestColumn() const {
        std::size_t idx = (head_ + cols_ - 1) % cols_;
        return data_.data() + idx * rows_;
    }

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    std::size_t filled() const { return filled_; }

private:
    std::size_t rows_, cols_;
    std::vector<double> data_;
    std::size_t head_ = 0;    // next column to write (circular)
    std::size_t filled_ = 0;  // columns written so far (capped at cols_)
};

// --------------------------- ACF processor (subscriber) -----------------------
class AcfProcessor : public ISampleSubscriber {
public:
    AcfProcessor(double sampleRateHz, int fMin, int fMax, int fStep,
                 std::size_t timeAxis)
        : fs_(sampleRateHz),
          fMin_(fMin), fMax_(fMax), fStep_(fStep),
          numFreq_(static_cast<std::size_t>((fMax - fMin) / fStep + 1)),
          ring_(static_cast<std::size_t>(sampleRateHz / fMin)),  // longest window
          matrix_(numFreq_, timeAxis),
          column_(numFreq_, 0.0) {
        // Precompute per-channel window (N) and lag (k).
        N_.resize(numFreq_);
        k_.resize(numFreq_);
        for (std::size_t i = 0; i < numFreq_; ++i) {
            const int f = fMin_ + static_cast<int>(i) * fStep_;
            const int N = static_cast<int>(fs_) / f;  // samples per period (T = 1/f)
            N_[i] = N;
            k_[i] = std::max(1, N / 4);                // quarter-period lag
        }
    }

    void onSample(std::uint64_t /*index*/, double value) override {
        ring_.push(value);

        // Recompute one ACF value per frequency channel -> one matrix column.
        const int avail = static_cast<int>(ring_.size());
        for (std::size_t i = 0; i < numFreq_; ++i) {
            column_[i] = computeAcf(N_[i], k_[i], avail);
        }
        matrix_.writeColumn(column_);
        ++columns_;
    }

    // ACF_f = sum_{i=k}^{N-1} g[i] * g[i-k], where g is the most recent N
    // samples (g[N-1] = newest). Expressed via ring 'ago' indices:
    //   g[i]   -> ago = N-1-i
    //   g[i-k] -> ago = N-1-i+k
    // During warm-up (avail < N) we skip terms whose history is not present yet.
    double computeAcf(int N, int k, int avail) const {
        double acc = 0.0;
        for (int i = k; i < N; ++i) {
            const int agoNew = N - 1 - i;
            const int agoOld = agoNew + k;
            if (agoOld >= avail) continue;  // not enough history yet
            acc += ring_.at(static_cast<std::size_t>(agoNew)) *
                   ring_.at(static_cast<std::size_t>(agoOld));
        }
        return acc;
    }

    int freqOf(std::size_t channel) const {
        return fMin_ + static_cast<int>(channel) * fStep_;
    }
    std::size_t numFreq() const { return numFreq_; }
    std::uint64_t columnsProduced() const { return columns_; }
    const AcfMatrix& matrix() const { return matrix_; }

private:
    double fs_;
    int fMin_, fMax_, fStep_;
    std::size_t numFreq_;
    RingBuffer<double> ring_;
    AcfMatrix matrix_;
    std::vector<double> column_;
    std::vector<int> N_, k_;
    std::uint64_t columns_ = 0;
};
