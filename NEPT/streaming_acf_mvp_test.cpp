// streaming_acf_mvp_test.cpp
// -----------------------------------------------------------------------------
// GoogleTest unit tests for the streaming multi-rate ACF MVP.
//
// Expects the classes (RingBuffer, ISampleSubscriber, SignalGenerator,
// AcfMatrix, AcfProcessor) to be declared in "streaming_acf_mvp.h".
//
// Build (see Makefile):
//   g++ -std=c++17 -O2 -pthread streaming_acf_mvp_test.cpp -lgtest -lgtest_main -o acf_mvp_test
//   ./acf_mvp_test
// -----------------------------------------------------------------------------

#include "streaming_acf_mvp.h"

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>

namespace {

// Push a sequence of samples into a processor (index 0..n-1).
void feed(AcfProcessor& p, std::initializer_list<double> xs) {
    std::uint64_t i = 0;
    for (double x : xs) p.onSample(i++, x);
}

// Subscriber that records everything it receives (for pub/sub tests).
struct RecordingSubscriber : ISampleSubscriber {
    std::vector<std::uint64_t> indices;
    std::vector<double> values;
    void onSample(std::uint64_t index, double value) override {
        indices.push_back(index);
        values.push_back(value);
    }
};

}  // namespace

// ------------------------------- RingBuffer ----------------------------------

TEST(RingBuffer, StartsEmpty) {
    RingBuffer<double> rb(4);
    EXPECT_EQ(rb.size(), 0u);
    EXPECT_EQ(rb.capacity(), 4u);
}

TEST(RingBuffer, NewestFirstOrdering) {
    RingBuffer<double> rb(4);
    rb.push(1);
    rb.push(2);
    rb.push(3);
    EXPECT_EQ(rb.size(), 3u);
    EXPECT_DOUBLE_EQ(rb.at(0), 3.0);  // newest
    EXPECT_DOUBLE_EQ(rb.at(1), 2.0);
    EXPECT_DOUBLE_EQ(rb.at(2), 1.0);  // oldest available
}

TEST(RingBuffer, OverwritesOldestWhenFull) {
    RingBuffer<double> rb(4);
    for (double v : {1.0, 2.0, 3.0, 4.0, 5.0}) rb.push(v);  // 1 falls off
    EXPECT_EQ(rb.size(), 4u);          // capped at capacity
    EXPECT_DOUBLE_EQ(rb.at(0), 5.0);   // newest
    EXPECT_DOUBLE_EQ(rb.at(3), 2.0);   // oldest still present
}

// ------------------------------- AcfMatrix -----------------------------------

TEST(AcfMatrix, ReportsDimensions) {
    AcfMatrix m(500, 2000);
    EXPECT_EQ(m.rows(), 500u);
    EXPECT_EQ(m.cols(), 2000u);
    EXPECT_EQ(m.filled(), 0u);
}

TEST(AcfMatrix, WriteColumnUpdatesLatest) {
    AcfMatrix m(2, 3);
    m.writeColumn({1.0, 2.0});
    m.writeColumn({3.0, 4.0});
    EXPECT_EQ(m.filled(), 2u);
    EXPECT_DOUBLE_EQ(m.latestColumn()[0], 3.0);
    EXPECT_DOUBLE_EQ(m.latestColumn()[1], 4.0);
}

TEST(AcfMatrix, CircularWrapKeepsLatestAndCapsFilled) {
    AcfMatrix m(2, 3);
    m.writeColumn({1.0, 2.0});
    m.writeColumn({3.0, 4.0});
    m.writeColumn({5.0, 6.0});
    m.writeColumn({7.0, 8.0});  // wraps over column 0
    EXPECT_EQ(m.filled(), 3u);  // never exceeds cols()
    EXPECT_DOUBLE_EQ(m.latestColumn()[0], 7.0);
    EXPECT_DOUBLE_EQ(m.latestColumn()[1], 8.0);
}

// ------------------------------ AcfProcessor ---------------------------------

TEST(AcfProcessor, ChannelCountAndMatrixShape) {
    AcfProcessor p(2000, 1, 500, 1, 2000);
    EXPECT_EQ(p.numFreq(), 500u);
    EXPECT_EQ(p.matrix().rows(), 500u);
    EXPECT_EQ(p.matrix().cols(), 2000u);
}

TEST(AcfProcessor, FrequencyMapping) {
    AcfProcessor p(2000, 1, 500, 1, 2000);
    EXPECT_EQ(p.freqOf(0), 1);
    EXPECT_EQ(p.freqOf(99), 100);
    EXPECT_EQ(p.freqOf(499), 500);
}

TEST(AcfProcessor, ColumnsProducedTracksSamples) {
    AcfProcessor p(8, 1, 2, 1, 4);
    feed(p, {1, 2, 3, 4, 5});
    EXPECT_EQ(p.columnsProduced(), 5u);
}

// Hand-computed full-window ACF with Fs=8 and the sequence 1..8.
//   f=1: N=8, k=2 -> sum_{i=2}^{7} g[i]*g[i-2] = 133
//   f=2: N=4, k=1 -> sum_{i=1}^{3} g[i]*g[i-1] = 128
TEST(AcfProcessor, KnownValueAcf) {
    AcfProcessor p(8, 1, 2, 1, 4);
    feed(p, {1, 2, 3, 4, 5, 6, 7, 8});
    const double* col = p.matrix().latestColumn();
    EXPECT_DOUBLE_EQ(col[0], 133.0);  // f = 1 Hz
    EXPECT_DOUBLE_EQ(col[1], 128.0);  // f = 2 Hz
}

TEST(AcfProcessor, ComputeAcfDirect) {
    AcfProcessor p(8, 1, 2, 1, 4);
    feed(p, {1, 2, 3, 4, 5, 6, 7, 8});
    EXPECT_DOUBLE_EQ(p.computeAcf(8, 2, 8), 133.0);
    EXPECT_DOUBLE_EQ(p.computeAcf(4, 1, 4), 128.0);
}

// During warm-up the window is not yet full, so terms whose history is
// missing are skipped. With 3 samples in an N=8, k=2 window only the
// newest*oldest term survives: at(0) * at(2) = 30 * 10.
TEST(AcfProcessor, WarmupSkipsMissingHistory) {
    AcfProcessor p(8, 1, 1, 1, 4);
    feed(p, {10, 20, 30});
    EXPECT_DOUBLE_EQ(p.computeAcf(8, 2, 3), 300.0);
}

// A constant unit signal makes every product 1, so the full-window ACF equals
// the term count (N - k). This verifies N = Fs/f and k = N/4 per channel
// through the public interface.
TEST(AcfProcessor, ConstantSignalCountsTerms) {
    AcfProcessor p(2000, 1, 500, 1, 2000);
    for (int i = 0; i < 2000; ++i) p.onSample(static_cast<std::uint64_t>(i), 1.0);
    const double* col = p.matrix().latestColumn();
    EXPECT_DOUBLE_EQ(col[0], 1500.0);   // f=1:   N=2000, k=500 -> 1500
    EXPECT_DOUBLE_EQ(col[99], 15.0);    // f=100: N=20,   k=5   -> 15
    EXPECT_DOUBLE_EQ(col[499], 3.0);    // f=500: N=4,    k=1   -> 3
}

// ---------------------------- SignalGenerator --------------------------------

TEST(SignalGenerator, PublishesToAllSubscribers) {
    RecordingSubscriber a, b;
    SignalGenerator g(2000, 50.0, 1.0, 0.0);
    g.subscribe(&a);
    g.subscribe(&b);
    g.run(16);
    EXPECT_EQ(a.values.size(), 16u);
    EXPECT_EQ(b.values.size(), 16u);
    EXPECT_EQ(a.indices.front(), 0u);
    EXPECT_EQ(a.indices.back(), 15u);
}

TEST(SignalGenerator, ProducesExpectedSineSamples) {
    RecordingSubscriber rec;
    SignalGenerator g(8, 1.0, 1.0, 0.0);  // Fs=8, 1 Hz -> sin(pi*n/4)
    g.subscribe(&rec);
    g.run(8);
    ASSERT_EQ(rec.values.size(), 8u);
    EXPECT_NEAR(rec.values[0], 0.0, 1e-12);  // sin(0)
    EXPECT_NEAR(rec.values[2], 1.0, 1e-12);  // sin(pi/2)
    EXPECT_NEAR(rec.values[4], 0.0, 1e-12);  // sin(pi)
    EXPECT_NEAR(rec.values[6], -1.0, 1e-12); // sin(3pi/2)
}
