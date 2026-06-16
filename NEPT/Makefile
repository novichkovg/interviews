CXX        ?= g++
CXXFLAGS   ?= -std=c++17 -O2 -Wall -Wextra -pthread
LDFLAGS    ?= -pthread

# GoogleTest libraries. Override if installed elsewhere, e.g.:
#   make test GTEST_LIBS="$(pkg-config --libs gtest_main)"
GTEST_LIBS ?= -lgtest -lgtest_main

HEADER      := streaming_acf_mvp.h

TARGET      := acf_mvp
SRC         := streaming_acf_mvp.cpp

TEST_TARGET := acf_mvp_test
TEST_SRC    := streaming_acf_mvp_test.cpp

.PHONY: all run test clean

all: $(TARGET)

$(TARGET): $(SRC) $(HEADER)
	$(CXX) $(CXXFLAGS) $(SRC) -o $@ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) 100 1.5 0.0

# Build and run the GoogleTest suite.
test: $(TEST_TARGET)
	./$(TEST_TARGET)

$(TEST_TARGET): $(TEST_SRC) $(HEADER)
	$(CXX) $(CXXFLAGS) $(TEST_SRC) -o $@ $(LDFLAGS) $(GTEST_LIBS)

clean:
	rm -f $(TARGET) $(TEST_TARGET)
