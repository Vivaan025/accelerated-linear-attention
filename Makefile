.PHONY: all build clean run throughput

# Detect OS
ifeq ($(OS),Windows_NT)
    RM_CMD = if exist build rmdir /s /q build
    EXE_EXT = .exe
    BUILD_DIR = build
    CMAKE_GENERATOR = -G "Visual Studio 17 2022" -A x64
    BUILD_CONFIG = --config Release
    BIN_PATH = $(BUILD_DIR)
else
    RM_CMD = rm -rf build
    EXE_EXT =
    BUILD_DIR = build
    CMAKE_GENERATOR =
    BUILD_CONFIG =
    BIN_PATH = $(BUILD_DIR)
endif

all: build

build:
	@cmake -S . -B $(BUILD_DIR) $(CMAKE_GENERATOR)
	@cmake --build $(BUILD_DIR) $(BUILD_CONFIG)

clean:
	@$(RM_CMD)

run: build
	@$(BIN_PATH)/benchmark$(EXE_EXT)

throughput: build
	@$(BIN_PATH)/benchmark_throughput$(EXE_EXT)
