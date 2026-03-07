FROM ubuntu:24.04 AS build

RUN apt-get update && apt-get install -y \
    git cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/ggml-org/llama.cpp.git /llama.cpp

WORKDIR /llama.cpp
RUN cmake -B build -DGGML_METAL=OFF -DBUILD_SHARED_LIBS=OFF && \
    cmake --build build --config Release -j$(nproc) --target llama-server

FROM ubuntu:24.04

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY --from=build /llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

ENTRYPOINT ["llama-server"]
