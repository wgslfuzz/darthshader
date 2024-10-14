Building the fuzzing target `libdxcompiler.so` works as follows:

```bash
git clone https://github.com/microsoft/DirectXShaderCompiler.git
cd DirectXShaderCompiler
git submodule init
git submodule update

cd DirectXShaderCompiler
mkdir -p out/build
cd out/build
CC=/path/to/afl-clang-fast CXX=/path/to/afl-clang-fast++ cmake ../../ -C ../../cmake/caches/PredefinedParams.cmake -DCMAKE_BUILD_TYPE=Release -DDXC_DISABLE_ALLOCATOR_OVERRIDES=ON -DENABLE_SPIRV_CODEGEN=OFF -DSPIRV_BUILD_TESTS=OFF -DLLVM_USE_SANITIZER=Address -DLLVM_ENABLE_LTO=Off -DLLVM_ENABLE_ASSERTIONS=1 -G Ninja
ninja

# We now have libdxcompiler.so, which can be used with the dawn harness.
```
