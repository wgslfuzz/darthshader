Building the fuzzing target `wgslfuzz` works as follows:

```bash
git clone https://github.com/WebKit/WebKit.git
cd WebKit
git checkout ad13d16
git apply webkit_ad13d16.diff
WEBKIT_OUTPUTDIR=build ./Tools/Scripts/build-jsc --jsc-only --debug --cmakeargs="-DENABLE_STATIC_JSC=ON -DCMAKE_C_COMPILER='/path/to/afl-clang-fast' -DCMAKE_CXX_COMPILER='/path/to/afl-clang-fast++' -DCMAKE_CXX_FLAGS='-O3 -lrt'"
```
