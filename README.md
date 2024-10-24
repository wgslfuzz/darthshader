# DarthShader

DarthShader is a fuzzer for fuzzing the WebGPU shader compilation pipeline in web browsers. To this end, the fuzzer uses a combination of code generation and mutation to produce WebGPU shaders (wgsl). These wgsl shaders are then processed by a shader translator (the front-end). In particular, we support the shader translators of Chrome (tint), Firefox (naga), and Safari (wgslc). These components translate the input shader into a representation native to the respective GPU stack of the host operating system. For example, on Windows the WebGPU shaders are translated to HLSL. Once this native representation has been generated, shaders are then further processed by the respective graphics stack (the back-end). On Windows, HLSL shaders are consumed by dxcompiler, the shader compiler of DirectX12. DarthShader also supports testing of this component: First, WebGPU shaders are translated with tint to HLSL. Next, the HLSL output is fed into dxcompiler. This two-stage approach replicates the shader translation pipeline and hence ensures that crashes in dxcompiler are reachable from the web.

## DarthShader Command Line Arguments

The usage of DarthShader is best explained by its `--help` switch:

```console
Usage: darthshader [OPTIONS] --output <out> <exec> [-- [arguments]...]

Arguments:
  <exec>          The instrumented binary we want to fuzz
  [arguments]...

Options:
  -o, --output <out>              The directory to place finds in ('corpus')
  -i, --input <in>                The directory to read initial inputs from ('seeds')
  -l, --logfile <logfile>         Duplicates all output to this file [default: libafl.log]
  -t, --timeout <timeout>         Timeout for each individual execution, in milliseconds [default: 1200]
  -d, --debug-child               If not set, the child's stdout and stderror will be redirected to /dev/null
      --iterations <iterations>   Number of iterations to fuzz
      --config <generatorconfig>  The .yaml generator config file
      --seed <generatorseed>      Seed for the generation process
      --discard-crashes           Do not store crashes but explore for coverage only
  -h, --help                      Print help
  -V, --version                   Print version
```

The most important arguments are:
- The `exec` argument specifies the fuzzing harness executable.
- The `--output` folder contains two sub-directories. There is the `queue` file which contains all samples contributing new coverage. In addition, there is a `crashes` directory containing, as the name implies, all crashes found during the fuzzing campaign. Note that the samples are serialized in `.ron` files. Converting them to `.wgsl` files is possible using the lifter. This utility program is build alongside the fuzzer.
- The `--input` folder will be traversed recursively for all shader files for importing them.

## Crash and Queue Analysis
The fuzzer will write crashes and samples increasing coverage to the `crashes/` and `queue/` folder, respectively. Those `.ron` files are a serialized version of the fuzzers internal representation, they're *not* wgsl files. When investigating crashes (or queue files), one needs to convert the internal representation to a wgsl file. To this end, the fuzzer ships a lifter, which converts the .ron files to .wgsl.
Standalone reproduction of crashes follows a general pattern. First, the .ron file must be lifted to a .wgsl, as described just above. The next step depends on whether we're dealing with a front-end translator crash (e.g., tint, naga, or wgslc) or a backend compiler crash (e.g., dxcompiler). All front-end translator support standalone invocation. There are 2 things to keep in mind. First, specify the same command line parameters as the harness + use the same set of sanitizers. Otherwise, the crash might not reproduce. Second, depending on the front-end, an entrypoint must be specified. The fuzzing harness sequentially compiles each entrypoint. This behavior eases mutations, but can result in shaders with multiple entrypoints. Attempt to compile all of the entrypoints, at least one of them *should* provoke a crash.
For backed compiler crashes, you first need to convert the wgsl file to the format supported by the backend compiler. As an example, a shader translated with tint and later compiled with dxc can be translated as follows: `./tint shader.wgsl -o shader.hlsl`. Next, the .hlsl shader can be compiled with dxc. Again, make sure to use the same sanitizer flags and runtime options. E.g., `./dxc-3.7 -T cs_6_2 -HV 2018 shader.hlsl`.

### cs\_6\_2, ps\_6\_3, whaaaat?
The backend translator dxc requires you to pass the type of shader you're translating as well as a shader version. The type of shader depends on the entrypoint. @compute/@vertex/@fragment correspond to cs_/vs_/ps_, respectively. The shader version should be between 6.2 and 6.6. The fuzzer defaults to 6.6. As an example, a vertex shader of version 6.6 results in `-T vs_6_6`.

### Minimizing Crashes
Some of the findings might be rather large in size and reducing them manually can be a tedious process. Luckily, the compiler testing community already build all the tooling we need. The `creduce` tool, originally developed for minimizing C compiler tests, features a `--not-c` mode which allows reducing testcases written in languages other than C. Its interface might take a moment to get accustomed to, but it works quite well. Assume we have a .wgsl file which triggers an ASAN violation in dxc. We'd like to reduce the testcase as long as it continues to trigger an ASAN violation. The `creduce` script would look something like (save as `interestingness.sh`):

```bash
#!/bin/bash

export ASAN_OPTIONS=abort_on_error=1,detect_leaks=0
rm standalone.hlsl
/path/to/tint pathToCrash.wgsl -o standalone.hlsl;
STDE=$((/path/to/dxc-3.7 -T cs_6_6 -opt-disable structurize-loop-exits-for-unroll /Gis /Zpr /enable-16bit-types -HV 2018 standalone.hlsl) 2>&1)
echo $STDE
if [[ $STDE == *"heap-use-after-free"* ]]; then
  exit 0
fi
if [[ $STDE == *"heap-buffer-overflow"* ]]; then
  exit 0
fi
if [[ $STDE == *"SEGV on"* ]]; then
  exit 0
fi
exit 1
```

We can now invoke `creduce --not-c ./interestingness.sh pathToCrash.wgsl` which should reduce the wgsl file quite substantially.

## Further Components

### ****Seeds****
The fuzzer supports importing seeds from in the following formats: wgsl, spv, and glsl. In our evaluation (and as a good starting point), we use the [`dawn`](https://dawn.googlesource.com/dawn) repository as source, as it contains thousands of shaders.

### ****Compiler pass****
The fuzzing targets are instrumented with a variant of the well-known AFL++ instrumentation. More precisely, the only difference is that instead of collecting hit-counts, we only collect information about whether an edge was hit (or not). Please note that the AFL++ forkserver was recently updated. The libAFL version used by DarthShader is incompatible with these changes. Hence building with an older version of AFL++ (e.g., 4.10c) is required.

### ****Targets****
DarthShader currently provided fuzzing harnesses for 4 different shader compilers. These are 3 front-end translators used directly in browsers (tint, naga, wgslc) and the back-end compiler of DirectX, dxcompiler. For each of the fuzzing targets, the repository contains patches that add an AFL forkserver to the target.

#### ****tint****
This is the shader compiler of Chrome, supporting compilation from WGSL to the HLSL (Windows), SPIR-V (Linux & Android), and Metal (OSX). Our fuzzing harness for tint translates a single WGSL shader to each of the three target languages.

#### ****dxcompiler****
This is the DirectX shader compiler taking HLSL as input and producing an output format based on LLVM IR. Our fuzzing harness first translates WGSL shaders to HLSL via tint and subsequently passes the HLSL code to dxc, our setup hence replicates browser usage.

#### ****naga****
This is the shader compiler of Firefox supporting compilation from WGSL to HLSL, SPIR-V, and Metal. Analogue to the tint harness, our naga harness translates a single WGSL shader to HLSL, SPIR-V, and Metal.

#### ****wgslc****
The shader compiler of Safari translating WGSL to Metal. No other output languages are supported.
