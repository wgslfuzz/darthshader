#![allow(clippy::manual_strip)]
#[allow(unused_imports)]
use std::fs;
use std::{error::Error, fmt, path::{Path, PathBuf}, str::FromStr};
//use spirv_tools::val::{Validator, compiled::CompiledValidator};

/// Translate shaders to different formats.
#[derive(argh::FromArgs, Debug, Clone)]
struct Args {
    /// bitmask of the ValidationFlags to be used, use 0 to disable validation
    #[argh(option)]
    validate: Option<u8>,

    /// what policy to use for index bounds checking for arrays, vectors, and
    /// matrices.
    ///
    /// May be `Restrict` (force all indices in-bounds), `ReadZeroSkipWrite`
    /// (out-of-bounds indices read zeros, and don't write at all), or
    /// `Unchecked` (generate the simplest code, and whatever happens, happens)
    ///
    /// `Unchecked` is the default.
    #[argh(option)]
    index_bounds_check_policy: Option<BoundsCheckPolicyArg>,

    /// what policy to use for index bounds checking for arrays, vectors, and
    /// matrices, when they are stored in globals in the `storage` or `uniform`
    /// storage classes.
    ///
    /// Possible values are the same as for `index-bounds-check-policy`. If
    /// omitted, defaults to the index bounds check policy.
    #[argh(option)]
    buffer_bounds_check_policy: Option<BoundsCheckPolicyArg>,

    /// what policy to use for texture loads bounds checking.
    ///
    /// Possible values are the same as for `index-bounds-check-policy`. If
    /// omitted, defaults to the index bounds check policy.
    #[argh(option)]
    image_load_bounds_check_policy: Option<BoundsCheckPolicyArg>,

    /// what policy to use for texture stores bounds checking.
    ///
    /// Possible values are the same as for `index-bounds-check-policy`. If
    /// omitted, defaults to the index bounds check policy.
    #[argh(option)]
    image_store_bounds_check_policy: Option<BoundsCheckPolicyArg>,

    /// the shader entrypoint to use when compiling to GLSL
    #[argh(option)]
    entry_point: Option<String>,

    /// the shader model to use if targeting HLSL
    ///
    /// May be `50`, 51`, or `60`
    #[argh(option)]
    shader_model: Option<ShaderModelArg>,

    /// the metal version to use, for example, 1.0, 1.1, 1.2, etc.
    #[argh(option)]
    metal_version: Option<MslVersionArg>,

    /// if the selected frontends/backends support coordinate space conversions,
    /// disable them
    #[argh(switch)]
    keep_coordinate_space: bool,

    /// generate debug symbols, only works for spv-out for now
    #[argh(switch, short = 'g')]
    generate_debug_symbols: bool,

    /// compact the module's IR and revalidate.
    ///
    /// Output files will reflect the compacted IR. If you want to see the IR as
    /// it was before compaction, use the `--before-compaction` option.
    #[argh(switch)]
    compact: bool,

    /// write the module's IR before compaction to the given file.
    ///
    /// This implies `--compact`. Like any other output file, the filename
    /// extension determines the form in which the module is written.
    #[argh(option)]
    before_compaction: Option<String>,

    /// the input and output file
    #[argh(positional)]
    file: Option<PathBuf>,
}

/// Newtype so we can implement [`FromStr`] for `BoundsCheckPolicy`.
#[derive(Debug, Clone, Copy)]
struct BoundsCheckPolicyArg(naga::proc::BoundsCheckPolicy);

impl FromStr for BoundsCheckPolicyArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use naga::proc::BoundsCheckPolicy;
        Ok(Self(match s.to_lowercase().as_str() {
            "restrict" => BoundsCheckPolicy::Restrict,
            "readzeroskipwrite" => BoundsCheckPolicy::ReadZeroSkipWrite,
            "unchecked" => BoundsCheckPolicy::Unchecked,
            _ => {
                return Err(format!(
                    "Invalid value for --index-bounds-check-policy: {s}"
                ))
            }
        }))
    }
}

/// Newtype so we can implement [`FromStr`] for `ShaderModel`.
#[derive(Debug, Clone)]
struct ShaderModelArg(naga::back::hlsl::ShaderModel);

impl FromStr for ShaderModelArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use naga::back::hlsl::ShaderModel;
        Ok(Self(match s.to_lowercase().as_str() {
            "50" => ShaderModel::V5_0,
            "51" => ShaderModel::V5_1,
            "60" => ShaderModel::V6_0,
            _ => return Err(format!("Invalid value for --shader-model: {s}")),
        }))
    }
}

/// Newtype so we can implement [`FromStr`] for a Metal Language Version.
#[derive(Clone, Debug)]
struct MslVersionArg((u8, u8));

impl FromStr for MslVersionArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut iter = s.split('.');

        let check_value = |iter: &mut core::str::Split<_>| {
            iter.next()
                .ok_or_else(|| format!("Invalid value for --metal-version: {s}"))?
                .parse::<u8>()
                .map_err(|err| format!("Invalid value for --metal-version: '{s}': {err}"))
        };

        let major = check_value(&mut iter)?;
        let minor = check_value(&mut iter)?;

        Ok(Self((major, minor)))
    }
}

#[derive(Default)]
struct Parameters<'a> {
    validation_flags: naga::valid::ValidationFlags,
    bounds_check_policies: naga::proc::BoundsCheckPolicies,
    entry_point: Option<String>,
    keep_coordinate_space: bool,
    spv_out: naga::back::spv::Options<'a>,
    msl: naga::back::msl::Options,
    hlsl: naga::back::hlsl::Options,
}

trait PrettyResult {
    type Target;
    fn unwrap_pretty(self) -> Self::Target;
}

fn print_err(error: &dyn Error) {
    eprint!("{error}");

    let mut e = error.source();
    if e.is_some() {
        eprintln!(": ");
    } else {
        eprintln!();
    }

    while let Some(source) = e {
        eprintln!("\t{source}");
        e = source.source();
    }
}

impl<T, E: Error> PrettyResult for Result<T, E> {
    type Target = T;
    fn unwrap_pretty(self) -> T {
        match self {
            Result::Ok(value) => value,
            Result::Err(error) => {
                print_err(&error);
                std::process::exit(1);
            }
        }
    }
}

fn main() {
    env_logger::init();
    let mut fallback: [u8; 2] = Default::default();

    // Parse commandline arguments
    let args: Args = argh::from_env();

    let exit_code = match std::env::var("__LIBAFL_EXIT_ID") {
         Ok(id) => {
             let key = id.parse::<libc::key_t>().expect("__LIBAFL_EXIT_ID invalid");
             unsafe {
                 let ptr = libc::shmat(key, std::ptr::null(), 0);
                 if ptr.is_null() {
                     panic!("Failed to attach __LIBAFL_EXIT_ID");
                 }
                 let ptr = ptr as *mut libc::c_uchar;
                 std::slice::from_raw_parts_mut(ptr, 2)
             }
         },
         Err(_) => &mut fallback
    };

    #[cfg(fuzzing)] {
        afl::fuzz!(|data: &[u8]| {
            if let Err(e) = run(data, &args) {
                print_err(e.as_ref());
                exit_code[0] = 1;
                exit_code[1] = 1;
            }
            else {
                exit_code[0] = 0;
                exit_code[1] = 1;
            }
        });
    }
    #[cfg(not(fuzzing))] {
        let file_path = args.file.as_ref().expect("must have file set if not fuzzing");
        let input = fs::read(file_path).expect("failed to read file");
        if let Err(e) = run(input.as_slice(), &args) {
            print_err(e.as_ref());
            exit_code[0] = 1;
            exit_code[1] = 1;
        }
        else {
            exit_code[0] = 0;
            exit_code[1] = 1;
        }
    }
}

/// Error type for the CLI
#[derive(Debug, Clone)]
struct CliError(&'static str);
impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for CliError {}

fn run(data: &[u8], args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize default parameters
    //TODO: read the parameters from RON?
    let mut params = Parameters::default();

    // Update parameters from commandline arguments
    if let Some(bits) = args.validate {
        params.validation_flags = naga::valid::ValidationFlags::from_bits(bits)
            .ok_or(CliError("Invalid validation flags"))?;
    }
    if let Some(policy) = args.index_bounds_check_policy {
        params.bounds_check_policies.index = policy.0;
    }
    params.bounds_check_policies.buffer = match args.buffer_bounds_check_policy {
        Some(arg) => arg.0,
        None => params.bounds_check_policies.index,
    };
    params.bounds_check_policies.image_load = match args.image_load_bounds_check_policy {
        Some(arg) => arg.0,
        None => params.bounds_check_policies.index,
    };
    params.bounds_check_policies.image_store = match args.image_store_bounds_check_policy {
        Some(arg) => arg.0,
        None => params.bounds_check_policies.index,
    };

    params.entry_point = args.entry_point.clone();
    if let Some(ref model) = args.shader_model {
        params.hlsl.shader_model = model.0;
    }
    if let Some(ref version) = args.metal_version {
        params.msl.lang_version = version.0;
    }
    params.keep_coordinate_space = args.keep_coordinate_space;

    params.spv_out.bounds_check_policies = params.bounds_check_policies;
    params.spv_out.flags.set(
        naga::back::spv::WriterFlags::ADJUST_COORDINATE_SPACE,
        !params.keep_coordinate_space,
    );

    let input_path = Path::new("fuzz.wgsl");
    let input = data;

    let Parsed {
        mut module,
        input_text,
    } = parse_input(input.to_vec())?;

    // Include debugging information if requested.
    if args.generate_debug_symbols {
        if let Some(ref input_text) = input_text {
            params
                .spv_out
                .flags
                .set(naga::back::spv::WriterFlags::DEBUG, true);
            params.spv_out.debug_info = Some(naga::back::spv::DebugInfo {
                source_code: input_text,
                file_name: input_path,
            })
        } else {
            eprintln!(
                "warning: `--generate-debug-symbols` was passed, \
                       but input is not human-readable: {}",
                input_path.display()
            );
        }
    }

    //let output_paths = args.files.get(1..).unwrap_or(&[]);

    //// Decide which capabilities our output formats can support.
    let validation_caps = naga::valid::Capabilities::all();
    //    output_paths
    //        .iter()
    //        .fold(naga::valid::Capabilities::all(), |caps, path| {
    //            use naga::valid::Capabilities as C;
    //            let missing = match Path::new(path).extension().and_then(|ex| ex.to_str()) {
    //                Some("wgsl") => C::CLIP_DISTANCE | C::CULL_DISTANCE,
    //                Some("metal") => C::CULL_DISTANCE,
    //                _ => C::empty(),
    //            };
    //            caps & !missing
    //        });

    // Validate the IR before compaction.
    let info = match naga::valid::Validator::new(params.validation_flags, validation_caps)
        .validate(&module)
    {
        Ok(info) => Some(info),
        Err(error) => {
            // Validation failure is not fatal. Just report the error.
            if let Some(input) = &input_text {
                let filename = input_path.file_name().and_then(std::ffi::OsStr::to_str);
                emit_annotated_error(&error, filename.unwrap_or("input"), input);
            }
            print_err(&error);
            None
        }
    };

    // Compact the module, if requested.
    let info = if args.compact || args.before_compaction.is_some() {
        // Compact only if validation succeeded. Otherwise, compaction may panic.
        if info.is_some() {
            // Write out the module state before compaction, if requested.
            if args.before_compaction.is_some() {
                write_output(&module, &info, &params)?;
            }

            naga::compact::compact(&mut module);

            // Re-validate the IR after compaction.
            match naga::valid::Validator::new(params.validation_flags, validation_caps)
                .validate(&module)
            {
                Ok(info) => Some(info),
                Err(error) => {
                    // Validation failure is not fatal. Just report the error.
                    eprintln!("Error validating compacted module:");
                    if let Some(input) = &input_text {
                        let filename = input_path.file_name().and_then(std::ffi::OsStr::to_str);
                        emit_annotated_error(&error, filename.unwrap_or("input"), input);
                    }
                    print_err(&error);
                    None
                }
            }
        } else {
            eprintln!("Skipping compaction due to validation failure.");
            None
        }
    } else {
        info
    };

    write_output(&module, &info, &params)?;

    Ok(())
}

struct Parsed {
    module: naga::Module,
    input_text: Option<String>,
}

fn parse_input(
    input: Vec<u8>,
) -> Result<Parsed, Box<dyn std::error::Error>> {
     let input = String::from_utf8(input)?;
     let result = naga::front::wgsl::parse_str(&input);
     let (module, input_text) = match result {
         Ok(v) => (v, Some(input)),
         Err(ref e) => {
             let message = format!(
                 "Could not parse WGSL {}:\n", e
             );
             return Err(message.into());
         }
     };

    Ok(Parsed { module, input_text })
}

fn write_output(
    module: &naga::Module,
    info: &Option<naga::valid::ModuleInfo>,
    params: &Parameters,
) -> Result<(), Box<dyn std::error::Error>> {
    use naga::back::msl;
    use naga::back::spv;
    use naga::back::hlsl;

    let mut options = params.msl.clone();
    options.bounds_check_policies = params.bounds_check_policies;

    let pipeline_options = msl::PipelineOptions::default();
    msl::write_string(
        module,
        info.as_ref().ok_or(CliError(
            "Generating metal output requires validation to \
             succeed, and it failed in a previous step",
        ))?,
        &options,
        &pipeline_options,
    )
    .unwrap_pretty();

    let pipeline_options_owned;
    let pipeline_options = match params.entry_point {
        Some(ref name) => {
            let ep_index = module
                .entry_points
                .iter()
                .position(|ep| ep.name == *name)
                .expect("Unable to find the entry point");
            pipeline_options_owned = spv::PipelineOptions {
                entry_point: name.clone(),
                shader_stage: module.entry_points[ep_index].stage,
            };
            Some(&pipeline_options_owned)
        }
        None => None,
    };

    spv::write_vec(
        module,
        info.as_ref().ok_or(CliError(
            "Generating SPIR-V output requires validation to \
             succeed, and it failed in a previous step",
        ))?,
        &params.spv_out,
        pipeline_options,
    )
    .unwrap_pretty();

    //// changes for naga spirv testing
    //let blob = spv::write_vec(
    //    module,
    //    info.as_ref().ok_or(CliError(
    //        "Generating SPIR-V output requires validation to \
    //         succeed, and it failed in a previous step",
    //    ))?,
    //    &params.spv_out,
    //    pipeline_options,
    //)
    //.unwrap_pretty();
    //CompiledValidator::default().validate(blob, None).unwrap();

    let mut buffer = String::new();
    let mut writer = hlsl::Writer::new(&mut buffer, &params.hlsl);
    writer
        .write(
            module,
            info.as_ref().ok_or(CliError(
                "Generating hlsl output requires validation to \
                 succeed, and it failed in a previous step",
            ))?,
        )
        .unwrap_pretty();

    Ok(())
}

use codespan_reporting::{
    diagnostic::{Diagnostic, Label},
    files::SimpleFile,
    term::{
        self,
        termcolor::{ColorChoice, StandardStream},
    },
};
use naga::WithSpan;

pub fn emit_annotated_error<E: Error>(ann_err: &WithSpan<E>, filename: &str, source: &str) {
    let files = SimpleFile::new(filename, source);
    let config = codespan_reporting::term::Config::default();
    let writer = StandardStream::stderr(ColorChoice::Auto);

    let diagnostic = Diagnostic::error().with_labels(
        ann_err
            .spans()
            .map(|(span, desc)| {
                Label::primary((), span.to_range().unwrap()).with_message(desc.to_owned())
            })
            .collect(),
    );

    term::emit(&mut writer.lock(), &config, &files, &diagnostic).expect("cannot write error");
}
