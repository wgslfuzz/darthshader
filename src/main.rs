#![feature(variant_count)]
#![feature(lazy_cell)]
#![deny(
    clippy::correctness,
    clippy::cast_possible_wrap,
    unused_lifetimes,
    unused_unsafe,
    single_use_lifetimes,
    missing_debug_implementations
)]
#![recursion_limit = "256"]

extern crate link_cplusplus;

use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use core::{cell::RefCell, time::Duration};
use glob::glob;
use std::{
    env,
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
    process,
};

use clap::{Arg, ArgAction, Command};
use libafl::{
    corpus::{Corpus, InMemoryOnDiskCorpus, OnDiskCorpus},
    events::SimpleEventManager,
    executors::forkserver::{ForkserverExecutor, TimeoutForkserverExecutor},
    feedback_and_fast, feedback_or,
    feedbacks::{ConstFeedback, CrashFeedback, MaxMapFeedback, TimeFeedback},
    fuzzer::{Fuzzer, StdFuzzer},
    inputs::Input,
    monitors::SimpleMonitor,
    mutators::{StdMOptMutator, Tokens},
    observers::{HitcountsMapObserver, StdMapObserver, TimeObserver},
    prelude::{forkserver::HasForkserver, CorpusId},
    schedulers::{
        powersched::PowerSchedule, IndexesLenTimeMinimizerScheduler, StdWeightedScheduler,
    },
    stages::{calibrate::CalibrationStage, power::StdPowerMutationalStage, IfStage},
    state::{HasCorpus, HasMetadata, HasRand, StdState},
    Error,
};
use libafl_bolts::{
    current_nanos, current_time,
    rands::{Rand, StdRand},
    shmem::{unix_shmem::UnixShMem, ShMem, ShMemProvider, UnixShMemProvider},
    tuples::tuple_list,
    AsMutSlice,
};

use nix::sys::signal::Signal;

mod ast;
mod exit;
mod generator;
mod ir;
mod ladder;
mod layeredinput;
mod minimizer;
mod randomext;

use crate::generator::{GeneratorConfig, IRGenerator};
use crate::{
    ast::mutate::ast_mutations,
    exit::{ExitFeedback, ExitObserver},
    ir::mutate::ir_mutations,
    ladder::LadderStage,
    minimizer::LayeredMinimizerStage,
};
use layeredinput::LayeredInput;

pub fn main() {
    let res = Command::new(env!("CARGO_PKG_NAME"))
        .version(env!("CARGO_PKG_VERSION"))
        .author("Lukas Bernhard")
        .about("Darthshader fuzzer")
        .arg(
            Arg::new("out")
                .short('o')
                .long("output")
                .required(true)
                .help("The directory to place finds in ('corpus')"),
        )
        .arg(
            Arg::new("in")
                .short('i')
                .long("input")
                .value_parser(clap::value_parser!(PathBuf))
                .help("The directory to read initial inputs from ('seeds')"),
        )
        .arg(
            Arg::new("logfile")
                .short('l')
                .long("logfile")
                .help("Duplicates all output to this file")
                .default_value("libafl.log"),
        )
        .arg(
            Arg::new("timeout")
                .short('t')
                .long("timeout")
                .help("Timeout for each individual execution, in milliseconds")
                .value_parser(clap::value_parser!(u64))
                .default_value("1200"),
        )
        .arg(
            Arg::new("exec")
                .help("The instrumented binary we want to fuzz")
                .required(true),
        )
        .arg(
            Arg::new("debug-child")
                .short('d')
                .long("debug-child")
                .help("If not set, the child's stdout and stderror will be redirected to /dev/null")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("iterations")
                .long("iterations")
                .help("Number of iterations to fuzz")
                .value_parser(clap::value_parser!(u64)),
        )
        .arg(
            Arg::new("generatorconfig")
                .long("config")
                .value_parser(clap::value_parser!(PathBuf))
                .help("The .yaml generator config file"),
        )
        .arg(
            Arg::new("generatorseed")
                .long("seed")
                .help("Seed for the generation process")
                .value_parser(clap::value_parser!(u64)),
        )
        .arg(
            Arg::new("discard-crashes")
                .long("discard-crashes")
                .help("Do not store crashes but explore for coverage only")
                .action(ArgAction::SetTrue),
        )
        .arg(Arg::new("arguments").num_args(0..).last(true))
        .get_matches();

    println!(
        "Workdir: {:?}",
        env::current_dir().unwrap().to_string_lossy().to_string()
    );

    let iters: Option<u64> = res.get_one("iterations").copied();
    let seed: Option<u64> = res.get_one("generatorseed").copied();

    let mut out_dir = PathBuf::from(
        res.get_one::<String>("out")
            .expect("The --output parameter is missing")
            .to_string(),
    );
    if fs::create_dir(&out_dir).is_err() {
        println!("Out dir at {:?} already exists.", &out_dir);
        if !out_dir.is_dir() {
            println!("Out dir at {:?} is not a valid directory!", &out_dir);
            return;
        }
    }
    let mut crashes = out_dir.clone();
    crashes.push("crashes");
    out_dir.push("queue");
    let corpus_dir = out_dir;

    let in_dir: Option<PathBuf> = res.get_one("in").cloned();
    if let Some(in_dir) = &in_dir {
        if !in_dir.is_dir() {
            println!("In dir at {:?} is not a valid directory!", &in_dir);
            return;
        }
    }

    let config = {
        let gen_config: Option<PathBuf> = res.get_one("generatorconfig").cloned();
        if let Some(gen_config) = gen_config {
            let Ok(file) = fs::File::open(&gen_config) else {
                println!("Cannot open generator config file {:?}", gen_config);
                return;
            };
            let config: Result<GeneratorConfig, _> = serde_yaml::from_reader(file);
            match config {
                Ok(config) => config,
                Err(err) => {
                    println!(
                        "Cannot parse generator config file {:?} {:?}",
                        gen_config, err
                    );
                    return;
                }
            }
        } else {
            GeneratorConfig::default()
        }
    };

    let logfile = PathBuf::from(res.get_one::<String>("logfile").unwrap().to_string());

    let timeout = Duration::from_millis(*res.get_one::<u64>("timeout").unwrap());

    let executable = res.get_one::<String>("exec").unwrap().to_string();

    let debug_child = res.get_flag("debug-child");

    let discard_crashes = res.get_flag("discard-crashes");

    let arguments = res
        .get_many::<String>("arguments")
        .map(|v| v.map(std::string::ToString::to_string).collect::<Vec<_>>())
        .unwrap_or_default();

    fuzz(
        corpus_dir,
        crashes,
        in_dir,
        &logfile,
        timeout,
        executable,
        debug_child,
        &arguments,
        iters,
        config,
        seed,
        discard_crashes,
    )
    .expect("An error occurred while fuzzing");
}

/// The actual fuzzer
fn fuzz(
    corpus_dir: PathBuf,
    objective_dir: PathBuf,
    seed_dir: Option<PathBuf>,
    logfile: &PathBuf,
    timeout: Duration,
    executable: String,
    debug_child: bool,
    arguments: &[String],
    iters: Option<u64>,
    config: GeneratorConfig,
    seed: Option<u64>,
    discard_crashes: bool,
) -> Result<(), Error> {
    const MAP_SIZE: usize = 2_621_440;

    let log = RefCell::new(OpenOptions::new().append(true).create(true).open(logfile)?);

    let monitor = SimpleMonitor::with_user_monitor(
        |s| {
            println!("{s}");
            writeln!(log.borrow_mut(), "{:?} {}", current_time(), s).unwrap();
        },
        true,
    );

    let mut mgr = SimpleEventManager::new(monitor);

    let mut shmem_provider = UnixShMemProvider::new().unwrap();

    let mut shmem = shmem_provider.new_shmem(MAP_SIZE).unwrap();
    let shmap_to_drop =
        UnixShMem::shmem_from_id_and_size(shmem.id(), shmem.len()).expect("Failed to reopen map");
    shmem.write_to_env("__AFL_SHM_ID").unwrap();
    let shmem_buf = shmem.as_mut_slice();
    std::env::set_var("AFL_MAP_SIZE", format!("{}", MAP_SIZE));

    let mut shexit = shmem_provider.new_shmem(0x1000).unwrap();
    let shexit_to_drop =
        UnixShMem::shmem_from_id_and_size(shexit.id(), shexit.len()).expect("Failed to reopen map");
    shexit.write_to_env("__LIBAFL_EXIT_ID").unwrap();

    let edges_observer =
        unsafe { HitcountsMapObserver::new(StdMapObserver::new("shared_mem", shmem_buf)) };

    let time_observer = TimeObserver::new("time");
    let exit_observer = ExitObserver::new("exit", shexit.as_mut_slice());

    let map_feedback = MaxMapFeedback::tracking(&edges_observer, true, true);

    let calibration = CalibrationStage::new(&map_feedback);
    let minimize = LayeredMinimizerStage::new(&map_feedback);
    let ladder = LadderStage::new();

    let mut feedback = feedback_or!(
        map_feedback,
        TimeFeedback::with_observer(&time_observer),
        ExitFeedback::with_observer(&exit_observer)
    );

    let mut objective =
        feedback_and_fast!(ConstFeedback::new(!discard_crashes), CrashFeedback::new());

    let mut state = StdState::new(
        StdRand::with_seed(current_nanos()),
        InMemoryOnDiskCorpus::<LayeredInput>::new(corpus_dir).unwrap(),
        OnDiskCorpus::new(objective_dir).unwrap(),
        &mut feedback,
        &mut objective,
    )
    .unwrap();

    println!("Let's fuzz :)");

    let mutator_ir = StdMOptMutator::new(&mut state, ir_mutations(), 2, 8)?;
    let mutator_ast = StdMOptMutator::new(&mut state, ast_mutations(), 2, 8)?;

    let cb_ast_mutate = |_fuzzer: &mut _,
                         _executor: &mut _,
                         state: &mut StdState<_, InMemoryOnDiskCorpus<_>, _, _>,
                         _event_manager: &mut _,
                         corpus_id: CorpusId|
     -> Result<bool, libafl::Error> {
        let corpus = state.corpus().get(corpus_id)?.borrow();
        let input = corpus.input().as_ref().unwrap();
        Ok(matches!(input, LayeredInput::Ast(_)))
    };
    let cb_ir_mutate = |_fuzzer: &mut _,
                        _executor: &mut _,
                        state: &mut StdState<_, InMemoryOnDiskCorpus<_>, _, _>,
                        _event_manager: &mut _,
                        corpus_id: CorpusId|
     -> Result<bool, libafl::Error> {
        let corpus = state.corpus().get(corpus_id)?.borrow();
        let input = corpus.input().as_ref().unwrap();
        Ok(matches!(input, LayeredInput::IR(_)))
    };
    let mutation_stage_ir = StdPowerMutationalStage::new(mutator_ir);
    let mutation_stage_ast = StdPowerMutationalStage::new(mutator_ast);
    let maybe_ir_mutate_stage = IfStage::new(cb_ir_mutate, tuple_list!(mutation_stage_ir));
    let maybe_ast_mutate_stage = IfStage::new(cb_ast_mutate, tuple_list!(mutation_stage_ast));

    let scheduler = IndexesLenTimeMinimizerScheduler::new(StdWeightedScheduler::with_schedule(
        &mut state,
        &edges_observer,
        Some(PowerSchedule::EXPLOIT),
    ));

    let mut fuzzer = StdFuzzer::new(scheduler, feedback, objective);

    let forkserver = ForkserverExecutor::builder()
        .program(executable)
        .env(
            "ASAN_OPTIONS",
            "symbolize=0,detect_leaks=0,malloc_context_size=0",
        )
        .debug_child(debug_child)
        .shmem_provider(&mut shmem_provider)
        .parse_afl_cmdline(arguments)
        .coverage_map_size(MAP_SIZE)
        .is_persistent(true)
        .is_deferred_frksrv(true)
        .build_dynamic_map(edges_observer, tuple_list!(time_observer, exit_observer))
        .unwrap();

    let shinput_to_drop = forkserver
        .shmem()
        .as_ref()
        .map(|shmem| UnixShMem::shmem_from_id_and_size(shmem.id(), shmem.len()));
    println!("Coverage map size: {:?}", forkserver.coverage_map_size());

    let mut executor = TimeoutForkserverExecutor::with_signal(forkserver, timeout, Signal::SIGKILL)
        .expect("Failed to create the executor.");

    std::mem::drop(shmap_to_drop);
    std::mem::drop(shexit_to_drop);
    std::mem::drop(shinput_to_drop);

    let tokens = {
        let s = include_str!("dictionary.txt");
        let v: Vec<Vec<u8>> = s.lines().map(|l| l.as_bytes().to_vec()).collect();
        let mut tokens = Tokens::new();
        tokens.add_tokens(v.iter());
        assert!(!tokens.is_empty());
        tokens
    };
    state.add_metadata(tokens);

    if let Some(mut seed_dir) = seed_dir {
        seed_dir.push("**/*");
        println!("{:?}", seed_dir);
        let seed_dir = seed_dir.as_path().to_str().unwrap();
        let mut seeds = Vec::new();
        for entry in glob(seed_dir).expect("Failed to traverse seed dir") {
            let entry = entry.expect("Failed to traverse seed dir");
            if !entry.is_file() {
                continue;
            }
            let size: u64 = std::fs::metadata(&entry)
                .expect("Failed to read glob pattern")
                .len();
            seeds.push((size, entry));
        }
        seeds.sort_by(|a, b| b.cmp(a));
        let seeds: Vec<PathBuf> = seeds.into_iter().map(|(_, p)| p).collect();
        // filter all paths which cannot be parsed successfully
        let seeds: Vec<PathBuf> = seeds
            .into_iter()
            .filter_map(|p| LayeredInput::from_file(&p).is_ok().then_some(p))
            .collect();

        state
            .load_initial_inputs_by_filenames(&mut fuzzer, &mut executor, &mut mgr, &seeds)
            .unwrap_or_else(|e| {
                println!("Failed to load initial corpus at {:?} {:?}", &seed_dir, e);
                process::exit(0);
            });
        println!("We imported {} inputs from disk.", state.corpus().count());
    }

    if let Some(seed) = seed {
        state.rand_mut().set_seed(seed);
    }

    let mut generator = IRGenerator::new(config);
    let mut corpus_size = state.corpus().count();
    loop {
        let _ = state.generate_initial_inputs(
            &mut fuzzer,
            &mut executor,
            &mut generator,
            &mut mgr,
            100,
        );
        if corpus_size == state.corpus().count() {
            break;
        } else {
            corpus_size = state.corpus().count();
        }
    }

    let mut stages = tuple_list!(
        minimize,
        calibration,
        ladder,
        maybe_ast_mutate_stage,
        maybe_ir_mutate_stage
    );

    match iters {
        None => {
            fuzzer.fuzz_loop(&mut stages, &mut executor, &mut state, &mut mgr)?;
        }
        Some(0) => (),
        Some(iters) => {
            fuzzer.fuzz_loop_for(&mut stages, &mut executor, &mut state, &mut mgr, iters)?;
        }
    }

    Ok(())
}
