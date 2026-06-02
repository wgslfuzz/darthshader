use core::{fmt::Debug, marker::PhantomData};

use libafl_bolts::{
    tuples::{Handle, MatchNameRef, NamedTuple},
    AsSlice, HasLen,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use libafl::{
    common::{HasMetadata, HasNamedMetadata},
    corpus::HasCurrentCorpusId,
    events::{EventFirer, LogSeverity},
    executors::{Executor, ExitKind, HasObservers},
    feedbacks::map::MapNoveltiesMetadata,
    fuzzer::Evaluator,
    mutators::{MutationResult, Mutator, MutatorsTuple},
    observers::{MapObserver, ObserversTuple},
    stages::{Restartable, Stage},
    state::{HasClientPerfMonitor, HasCorpus, HasCurrentTestcase, HasExecutions, HasRand},
    Error,
};

use crate::{
    ast::{mutate::ASTDeleteMutator, Ast},
    ir::{funcext::FuncExt, iter::IterFuncs, minimizer::ir_minimizers},
    layeredinput::LayeredInput,
    randomext::RandExt,
};

#[derive(Clone, Debug)]
pub struct LayeredMinimizerStage<O, MO, OT, S> {
    map_observer_handle: Handle<O>,
    phantom: PhantomData<(MO, OT, S)>,
}

impl<O, MO, OT, S> LayeredMinimizerStage<O, MO, OT, S> {
    /// Create a new [`LayeredMinimizerStage`].
    #[must_use]
    pub fn new(map_observer_handle: Handle<O>) -> Self {
        Self {
            map_observer_handle,
            phantom: PhantomData,
        }
    }
}

impl<O, MO, OT, S> LayeredMinimizerStage<O, MO, OT, S>
where
    O: AsRef<MO>,
    MO: MapObserver,
    OT: MatchNameRef,
{
    fn minimize_ir(&self, state: &mut S, input: &LayeredInput) -> Option<LayeredInput>
    where
        S: HasRand + HasCorpus<LayeredInput>,
    {
        let mut minimizers = ir_minimizers();
        let mut idxs: Vec<_> = (0..minimizers.len()).collect();
        state.rand_mut().shuffle(&mut idxs);
        let mut input = input.clone();

        for rand_idx in idxs {
            let result = minimizers
                .get_and_mutate(rand_idx.into(), state, &mut input)
                .ok()?;

            if matches!(result, MutationResult::Mutated) {
                let LayeredInput::IR(ir) = &input else {
                    unreachable!();
                };
                for (_, func) in ir.iter_funcs() {
                    assert!(func.validate_dag().is_ok());
                }
                if let Err(err) = ir.try_get_text() {
                    println!(
                        "Minimizer {} broke sample: {}",
                        minimizers.name(rand_idx).unwrap(),
                        err
                    );
                }
                return Some(input);
            }
        }
        None
    }

    fn minimize_ast(&self, state: &mut S, input: &LayeredInput) -> Option<LayeredInput>
    where
        S: HasRand + HasCorpus<LayeredInput>,
    {
        let mut mutator = ASTDeleteMutator::new();
        let mut input = input.clone();
        match mutator.mutate(state, &mut input) {
            Ok(MutationResult::Mutated) => Some(input),
            _ => None,
        }
    }

    fn clone_map<E>(&self, executor: &E) -> Result<Vec<<MO as MapObserver>::Entry>, Error>
    where
        E: HasObservers<Observers = OT>,
    {
        let observers = executor.observers();
        let Some(observer) = observers.get(&self.map_observer_handle) else {
            return Err(Error::key_not_found(format!(
                "Observer not found: {:?}",
                &self.map_observer_handle
            )));
        };

        Ok(observer.as_ref().to_vec())
    }
}

const CAL_STAGE_START: usize = 4;
const CAL_STAGE_MAX: usize = 8;

impl<E, EM, O, MO, OT, S, Z> Stage<E, EM, S, Z> for LayeredMinimizerStage<O, MO, OT, S>
where
    E: Executor<EM, LayeredInput, S, Z> + HasObservers<Observers = OT>,
    EM: EventFirer<LayeredInput, S>,
    O: AsRef<MO>,
    MO: MapObserver,
    for<'de> <MO as MapObserver>::Entry: Serialize + Deserialize<'de> + 'static,
    OT: ObserversTuple<LayeredInput, S>,
    S: HasCorpus<LayeredInput>
        + HasMetadata
        + HasCurrentCorpusId
        + HasClientPerfMonitor
        + HasExecutions
        + HasNamedMetadata
        + HasRand,
    Z: Evaluator<E, EM, LayeredInput, S>,
{
    #[inline]
    #[allow(
        clippy::let_and_return,
        clippy::too_many_lines,
        clippy::cast_precision_loss
    )]
    fn perform(
        &mut self,
        fuzzer: &mut Z,
        executor: &mut E,
        state: &mut S,
        mgr: &mut EM,
    ) -> Result<(), Error> {
        if state.current_testcase()?.scheduled_count() > 0 {
            return Ok(());
        }

        let mut iter = CAL_STAGE_START;

        let input = state.current_input_cloned()?;

        executor.observers_mut().pre_exec_all(state, &input)?;

        let exit_kind = executor.run_target(fuzzer, state, mgr, &input)?;

        executor
            .observers_mut()
            .post_exec_all(state, &input, &exit_kind)?;

        let map_first = self.clone_map(executor)?;

        let mut unstable_entries = HashSet::new();
        let mut i = 1;
        let mut has_errors = false;

        while i < iter {
            executor.observers_mut().pre_exec_all(state, &input)?;

            let exit_kind = executor.run_target(fuzzer, state, mgr, &input)?;
            if exit_kind != ExitKind::Ok {
                if !has_errors {
                    mgr.log(
                        state,
                        LogSeverity::Warn,
                        "Corpus entry errored on execution!".into(),
                    )?;

                    has_errors = true;
                }
                iter = std::cmp::min(CAL_STAGE_MAX, iter + 2);
            };

            executor
                .observers_mut()
                .post_exec_all(state, &input, &exit_kind)?;

            let map = self.clone_map(executor)?;

            for (idx, (first, cur)) in map_first.iter().zip(map.iter()).enumerate() {
                if *first != *cur {
                    unstable_entries.insert(idx);
                };
            }

            if !unstable_entries.is_empty() && iter < CAL_STAGE_MAX {
                iter += 2;
            }
            i += 1;
        }

        let novelties: HashSet<usize> = {
            let testcase = state.current_testcase()?;
            let novelties = testcase
                .metadata_map()
                .get::<MapNoveltiesMetadata>()
                .expect("check arguments of MapFeedback::new");
            HashSet::from_iter(novelties.as_slice().iter().cloned())
        };
        let mut stable_novelties: Vec<usize> =
            novelties.difference(&unstable_entries).cloned().collect();
        stable_novelties.sort();
        println!(
            "{} has {} unstable entries, {} stable entries",
            state.current_corpus_id().unwrap().unwrap(),
            unstable_entries.len(),
            stable_novelties.len()
        );

        let mut iter = 100;
        let mut smallest = state.current_input_cloned()?;
        let mut smallest_size = smallest.len();
        'retry: while iter > 0 {
            iter -= 1;

            let mutated = match smallest {
                LayeredInput::IR(_) => self.minimize_ir(state, &smallest),
                LayeredInput::Ast(_) => self.minimize_ast(state, &smallest),
            };
            let Some(mutated) = mutated else {
                println!("warning: mutator failed to mutate");
                continue;
            };

            executor.observers_mut().pre_exec_all(state, &mutated)?;
            if executor.run_target(fuzzer, state, mgr, &mutated)? != exit_kind {
                println!("warning: mutation changed exit_kind, discarding");
                continue;
            }

            executor
                .observers_mut()
                .post_exec_all(state, &mutated, &exit_kind)?;

            let map = self.clone_map(executor)?;

            for &idx in stable_novelties.iter() {
                if map[idx] != map_first[idx] {
                    continue 'retry;
                }
            }

            if mutated.len() >= smallest_size {
                println!(
                    "warning: minimizer didn't produce smaller sample {} vs {}",
                    mutated.len(),
                    smallest_size
                );
            } else {
                iter = 100;
            }

            smallest = mutated;
            smallest_size = smallest.len();
        }

        if let LayeredInput::Ast(ref mut ast) = smallest {
            let bytes = ast.get_text().as_bytes();
            let reparsed = Ast::try_from_wgsl(bytes);
            if let Ok(reparsed) = reparsed {
                *ast = reparsed;
            } else {
                ast.deflate();
            }
        }

        let mut entry = state.current_testcase_mut()?;
        entry.set_input(smallest);

        Ok(())
    }
}

// TODO: Either rework this custom stage to not be a stage, or do something
// more intelligent here around restarts. Given that this stage executes inputs,
// it might crash the process (e.g. if using in-process execution). When libafl
// restarts, we should not attempt to run the crashing input again and again,
// lest we get stuck in a crashing loop.
impl<O, MO, OT, S> Restartable<S> for LayeredMinimizerStage<O, MO, OT, S> {
    fn should_restart(&mut self, _state: &mut S) -> Result<bool, Error> {
        Ok(true)
    }

    fn clear_progress(&mut self, _state: &mut S) -> Result<(), Error> {
        Ok(())
    }
}
