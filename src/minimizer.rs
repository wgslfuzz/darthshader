use core::{fmt::Debug, marker::PhantomData};

use libafl_bolts::{
    tuples::{HasConstLen, NamedTuple},
    AsIter, AsSlice, HasLen, Named,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use libafl::{
    corpus::{Corpus, CorpusId},
    events::{EventFirer, LogSeverity},
    executors::{Executor, ExitKind, HasObservers},
    feedbacks::HasObserverName,
    fuzzer::Evaluator,
    inputs::UsesInput,
    mutators::MutatorsTuple,
    observers::{MapObserver, ObserversTuple, UsesObserver},
    prelude::{MapNoveltiesMetadata, MutationResult, Mutator},
    stages::Stage,
    state::{
        HasClientPerfMonitor, HasCorpus, HasMetadata, HasNamedMetadata, HasRand, State, UsesState,
    },
    Error,
};

use crate::{
    ast::{mutate::ASTDeleteMutator, Ast},
    ir::{funcext::FuncExt, iter::IterFuncs, minimizer::ir_minimizers},
    layeredinput::LayeredInput,
    randomext::RandExt,
};

#[derive(Clone, Debug)]
pub struct LayeredMinimizerStage<O, OT, S> {
    map_observer_name: String,
    stage_max: usize,
    phantom: PhantomData<(O, OT, S)>,
}

impl<O, OT, S> LayeredMinimizerStage<O, OT, S> {
    fn minimize_ir(&self, state: &mut S, input: &LayeredInput) -> Option<LayeredInput>
    where
        S: HasRand + HasCorpus<Input = LayeredInput>,
    {
        let mut minimizers = ir_minimizers();
        let mut idxs: Vec<_> = (0..minimizers.len()).collect();
        state.rand_mut().shuffle(&mut idxs);
        let mut input = input.clone();

        for rand_idx in idxs {
            let result = minimizers
                .get_and_mutate(rand_idx.into(), state, &mut input, 0)
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
        S: HasRand + HasCorpus<Input = LayeredInput>,
    {
        let mut mutator = ASTDeleteMutator::new();
        let mut input = input.clone();
        match mutator.mutate(state, &mut input, 0) {
            Ok(MutationResult::Mutated) => Some(input),
            _ => None,
        }
    }
}

const CAL_STAGE_START: usize = 4;
const CAL_STAGE_MAX: usize = 8;

impl<O, OT, S> UsesState for LayeredMinimizerStage<O, OT, S>
where
    S: UsesInput + State,
{
    type State = S;
}

impl<E, EM, O, OT, Z> Stage<E, EM, Z> for LayeredMinimizerStage<O, OT, E::State>
where
    E: Executor<EM, Z> + HasObservers<Observers = OT>,
    EM: EventFirer<State = E::State>,
    O: MapObserver,
    for<'de> <O as MapObserver>::Entry: Serialize + Deserialize<'de> + 'static,
    OT: ObserversTuple<E::State>,
    E::State: HasCorpus<Input = LayeredInput>
        + HasMetadata
        + HasClientPerfMonitor
        + HasNamedMetadata
        + HasRand,
    Z: Evaluator<E, EM, State = E::State>,
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
        state: &mut E::State,
        mgr: &mut EM,
        corpus_idx: CorpusId,
    ) -> Result<(), Error> {
        let entry = state.corpus().get(corpus_idx)?.borrow();
        if entry.scheduled_count() > 0 {
            return Ok(());
        }
        drop(entry);

        let mut iter = self.stage_max;

        let input = state.corpus().cloned_input_for_id(corpus_idx)?;

        executor.observers_mut().pre_exec_all(state, &input)?;

        let exit_kind = executor.run_target(fuzzer, state, mgr, &input)?;

        executor
            .observers_mut()
            .post_exec_all(state, &input, &exit_kind)?;

        let map_first = &executor
            .observers()
            .match_name::<O>(&self.map_observer_name)
            .ok_or_else(|| Error::key_not_found("MapObserver not found".to_string()))?
            .to_vec();

        let mut unstable_entries = HashSet::new();
        let mut i = 1;
        let mut has_errors = false;

        while i < iter {
            let input = state.corpus().cloned_input_for_id(corpus_idx)?;

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

            let map = &executor
                .observers()
                .match_name::<O>(&self.map_observer_name)
                .ok_or_else(|| Error::key_not_found("MapObserver not found".to_string()))?
                .to_vec();

            for (idx, (first, cur)) in map_first.iter().zip(map).enumerate() {
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
            let entry = state.corpus().get(corpus_idx)?.borrow_mut();
            let novelties = entry
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
            corpus_idx,
            unstable_entries.len(),
            stable_novelties.len()
        );

        let mut iter = 100;
        let mut smallest = state.corpus().cloned_input_for_id(corpus_idx)?;
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

            let map = &executor
                .observers()
                .match_name::<O>(&self.map_observer_name)
                .ok_or_else(|| Error::key_not_found("MapObserver not found".to_string()))?
                .to_vec();

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

        let mut entry = state.corpus().get(corpus_idx)?.borrow_mut();
        entry.set_input(smallest);

        Ok(())
    }
}

impl<O, OT, S> LayeredMinimizerStage<O, OT, S>
where
    O: MapObserver,
    OT: ObserversTuple<S>,
    S: HasCorpus + HasMetadata + HasNamedMetadata,
{
    /// Create a new [`LayeredMinimizerStage`].
    #[must_use]
    pub fn new<F>(map_feedback: &F) -> Self
    where
        F: HasObserverName + Named + UsesObserver<S, Observer = O>,
        for<'it> O: AsIter<'it, Item = O::Entry>,
    {
        Self {
            map_observer_name: map_feedback.observer_name().to_string(),
            stage_max: CAL_STAGE_START,
            phantom: PhantomData,
        }
    }
}
