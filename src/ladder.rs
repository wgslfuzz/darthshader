// Up and Down the Ladder of Abstraction

use core::{fmt::Debug, marker::PhantomData};

use libafl::{
    common::{HasMetadata, HasNamedMetadata},
    corpus::{Corpus, CorpusId, HasCurrentCorpusId, Testcase},
    events::EventFirer,
    executors::Executor,
    fuzzer::Evaluator,
    stages::{Restartable, Stage},
    state::{HasClientPerfMonitor, HasCorpus, HasCurrentTestcase, HasRand},
    Error, HasScheduler,
};
use libafl_bolts::impl_serdeany;
use log::debug;
use serde::{Deserialize, Serialize};

use crate::{
    ast::Ast,
    layeredinput::{LayeredInput, IR},
};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LadderNoAscendMetadata {}
impl_serdeany!(LadderNoAscendMetadata);

impl LadderNoAscendMetadata {
    #[must_use]
    /// Create a new [`struct@LadderNoAscendMetadata`]
    pub fn new() -> Self {
        Self {}
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LadderNoDescendMetadata {}
impl_serdeany!(LadderNoDescendMetadata);

impl LadderNoDescendMetadata {
    #[must_use]
    /// Create a new [`struct@LadderNoDescendMetadata`]
    pub fn new() -> Self {
        Self {}
    }
}

fn may_ascend(testcase: &Testcase<LayeredInput>) -> bool {
    !testcase.has_metadata::<LadderNoAscendMetadata>()
}

fn may_descend(testcase: &Testcase<LayeredInput>) -> bool {
    !testcase.has_metadata::<LadderNoDescendMetadata>()
}

/// The calibration stage will measure the average exec time and the target's stability for this input.
#[derive(Clone, Debug)]
pub struct LadderStage<S> {
    phantom: PhantomData<S>,
}

impl<S> LadderStage<S>
where
    S: HasCorpus<LayeredInput> + HasMetadata + HasNamedMetadata,
{
    /// Create a new [`LadderStage`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }

    fn prohibit_ascend(state: &S, corpus_idx: CorpusId) {
        state
            .corpus()
            .get(corpus_idx)
            .unwrap()
            .borrow_mut()
            .add_metadata(LadderNoAscendMetadata::new());
    }

    fn prohibit_descend(state: &S, corpus_idx: CorpusId) {
        state
            .corpus()
            .get(corpus_idx)
            .unwrap()
            .borrow_mut()
            .add_metadata(LadderNoDescendMetadata::new());
    }
}

impl<E, EM, S, Z> Stage<E, EM, S, Z> for LadderStage<S>
where
    E: Executor<EM, LayeredInput, S, Z>,
    EM: EventFirer<LayeredInput, S>,
    S: HasCorpus<LayeredInput>
        + HasCurrentCorpusId
        + HasMetadata
        + HasClientPerfMonitor
        + HasNamedMetadata
        + HasRand,
    Z: Evaluator<E, EM, LayeredInput, S> + HasScheduler<LayeredInput, S>,
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

        let id = state.current_corpus_id()?.unwrap();
        debug!("scheduling for ladder {}", id);

        let input = state.current_input_cloned()?;

        match input {
            LayeredInput::IR(ir) => {
                debug!("IR ladder");
                if may_descend(&*state.current_testcase().unwrap()) {
                    let Ok(text) = ir.try_get_text() else {
                        return Ok(());
                    };
                    let bytes = text.as_bytes();
                    let ast = Ast::try_from_wgsl(bytes);
                    if let Ok(ast) = ast {
                        let input = LayeredInput::Ast(ast);
                        let nidx = fuzzer.add_input(state, executor, mgr, input).unwrap();
                        Self::prohibit_ascend(state, nidx);
                    }
                }
            }
            LayeredInput::Ast(ast) => {
                if may_ascend(&*state.current_testcase().unwrap()) {
                    let text = ast.get_text().to_owned();
                    let result = std::panic::catch_unwind(|| {
                        let result: Result<IR, _> = text.as_str().try_into();
                        result
                    });

                    if let Ok(Ok(ir)) = result {
                        let input = LayeredInput::IR(ir);
                        let nidx = fuzzer.add_input(state, executor, mgr, input).unwrap();
                        Self::prohibit_descend(state, nidx);
                    }
                } else {
                    debug!("not allowed to accend");
                }
            }
        };

        Ok(())
    }
}

// TODO: Either rework this custom stage to be a custom mutator, or do something
// more intelligent here around restarts. Given that this stage executes inputs,
// it might crash the process (e.g. if using in-process execution). When libafl
// restarts, we should not attempt to run the crashing input again and again,
// lest we get stuck in a crashing loop.
impl<S> Restartable<S> for LadderStage<S> {
    fn should_restart(&mut self, _state: &mut S) -> Result<bool, Error> {
        Ok(true)
    }

    fn clear_progress(&mut self, _state: &mut S) -> Result<(), Error> {
        Ok(())
    }
}
