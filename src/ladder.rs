// Up and Down the Ladder of Abstraction

use core::{fmt::Debug, marker::PhantomData};

use libafl_bolts::impl_serdeany;
use serde::{Deserialize, Serialize};

use libafl::{
    corpus::{Corpus, CorpusId},
    events::EventFirer,
    executors::Executor,
    fuzzer::Evaluator,
    inputs::UsesInput,
    stages::Stage,
    state::{
        HasClientPerfMonitor, HasCorpus, HasMetadata, HasNamedMetadata, HasRand, State, UsesState,
    },
    Error, HasScheduler,
};

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

/// The calibration stage will measure the average exec time and the target's stability for this input.
#[derive(Clone, Debug)]
pub struct LadderStage<S> {
    phantom: PhantomData<S>,
}

impl<S> UsesState for LadderStage<S>
where
    S: UsesInput + State,
{
    type State = S;
}

impl<S> LadderStage<S>
where
    S: HasCorpus + HasMetadata + HasNamedMetadata,
{
    /// Create a new [`LadderStage`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }

    fn may_ascend(state: &S, corpus_idx: CorpusId) -> bool {
        !state
            .corpus()
            .get(corpus_idx)
            .unwrap()
            .borrow()
            .has_metadata::<LadderNoAscendMetadata>()
    }

    fn may_descend(state: &S, corpus_idx: CorpusId) -> bool {
        !state
            .corpus()
            .get(corpus_idx)
            .unwrap()
            .borrow()
            .has_metadata::<LadderNoDescendMetadata>()
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

impl<E, EM, Z> Stage<E, EM, Z> for LadderStage<E::State>
where
    E: Executor<EM, Z>,
    EM: EventFirer<State = E::State>,
    E::State: HasCorpus<Input = LayeredInput>
        + HasMetadata
        + HasClientPerfMonitor
        + HasNamedMetadata
        + HasRand,
    Z: Evaluator<E, EM, State = E::State> + HasScheduler,
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
        {
            let entry = state.corpus().get(corpus_idx)?.borrow();
            if entry.scheduled_count() > 0 {
                return Ok(());
            }
        }
        println!("scheduling for ladder {}", corpus_idx);

        let input = state.corpus().cloned_input_for_id(corpus_idx)?;

        match input {
            LayeredInput::IR(ir) => {
                println!("IR ladder");
                if Self::may_descend(state, corpus_idx) {
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
                if Self::may_ascend(state, corpus_idx) {
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
                    println!("not allowed to accend");
                }
            }
        };

        Ok(())
    }
}
