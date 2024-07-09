use core::marker::PhantomData;
use libafl::{
    monitors::{AggregatorOps, UserStats, UserStatsValue},
    prelude::{
        Event, EventFirer, ExitKind, Feedback, Observer, ObserversTuple, Testcase, UsesInput,
    },
    state::{HasClientPerfMonitor, HasMetadata, State},
};
use libafl_bolts::{
    current_nanos, impl_serdeany, prelude::OwnedMutSlice, AsMutSlice, AsSlice, Error, Named,
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ExitMetadata {
    pub code: u8,
}
impl_serdeany!(ExitMetadata);

impl ExitMetadata {
    #[must_use]
    /// Create a new [`struct@ExitMetadata`]
    pub fn new(code: u8) -> Self {
        Self { code }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ExitObserver<'a> {
    name: String,
    owned: OwnedMutSlice<'a, u8>,
    last_exit: Option<u8>,
}

impl<'a> ExitObserver<'a> {
    /// Creates a new [`ExitObserver`] with the given name.
    #[must_use]
    pub fn new(name: &'static str, shmem: &'a mut [u8]) -> Self {
        Self {
            name: name.to_string(),
            owned: OwnedMutSlice::from(shmem),
            last_exit: None,
        }
    }

    /// Gets the runtime for the last execution of this target.
    #[must_use]
    pub fn last_exit(&self) -> &Option<u8> {
        &self.last_exit
    }
}

impl<S> Observer<S> for ExitObserver<'_>
where
    S: UsesInput,
{
    fn pre_exec(&mut self, _state: &mut S, _input: &S::Input) -> Result<(), Error> {
        self.owned.as_mut_slice()[0] = 0;
        self.owned.as_mut_slice()[1] = 0;
        self.last_exit = None;
        Ok(())
    }

    fn post_exec(
        &mut self,
        _state: &mut S,
        _input: &S::Input,
        _exit_kind: &ExitKind,
    ) -> Result<(), Error> {
        if self.owned.as_mut_slice()[1] == 1 {
            self.last_exit = Some(self.owned.as_slice()[0]);
        }
        Ok(())
    }
}

impl Named for ExitObserver<'_> {
    fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ExitFeedback {
    name: String,
    last_update: u64,
    execs_suc: u64,
    execs_err: u64,
}

impl<S> Feedback<S> for ExitFeedback
where
    S: UsesInput + HasClientPerfMonitor + State,
{
    #[allow(clippy::wrong_self_convention)]
    fn is_interesting<EM, OT>(
        &mut self,
        state: &mut S,
        manager: &mut EM,
        _input: &S::Input,
        observers: &OT,
        _exit_kind: &ExitKind,
    ) -> Result<bool, Error>
    where
        EM: EventFirer<State = S>,
        OT: ObserversTuple<S>,
    {
        let observer = observers.match_name::<ExitObserver>(self.name()).unwrap();
        if let Some(last_exit) = *observer.last_exit() {
            if last_exit == 0 {
                self.execs_suc += 1;
            } else {
                self.execs_err += 1;
            }
        }

        let now = current_nanos();
        if now < self.last_update {
            self.last_update = now;
        } else if now - self.last_update > 60_000_000_000 {
            self.last_update = now;
            manager.fire(
                state,
                Event::UpdateUserStats {
                    name: "execs_err".to_string(),
                    value: UserStats::new(
                        UserStatsValue::Number(self.execs_err),
                        AggregatorOps::Sum,
                    ),
                    phantom: PhantomData,
                },
            )?;
            manager.fire(
                state,
                Event::UpdateUserStats {
                    name: "execs_suc".to_string(),
                    value: UserStats::new(
                        UserStatsValue::Number(self.execs_suc),
                        AggregatorOps::Sum,
                    ),
                    phantom: PhantomData,
                },
            )?;
        }
        Ok(false)
    }

    #[inline]
    fn append_metadata<OT>(
        &mut self,
        _state: &mut S,
        observers: &OT,
        testcase: &mut Testcase<S::Input>,
    ) -> Result<(), Error>
    where
        OT: ObserversTuple<S>,
    {
        let observer = observers.match_name::<ExitObserver>(self.name()).unwrap();
        if let Some(last_exit) = *observer.last_exit() {
            testcase.add_metadata(ExitMetadata::new(last_exit));
        }

        Ok(())
    }

    /// Discard the stored metadata in case that the testcase is not added to the corpus
    #[inline]
    fn discard_metadata(&mut self, _state: &mut S, _input: &S::Input) -> Result<(), Error> {
        Ok(())
    }
}

impl Named for ExitFeedback {
    #[inline]
    fn name(&self) -> &str {
        self.name.as_str()
    }
}

impl ExitFeedback {
    /// Creates a new [`ExitFeedback`], deciding if the given [`ExitObserver`] value of a run is interesting.
    #[must_use]
    pub fn with_observer(observer: &ExitObserver) -> Self {
        Self {
            name: observer.name().to_string(),
            last_update: current_nanos(),
            execs_err: 0,
            execs_suc: 0,
        }
    }
}
