use core::{marker::PhantomData, time::Duration};
use std::borrow::Cow;

use libafl::{
    common::HasMetadata,
    corpus::Testcase,
    events::{Event, EventFirer, EventWithStats, ExecStats},
    executors::ExitKind,
    feedbacks::{Feedback, StateInitializer},
    monitors::stats::{AggregatorOps, UserStats, UserStatsValue},
    observers::{Observer, ObserversTuple},
    state::{HasClientPerfMonitor, HasExecutions},
};
use libafl_bolts::{
    current_time, impl_serdeany,
    ownedref::OwnedMutSlice,
    tuples::{Handle, Handled, MatchNameRef},
    AsSlice, Error, Named,
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
    name: Cow<'static, str>,
    owned: OwnedMutSlice<'a, u8>,
    last_exit: Option<u8>,
}

impl<'a> ExitObserver<'a> {
    /// Creates a new [`ExitObserver`] with the given name.
    #[must_use]
    pub fn new(name: &'static str, shmem: &'a mut [u8]) -> Self {
        Self {
            name: Cow::from(name),
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

impl<I, S> Observer<I, S> for ExitObserver<'_> {
    fn pre_exec(&mut self, _state: &mut S, _input: &I) -> Result<(), Error> {
        self.owned[0] = 0;
        self.owned[1] = 0;
        self.last_exit = None;
        Ok(())
    }

    fn post_exec(
        &mut self,
        _state: &mut S,
        _input: &I,
        _exit_kind: &ExitKind,
    ) -> Result<(), Error> {
        if self.owned[1] == 1 {
            self.last_exit = Some(self.owned.as_slice()[0]);
        }
        Ok(())
    }
}

impl Named for ExitObserver<'_> {
    fn name(&self) -> &Cow<'static, str> {
        &self.name
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ExitFeedback<'a> {
    name: Cow<'static, str>,
    observer_handle: Handle<ExitObserver<'a>>,
    last_update: Duration,
    execs_suc: u64,
    execs_err: u64,
}

impl<EM, I, OT, S> Feedback<EM, I, OT, S> for ExitFeedback<'_>
where
    EM: EventFirer<I, S>,
    OT: ObserversTuple<I, S>,
    S: HasClientPerfMonitor + HasExecutions,
{
    #[allow(clippy::wrong_self_convention)]
    fn is_interesting(
        &mut self,
        state: &mut S,
        manager: &mut EM,
        _input: &I,
        observers: &OT,
        _exit_kind: &ExitKind,
    ) -> Result<bool, Error> {
        let observer = observers.get(&self.observer_handle).unwrap();
        if let Some(last_exit) = *observer.last_exit() {
            if last_exit == 0 {
                self.execs_suc += 1;
            } else {
                self.execs_err += 1;
            }
        }

        let now = current_time();
        if now < self.last_update {
            self.last_update = now;
        } else if now - self.last_update > Duration::from_secs(60) {
            self.last_update = now;
            manager.fire(
                state,
                EventWithStats::new(
                    Event::UpdateUserStats {
                        name: Cow::from("execs_err"),
                        value: UserStats::new(
                            UserStatsValue::Number(self.execs_err),
                            AggregatorOps::Sum,
                        ),
                        phantom: PhantomData,
                    },
                    ExecStats::new(now, *state.executions()),
                ),
            )?;
            manager.fire(
                state,
                EventWithStats::new(
                    Event::UpdateUserStats {
                        name: Cow::from("execs_suc"),
                        value: UserStats::new(
                            UserStatsValue::Number(self.execs_suc),
                            AggregatorOps::Sum,
                        ),
                        phantom: PhantomData,
                    },
                    ExecStats::new(now, *state.executions()),
                ),
            )?;
        }
        Ok(false)
    }

    #[inline]
    fn append_metadata(
        &mut self,
        _state: &mut S,
        _event_manager: &mut EM,
        observers: &OT,
        testcase: &mut Testcase<I>,
    ) -> Result<(), Error> {
        let observer = observers.get(&self.observer_handle).unwrap();
        if let Some(last_exit) = *observer.last_exit() {
            testcase.add_metadata(ExitMetadata::new(last_exit));
        }

        Ok(())
    }
}

impl Named for ExitFeedback<'_> {
    #[inline]
    fn name(&self) -> &Cow<'static, str> {
        &self.name
    }
}

impl<S> StateInitializer<S> for ExitFeedback<'_> {}

impl<'a> ExitFeedback<'a> {
    /// Creates a new [`ExitFeedback`], deciding if the given [`ExitObserver`] value of a run is interesting.
    #[must_use]
    pub fn with_observer(observer: &ExitObserver<'a>) -> Self {
        Self {
            name: observer.name().clone(),
            observer_handle: observer.handle(),
            last_update: current_time(),
            execs_err: 0,
            execs_suc: 0,
        }
    }
}
