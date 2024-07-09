use libafl::{prelude::{Observer, UsesInput, ExitKind, Feedback, ObserversTuple, EventFirer, Testcase}, state::{HasClientPerfMonitor, HasMetadata}};
use libafl_bolts::{Named, Error, AsSlice, AsMutSlice, prelude::OwnedMutSlice, impl_serdeany};
use serde::{Serialize, Deserialize};

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
            last_exit: None
        }
    }

    /// Gets the runtime for the last execution of this target.
    #[must_use]
    pub fn last_exit(&self) -> &Option<u8> {
        &self.last_exit
    }
}

impl< S> Observer<S> for ExitObserver<'_>
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
}

impl<S> Feedback<S> for ExitFeedback
where
    S: UsesInput + HasClientPerfMonitor,
{
    #[allow(clippy::wrong_self_convention)]
    fn is_interesting<EM, OT>(
        &mut self,
        _state: &mut S,
        _manager: &mut EM,
        _input: &S::Input,
        _observers: &OT,
        _exit_kind: &ExitKind,
    ) -> Result<bool, Error>
    where
        EM: EventFirer<State = S>,
        OT: ObserversTuple<S>,
    {
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
        testcase.add_metadata(ExitMetadata::new(observer.last_exit().unwrap()));
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
        }
    }
}
