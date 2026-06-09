use core::fmt::{Display, Error as FmtError, Formatter};

use naga::{Arena, Expression};

/// Newtype for lazily printing expressions in case of error.
pub struct ExpressionsPrinter<'a>(pub &'a Arena<Expression>);

impl Display for ExpressionsPrinter<'_> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result<(), FmtError> {
        for (handle, expr) in self.0.iter() {
            write!(fmt, "\n{:?} {:?}", handle, expr)?
        }
        Ok(())
    }
}
