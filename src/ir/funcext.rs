use naga::{Expression, Function, Handle, RayQueryFunction, Statement};

use crate::ir::iter::{StatementVisitor, UsesExprIter};

pub trait FuncExt {
    fn validate_dag(&self) -> Result<(), ()>;
}

impl FuncExt for Function {
    fn validate_dag(&self) -> Result<(), ()> {
        let mut done = vec![false; self.expressions.len()];
        let mut in_stack = vec![false; self.expressions.len()];

        fn log_err(func: &Function) {
            println!("failing graph:");
            for (handle, expr) in func.expressions.iter() {
                println!("{:?} {:?}", handle, expr);
            }
        }

        fn is_cyclic(
            func: &Function,
            handle: Handle<Expression>,
            in_stack: &mut Vec<bool>,
            done: &mut Vec<bool>,
        ) -> bool {
            if done[handle.index()] {
                return false;
            }
            in_stack[handle.index()] = true;
            for used_handle in func.expressions[handle].iter_used_exprs() {
                if in_stack[used_handle.index()] {
                    return true;
                }
                if is_cyclic(func, used_handle, in_stack, done) {
                    return true;
                }
            }
            in_stack[handle.index()] = false;
            done[handle.index()] = true;
            false
        }

        for (handle, _) in self.expressions.iter() {
            if is_cyclic(self, handle, &mut in_stack, &mut done) {
                log_err(self);
                return Err(());
            }
        }

        let mut all_okay = Ok(());
        let mut traversed = Vec::new();
        let validate_result = |stmt: &Statement| {
            let target;
            let mut worklist;
            match stmt {
                Statement::Atomic {
                    pointer,
                    value,
                    result,
                    ..
                } => {
                    target = *result;
                    worklist = vec![*pointer, *value];
                }
                Statement::WorkGroupUniformLoad { pointer, result } => {
                    target = *result;
                    worklist = vec![*pointer];
                }
                Statement::Call {
                    arguments,
                    result: Some(result),
                    ..
                } => {
                    target = *result;
                    worklist = arguments.clone();
                }
                Statement::RayQuery {
                    query,
                    fun: RayQueryFunction::Proceed { result },
                } => {
                    target = *result;
                    worklist = vec![*query];
                }
                _ => return true,
            }

            traversed.clear();
            traversed.resize(self.expressions.len(), false);
            while let Some(handle) = worklist.pop() {
                if traversed[handle.index()] {
                    continue;
                }
                if handle == target {
                    all_okay = Err(());
                    return false;
                }
                worklist.extend(self.expressions[handle].iter_used_exprs());
                traversed[handle.index()] = true;
            }

            true
        };
        self.visit_statements(validate_result);

        if all_okay.is_err() {
            log_err(self);
        }

        all_okay
    }
}
