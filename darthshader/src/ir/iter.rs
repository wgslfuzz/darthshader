use crate::layeredinput::IR;
use naga::{
    AtomicFunction, Block, Expression, Function, Handle, ImageQuery, Module, SampleLevel, Statement,
};
use smallvec::{smallvec, SmallVec};

#[derive(Clone, Copy, Debug)]
pub(crate) enum FunctionIdentifier {
    Function(Handle<Function>),
    EntryPoint(usize),
}

pub(crate) trait IterFuncs {
    fn iter_funcs(&self) -> impl Iterator<Item = (FunctionIdentifier, &Function)>;
    fn iter_funcs_mut(&mut self) -> impl Iterator<Item = (FunctionIdentifier, &mut Function)>;
}

impl IterFuncs for Module {
    fn iter_funcs(&self) -> impl Iterator<Item = (FunctionIdentifier, &Function)> {
        let entry_funcs = self
            .entry_points
            .iter()
            .enumerate()
            .map(|(idx, e)| (FunctionIdentifier::EntryPoint(idx), &e.function));
        let other_funcs = self
            .functions
            .iter()
            .map(|(handle, f)| (FunctionIdentifier::Function(handle), f));
        entry_funcs.chain(other_funcs)
    }

    fn iter_funcs_mut(&mut self) -> impl Iterator<Item = (FunctionIdentifier, &mut Function)> {
        let entry_funcs = self
            .entry_points
            .iter_mut()
            .enumerate()
            .map(|(idx, e)| (FunctionIdentifier::EntryPoint(idx), &mut e.function));
        let other_funcs = self
            .functions
            .iter_mut()
            .map(|(handle, f)| (FunctionIdentifier::Function(handle), f));
        entry_funcs.chain(other_funcs)
    }
}

impl IterFuncs for IR {
    fn iter_funcs(&self) -> impl Iterator<Item = (FunctionIdentifier, &Function)> {
        self.get_module().iter_funcs()
    }

    fn iter_funcs_mut(&mut self) -> impl Iterator<Item = (FunctionIdentifier, &mut Function)> {
        self.get_module_mut().iter_funcs_mut()
    }
}

pub(crate) trait BlockVisitor<F>
where
    F: FnMut(&Block) -> bool,
{
    fn visit_blocks(&self, visitor: F);
}

pub(crate) trait BlockVisitorMut<F>
where
    F: FnMut(&mut Block) -> bool,
{
    fn visit_blocks_mut(&mut self, visitor: F);
}

pub(crate) trait StatementVisitor<F>
where
    F: FnMut(&Statement) -> bool,
{
    fn visit_statements(&self, visitor: F);
}

pub(crate) trait StatementVisitorMut<F>
where
    F: FnMut(&mut Statement) -> bool,
{
    fn visit_statements_mut(&mut self, visitor: F);
}

impl<F> BlockVisitor<F> for Function
where
    F: FnMut(&Block) -> bool,
{
    fn visit_blocks(&self, mut visitor: F) {
        let mut blocks = vec![&self.body];
        while let Some(block) = blocks.pop() {
            if !visitor(block) {
                return;
            }
            for stmt in block.iter() {
                match stmt {
                    Statement::Block(b) => {
                        blocks.push(b);
                    }
                    Statement::If { accept, reject, .. } => {
                        blocks.push(accept);
                        blocks.push(reject);
                    }
                    Statement::Loop {
                        body, continuing, ..
                    } => {
                        blocks.push(body);
                        blocks.push(continuing);
                    }
                    Statement::Switch { cases, .. } => {
                        for case in cases.iter() {
                            blocks.push(&case.body);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

impl<F> BlockVisitorMut<F> for Function
where
    F: FnMut(&mut Block) -> bool,
{
    fn visit_blocks_mut(&mut self, mut visitor: F) {
        let mut blocks = vec![&mut self.body];
        while let Some(block) = blocks.pop() {
            if !visitor(block) {
                return;
            }
            for stmt in block.iter_mut() {
                match stmt {
                    Statement::Block(b) => {
                        blocks.push(b);
                    }
                    Statement::If { accept, reject, .. } => {
                        blocks.push(accept);
                        blocks.push(reject);
                    }
                    Statement::Loop {
                        body, continuing, ..
                    } => {
                        blocks.push(body);
                        blocks.push(continuing);
                    }
                    Statement::Switch { cases, .. } => {
                        for case in cases.iter_mut() {
                            blocks.push(&mut case.body);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

impl<F> StatementVisitor<F> for Function
where
    F: FnMut(&Statement) -> bool,
{
    fn visit_statements(&self, mut visitor: F) {
        let block_visitor = |block: &Block| {
            for stmt in block.iter() {
                if !visitor(stmt) {
                    return false;
                }
            }
            true
        };
        self.visit_blocks(block_visitor)
    }
}

impl<F> StatementVisitorMut<F> for Function
where
    F: FnMut(&mut Statement) -> bool,
{
    fn visit_statements_mut(&mut self, mut visitor: F) {
        let block_visitor = |block: &mut Block| {
            for stmt in block.iter_mut() {
                if !visitor(stmt) {
                    return false;
                }
            }
            true
        };
        self.visit_blocks_mut(block_visitor)
    }
}

pub(crate) trait UsesExprIter {
    fn iter_used_exprs(&self) -> impl Iterator<Item = Handle<Expression>>;
    fn iter_used_exprs_mut(&mut self) -> impl Iterator<Item = &mut Handle<Expression>>;
}

impl UsesExprIter for Expression {
    fn iter_used_exprs(&self) -> impl Iterator<Item = Handle<Expression>> {
        let v: SmallVec<[_; 5]> = match self.clone() {
            Expression::Compose { components, .. } => SmallVec::from_iter(components),
            Expression::Access { base, index } => smallvec![base, index],
            Expression::AccessIndex { base, .. } => smallvec![base],
            Expression::Splat { value, .. } => smallvec![value],
            Expression::Swizzle { vector, .. } => smallvec![vector],
            Expression::Load { pointer } => smallvec![pointer],
            Expression::ImageSample {
                image,
                sampler,
                coordinate,
                array_index,
                level,
                depth_ref,
                ..
            } => {
                let mut v = smallvec![image, sampler, coordinate];
                if let Some(array_index) = array_index {
                    v.push(array_index)
                }
                if let Some(depth_ref) = depth_ref {
                    v.push(depth_ref)
                }
                match level {
                    SampleLevel::Exact(handle) | SampleLevel::Bias(handle) => v.push(handle),
                    SampleLevel::Gradient { x, y } => {
                        v.push(x);
                        v.push(y);
                    }
                    SampleLevel::Auto | SampleLevel::Zero => {}
                }
                v
            }
            Expression::ImageLoad {
                image,
                coordinate,
                array_index,
                sample,
                level,
            } => {
                let mut v = smallvec![image, coordinate];
                if let Some(array_index) = array_index {
                    v.push(array_index)
                }
                if let Some(sample) = sample {
                    v.push(sample)
                }
                if let Some(level) = level {
                    v.push(level)
                }
                v
            }
            Expression::ImageQuery { image, query } => {
                let mut v = smallvec![image];
                if let ImageQuery::Size { level: Some(level) } = query {
                    v.push(level)
                }
                v
            }
            Expression::Binary { left, right, .. } => smallvec![left, right],
            Expression::Select {
                condition,
                accept,
                reject,
            } => smallvec![condition, accept, reject],
            Expression::Relational { argument, .. } => smallvec![argument],
            Expression::Math {
                arg,
                arg1,
                arg2,
                arg3,
                ..
            } => {
                let mut v = smallvec![arg];
                if let Some(arg1) = arg1 {
                    v.push(arg1)
                }
                if let Some(arg2) = arg2 {
                    v.push(arg2)
                }
                if let Some(arg3) = arg3 {
                    v.push(arg3)
                }
                v
            }
            Expression::Derivative { expr, .. }
            | Expression::Unary { expr, .. }
            | Expression::As { expr, .. } => smallvec![expr],
            Expression::ArrayLength(handle) => smallvec![handle],
            Expression::RayQueryGetIntersection { query, .. } => smallvec![query],
            Expression::CallResult(_)
            | Expression::AtomicResult { .. }
            | Expression::WorkGroupUniformLoadResult { .. }
            | Expression::FunctionArgument(_)
            | Expression::GlobalVariable(_)
            | Expression::LocalVariable(_)
            | Expression::RayQueryProceedResult
            | Expression::Literal(_)
            | Expression::Constant(_)
            | Expression::ZeroValue(_) => smallvec![],
        };
        v.into_iter()
    }

    fn iter_used_exprs_mut(&mut self) -> impl Iterator<Item = &mut Handle<Expression>> {
        let v: SmallVec<[_; 5]> = match self {
            Expression::Compose { components, .. } => SmallVec::from_iter(components),
            Expression::Access { base, index } => smallvec![base, index],
            Expression::AccessIndex { base, .. } => smallvec![base],
            Expression::Splat { value, .. } => smallvec![value],
            Expression::Swizzle { vector, .. } => smallvec![vector],
            Expression::Load { pointer } => smallvec![pointer],
            Expression::ImageSample {
                image,
                sampler,
                coordinate,
                array_index,
                level,
                depth_ref,
                ..
            } => {
                let mut v = smallvec![image, sampler, coordinate];
                if let Some(array_index) = array_index {
                    v.push(array_index)
                }
                if let Some(depth_ref) = depth_ref {
                    v.push(depth_ref)
                }
                match level {
                    SampleLevel::Exact(handle) | SampleLevel::Bias(handle) => v.push(handle),
                    SampleLevel::Gradient { x, y } => {
                        v.push(x);
                        v.push(y);
                    }
                    SampleLevel::Auto | SampleLevel::Zero => {}
                }
                v
            }
            Expression::ImageLoad {
                image,
                coordinate,
                array_index,
                sample,
                level,
            } => {
                let mut v = smallvec![image, coordinate];
                if let Some(array_index) = array_index {
                    v.push(array_index)
                }
                if let Some(sample) = sample {
                    v.push(sample)
                }
                if let Some(level) = level {
                    v.push(level)
                }
                v
            }
            Expression::ImageQuery { image, query } => {
                let mut v = smallvec![image];
                if let ImageQuery::Size { level: Some(level) } = query {
                    v.push(level)
                }
                v
            }
            Expression::Binary { left, right, .. } => smallvec![left, right],
            Expression::Select {
                condition,
                accept,
                reject,
            } => smallvec![condition, accept, reject],
            Expression::Relational { argument, .. } => smallvec![argument],
            Expression::Math {
                arg,
                arg1,
                arg2,
                arg3,
                ..
            } => {
                let mut v = smallvec![arg];
                if let Some(arg1) = arg1 {
                    v.push(arg1)
                }
                if let Some(arg2) = arg2 {
                    v.push(arg2)
                }
                if let Some(arg3) = arg3 {
                    v.push(arg3)
                }
                v
            }
            Expression::Derivative { expr, .. }
            | Expression::Unary { expr, .. }
            | Expression::As { expr, .. } => smallvec![expr],
            Expression::ArrayLength(handle) => smallvec![handle],
            Expression::RayQueryGetIntersection { query, .. } => smallvec![query],
            Expression::CallResult(_)
            | Expression::AtomicResult { .. }
            | Expression::WorkGroupUniformLoadResult { .. }
            | Expression::FunctionArgument(_)
            | Expression::GlobalVariable(_)
            | Expression::LocalVariable(_)
            | Expression::RayQueryProceedResult
            | Expression::Literal(_)
            | Expression::Constant(_)
            | Expression::ZeroValue(_) => smallvec![],
        };
        v.into_iter()
    }
}

impl UsesExprIter for Statement {
    fn iter_used_exprs(&self) -> impl Iterator<Item = Handle<Expression>> {
        let v: SmallVec<[_; 4]> = match self.clone() {
            Statement::Emit(range) => SmallVec::from_iter(range),
            Statement::If { condition, .. } => smallvec![condition],
            Statement::Switch { selector, .. } => smallvec![selector],
            Statement::Loop {
                break_if: Some(break_if),
                ..
            } => smallvec![break_if],
            Statement::Return { value: Some(value) } => smallvec![value],
            Statement::Store { pointer, value } => smallvec![pointer, value],
            Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => match array_index {
                Some(array_index) => smallvec![image, coordinate, value, array_index],
                None => smallvec![image, coordinate, value],
            },
            Statement::Atomic {
                pointer,
                value,
                result,
                fun:
                    AtomicFunction::Exchange {
                        compare: Some(compare),
                    },
            } => smallvec![pointer, value, result, compare],
            Statement::Atomic {
                pointer,
                value,
                result,
                ..
            } => smallvec![pointer, value, result],
            Statement::WorkGroupUniformLoad { pointer, result } => smallvec![pointer, result],
            Statement::Call {
                arguments, result, ..
            } => {
                let mut v = SmallVec::from_iter(arguments);
                if let Some(result) = result {
                    v.push(result);
                }
                v
            }
            Statement::RayQuery { query, fun } => match fun {
                naga::RayQueryFunction::Initialize {
                    acceleration_structure,
                    descriptor,
                } => smallvec![query, acceleration_structure, descriptor],
                naga::RayQueryFunction::Proceed { result } => smallvec![query, result],
                naga::RayQueryFunction::Terminate => smallvec![],
            },
            Statement::Loop { break_if: None, .. }
            | Statement::Return { value: None }
            | Statement::Block(_)
            | Statement::Break
            | Statement::Continue
            | Statement::Kill
            | Statement::Barrier(_) => {
                smallvec![]
            }
        };
        v.into_iter()
    }

    fn iter_used_exprs_mut(&mut self) -> impl Iterator<Item = &mut Handle<Expression>> {
        let v: SmallVec<[_; 4]> = match self {
            Statement::Emit(_) => {
                panic!("Impossible to mutable iterate inputs of Statement::Emit")
            }
            Statement::If { condition, .. } => smallvec![condition],
            Statement::Switch { selector, .. } => smallvec![selector],
            Statement::Loop {
                break_if: Some(break_if),
                ..
            } => smallvec![break_if],
            Statement::Return { value: Some(value) } => smallvec![value],
            Statement::Store { pointer, value } => smallvec![pointer, value],
            Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => match array_index {
                Some(array_index) => smallvec![image, coordinate, value, array_index],
                None => smallvec![image, coordinate, value],
            },
            Statement::Atomic {
                pointer,
                value,
                result,
                fun:
                    AtomicFunction::Exchange {
                        compare: Some(compare),
                    },
            } => smallvec![pointer, value, result, compare],
            Statement::Atomic {
                pointer,
                value,
                result,
                ..
            } => smallvec![pointer, value, result],
            Statement::WorkGroupUniformLoad { pointer, result } => smallvec![pointer, result],
            Statement::Call {
                arguments, result, ..
            } => {
                let mut v = SmallVec::from_iter(arguments);
                if let Some(result) = result {
                    v.push(result);
                }
                v
            }
            Statement::RayQuery { query, fun } => match fun {
                naga::RayQueryFunction::Initialize {
                    acceleration_structure,
                    descriptor,
                } => smallvec![query, acceleration_structure, descriptor],
                naga::RayQueryFunction::Proceed { result } => smallvec![query, result],
                naga::RayQueryFunction::Terminate => smallvec![],
            },
            Statement::Loop { break_if: None, .. }
            | Statement::Return { value: None }
            | Statement::Block(_)
            | Statement::Break
            | Statement::Continue
            | Statement::Kill
            | Statement::Barrier(_) => {
                smallvec![]
            }
        };
        v.into_iter()
    }
}
