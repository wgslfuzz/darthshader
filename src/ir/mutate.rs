use crate::{
    generator::{FunctionGenCtx, GeneratorConfig, IRGenerator},
    ir::{
        exprscope::ExprScope,
        funcext::FuncExt,
        iter::{BlockVisitor, StatementVisitor, UsesExprIter},
    },
    layeredinput::LayeredInput,
    randomext::RandExt,
};
use libafl::{
    generators::Generator,
    prelude::{MutationResult, Mutator},
    state::{HasCorpus, HasRand},
    Error,
};
use libafl_bolts::{
    rands::{Rand, StdRand},
    tuples::{tuple_list, tuple_list_type},
    Named,
};
use naga::{
    AddressSpace, ArraySize, AtomicFunction, Barrier, BinaryOperator, Binding, Block, BuiltIn,
    Expression, Handle, ImageDimension, Literal, MathFunction, ScalarKind, Statement,
    StorageAccess, StructMember, Type, VectorSize,
};
use rand::{seq::IteratorRandom, Rng};

use crate::ir::iter::IterFuncs;

use super::iter::{FunctionIdentifier, StatementVisitorMut};

/// Tuple type of the mutations that compose the IR mutator
pub type IRMutationsType = tuple_list_type!(
    UnaryOpMutator,
    BinOpMutator,
    MathFuncMutator,
    LiteralMutator,
    RewireExpressionMutator,
    RewireStatementMutator,
    StatementMutator,
    TypeMutator,
    FullGenerationMutation,
    CodeGenerationMutation,
);

/// Get the mutations that compose the IR mutator
#[must_use]
pub fn ir_mutations() -> IRMutationsType {
    tuple_list!(
        UnaryOpMutator::new(),
        BinOpMutator::new(),
        MathFuncMutator::new(),
        LiteralMutator::new(),
        RewireExpressionMutator::new(false),
        RewireStatementMutator::new(false),
        StatementMutator::new(),
        TypeMutator::new(),
        FullGenerationMutation::new(),
        CodeGenerationMutation::new(),
    )
}

#[derive(Default, Debug)]
pub struct BinOpMutator;

impl BinOpMutator {
    /// Creates a new [`BinOpMutator`].
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }

    const BINOPS: [BinaryOperator; 10] = [
        BinaryOperator::Add,
        BinaryOperator::Subtract,
        BinaryOperator::Multiply,
        BinaryOperator::Divide,
        BinaryOperator::Modulo,
        BinaryOperator::ExclusiveOr,
        BinaryOperator::InclusiveOr,
        BinaryOperator::And,
        BinaryOperator::ShiftLeft,
        BinaryOperator::ShiftRight,
    ];

    const BOOLOPS: [BinaryOperator; 8] = [
        BinaryOperator::Equal,
        BinaryOperator::NotEqual,
        BinaryOperator::Less,
        BinaryOperator::LessEqual,
        BinaryOperator::Greater,
        BinaryOperator::GreaterEqual,
        BinaryOperator::LogicalAnd,
        BinaryOperator::LogicalOr,
    ];
}
const _: () = assert!(
    std::mem::variant_count::<BinaryOperator>()
        == BinOpMutator::BINOPS.len() + BinOpMutator::BOOLOPS.len()
);

impl Named for BinOpMutator {
    fn name(&self) -> &str {
        "IRBinOpMutator"
    }
}

impl<S> Mutator<S::Input, S> for BinOpMutator
where
    S: HasRand + HasCorpus<Input = LayeredInput>,
{
    fn mutate(
        &mut self,
        state: &mut S,
        input: &mut LayeredInput,
        _stage_idx: i32,
    ) -> Result<MutationResult, Error> {
        let LayeredInput::IR(ir) = input else {
            return Ok(MutationResult::Skipped);
        };

        let mut rng = StdRand::with_seed(state.rand_mut().next());
        let mut num_binary = 0u32;
        let mut target = None;
        for (_, expr) in ir
            .iter_funcs_mut()
            .flat_map(|(_, f)| f.expressions.iter_mut())
        {
            if let Expression::Binary { op, .. } = expr {
                num_binary += 1;
                if rng.gen_ratio(1, num_binary) {
                    target = Some(op);
                }
            }
        }

        let Some(op) = target else {
            return Ok(MutationResult::Skipped);
        };

        assert_eq!(
            std::mem::variant_count::<BinaryOperator>(),
            Self::BINOPS.len() + Self::BOOLOPS.len()
        );
        if Self::BOOLOPS.contains(op) {
            *op = *state.rand_mut().choose(&Self::BOOLOPS);
        } else {
            *op = *state.rand_mut().choose(&Self::BINOPS);
        }

        Ok(MutationResult::Mutated)
    }
}

#[derive(Default, Debug)]
pub struct MathFuncMutator;

impl MathFuncMutator {
    /// Creates a new [`MathFuncMutator`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    const MATHFUNCS: [naga::MathFunction; 69] = [
        MathFunction::Abs,
        MathFunction::Min,
        MathFunction::Max,
        MathFunction::Clamp,
        MathFunction::Saturate,
        MathFunction::Cos,
        MathFunction::Cosh,
        MathFunction::Sin,
        MathFunction::Sinh,
        MathFunction::Tan,
        MathFunction::Tanh,
        MathFunction::Acos,
        MathFunction::Asin,
        MathFunction::Atan,
        MathFunction::Atan2,
        MathFunction::Asinh,
        MathFunction::Acosh,
        MathFunction::Atanh,
        MathFunction::Radians,
        MathFunction::Degrees,
        MathFunction::Ceil,
        MathFunction::Floor,
        MathFunction::Round,
        MathFunction::Fract,
        MathFunction::Trunc,
        MathFunction::Modf,
        MathFunction::Frexp,
        MathFunction::Ldexp,
        MathFunction::Exp,
        MathFunction::Exp2,
        MathFunction::Log,
        MathFunction::Log2,
        MathFunction::Pow,
        MathFunction::Dot,
        MathFunction::Outer,
        MathFunction::Cross,
        MathFunction::Distance,
        MathFunction::Length,
        MathFunction::Normalize,
        MathFunction::FaceForward,
        MathFunction::Reflect,
        MathFunction::Refract,
        MathFunction::Sign,
        MathFunction::Fma,
        MathFunction::Mix,
        MathFunction::Step,
        MathFunction::SmoothStep,
        MathFunction::Sqrt,
        MathFunction::InverseSqrt,
        MathFunction::Transpose,
        MathFunction::Determinant,
        MathFunction::CountTrailingZeros,
        MathFunction::CountLeadingZeros,
        MathFunction::CountOneBits,
        MathFunction::ReverseBits,
        MathFunction::ExtractBits,
        MathFunction::InsertBits,
        MathFunction::FindLsb,
        MathFunction::FindMsb,
        MathFunction::Pack4x8snorm,
        MathFunction::Pack4x8unorm,
        MathFunction::Pack2x16snorm,
        MathFunction::Pack2x16unorm,
        MathFunction::Pack2x16float,
        MathFunction::Unpack4x8snorm,
        MathFunction::Unpack4x8unorm,
        MathFunction::Unpack2x16snorm,
        MathFunction::Unpack2x16unorm,
        MathFunction::Unpack2x16float,
    ];
}

impl Named for MathFuncMutator {
    fn name(&self) -> &str {
        "IRMathFuncMutator"
    }
}

impl<S> Mutator<S::Input, S> for MathFuncMutator
where
    S: HasRand + HasCorpus<Input = LayeredInput>,
{
    fn mutate(
        &mut self,
        state: &mut S,
        input: &mut LayeredInput,
        _stage_idx: i32,
    ) -> Result<MutationResult, Error> {
        let LayeredInput::IR(ir) = input else {
            return Ok(MutationResult::Skipped);
        };

        let mut rng = StdRand::with_seed(state.rand_mut().next());
        let mut num_math = 0u32;
        let mut target = None;
        for (_, expr) in ir
            .iter_funcs_mut()
            .flat_map(|(_, f)| f.expressions.iter_mut())
        {
            if let Expression::Math { fun, .. } = expr {
                num_math += 1;
                if rng.gen_ratio(1, num_math) {
                    target = Some(fun);
                }
            }
        }

        let Some(fun) = target else {
            return Ok(MutationResult::Skipped);
        };

        assert_eq!(
            std::mem::variant_count::<MathFunction>(),
            Self::MATHFUNCS.len() + 1
        );

        *fun = *state.rand_mut().choose(&Self::MATHFUNCS);
        Ok(MutationResult::Mutated)
    }
}

#[derive(Default, Debug)]
pub struct UnaryOpMutator;

impl UnaryOpMutator {
    /// Creates a new [`UnaryOpMutator`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Named for UnaryOpMutator {
    fn name(&self) -> &str {
        "IRUnaryOpMutator"
    }
}

impl<S> Mutator<S::Input, S> for UnaryOpMutator
where
    S: HasRand + HasCorpus<Input = LayeredInput>,
{
    fn mutate(
        &mut self,
        state: &mut S,
        input: &mut LayeredInput,
        _stage_idx: i32,
    ) -> Result<MutationResult, Error> {
        let LayeredInput::IR(ir) = input else {
            return Ok(MutationResult::Skipped);
        };

        let mut rng = StdRand::with_seed(state.rand_mut().next());
        let mut num_unary = 0u32;
        let mut target = None;
        for (_, expr) in ir
            .iter_funcs_mut()
            .flat_map(|(_, f)| f.expressions.iter_mut())
        {
            if let Expression::Unary { op, .. } = expr {
                num_unary += 1;
                if rng.gen_ratio(1, num_unary) {
                    target = Some(op);
                }
            }
        }

        let Some(op) = target else {
            return Ok(MutationResult::Skipped);
        };

        use naga::UnaryOperator as Uo;
        let unaryops = &[Uo::Negate, Uo::BitwiseNot, Uo::LogicalNot];
        assert_eq!(std::mem::variant_count::<Uo>(), unaryops.len());

        *op = *state.rand_mut().choose(unaryops.iter());
        Ok(MutationResult::Mutated)
    }
}

#[derive(Default, Debug)]
pub struct LiteralMutator;

impl LiteralMutator {
    /// Creates a new [`LiteralMutator`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    fn random_literal<R>(l: &Literal, r: &mut R) -> Literal
    where
        R: Rand,
    {
        match l {
            Literal::F64(_) => Literal::F64(r.random_f32() as f64),
            Literal::F32(_) => Literal::F32(r.random_f32()),
            Literal::U32(_) => Literal::U32(r.random_u32()),
            Literal::I32(_) => Literal::I32(r.random_i32()),
            Literal::Bool(_) => Literal::Bool(r.random_bool()),
        }
    }
}

impl Named for LiteralMutator {
    fn name(&self) -> &str {
        "IRLiteralMutator"
    }
}

impl<S> Mutator<S::Input, S> for LiteralMutator
where
    S: HasRand + HasCorpus<Input = LayeredInput>,
{
    fn mutate(
        &mut self,
        state: &mut S,
        input: &mut LayeredInput,
        _stage_idx: i32,
    ) -> Result<MutationResult, Error> {
        let LayeredInput::IR(ir) = input else {
            return Ok(MutationResult::Skipped);
        };

        let mut rng = StdRand::with_seed(state.rand_mut().next());
        let mut num_literals = 0u32;
        let mut target = None;
        for (_, expr) in ir
            .iter_funcs_mut()
            .flat_map(|(_, f)| f.expressions.iter_mut())
        {
            if let Expression::Literal(l) = expr {
                num_literals += 1;
                if rng.gen_ratio(1, num_literals) {
                    target = Some(l);
                }
            }
        }

        let Some(literal) = target else {
            return Ok(MutationResult::Skipped);
        };

        *literal = Self::random_literal(literal, state.rand_mut());
        Ok(MutationResult::Mutated)
    }
}

#[allow(dead_code)]
pub(crate) enum DfsItem<'a> {
    BlockOpen(&'a Block),
    BlockClose(&'a Block),
    Statement(&'a Statement),
}

pub(crate) struct DfsFuncIter<'a> {
    queue: Vec<DfsItem<'a>>,
}

impl<'a> DfsFuncIter<'a> {
    pub(crate) fn new(body: &'a Block) -> Self {
        let mut queue: Vec<DfsItem<'_>> = Vec::new();
        queue.push(DfsItem::BlockClose(body));
        queue.extend(body.iter().rev().map(DfsItem::Statement));
        queue.push(DfsItem::BlockOpen(body));
        Self { queue }
    }
}

impl<'a> Iterator for DfsFuncIter<'a> {
    type Item = DfsItem<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(item) = self.queue.pop() {
            let DfsItem::Statement(stm) = item else {
                return Some(item);
            };

            match stm {
                Statement::Block(b) => {
                    self.queue.push(DfsItem::BlockClose(b));
                    self.queue.extend(b.iter().rev().map(DfsItem::Statement));
                    self.queue.push(DfsItem::BlockOpen(b));
                }
                Statement::If { accept, reject, .. } => {
                    self.queue.push(DfsItem::BlockClose(accept));
                    self.queue
                        .extend(accept.iter().rev().map(DfsItem::Statement));
                    self.queue.push(DfsItem::BlockOpen(accept));

                    self.queue.push(DfsItem::BlockClose(reject));
                    self.queue
                        .extend(reject.iter().rev().map(DfsItem::Statement));
                    self.queue.push(DfsItem::BlockOpen(reject));
                }
                Statement::Loop {
                    body, continuing, ..
                } => {
                    self.queue.push(DfsItem::BlockClose(body));
                    self.queue.extend(body.iter().rev().map(DfsItem::Statement));
                    self.queue.push(DfsItem::BlockOpen(body));

                    self.queue.push(DfsItem::BlockClose(continuing));
                    self.queue
                        .extend(continuing.iter().rev().map(DfsItem::Statement));
                    self.queue.push(DfsItem::BlockOpen(continuing));
                }
                Statement::Switch { cases, .. } => {
                    for c in cases {
                        self.queue.push(DfsItem::BlockClose(&c.body));
                        self.queue
                            .extend(c.body.iter().rev().map(DfsItem::Statement));
                        self.queue.push(DfsItem::BlockOpen(&c.body));
                    }
                }
                _ => {}
            }
            Some(item)
        } else {
            None
        }
    }
}

#[derive(Default, Debug)]
pub struct RewireExpressionMutator {
    typed: bool,
}

impl RewireExpressionMutator {
    /// Creates a new [`ExpressionInputMutator`].
    #[must_use]
    pub const fn new(typed: bool) -> Self {
        Self { typed }
    }
}

impl Named for RewireExpressionMutator {
    fn name(&self) -> &str {
        if self.typed {
            "IRRewireExpressionMutator (typed)"
        } else {
            "IRRewireExpressionMutator (untyped)"
        }
    }
}

impl<S> Mutator<S::Input, S> for RewireExpressionMutator
where
    S: HasRand + HasCorpus<Input = LayeredInput>,
{
    fn mutate(
        &mut self,
        state: &mut S,
        input: &mut LayeredInput,
        _stage_idx: i32,
    ) -> Result<MutationResult, Error> {
        let LayeredInput::IR(ir) = input else {
            return Ok(MutationResult::Skipped);
        };

        let mut rng = StdRand::with_seed(state.rand_mut().next());
        let Some((fid, func)) = ir.iter_funcs().choose(&mut rng) else {
            return Ok(MutationResult::Skipped);
        };

        let Some(target_expr) = func
            .expressions
            .iter()
            .filter_map(|(handle, expr)| {
                if matches!(expr, Expression::As { .. }) {
                    return None;
                }
                expr.iter_used_exprs().next().is_some().then_some(handle)
            })
            .choose(&mut rng)
        else {
            return Ok(MutationResult::Skipped);
        };
        assert!(!matches!(
            func.expressions[target_expr],
            Expression::As { .. }
        ));
        assert!(func.validate_dag().is_ok());

        let mut emit = None;
        let searcher = |stmt: &Statement| match stmt {
            Statement::Emit(range) => {
                let Some((first, last)) = range.first_and_last() else {
                    unreachable!()
                };
                assert_eq!(first.index(), last.index());
                if first == target_expr {
                    emit = Some(stmt as *const Statement);
                    false
                } else {
                    true
                }
            }
            _ => true,
        };
        func.visit_statements(searcher);
        let Some(emit) = emit else {
            return Ok(MutationResult::Skipped);
        };

        let scope = ExprScope::new_before(fid, emit, ir.get_module(), false);
        if scope.is_empty() {
            return Ok(MutationResult::Skipped);
        }
        let (input_idx, input) = func.expressions[target_expr]
            .iter_used_exprs()
            .enumerate()
            .choose(&mut rng)
            .unwrap();

        for _ in 0..5 {
            let replacement = {
                if self.typed {
                    panic!("not supported");
                }
                scope.any().unwrap()
            };
            if replacement != input {
                let func = ir.get_func_mut(fid);
                let input = func.expressions[target_expr]
                    .iter_used_exprs_mut()
                    .nth(input_idx)
                    .unwrap();
                let org_input = *input;
                *input = replacement;

                if func.validate_dag().is_err() {
                    *func.expressions[target_expr]
                        .iter_used_exprs_mut()
                        .nth(input_idx)
                        .unwrap() = org_input;
                }

                assert!(func.validate_dag().is_ok());
                return Ok(MutationResult::Mutated);
            }
        }

        Ok(MutationResult::Skipped)
    }
}

#[derive(Default, Debug)]
pub struct RewireStatementMutator {
    typed: bool,
}

impl RewireStatementMutator {
    /// Creates a new [`StatementInputMutator`].
    #[must_use]
    pub fn new(typed: bool) -> Self {
        Self { typed }
    }
}

impl Named for RewireStatementMutator {
    fn name(&self) -> &str {
        if self.typed {
            "IRRewireStatementMutator (typed)"
        } else {
            "IRStatementInputMutator (untyped)"
        }
    }
}

impl<S> Mutator<S::Input, S> for RewireStatementMutator
where
    S: HasRand + HasCorpus<Input = LayeredInput>,
{
    fn mutate(
        &mut self,
        state: &mut S,
        input: &mut LayeredInput,
        _stage_idx: i32,
    ) -> Result<MutationResult, Error> {
        let LayeredInput::IR(ir) = input else {
            return Ok(MutationResult::Skipped);
        };

        let mut rng = StdRand::with_seed(state.rand_mut().next());
        let Some((fid, func)) = ir.iter_funcs().choose(&mut rng) else {
            return Ok(MutationResult::Skipped);
        };

        let mut stmts_with_inputs = 0u32;
        let mut target = None;
        let mut input_idx = None;
        let mut input_handle = None;
        let choose_stmt = |stmt: &Statement| {
            if matches!(stmt, Statement::Emit(_)) {
                return true;
            }
            if let Statement::Call { arguments, .. } = stmt {
                if arguments.is_empty() {
                    return true;
                }
            }
            for (idx, handle) in stmt.iter_used_exprs().enumerate() {
                let expr = &func.expressions[handle];
                if matches!(
                    expr,
                    Expression::CallResult(..)
                        | Expression::AtomicResult { .. }
                        | Expression::RayQueryProceedResult
                        | Expression::WorkGroupUniformLoadResult { .. }
                ) {
                    continue;
                }
                stmts_with_inputs += 1;
                if rng.gen_ratio(1, stmts_with_inputs) {
                    target = Some(stmt as *const Statement);
                    input_idx = Some(idx);
                    input_handle = Some(handle);
                }
            }
            true
        };
        func.visit_statements(choose_stmt);
        let Some(target) = target else {
            return Ok(MutationResult::Skipped);
        };
        let input_idx = input_idx.unwrap();
        let input_handle = input_handle.unwrap();

        assert!(func.validate_dag().is_ok());

        let scope = ExprScope::new_before(fid, target, ir.get_module(), false);
        if scope.is_empty() {
            return Ok(MutationResult::Skipped);
        }
        let module = ir.get_module_mut();
        let func = match fid {
            FunctionIdentifier::Function(handle) => &mut module.functions[handle],
            FunctionIdentifier::EntryPoint(idx) => &mut module.entry_points[idx].function,
        };

        let update_input = |stmt: &mut Statement| {
            if stmt as *const Statement != target {
                return true;
            }
            let target_input = stmt.iter_used_exprs_mut().nth(input_idx).unwrap();
            for _ in 0..5 {
                let replacement = {
                    if self.typed {
                        panic!("not supported");
                    }
                    scope.any().unwrap()
                };
                if replacement != *target_input {
                    *target_input = replacement;
                    return false;
                }
            }

            true
        };
        func.visit_statements_mut(update_input);

        if func.validate_dag().is_err() {
            let undo_rewire = |stmt: &mut Statement| {
                if stmt as *const Statement == target {
                    *stmt.iter_used_exprs_mut().nth(input_idx).unwrap() = input_handle;
                    return false;
                }
                true
            };
            func.visit_statements_mut(undo_rewire);
        }
        assert!(func.validate_dag().is_ok());

        Ok(MutationResult::Mutated)
    }
}

#[derive(Default, Debug)]
pub struct StatementMutator;

impl StatementMutator {
    /// Creates a new [`StatementMutator`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Named for StatementMutator {
    fn name(&self) -> &str {
        "IRStatementMutator"
    }
}

impl<S> Mutator<S::Input, S> for StatementMutator
where
    S: HasRand + HasCorpus<Input = LayeredInput>,
{
    fn mutate(
        &mut self,
        state: &mut S,
        input: &mut LayeredInput,
        _stage_idx: i32,
    ) -> Result<MutationResult, Error> {
        let LayeredInput::IR(ir) = input else {
            return Ok(MutationResult::Skipped);
        };

        let mut rng = StdRand::with_seed(state.rand_mut().next());
        let Some((fid, func)) = ir.iter_funcs_mut().choose(&mut rng) else {
            return Ok(MutationResult::Skipped);
        };

        assert!(func.validate_dag().is_ok());
        let func = ir.get_func_mut(fid);

        let Some(target_stmt) = DfsFuncIter::new(&func.body)
            .filter_map(|item| {
                let DfsItem::Statement(stmt) = item else {
                    return None;
                };
                match stmt {
                    Statement::Break
                    | Statement::Kill
                    | Statement::Emit(_)
                    | Statement::Return { value: None }
                    | Statement::Call { .. }
                    | Statement::WorkGroupUniformLoad { .. }
                    | Statement::RayQuery { .. }
                    | Statement::Continue => None,
                    Statement::If { .. }
                    | Statement::Block(_)
                    | Statement::Switch { .. }
                    | Statement::Loop { .. }
                    | Statement::Return { .. }
                    | Statement::Barrier(_)
                    | Statement::Store { .. }
                    | Statement::ImageStore { .. }
                    | Statement::Atomic { .. } => Some(stmt),
                }
            })
            .choose(&mut rng)
        else {
            return Ok(MutationResult::Skipped);
        };

        let mut available_exprs = Vec::new();
        for item in DfsFuncIter::new(&func.body) {
            match item {
                DfsItem::BlockOpen(_) => {
                    available_exprs.push(Vec::new());
                }
                DfsItem::BlockClose(_) => {
                    available_exprs.pop();
                }
                DfsItem::Statement(stmt) => {
                    if stmt as *const Statement == target_stmt {
                        break;
                    }
                    let Statement::Emit(exprs) = stmt else {
                        continue;
                    };
                    for e in exprs.clone() {
                        available_exprs.last_mut().unwrap().push(e);
                    }
                }
            }
        }
        let mut available_exprs: Vec<_> = available_exprs.into_iter().flatten().collect();
        for (handle, expr) in func.expressions.iter() {
            match expr {
                Expression::FunctionArgument(_)
                | Expression::LocalVariable(_)
                | Expression::GlobalVariable(_)
                | Expression::Constant(_)
                | Expression::Literal(_) => {
                    available_exprs.push(handle);
                }
                _ => {}
            }
        }

        if available_exprs.is_empty() {
            return Ok(MutationResult::Skipped);
        }
        let target_stmt = target_stmt as *const Statement;

        let mut blocks = Vec::new();
        blocks.push(&mut func.body);
        'outer: while let Some(block) = blocks.pop() {
            for stmt in block.iter_mut() {
                if stmt as *const Statement == target_stmt {
                    match stmt {
                        Statement::Break
                        | Statement::Kill
                        | Statement::Continue
                        | Statement::Emit(_)
                        | Statement::Call { .. }
                        | Statement::WorkGroupUniformLoad { .. }
                        | Statement::RayQuery { .. }
                        | Statement::Return { value: None } => {
                            unreachable!()
                        }
                        Statement::If {
                            condition,
                            accept,
                            reject,
                        } => {
                            if state.rand_mut().below(2) == 0 {
                                *condition = state.rand_mut().choose(available_exprs);
                            } else {
                                std::mem::swap(accept, reject);
                            }
                        }
                        Statement::Switch { selector, cases } => {
                            if state.rand_mut().probability(0.5) || cases.is_empty() {
                                *selector = state.rand_mut().choose(available_exprs);
                            } else {
                                let case = state.rand_mut().choose(cases);
                                case.fall_through = state.rand_mut().below(2) == 0;
                                match case.value {
                                    naga::SwitchValue::I32(ref mut v) => {
                                        *v = state.rand_mut().random_i32()
                                    }
                                    naga::SwitchValue::U32(ref mut v) => {
                                        *v = state.rand_mut().random_u32()
                                    }
                                    naga::SwitchValue::Default => {}
                                }
                            }
                        }
                        Statement::Loop { break_if, .. } => match state.rand_mut().below(2) {
                            0 => *break_if = None,
                            1 => *break_if = Some(state.rand_mut().choose(available_exprs)),
                            _ => unreachable!(),
                        },
                        Statement::Return { value } => {
                            *value = Some(state.rand_mut().choose(available_exprs));
                        }
                        Statement::Barrier(b) => {
                            *b = state
                                .rand_mut()
                                .choose([Barrier::STORAGE, Barrier::WORK_GROUP]);
                        }
                        Statement::Store { pointer, value } => match state.rand_mut().below(2) {
                            0 => *pointer = state.rand_mut().choose(available_exprs),
                            1 => *value = state.rand_mut().choose(available_exprs),
                            _ => unreachable!(),
                        },
                        Statement::ImageStore {
                            image,
                            coordinate,
                            array_index,
                            value,
                        } => match state.rand_mut().below(4) {
                            0 => *image = state.rand_mut().choose(available_exprs),
                            1 => *coordinate = state.rand_mut().choose(available_exprs),
                            2 => {
                                *array_index = Some(state.rand_mut().choose(available_exprs));
                            }
                            3 => *value = state.rand_mut().choose(available_exprs),
                            _ => unimplemented!(),
                        },
                        Statement::Block(b) => {
                            if !b.is_empty() {
                                if state.rand_mut().below(2) == 0 {
                                    let del_idx = state.rand_mut().below(b.len() as u64) as usize;
                                    b.cull(del_idx..=del_idx);
                                } else {
                                    let mut stmts: Vec<Statement> = b.iter().cloned().collect();
                                    state.rand_mut().shuffle(&mut stmts);
                                    *b = Block::from_vec(stmts);
                                }
                            }
                        }
                        Statement::Atomic {
                            pointer: _,
                            fun,
                            value: _,
                            result: _,
                        } => {
                            let from = [
                                AtomicFunction::Add,
                                AtomicFunction::Subtract,
                                AtomicFunction::And,
                                AtomicFunction::ExclusiveOr,
                                AtomicFunction::InclusiveOr,
                                AtomicFunction::Min,
                                AtomicFunction::Max,
                                AtomicFunction::Exchange {
                                    compare: Some(state.rand_mut().choose(available_exprs)),
                                },
                            ];
                            *fun = state.rand_mut().choose(from);
                        }
                    }
                    break 'outer;
                } else {
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

        assert!(func.validate_dag().is_ok());
        Ok(MutationResult::Mutated)
    }
}

#[derive(Default, Debug)]
pub struct TypeMutator;

impl TypeMutator {
    /// Creates a new [`TypeMutator`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    fn random_scalar_kind<R>(r: &mut R) -> ScalarKind
    where
        R: Rand,
    {
        r.choose([
            ScalarKind::Bool,
            ScalarKind::Float,
            ScalarKind::Sint,
            ScalarKind::Uint,
        ])
    }

    fn random_vector_size<R>(r: &mut R) -> VectorSize
    where
        R: Rand,
    {
        r.choose([VectorSize::Bi, VectorSize::Tri, VectorSize::Quad])
    }

    fn random_array_size<R>(r: &mut R) -> ArraySize
    where
        R: Rand,
    {
        if r.below(10) == 0 {
            ArraySize::Dynamic
        } else {
            let r = r.below(u16::MAX as u64) + 1;
            let r = std::num::NonZeroU32::new(r as u32).unwrap();
            ArraySize::Constant(r)
        }
    }

    fn random_address_space<R>(r: &mut R) -> AddressSpace
    where
        R: Rand,
    {
        r.choose([
            AddressSpace::Function,
            AddressSpace::Private,
            AddressSpace::WorkGroup,
            AddressSpace::Uniform,
            AddressSpace::Storage {
                access: StorageAccess::LOAD,
            },
            AddressSpace::Storage {
                access: StorageAccess::LOAD | StorageAccess::STORE,
            },
            AddressSpace::Handle,
            AddressSpace::PushConstant,
        ])
    }

    fn random_image_dimension<R>(r: &mut R) -> ImageDimension
    where
        R: Rand,
    {
        r.choose([
            ImageDimension::D1,
            ImageDimension::D2,
            ImageDimension::D3,
            ImageDimension::Cube,
        ])
    }

    fn random_builtin<R>(r: &mut R) -> BuiltIn
    where
        R: Rand,
    {
        r.choose([
            BuiltIn::Position { invariant: false },
            BuiltIn::Position { invariant: true },
            BuiltIn::ViewIndex,
            BuiltIn::InstanceIndex,
            BuiltIn::VertexIndex,
            BuiltIn::FragDepth,
            BuiltIn::FrontFacing,
            BuiltIn::PrimitiveIndex,
            BuiltIn::SampleIndex,
            BuiltIn::SampleMask,
            BuiltIn::GlobalInvocationId,
            BuiltIn::LocalInvocationId,
            BuiltIn::LocalInvocationIndex,
            BuiltIn::WorkGroupId,
            BuiltIn::NumWorkGroups,
        ])
    }
}

impl Named for TypeMutator {
    fn name(&self) -> &str {
        "IRTypeMutator"
    }
}

impl<S> Mutator<S::Input, S> for TypeMutator
where
    S: HasRand + HasCorpus<Input = LayeredInput>,
{
    fn mutate(
        &mut self,
        state: &mut S,
        input: &mut LayeredInput,
        _stage_idx: i32,
    ) -> Result<MutationResult, Error> {
        let LayeredInput::IR(ir) = input else {
            return Ok(MutationResult::Skipped);
        };

        let module = ir.get_module();
        let types: Vec<Handle<Type>> = module
            .types
            .iter()
            .filter_map(|(handle, ir_type)| match ir_type.inner {
                Ti::Scalar { .. } => None,
                Ti::AccelerationStructure { .. } => None,
                Ti::RayQuery { .. } => None,
                Ti::Vector { .. } => None,
                _ => Some(handle),
            })
            .collect();
        if types.is_empty() {
            return Ok(MutationResult::Skipped);
        }

        let type_handle = *state.rand_mut().choose(&types);
        let ir_type = module.types.get_handle(type_handle).unwrap().clone();

        use naga::TypeInner as Ti;
        let new_inner = match ir_type.inner {
            Ti::Scalar { .. } => {
                unreachable!();
            }
            Ti::Vector { .. } => {
                unreachable!();
            }
            Ti::Matrix {
                columns,
                rows,
                width,
            } => match state.rand_mut().below(3) {
                0 => {
                    let columns = Self::random_vector_size(state.rand_mut());
                    Ti::Matrix {
                        columns,
                        rows,
                        width,
                    }
                }
                1 => {
                    let rows = Self::random_vector_size(state.rand_mut());
                    Ti::Matrix {
                        columns,
                        rows,
                        width,
                    }
                }
                2 => {
                    let width = state.rand_mut().random_u8();
                    Ti::Matrix {
                        columns,
                        rows,
                        width,
                    }
                }
                _ => {
                    unreachable!()
                }
            },
            Ti::Atomic { kind, width } => match state.rand_mut().below(2) {
                0 => {
                    let kind = Self::random_scalar_kind(state.rand_mut());
                    Ti::Atomic { kind, width }
                }
                1 => {
                    let width = state.rand_mut().random_u8();
                    Ti::Atomic { kind, width }
                }
                _ => {
                    unreachable!()
                }
            },
            Ti::Pointer { base, space } => match state.rand_mut().below(2) {
                0 => {
                    let base = *state.rand_mut().choose(&types);
                    Ti::Pointer { base, space }
                }
                1 => {
                    let space = Self::random_address_space(state.rand_mut());
                    Ti::Pointer { base, space }
                }
                _ => {
                    unreachable!()
                }
            },
            Ti::ValuePointer {
                size,
                kind,
                width,
                space,
            } => match state.rand_mut().below(4) {
                0 => {
                    let mut size = Some(Self::random_vector_size(state.rand_mut()));
                    if state
                        .rand_mut()
                        .below(std::mem::variant_count::<VectorSize>() as u64 + 1)
                        == 0
                    {
                        size = None;
                    }
                    Ti::ValuePointer {
                        size,
                        kind,
                        width,
                        space,
                    }
                }
                1 => {
                    let kind = Self::random_scalar_kind(state.rand_mut());
                    Ti::ValuePointer {
                        size,
                        kind,
                        width,
                        space,
                    }
                }
                2 => {
                    let width = state.rand_mut().random_u8();
                    Ti::ValuePointer {
                        size,
                        kind,
                        width,
                        space,
                    }
                }
                3 => {
                    let space = Self::random_address_space(state.rand_mut());
                    Ti::ValuePointer {
                        size,
                        kind,
                        width,
                        space,
                    }
                }
                _ => {
                    unreachable!()
                }
            },
            Ti::Array { base, size, stride } => match state.rand_mut().below(3) {
                0 => {
                    let base = *state.rand_mut().choose(&types);
                    Ti::Array { base, size, stride }
                }
                1 => {
                    let size = Self::random_array_size(state.rand_mut());
                    Ti::Array { base, size, stride }
                }
                2 => {
                    let stride = state.rand_mut().random_u32();
                    Ti::Array { base, size, stride }
                }
                _ => {
                    unreachable!()
                }
            },
            Ti::Struct { ref members, span } => {
                if members.is_empty() {
                    return Ok(MutationResult::Skipped);
                }
                let mut members = members.clone();

                let StructMember {
                    name: _,
                    ty,
                    binding,
                    offset: _,
                } = state.rand_mut().choose(&mut members);

                match state.rand_mut().below(2) {
                    0 => {
                        *ty = *state.rand_mut().choose(&types);
                    }
                    1 => {
                        *binding = match state.rand_mut().below(3) {
                            0 => None,
                            1 => Some(Binding::BuiltIn(Self::random_builtin(state.rand_mut()))),
                            2 => Some(Binding::Location {
                                location: state.rand_mut().random_u32(),
                                second_blend_source: state.rand_mut().random_bool(),
                                interpolation: None,
                                sampling: None,
                            }),
                            _ => {
                                unreachable!()
                            }
                        };
                    }
                    _ => {
                        unreachable!()
                    }
                }
                Ti::Struct { members, span }
            }
            Ti::Image {
                dim,
                arrayed,
                class,
            } => match state.rand_mut().below(3) {
                0 => {
                    let dim = Self::random_image_dimension(state.rand_mut());
                    Ti::Image {
                        dim,
                        arrayed,
                        class,
                    }
                }
                1 => {
                    let arrayed = !arrayed;
                    Ti::Image {
                        dim,
                        arrayed,
                        class,
                    }
                }
                2 => Ti::Image {
                    dim,
                    arrayed,
                    class,
                },
                _ => {
                    unreachable!()
                }
            },
            Ti::Sampler { comparison } => {
                let comparison = !comparison;
                Ti::Sampler { comparison }
            }
            Ti::BindingArray { base, size } => match state.rand_mut().below(2) {
                0 => {
                    let base = *state.rand_mut().choose(&types);
                    Ti::BindingArray { base, size }
                }
                1 => {
                    let size = Self::random_array_size(state.rand_mut());
                    Ti::BindingArray { base, size }
                }
                _ => {
                    unreachable!()
                }
            },
            Ti::AccelerationStructure | Ti::RayQuery => {
                unreachable!();
            }
        };
        let new_type = Type {
            name: ir_type.name.clone(),
            inner: new_inner,
        };
        if module.types.get(&new_type).is_some() {
            return Ok(MutationResult::Skipped);
        }

        Ok(MutationResult::Mutated)
    }
}

#[derive(Debug)]
pub struct FullGenerationMutation {
    generator: IRGenerator,
}

impl FullGenerationMutation {
    /// Creates a new [`FullGenerationMutation`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            generator: IRGenerator::new(Default::default()),
        }
    }
}

impl Named for FullGenerationMutation {
    fn name(&self) -> &str {
        "IRFullGenerationMutation"
    }
}

impl<S> Mutator<S::Input, S> for FullGenerationMutation
where
    S: HasRand + HasCorpus<Input = LayeredInput>,
{
    fn mutate(
        &mut self,
        state: &mut S,
        input: &mut LayeredInput,
        stage_idx: i32,
    ) -> Result<MutationResult, Error> {
        if stage_idx > 0 {
            return Ok(MutationResult::Skipped);
        }

        let result = self.generator.generate(state)?;
        if matches!(result, LayeredInput::IR(..)) {
            *input = result;
            Ok(MutationResult::Mutated)
        } else {
            Ok(MutationResult::Skipped)
        }
    }
}

#[derive(Default, Debug)]
pub struct CodeGenerationMutation {
    config: GeneratorConfig,
}

impl CodeGenerationMutation {
    /// Creates a new [`CodeGenerationMutation`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: Default::default(),
        }
    }
}

impl Named for CodeGenerationMutation {
    fn name(&self) -> &str {
        "IRCodeGenerationMutation"
    }
}

impl<S> Mutator<S::Input, S> for CodeGenerationMutation
where
    S: HasRand + HasCorpus<Input = LayeredInput>,
{
    fn mutate(
        &mut self,
        state: &mut S,
        input: &mut LayeredInput,
        _stage_idx: i32,
    ) -> Result<MutationResult, Error> {
        let LayeredInput::IR(ir) = input else {
            return Ok(MutationResult::Skipped);
        };

        let mut rng = StdRand::with_seed(state.rand_mut().next());
        let Some((fid, func)) = ir.iter_funcs().choose(&mut rng) else {
            return Ok(MutationResult::Skipped);
        };
        assert!(func.validate_dag().is_ok());

        let mut blocks = 0u32;
        let mut target = None;
        let block_chooser = |block: &Block| {
            blocks += 1;
            if rng.gen_ratio(1, blocks) {
                target = Some(block as *const Block);
            }
            true
        };
        func.visit_blocks(block_chooser);

        let budget = rng.between(10, 40) as u32;
        let mut gen =
            FunctionGenCtx::new(&self.config, ir.get_module_mut(), fid, rng.next(), false);
        gen.generate_at(target.unwrap(), budget);
        let func = ir.get_func(fid);
        assert!(func.validate_dag().is_ok());

        Ok(MutationResult::Mutated)
    }
}
