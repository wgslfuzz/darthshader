use libafl::{
    mutators::{MutationResult, Mutator},
    state::{HasCorpus, HasRand},
};
use libafl_bolts::{
    rands::{Rand, StdRand},
    tuples::{tuple_list, tuple_list_type},
    Error, Named,
};
use naga::{
    Arena, Block, Expression, FastIndexMap, Function, GlobalVariable, Handle, LocalVariable, Range,
    Span, Statement,
};
use rand::seq::{IteratorRandom, SliceRandom};

use crate::{
    ir::{
        funcext::FuncExt,
        iter::{BlockVisitorMut, IterFuncs, StatementVisitor, StatementVisitorMut, UsesExprIter},
    },
    layeredinput::LayeredInput,
};

use super::exprscope::ExprScope;

pub type IRMinimizersType = tuple_list_type!(
    EntryPointDeleteMutator,
    FunctionDeleteMutator,
    GlobalDeleteMutator,
    ResultDeleteMutator,
    LocalVarDeleteMutator,
    BlockSimplifyMutator,
    IfSimplifyMutator,
    SwitchSimplifyMutator,
    StatementDeleteMutator,
    ExpressionDeleteMutator,
    DuplicateDeleteMutator,
    ConstExpressionDeleteMutator,
);

/// Get the minimizers that simplify the IR module
#[must_use]
pub const fn ir_minimizers() -> IRMinimizersType {
    tuple_list!(
        EntryPointDeleteMutator::new(),
        FunctionDeleteMutator::new(),
        GlobalDeleteMutator::new(),
        ResultDeleteMutator::new(),
        LocalVarDeleteMutator::new(),
        BlockSimplifyMutator::new(),
        IfSimplifyMutator::new(),
        SwitchSimplifyMutator::new(),
        StatementDeleteMutator::new(),
        ExpressionDeleteMutator::new(),
        DuplicateDeleteMutator::new(),
        ConstExpressionDeleteMutator::new(),
    )
}

#[derive(Default, Debug)]
pub struct EntryPointDeleteMutator;

impl EntryPointDeleteMutator {
    /// Creates a new [`EntryPointDeleteMutator`].
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Named for EntryPointDeleteMutator {
    fn name(&self) -> &str {
        "IREntryPointDeleteMutator"
    }
}

impl<S> Mutator<S::Input, S> for EntryPointDeleteMutator
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

        let eps = &mut ir.get_module_mut().entry_points;
        if eps.is_empty() {
            return Ok(MutationResult::Skipped);
        }
        let idx = state.rand_mut().below(eps.len() as u64) as usize;
        eps.remove(idx);

        Ok(MutationResult::Mutated)
    }
}

#[derive(Default, Debug)]
pub struct FunctionDeleteMutator;

impl FunctionDeleteMutator {
    /// Creates a new [`FunctionDeleteMutator`].
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Named for FunctionDeleteMutator {
    fn name(&self) -> &str {
        "IRFunctionDeleteMutator"
    }
}

impl<S> Mutator<S::Input, S> for FunctionDeleteMutator
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

        let mut unused = vec![true; ir.get_module().functions.len()];
        let mut gather_unused = |stmt: &Statement| {
            if let Statement::Call { function, .. } = stmt {
                unused[function.index()] = false;
            }
            true
        };
        for (_, func) in ir.iter_funcs() {
            func.visit_statements(&mut gather_unused);
        }
        for (_, func) in ir.iter_funcs() {
            for (_, expr) in func.expressions.iter() {
                if let Expression::CallResult(function) = expr {
                    unused[function.index()] = false;
                }
            }
        }

        let mut rng = StdRand::with_seed(state.rand_mut().next());
        let Some(idx) = unused
            .iter()
            .enumerate()
            .filter_map(|(idx, unused)| unused.then_some(idx))
            .choose(&mut rng)
        else {
            return Ok(MutationResult::Skipped);
        };

        let mut mapping: Vec<_> = ir
            .get_module()
            .functions
            .iter()
            .map(|(handle, _)| Some(handle))
            .collect();
        mapping.pop().unwrap();
        mapping.insert(idx, None);

        let functions = std::mem::take(&mut ir.get_module_mut().functions);
        let mut functions = functions.into_inner();
        functions.remove(idx);

        let mut new_functions: Arena<Function> = Default::default();
        for func in functions.into_iter() {
            new_functions.append(func, Span::UNDEFINED);
        }
        ir.get_module_mut().functions = new_functions;

        let update_mapping = |stmt: &mut Statement| {
            if let Statement::Call { function, .. } = stmt {
                *function = mapping[function.index()].unwrap();
            }
            true
        };
        for (_, func) in ir.iter_funcs_mut() {
            func.visit_statements_mut(update_mapping);
        }
        for (_, func) in ir.iter_funcs_mut() {
            for (_, expr) in func.expressions.iter_mut() {
                if let Expression::CallResult(function) = expr {
                    *function = mapping[function.index()].unwrap();
                }
            }
        }

        Ok(MutationResult::Mutated)
    }
}

#[derive(Default, Debug)]
pub struct LocalVarDeleteMutator;

impl LocalVarDeleteMutator {
    /// Creates a new [`LocalVarDeleteMutator`].
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Named for LocalVarDeleteMutator {
    fn name(&self) -> &str {
        "IRLocalVarDeleteMutator"
    }
}

impl<S> Mutator<S::Input, S> for LocalVarDeleteMutator
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
        let mut funcs: Vec<_> = ir.iter_funcs_mut().map(|(_, func)| func).collect();
        funcs.shuffle(&mut rng);
        let mut unused = Vec::new();

        for func in funcs.into_iter() {
            unused.clear();
            unused.resize(func.local_variables.len(), true);
            for (_, expr) in func.expressions.iter() {
                if let Expression::LocalVariable(handle) = expr {
                    unused[handle.index()] = false;
                }
            }

            let Some(idx) = unused
                .iter()
                .enumerate()
                .filter_map(|(idx, unused)| unused.then_some(idx))
                .choose(&mut rng)
            else {
                continue;
            };

            let mut mapping: Vec<_> = func
                .local_variables
                .iter()
                .map(|(handle, _)| Some(handle))
                .collect();
            mapping.pop().unwrap();
            mapping.insert(idx, None);

            let local_vars = std::mem::take(&mut func.local_variables);
            let mut local_vars = local_vars.into_inner();
            local_vars.remove(idx);

            let mut new_local_vars: Arena<LocalVariable> = Default::default();
            for local_var in local_vars.into_iter() {
                new_local_vars.append(local_var, Span::UNDEFINED);
            }
            func.local_variables = new_local_vars;

            for (_, expr) in func.expressions.iter_mut() {
                if let Expression::LocalVariable(handle) = expr {
                    *handle = mapping[handle.index()].unwrap();
                }
            }
            return Ok(MutationResult::Mutated);
        }

        Ok(MutationResult::Skipped)
    }
}

#[derive(Default, Debug)]
pub struct GlobalDeleteMutator;

impl GlobalDeleteMutator {
    /// Creates a new [`CallDeleteMutator`].
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Named for GlobalDeleteMutator {
    fn name(&self) -> &str {
        "IRGlobalDeleteMutator"
    }
}

impl<S> Mutator<S::Input, S> for GlobalDeleteMutator
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
        let mut unused = vec![true; ir.get_module().global_variables.len()];
        for (_, func) in ir.iter_funcs() {
            for (_, expr) in func.expressions.iter() {
                if let Expression::GlobalVariable(handle) = expr {
                    unused[handle.index()] = false;
                }
            }
        }

        let Some(idx) = unused
            .iter()
            .enumerate()
            .filter_map(|(idx, unused)| unused.then_some(idx))
            .choose(&mut rng)
        else {
            return Ok(MutationResult::Skipped);
        };

        let mut mapping: Vec<_> = ir
            .get_module()
            .global_variables
            .iter()
            .map(|(handle, _)| Some(handle))
            .collect();
        mapping.pop().unwrap();
        mapping.insert(idx, None);

        let global_vars = std::mem::take(&mut ir.get_module_mut().global_variables);
        let mut global_vars = global_vars.into_inner();
        global_vars.remove(idx);
        let mut new_global_vars: Arena<GlobalVariable> = Default::default();
        for global_var in global_vars.into_iter() {
            new_global_vars.append(global_var, Span::UNDEFINED);
        }
        ir.get_module_mut().global_variables = new_global_vars;

        for (_, func) in ir.iter_funcs_mut() {
            for (_, expr) in func.expressions.iter_mut() {
                if let Expression::GlobalVariable(handle) = expr {
                    *handle = mapping[handle.index()].unwrap();
                }
            }
        }

        Ok(MutationResult::Mutated)
    }
}

#[derive(Default, Debug)]
pub struct ResultDeleteMutator;

impl ResultDeleteMutator {
    /// Creates a new [`ResultDeleteMutator`].
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Named for ResultDeleteMutator {
    fn name(&self) -> &str {
        "IRResultDeleteMutator"
    }
}

impl<S> Mutator<S::Input, S> for ResultDeleteMutator
where
    S: HasRand + HasCorpus<Input = LayeredInput>,
{
    fn mutate(
        &mut self,
        state: &mut S,
        input: &mut LayeredInput,
        _stage_idx: i32,
    ) -> Result<MutationResult, Error> {
        enum ResultStatement {
            Call(Handle<Function>, Option<Handle<Expression>>),
            Atomic(Handle<Expression>),
            WorkgroupLoad(Handle<Expression>),
        }

        let LayeredInput::IR(ir) = input else {
            return Ok(MutationResult::Skipped);
        };

        let mut rng = StdRand::with_seed(state.rand_mut().next());
        let mut funcs: Vec<_> = ir.iter_funcs().map(|(fid, _)| fid).collect();
        funcs.shuffle(&mut rng);
        for fid in funcs.into_iter() {
            let func = ir.get_func_mut(fid);
            let mut results = 0u32;
            let count_results = |stmt: &Statement| {
                if matches!(
                    stmt,
                    Statement::Call { .. }
                        | Statement::Atomic { .. }
                        | Statement::WorkGroupUniformLoad { .. }
                ) {
                    results += 1;
                }
                true
            };
            func.visit_statements(count_results);

            if results > 0 {
                let mut countdown = rng.below(results as u64);
                let mut result_stmt = None;

                let replace_result = |block: &mut Block| {
                    for (idx, stmt) in block.iter().enumerate() {
                        if !matches!(
                            stmt,
                            Statement::Call { .. }
                                | Statement::Atomic { .. }
                                | Statement::WorkGroupUniformLoad { .. }
                        ) {
                            continue;
                        }
                        if countdown != 0 {
                            countdown -= 1;
                            continue;
                        }
                        result_stmt = match stmt {
                            Statement::Call {
                                function, result, ..
                            } => Some(ResultStatement::Call(*function, *result)),
                            Statement::Atomic { result, .. } => {
                                Some(ResultStatement::Atomic(*result))
                            }
                            Statement::WorkGroupUniformLoad { result, .. } => {
                                Some(ResultStatement::WorkgroupLoad(*result))
                            }
                            _ => unreachable!(),
                        };
                        block.cull(idx..=idx);
                        return false;
                    }
                    true
                };
                func.visit_blocks_mut(replace_result);

                let result_stmt = result_stmt.unwrap();
                let result_ty = match result_stmt {
                    ResultStatement::Call(function, result) => result.map(|result| {
                        (
                            result,
                            ir.get_module()
                                .functions
                                .try_get(function)
                                .unwrap()
                                .result
                                .as_ref()
                                .unwrap()
                                .ty,
                        )
                    }),
                    ResultStatement::Atomic(result) => match func.expressions[result] {
                        Expression::AtomicResult { ty, .. } => Some((result, ty)),
                        _ => None,
                    },
                    ResultStatement::WorkgroupLoad(result) => match func.expressions[result] {
                        Expression::WorkGroupUniformLoadResult { ty } => Some((result, ty)),
                        _ => None,
                    },
                };

                if let Some((result, ty)) = result_ty {
                    ir.get_func_mut(fid).expressions[result] = Expression::ZeroValue(ty);
                }
                let func = ir.get_func(fid);
                assert!(func.validate_dag().is_ok());
                return Ok(MutationResult::Mutated);
            }
        }

        Ok(MutationResult::Skipped)
    }
}

#[derive(Default, Debug)]
pub struct BlockSimplifyMutator;

impl BlockSimplifyMutator {
    /// Creates a new [`BlockSimplifyMutator`].
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Named for BlockSimplifyMutator {
    fn name(&self) -> &str {
        "IRBlockSimplifyMutator"
    }
}

impl<S> Mutator<S::Input, S> for BlockSimplifyMutator
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
        let mut funcs: Vec<_> = ir.iter_funcs_mut().map(|(fid, _)| fid).collect();
        funcs.shuffle(&mut rng);
        for fid in funcs.into_iter() {
            let func = ir.get_func_mut(fid);
            let mut blocks = 0u32;
            let count_blocks = |stmt: &Statement| {
                if matches!(stmt, Statement::Block { .. }) {
                    blocks += 1;
                }
                true
            };
            func.visit_statements(count_blocks);

            if blocks > 0 {
                let mut countdown = rng.below(blocks as u64);
                let outline_block = |block: &mut Block| {
                    for (idx, stmt) in block.iter_mut().enumerate() {
                        if let Statement::Block(inner_block) = stmt {
                            if countdown == 0 {
                                let inner_block = std::mem::take(inner_block);
                                block.splice(idx..=idx, inner_block);
                                return false;
                            }
                            countdown -= 1;
                        }
                    }
                    true
                };
                func.visit_blocks_mut(outline_block);
                assert!(func.validate_dag().is_ok());
                return Ok(MutationResult::Mutated);
            }
        }

        Ok(MutationResult::Skipped)
    }
}

#[derive(Default, Debug)]
pub struct IfSimplifyMutator;

impl IfSimplifyMutator {
    /// Creates a new [`IfSimplifyMutator`].
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Named for IfSimplifyMutator {
    fn name(&self) -> &str {
        "IRIfSimplifyMutator"
    }
}

impl<S> Mutator<S::Input, S> for IfSimplifyMutator
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
        let mut funcs: Vec<_> = ir.iter_funcs().map(|(fid, _)| fid).collect();
        funcs.shuffle(&mut rng);
        for fid in funcs.into_iter() {
            let func = ir.get_func_mut(fid);
            let mut ifs = 0u32;
            let count_ifs = |stmt: &Statement| {
                if matches!(stmt, Statement::If { .. }) {
                    ifs += 1;
                }
                true
            };
            func.visit_statements(count_ifs);

            if ifs > 0 {
                let mut countdown = rng.below(ifs as u64);
                let replace_if = |block: &mut Block| {
                    for (idx, stmt) in block.iter_mut().enumerate() {
                        if let Statement::If { accept, reject, .. } = stmt {
                            if countdown == 0 {
                                let accept = std::mem::take(accept);
                                let reject = std::mem::take(reject);
                                let mut new_block = Block::new();
                                new_block.push(Statement::Block(accept), Span::UNDEFINED);
                                new_block.push(Statement::Block(reject), Span::UNDEFINED);
                                block.splice(idx..=idx, new_block);
                                return false;
                            }
                            countdown -= 1;
                        }
                    }
                    true
                };
                func.visit_blocks_mut(replace_if);
                assert!(func.validate_dag().is_ok());
                return Ok(MutationResult::Mutated);
            }
        }

        Ok(MutationResult::Skipped)
    }
}

#[derive(Default, Debug)]
pub struct SwitchSimplifyMutator;

impl SwitchSimplifyMutator {
    /// Creates a new [`SwitchSimplifyMutator`].
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Named for SwitchSimplifyMutator {
    fn name(&self) -> &str {
        "IRSwitchSimplifyMutator"
    }
}

impl<S> Mutator<S::Input, S> for SwitchSimplifyMutator
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
        let mut funcs: Vec<_> = ir.iter_funcs().map(|(fid, _)| fid).collect();
        funcs.shuffle(&mut rng);
        for fid in funcs.into_iter() {
            let func = ir.get_func_mut(fid);
            let mut switches = 0u32;
            let count_switches = |stmt: &Statement| {
                if matches!(stmt, Statement::Switch { .. }) {
                    switches += 1;
                }
                true
            };
            func.visit_statements(count_switches);

            let mut result = MutationResult::Skipped;
            if switches > 0 {
                let mut countdown = rng.below(switches as u64);
                let simplify_switch = |block: &mut Block| {
                    for stmt in block.iter_mut() {
                        if let Statement::Switch { cases, .. } = stmt {
                            if countdown == 0 {
                                if cases.is_empty() {
                                    return false;
                                }
                                let case_idx = rng.below(cases.len() as u64) as usize;
                                let _ = cases.remove(case_idx);
                                result = MutationResult::Mutated;
                                return false;
                            }
                            countdown -= 1;
                        }
                    }
                    true
                };
                func.visit_blocks_mut(simplify_switch);

                assert!(func.validate_dag().is_ok());
                return Ok(result);
            }
        }

        Ok(MutationResult::Skipped)
    }
}

#[derive(Default, Debug)]
pub struct ExpressionDeleteMutator;

impl ExpressionDeleteMutator {
    /// Creates a new [`ExpressionDeleteMutator`].
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Named for ExpressionDeleteMutator {
    fn name(&self) -> &str {
        "IRExpressionDeleteMutator"
    }
}

impl<S> Mutator<S::Input, S> for ExpressionDeleteMutator
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
        let mut funcs: Vec<_> = ir.iter_funcs_mut().collect();
        funcs.shuffle(&mut rng);
        let mut unused = Vec::new();
        for (_, func) in funcs.into_iter() {
            unused.clear();
            unused.resize(func.expressions.len(), true);

            for (_, loc_var) in func.local_variables.iter() {
                if let Some(handle) = loc_var.init {
                    unused[handle.index()] = false;
                }
            }
            let visitor = |stmt: &Statement| {
                if let Statement::Emit(range) = stmt {
                    assert_eq!(range.zero_based_index_range().len(), 1);
                } else {
                    for handle in stmt.iter_used_exprs() {
                        if unused[handle.index()] {
                            unused[handle.index()] = false;
                        }
                    }
                }
                true
            };
            func.visit_statements(visitor);
            for (_, expr) in func.expressions.iter() {
                for handle in expr.iter_used_exprs() {
                    unused[handle.index()] = false;
                }
            }

            let Some(idx) = unused
                .iter()
                .enumerate()
                .filter_map(|(idx, unused)| unused.then_some(idx))
                .choose(&mut rng)
            else {
                continue;
            };

            let mut mapping: Vec<_> = func
                .expressions
                .iter()
                .map(|(handle, _)| Some(handle))
                .collect();
            mapping.pop().unwrap();
            mapping.insert(idx, None);
            let remove_emit = |block: &mut Block| {
                while let Some(stmt_idx) = block.iter().position(|stmt| {
                    let Statement::Emit(range) = stmt else {
                        return false;
                    };
                    let Some((first, last)) = range.first_and_last() else {
                        return false;
                    };
                    assert_eq!(first.index(), last.index());
                    first.index() == idx
                }) {
                    block.cull(stmt_idx..=stmt_idx);
                }
                true
            };
            func.visit_blocks_mut(remove_emit);

            let update_stmts = |stmt: &mut Statement| {
                match stmt {
                    Statement::Emit(range) => {
                        if let Some((first, last)) = range.first_and_last() {
                            assert_eq!(first.index(), last.index());
                            let new_handle = mapping[first.index()].unwrap();
                            *range = Range::new_from_bounds(new_handle, new_handle);
                        }
                    }
                    stmt => {
                        for handle in stmt.iter_used_exprs_mut() {
                            *handle = mapping[handle.index()].unwrap();
                        }
                    }
                }
                true
            };
            func.visit_statements_mut(update_stmts);

            for (_, expr) in func.expressions.iter_mut() {
                for handle in expr.iter_used_exprs_mut() {
                    *handle = mapping[handle.index()].unwrap();
                }
            }

            for (_, loc_var) in func.local_variables.iter_mut() {
                if let Some(handle) = &mut loc_var.init {
                    *handle = mapping[handle.index()].unwrap();
                }
            }

            let old_exprs = std::mem::take(&mut func.expressions);
            let mut old_exprs = old_exprs.into_inner();
            old_exprs.remove(idx);
            let mut new_exprs: Arena<Expression> = Default::default();
            for expr in old_exprs.into_iter() {
                new_exprs.append(expr, Span::UNDEFINED);
            }
            func.expressions = new_exprs;

            let old_named_exprs = std::mem::take(&mut func.named_expressions);
            let mut new_named_exprs: FastIndexMap<Handle<Expression>, String> =
                FastIndexMap::default();
            for (handle, name) in old_named_exprs.into_iter() {
                if handle.index() != idx {
                    new_named_exprs.insert(mapping[handle.index()].unwrap(), name);
                }
            }
            func.named_expressions = new_named_exprs;

            assert!(func.validate_dag().is_ok());
            return Ok(MutationResult::Mutated);
        }

        Ok(MutationResult::Skipped)
    }
}

#[derive(Default, Debug)]
pub struct DuplicateDeleteMutator;

impl DuplicateDeleteMutator {
    /// Creates a new [`RedundantDeleteMutator`].
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Named for DuplicateDeleteMutator {
    fn name(&self) -> &str {
        "IRDuplicateDeleteMutator"
    }
}

impl<S> Mutator<S::Input, S> for DuplicateDeleteMutator
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
        let mut funcs: Vec<_> = ir.iter_funcs_mut().collect();
        funcs.shuffle(&mut rng);
        for (_, func) in funcs.into_iter() {
            let mut simplified_any = false;
            let mut new_exprs: Arena<Expression> = Arena::new();

            let mut mapping = vec![None; func.expressions.len()];
            for (handle, expr) in func.expressions.iter() {
                let mut found = false;
                if ExprScope::is_always_available(expr) {
                    for (other_handle, other_expr) in new_exprs.iter() {
                        if other_expr == expr {
                            mapping[handle.index()] = Some(other_handle);
                            simplified_any = true;
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        let new_handle = new_exprs.append(expr.clone(), Span::default());
                        mapping[handle.index()] = Some(new_handle);
                    }
                }
            }

            for (handle, expr) in func.expressions.iter() {
                if !ExprScope::is_always_available(expr) {
                    let new_handle = new_exprs.append(expr.clone(), Span::default());
                    mapping[handle.index()] = Some(new_handle);
                }
            }
            assert_eq!(mapping.len(), func.expressions.len());

            if !simplified_any {
                continue;
            }
            func.expressions = new_exprs;

            for (_, expr) in func.expressions.iter_mut() {
                for handle in expr.iter_used_exprs_mut() {
                    *handle = mapping[handle.index()].unwrap();
                }
            }

            for (_, loc_var) in func.local_variables.iter_mut() {
                if let Some(handle) = &mut loc_var.init {
                    *handle = mapping[handle.index()].unwrap();
                }
            }

            let update_stmts = |stmt: &mut Statement| {
                match stmt {
                    Statement::Emit(range) => {
                        if let Some((first, last)) = range.first_and_last() {
                            assert_eq!(first.index(), last.index());
                            let new_handle = mapping[first.index()].unwrap();
                            *range = Range::new_from_bounds(new_handle, new_handle);
                        }
                    }
                    stmt => {
                        for handle in stmt.iter_used_exprs_mut() {
                            *handle = mapping[handle.index()].unwrap();
                        }
                    }
                }
                true
            };
            func.visit_statements_mut(update_stmts);

            let old_named_exprs = std::mem::take(&mut func.named_expressions);
            let mut new_named_exprs: FastIndexMap<Handle<Expression>, String> =
                FastIndexMap::default();
            for (handle, name) in old_named_exprs.into_iter() {
                new_named_exprs.insert(mapping[handle.index()].unwrap(), name);
            }
            func.named_expressions = new_named_exprs;

            return Ok(MutationResult::Mutated);
        }
        Ok(MutationResult::Skipped)
    }
}

#[derive(Default, Debug)]
pub struct ConstExpressionDeleteMutator;

impl ConstExpressionDeleteMutator {
    /// Creates a new [`ConstExpressionDeleteMutator`].
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Named for ConstExpressionDeleteMutator {
    fn name(&self) -> &str {
        "IRConstExpressionDeleteMutator"
    }
}

impl<S> Mutator<S::Input, S> for ConstExpressionDeleteMutator
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
        let mut unused = vec![true; ir.get_module().const_expressions.len()];
        for handle in ir
            .iter_funcs()
            .flat_map(|(_, func)| func.expressions.iter())
            .filter_map(|(_, expr)| match expr {
                Expression::ImageSample { offset, .. } => *offset,
                _ => None,
            })
        {
            unused[handle.index()] = false;
        }

        for (_, global_var) in ir.get_module().global_variables.iter() {
            if let Some(handle) = global_var.init {
                unused[handle.index()] = false;
            }
        }

        for (_, expr) in ir.get_module().const_expressions.iter() {
            for handle in expr.iter_used_exprs() {
                unused[handle.index()] = false;
            }
        }

        for (_, constant) in ir.get_module().constants.iter() {
            unused[constant.init.index()] = false;
        }

        let Some(idx) = unused
            .iter()
            .enumerate()
            .filter_map(|(idx, unused)| unused.then_some(idx))
            .choose(&mut rng)
        else {
            return Ok(MutationResult::Skipped);
        };

        let mut mapping: Vec<_> = ir
            .get_module()
            .const_expressions
            .iter()
            .map(|(handle, _)| Some(handle))
            .collect();
        mapping.pop().unwrap();
        mapping.insert(idx, None);

        for (_, expr) in ir.get_module_mut().const_expressions.iter_mut() {
            for handle in expr.iter_used_exprs_mut() {
                *handle = mapping[handle.index()].unwrap();
            }
        }

        for (_, global_var) in ir.get_module_mut().global_variables.iter_mut() {
            if let Some(handle) = &mut global_var.init {
                *handle = mapping[handle.index()].unwrap();
            }
        }

        for (_, constant) in ir.get_module_mut().constants.iter_mut() {
            let handle = &mut constant.init;
            *handle = mapping[handle.index()].unwrap();
        }

        for (_, expr) in ir
            .iter_funcs_mut()
            .flat_map(|(_, func)| func.expressions.iter_mut())
        {
            if let Expression::ImageSample {
                offset: Some(offset),
                ..
            } = expr
            {
                *offset = mapping[offset.index()].unwrap();
            }
        }

        let old_exprs = std::mem::take(&mut ir.get_module_mut().const_expressions);
        let mut old_exprs = old_exprs.into_inner();
        old_exprs.remove(idx);
        let mut new_exprs: Arena<Expression> = Default::default();
        for expr in old_exprs.into_iter() {
            new_exprs.append(expr, Span::UNDEFINED);
        }
        ir.get_module_mut().const_expressions = new_exprs;

        Ok(MutationResult::Mutated)
    }
}

#[derive(Default, Debug)]
pub struct StatementDeleteMutator;

impl StatementDeleteMutator {
    /// Creates a new [`StatementDeleteMutator`].
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Named for StatementDeleteMutator {
    fn name(&self) -> &str {
        "IRStatementDeleteMutator"
    }
}

impl<S> Mutator<S::Input, S> for StatementDeleteMutator
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

        let allowed_stmt = |stmt: &Statement| {
            !matches!(
                stmt,
                Statement::Call { .. }
                    | Statement::Atomic { .. }
                    | Statement::WorkGroupUniformLoad { .. }
                    | Statement::Emit(_)
            )
        };

        let mut rng = StdRand::with_seed(state.rand_mut().next());
        let mut funcs: Vec<_> = ir.iter_funcs_mut().map(|(_, func)| func).collect();
        funcs.shuffle(&mut rng);
        for func in funcs.into_iter() {
            let mut results = 0u32;
            let count_results = |stmt: &Statement| {
                if allowed_stmt(stmt) {
                    results += 1;
                }
                true
            };
            func.visit_statements(count_results);

            if results > 0 {
                let mut countdown = rng.below(results as u64);

                let remove_stmt = |block: &mut Block| {
                    for (idx, stmt) in block.iter().enumerate() {
                        if !allowed_stmt(stmt) {
                            continue;
                        }
                        if countdown != 0 {
                            countdown -= 1;
                            continue;
                        }
                        block.cull(idx..=idx);
                        return false;
                    }
                    true
                };
                func.visit_blocks_mut(remove_stmt);
            }
        }

        Ok(MutationResult::Mutated)
    }
}
