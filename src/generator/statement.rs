use crate::randomext::RandExt;
use libafl_bolts::rands::Rand;
use naga::{
    valid::ShaderStages, AddressSpace, AtomicFunction, Barrier, Expression, FunctionResult, Handle,
    RayQueryFunction, ScalarKind, Span, Statement, StorageAccess, SwitchCase, SwitchValue, Type,
    TypeInner,
};
use serde::Deserialize;

use super::generateir::{CodeContext, FunctionGenCtx};

trait StatementGenerator {
    fn generate(ctx: &mut FunctionGenCtx, budget: u32) -> Option<(Statement, u32)>;
}
#[derive(Clone, Copy, Debug, Deserialize, Hash, Eq, PartialEq)]
pub(super) enum StatementGenerators {
    Atomic,
    Barrier,
    Block,
    Break,
    Call,
    Continue,
    If,
    Kill,
    Loop,
    Return,
    Store,
    Switch,
    WorkgroupLoad,
}

impl StatementGenerators {
    pub fn generate(&self, ctx: &mut FunctionGenCtx, budget: u32) -> Option<(Statement, u32)> {
        match self {
            Self::Atomic => AtomicGenerator::generate(ctx, budget),
            Self::Barrier => BarrierGenerator::generate(ctx, budget),
            Self::Block => BlockGenerator::generate(ctx, budget),
            Self::Break => BreakGenerator::generate(ctx, budget),
            Self::Call => CallGenerator::generate(ctx, budget),
            Self::Continue => ContinueGenerator::generate(ctx, budget),
            Self::If => IfGenerator::generate(ctx, budget),
            Self::Kill => KillGenerator::generate(ctx, budget),
            Self::Loop => LoopGenerator::generate(ctx, budget),
            Self::Return => ReturnGenerator::generate(ctx, budget),
            Self::Store => StoreGenerator::generate(ctx, budget),
            Self::Switch => SwitchGenerator::generate(ctx, budget),
            Self::WorkgroupLoad => WorkgroupLoadGenerator::generate(ctx, budget),
        }
    }

    pub fn iter() -> impl Iterator<Item = StatementGenerators> {
        use StatementGenerators as SG;
        static VALUES: [SG; std::mem::variant_count::<SG>()] = [
            SG::Atomic,
            SG::Barrier,
            SG::Block,
            SG::Break,
            SG::Call,
            SG::Continue,
            SG::If,
            SG::Kill,
            SG::Loop,
            SG::Return,
            SG::Store,
            SG::Switch,
            SG::WorkgroupLoad,
        ];
        VALUES.iter().cloned()
    }

    pub fn may_recurse(&self) -> bool {
        matches!(self, Self::Block | Self::If | Self::Loop | Self::Switch)
    }

    pub fn allowed_shader_stages(&self) -> ShaderStages {
        match self {
            Self::Barrier => ShaderStages::COMPUTE,
            Self::Kill => ShaderStages::FRAGMENT,
            Self::WorkgroupLoad => ShaderStages::COMPUTE,
            _ => ShaderStages::all(),
        }
    }

    pub fn allowed_code_context(&self, code_ctx: CodeContext) -> bool {
        match self {
            Self::Return | Self::Kill => !code_ctx.contains(CodeContext::LOOP_CONTINUING),
            Self::Continue => code_ctx.contains(CodeContext::LOOP_BODY),
            Self::Break => !code_ctx.intersection(CodeContext::LOOP_BODY).is_empty(),
            _ => true,
        }
    }
}

#[derive(Clone, Copy)]
struct BlockGenerator;
impl StatementGenerator for BlockGenerator {
    fn generate(ctx: &mut FunctionGenCtx, budget: u32) -> Option<(Statement, u32)> {
        let budget = ctx.recursive_budget(budget, 1);
        let (body, cost) = ctx.recursive_generate(budget);
        if body.is_empty() {
            None
        } else {
            Some((Statement::Block(body), cost))
        }
    }
}

struct SwitchGenerator;
impl StatementGenerator for SwitchGenerator {
    fn generate(ctx: &mut FunctionGenCtx, budget: u32) -> Option<(Statement, u32)> {
        let budget = std::cmp::max(ctx.recursive_budget_required, budget);
        let adjacent_values = ctx.rng.probability(0.5);
        let num_cases = ctx.config.switch_cases.choose(&mut ctx.rng);
        let num_cases = std::cmp::min(num_cases, budget / ctx.recursive_budget_required);

        let filter = |_, ty: &TypeInner| {
            matches!(
                ty,
                TypeInner::Scalar {
                    kind: ScalarKind::Uint | ScalarKind::Sint,
                    ..
                }
            )
        };
        let (selector, ty) = ctx.expr_matching(filter)?;

        let mut values = Vec::new();
        match ty {
            TypeInner::Scalar {
                kind: ScalarKind::Sint,
                ..
            } => {
                if adjacent_values {
                    let num_cases = num_cases as u64 as i32;
                    let first = ctx.rng.random_i32().saturating_add(num_cases) - num_cases;
                    let last = first + num_cases;
                    values.extend((first..=last).map(SwitchValue::I32));
                } else {
                    values.extend((0..num_cases).map(|_| SwitchValue::I32(ctx.rng.random_i32())));
                }
            }
            TypeInner::Scalar {
                kind: ScalarKind::Uint,
                ..
            } => {
                if adjacent_values {
                    let num_cases = num_cases as u32;
                    let first = ctx.rng.random_u32().saturating_add(num_cases) - num_cases;
                    let last = first + num_cases;
                    values.extend((first..=last).map(SwitchValue::U32));
                } else {
                    values.extend((0..num_cases).map(|_| SwitchValue::U32(ctx.rng.random_u32())));
                }
            }
            _ => unreachable!(),
        }
        values.push(SwitchValue::Default);

        let current_code_ctx = ctx.code_ctx;
        let mut total_cost = 0;
        let mut cases = Vec::new();
        for value in values.into_iter() {
            ctx.code_ctx = current_code_ctx.union(CodeContext::SWITCH_CASE);
            let case_budget = ctx.recursive_budget(budget, num_cases + 1);
            let (body, cost) = ctx.recursive_generate(case_budget as u32);
            cases.push(SwitchCase {
                value,
                body,
                fall_through: false,
            });
            total_cost += cost;
        }
        ctx.code_ctx = current_code_ctx;
        Some((Statement::Switch { selector, cases }, total_cost))
    }
}

struct IfGenerator;
impl StatementGenerator for IfGenerator {
    fn generate(ctx: &mut FunctionGenCtx, budget: u32) -> Option<(Statement, u32)> {
        let filter = |_, ty: &TypeInner| {
            matches!(
                ty,
                TypeInner::Scalar {
                    kind: ScalarKind::Bool,
                    ..
                }
            )
        };

        let Some((condition, _)) = ctx.expr_matching(filter) else {
            return None;
        };

        let current_code_ctx = ctx.code_ctx;
        let if_budget = ctx.recursive_budget(budget, 1);
        let else_budget = ctx.recursive_budget(budget, 1);
        let (accept, if_cost) = ctx.recursive_generate(if_budget);
        ctx.code_ctx = current_code_ctx;
        let (reject, else_cost) = ctx.recursive_generate(else_budget);
        ctx.code_ctx = current_code_ctx;
        Some((
            Statement::If {
                condition,
                accept,
                reject,
            },
            if_cost + else_cost,
        ))
    }
}

struct LoopGenerator;
impl StatementGenerator for LoopGenerator {
    fn generate(ctx: &mut FunctionGenCtx, budget: u32) -> Option<(Statement, u32)> {
        let body_budget = ctx.recursive_budget(budget, 1);
        let continuing_budget = ctx.recursive_budget(budget, 1);

        let current_code_ctx = ctx.code_ctx;
        ctx.code_ctx = current_code_ctx.union(CodeContext::LOOP_BODY);
        let (body, body_cost) = ctx.recursive_generate(body_budget);
        ctx.code_ctx = current_code_ctx.union(CodeContext::LOOP_CONTINUING);
        ctx.code_ctx.remove(CodeContext::LOOP_BODY);
        let (continuing, continuing_cost) = ctx.recursive_generate(continuing_budget);
        ctx.code_ctx = current_code_ctx;

        let marker = ctx.expr_scope.scope_available.len();
        for stmt in body.iter().chain(continuing.iter()) {
            match stmt {
                Statement::Emit(range) => {
                    ctx.expr_scope.scope_available.extend(range.clone());
                }
                Statement::Call {
                    result: Some(result),
                    ..
                }
                | Statement::WorkGroupUniformLoad { result, .. }
                | Statement::RayQuery {
                    fun: RayQueryFunction::Proceed { result },
                    ..
                } => {
                    ctx.expr_scope.scope_available.push(*result);
                }
                _ => {}
            }
        }

        let filter = |_, ty: &TypeInner| {
            matches!(
                ty,
                TypeInner::Scalar {
                    kind: ScalarKind::Bool,
                    ..
                }
            )
        };
        let break_if = ctx.expr_matching(filter).map(|(handle, _)| handle);
        ctx.expr_scope.scope_available.truncate(marker);
        Some((
            Statement::Loop {
                body,
                continuing,
                break_if,
            },
            body_cost + continuing_cost,
        ))
    }
}

struct BarrierGenerator;
impl StatementGenerator for BarrierGenerator {
    fn generate(ctx: &mut FunctionGenCtx, _: u32) -> Option<(Statement, u32)> {
        let barriers = [Barrier::STORAGE, Barrier::WORK_GROUP];
        let barrier = *ctx.rng.choose(&barriers);
        Some((Statement::Barrier(barrier), 1))
    }
}

struct KillGenerator;
impl StatementGenerator for KillGenerator {
    fn generate(ctx: &mut FunctionGenCtx, _: u32) -> Option<(Statement, u32)> {
        ctx.code_ctx = CodeContext::DEAD;
        Some((Statement::Kill, 1))
    }
}

struct BreakGenerator;
impl StatementGenerator for BreakGenerator {
    fn generate(ctx: &mut FunctionGenCtx, _: u32) -> Option<(Statement, u32)> {
        ctx.code_ctx = CodeContext::DEAD;
        Some((Statement::Break, 1))
    }
}

struct ContinueGenerator;
impl StatementGenerator for ContinueGenerator {
    fn generate(ctx: &mut FunctionGenCtx, _: u32) -> Option<(Statement, u32)> {
        ctx.code_ctx = CodeContext::DEAD;
        Some((Statement::Continue, 1))
    }
}

struct ReturnGenerator;
impl StatementGenerator for ReturnGenerator {
    fn generate(ctx: &mut FunctionGenCtx, _: u32) -> Option<(Statement, u32)> {
        let value = match ctx.get_function().result {
            Some(FunctionResult { ty, .. }) => {
                let ty = &ctx.module.types[ty].inner;
                Some(ctx.expr_of_type(ty)?)
            }
            None => None,
        };
        ctx.code_ctx = CodeContext::DEAD;
        Some((Statement::Return { value }, 1))
    }
}

struct StoreGenerator;
impl StatementGenerator for StoreGenerator {
    fn generate(ctx: &mut FunctionGenCtx, _: u32) -> Option<(Statement, u32)> {
        let filter = |_, ty: &TypeInner| match ty {
            TypeInner::ValuePointer { space, .. } | TypeInner::Pointer { space, .. } => match space
            {
                AddressSpace::Function | AddressSpace::Private => true,
                AddressSpace::Storage { access } => access.contains(StorageAccess::STORE),
                _ => false,
            },
            _ => false,
        };

        let (pointer, ty) = ctx.expr_matching(filter)?;
        let value = match *ty {
            TypeInner::Pointer { base, space: _ } => match ctx.module.types[base].inner {
                TypeInner::Atomic { kind, width } => {
                    let value_ty = TypeInner::Scalar { kind, width };
                    ctx.expr_of_type(&value_ty)
                }
                ref other => ctx.expr_of_type(other),
            },
            TypeInner::ValuePointer {
                size: Some(size),
                kind,
                width,
                ..
            } => {
                let value_ty = TypeInner::Vector { size, kind, width };
                ctx.expr_of_type(&value_ty)
            }
            TypeInner::ValuePointer {
                size: None,
                kind,
                width,
                ..
            } => {
                let value_ty = TypeInner::Scalar { kind, width };
                ctx.expr_of_type(&value_ty)
            }
            _ => unreachable!(),
        }?;
        Some((Statement::Store { pointer, value }, 1))
    }
}

struct WorkgroupLoadGenerator;
impl StatementGenerator for WorkgroupLoadGenerator {
    fn generate(ctx: &mut FunctionGenCtx, _: u32) -> Option<(Statement, u32)> {
        let filter = |_, ty: &TypeInner| match ty {
            TypeInner::Pointer {
                space: AddressSpace::WorkGroup,
                ..
            } => true,
            TypeInner::ValuePointer {
                space: AddressSpace::WorkGroup,
                ..
            } => true,
            _ => false,
        };
        let (pointer, ty_pointer) = ctx.expr_matching(filter)?;

        let (ty, _) = ctx
            .module
            .types
            .iter()
            .find(|(_, ty)| ty_pointer.eq(&ty.inner))?;

        let expr = Expression::WorkGroupUniformLoadResult { ty };
        let result = ctx
            .get_function_mut()
            .expressions
            .append(expr, Span::UNDEFINED);
        ctx.expr_scope.add_available(ctx.module, result);
        Some((Statement::WorkGroupUniformLoad { pointer, result }, 1))
    }
}

struct CallGenerator;
impl StatementGenerator for CallGenerator {
    fn generate(ctx: &mut FunctionGenCtx, _: u32) -> Option<(Statement, u32)> {
        if ctx.available_funcs.is_empty() {
            return None;
        }

        let function = *ctx.rng.choose(&ctx.available_funcs);
        let callee = &ctx.module.functions[function];
        let arguments: Option<Vec<Handle<Expression>>> = callee
            .arguments
            .iter()
            .map(|arg| {
                let ty = &ctx.module.types[arg.ty].inner;
                ctx.expr_of_type(ty)
            })
            .collect();
        let arguments = arguments?;
        let result = {
            match callee.result.as_ref() {
                Some(_) => {
                    let expr = Expression::CallResult(function);
                    let result = ctx
                        .get_function_mut()
                        .expressions
                        .append(expr, Span::UNDEFINED);
                    ctx.expr_scope.add_available(ctx.module, result);
                    Some(result)
                }
                None => None,
            }
        };

        Some((
            Statement::Call {
                function,
                arguments,
                result,
            },
            1,
        ))
    }
}

struct AtomicGenerator;
impl StatementGenerator for AtomicGenerator {
    fn generate(ctx: &mut FunctionGenCtx, _: u32) -> Option<(Statement, u32)> {
        use AddressSpace as A;
        use AtomicFunction as AF;
        let filter = |_, ty: &TypeInner| match ty {
            TypeInner::Pointer {
                base,
                space:
                    A::WorkGroup
                    | A::Storage {
                        access: StorageAccess::LOAD | StorageAccess::STORE,
                    },
            } => {
                let inner = &ctx.module.types[*base].inner;
                matches!(
                    inner,
                    TypeInner::Atomic {
                        kind: ScalarKind::Sint | ScalarKind::Uint,
                        width: 4
                    }
                )
            }
            _ => false,
        };
        let (pointer, ty) = ctx.expr_matching(filter)?;
        let TypeInner::Pointer { base, .. } = ty else {
            unreachable!();
        };
        let inner = &ctx.module.types[*base].inner;
        let TypeInner::Atomic { kind, width } = inner else {
            unreachable!();
        };
        let value_ty = TypeInner::Scalar {
            kind: *kind,
            width: *width,
        };
        let value = ctx.expr_of_type(&value_ty)?;

        let atomic_funcs = [
            AF::Add,
            AF::And,
            AF::ExclusiveOr,
            AF::InclusiveOr,
            AF::Max,
            AF::Min,
            AF::Subtract,
        ];
        assert_eq!(std::mem::variant_count::<AF>(), atomic_funcs.len() + 1);
        let fun = *ctx.rng.choose(&atomic_funcs);

        let ty = ctx.module.types.insert(
            Type {
                name: None,
                inner: value_ty,
            },
            Span::UNDEFINED,
        );
        let expr = Expression::AtomicResult {
            ty,
            comparison: false,
        };
        let result = ctx
            .get_function_mut()
            .expressions
            .append(expr, Span::UNDEFINED);
        ctx.expr_scope.add_available(ctx.module, result);

        Some((
            Statement::Atomic {
                pointer,
                fun,
                value,
                result,
            },
            1,
        ))
    }
}
