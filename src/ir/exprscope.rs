use libafl_bolts::current_nanos;
use naga::{
    proc::{ResolveContext, ResolveError, TypeResolution},
    Arena, Block, Expression, Handle, LocalVariable, Module, RayQueryFunction, Statement, Type,
    TypeInner, UniqueArena,
};
use rand::{rngs::SmallRng, seq::IteratorRandom, Rng, SeedableRng};

use crate::ir::iter::FunctionIdentifier;

use super::iter::UsesExprIter;

pub(crate) struct BestEffortTypifier {
    resolutions: Vec<Option<Option<TypeResolution>>>,
    strict_typing: bool,
}

impl BestEffortTypifier {
    const fn new(strict_typing: bool) -> Self {
        Self {
            resolutions: Vec::new(),
            strict_typing,
        }
    }

    fn add(
        &mut self,
        expr_handle: Handle<Expression>,
        expressions: &Arena<Expression>,
        ctx: &ResolveContext,
    ) {
        let mut worklist = Vec::new();
        worklist.push(expr_handle);
        let mut work_done = 0;
        while let Some(eh) = worklist.pop() {
            work_done += 1;
            if self.resolutions.len() <= eh.index() {
                self.resolutions.resize(eh.index() + 1, None);
            }
            let mut used_resolved = true;
            for eh_used in expressions[eh].iter_used_exprs() {
                if self.resolutions.get(eh_used.index()).is_none()
                    || self.resolutions[eh_used.index()].is_none()
                {
                    if used_resolved {
                        worklist.push(eh);
                        used_resolved = false;
                    }
                    if let Some(pos) = worklist.iter().position(|handle| *handle == eh_used) {
                        let last = worklist.len() - 1;
                        worklist.swap(pos, last);
                    } else {
                        worklist.push(eh_used);
                    }
                }
            }
            if used_resolved {
                let expr = &expressions[eh];
                let resolution = ctx.resolve(expr, |h| {
                    if let Some(res) = self.resolutions[h.index()].as_ref().unwrap().as_ref() {
                        Ok(res)
                    } else {
                        Err(ResolveError::OutOfBoundsIndex {
                            expr: h,
                            index: h.index() as u32,
                        })
                    }
                });
                match resolution {
                    Ok(resolution) => {
                        self.resolutions[eh.index()] = Some(Some(resolution));
                    }
                    Err(_) => {
                        assert!(!self.strict_typing);
                        self.resolutions[eh.index()] = Some(None);
                    }
                }
            }
            if work_done > expressions.len() * 2 {
                for (eh, expr) in expressions.iter() {
                    println!("{:?} {:?}", eh, expr);
                }
                panic!("Expressions contain a cycle when adding {:?}", expr_handle);
            }
        }
    }

    pub fn get<'a>(
        &'a self,
        expr_handle: Handle<Expression>,
        types: &'a UniqueArena<Type>,
    ) -> Option<&'a TypeInner> {
        self.resolutions[expr_handle.index()]
            .as_ref()
            .unwrap()
            .as_ref()
            .map(|res| res.inner_with(types))
    }
}

pub(crate) struct ExprScope {
    fid: Option<FunctionIdentifier>,
    pub typifier: BestEffortTypifier,
    pub always_available: Vec<Handle<Expression>>,
    pub scope_available: Vec<Handle<Expression>>,
    used: Vec<bool>,
}

impl ExprScope {
    pub const fn new(fid: Option<FunctionIdentifier>, strict_typing: bool) -> Self {
        Self {
            fid,
            typifier: BestEffortTypifier::new(strict_typing),
            always_available: Vec::new(),
            scope_available: Vec::new(),
            used: Vec::new(),
        }
    }

    pub fn new_before(
        fid: FunctionIdentifier,
        at: *const Statement,
        module: &Module,
        strict_typing: bool,
    ) -> Self {
        let func = match fid {
            FunctionIdentifier::Function(handle) => &module.functions[handle],
            FunctionIdentifier::EntryPoint(idx) => &module.entry_points[idx].function,
        };

        let resolve_ctx =
            ResolveContext::with_locals(module, &func.local_variables, &func.arguments);
        let mut typifier = BestEffortTypifier::new(strict_typing);

        let mut always_available: Vec<_> = Vec::new();
        for (handle, expr) in func.expressions.iter() {
            if Self::is_always_available(expr) {
                always_available.push(handle);
                typifier.add(handle, &func.expressions, &resolve_ctx);
            }
        }

        let mut scope_available: Vec<_> = Vec::new();
        'outer: {
            let mut num_available = Vec::new();
            for item in DfsFuncIter::new(&func.body) {
                match item {
                    DfsItem::BlockOpen(_) => {
                        num_available.push(scope_available.len());
                    }
                    DfsItem::BlockClose(_) => {
                        let available = num_available.pop().unwrap();
                        scope_available.truncate(available);
                    }
                    DfsItem::Statement(stmt) => {
                        if let Statement::Emit(range) = stmt {
                            for eh in range.clone() {
                                for input in func.expressions[eh].iter_used_exprs() {
                                    typifier.add(input, &func.expressions, &resolve_ctx);
                                }
                            }
                        }

                        if stmt as *const Statement == at {
                            break 'outer;
                        }

                        match stmt {
                            Statement::Emit(range) => {
                                for eh in range.clone() {
                                    scope_available.push(eh);
                                    typifier.add(eh, &func.expressions, &resolve_ctx);
                                }
                            }
                            Statement::Call {
                                result: Some(result),
                                ..
                            }
                            | Statement::Atomic { result, .. }
                            | Statement::WorkGroupUniformLoad { result, .. }
                            | Statement::RayQuery {
                                fun: RayQueryFunction::Proceed { result },
                                ..
                            } => {
                                scope_available.push(*result);
                                typifier.add(*result, &func.expressions, &resolve_ctx);
                            }
                            _ => {}
                        }
                    }
                }
            }
            panic!("Statement not part of Function");
        };

        Self {
            fid: Some(fid),
            typifier,
            always_available,
            scope_available,
            used: vec![false; func.expressions.len()],
        }
    }

    pub fn matching<'a, F>(
        &'a self,
        filter: F,
        types: &'a UniqueArena<Type>,
    ) -> Option<(Handle<Expression>, &'a TypeInner)>
    where
        F: Fn(Handle<Expression>, &TypeInner) -> bool,
    {
        let mut rng = SmallRng::seed_from_u64(current_nanos());

        let f = |expr: &Handle<Expression>| {
            let expr = *expr;
            let inner = self.typifier.get(expr, types);
            if let Some(inner) = inner {
                filter(expr, inner).then_some((expr, inner))
            } else {
                assert!(!self.typifier.strict_typing);
                None
            }
        };

        if let Some(unused_expr) = self
            .always_available
            .iter()
            .chain(self.scope_available.iter())
            .filter(|handle| !self.used[handle.index()])
            .filter_map(f)
            .choose(&mut rng)
        {
            return Some(unused_expr);
        }

        self.always_available
            .iter()
            .chain(self.scope_available.iter())
            .filter_map(f)
            .choose(&mut rng)
    }

    pub fn of_type(&self, ty: &TypeInner, types: &UniqueArena<Type>) -> Option<Handle<Expression>> {
        let filter = |_, other_ty: &TypeInner| other_ty == ty;
        self.matching(filter, types).map(|(expr, _)| expr)
    }

    pub fn any(&self) -> Option<Handle<Expression>> {
        if self.is_empty() {
            return None;
        }

        let mut rng = SmallRng::seed_from_u64(current_nanos());
        let scope_len = self.scope_available.len();
        let always_len = self.always_available.len();
        let total = scope_len + always_len;
        let idx = rng.gen_range(0..total);

        if idx < scope_len {
            Some(self.scope_available[idx])
        } else {
            Some(self.always_available[idx - scope_len])
        }
    }

    pub fn add_available(&mut self, module: &Module, handle: Handle<Expression>) {
        let dummy_local_vars = Arena::<LocalVariable>::default();

        let (resolve_ctx, exprs) = match self.fid {
            None => (
                ResolveContext::with_locals(module, &dummy_local_vars, &[]),
                &module.const_expressions,
            ),
            Some(func) => {
                let func = match func {
                    FunctionIdentifier::Function(handle) => &module.functions[handle],
                    FunctionIdentifier::EntryPoint(idx) => &module.entry_points[idx].function,
                };
                (
                    ResolveContext::with_locals(module, &func.local_variables, &func.arguments),
                    &func.expressions,
                )
            }
        };
        self.typifier.add(handle, exprs, &resolve_ctx);

        if Self::is_always_available(&exprs[handle]) {
            self.always_available.push(handle);
        } else {
            self.scope_available.push(handle);
        }

        if self.used.len() <= handle.index() {
            self.used.resize(handle.index() + 1, false);
        }
    }

    pub fn add_use(&mut self, expr: &Expression) {
        for handle in expr.iter_used_exprs() {
            self.used[handle.index()] = true;
        }
    }

    pub fn is_always_available(expr: &Expression) -> bool {
        matches!(
            expr,
            Expression::FunctionArgument(_)
                | Expression::LocalVariable(_)
                | Expression::GlobalVariable(_)
                | Expression::Literal(_)
                | Expression::Constant(_)
                | Expression::ZeroValue(_)
        )
    }

    pub fn is_empty(&self) -> bool {
        self.scope_available.is_empty() && self.always_available.is_empty()
    }
}

#[allow(dead_code)]
enum DfsItem<'a> {
    BlockOpen(&'a Block),
    BlockClose(&'a Block),
    Statement(&'a Statement),
}

struct DfsFuncIter<'a> {
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
