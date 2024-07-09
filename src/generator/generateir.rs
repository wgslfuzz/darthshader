use std::num::NonZeroU32;

use libafl::{generators::Generator, state::HasRand};
use libafl_bolts::{
    rands::{Rand, StdRand},
    Error, Named,
};
use naga::{
    valid::{ShaderStages, TypeFlags},
    AddressSpace, ArraySize, Binding, Block, BuiltIn, EntryPoint, Expression, Function,
    FunctionArgument, FunctionResult, GlobalVariable, Handle, Interpolation, LocalVariable, Module,
    Range, ResourceBinding, ScalarKind, ShaderStage, Span, Statement, StorageAccess, StructMember,
    Type, TypeInner, VectorSize,
};
use rand::{
    seq::{IteratorRandom, SliceRandom},
    RngCore,
};

use crate::{
    ast::Ast,
    ir::{
        exprscope::ExprScope,
        iter::BlockVisitorMut,
        mutate::{DfsFuncIter, DfsItem},
    },
    layeredinput::{LayeredInput, IR},
    randomext::RandExt,
};

use super::config::GeneratorConfig;
use super::expression::{ConstExpressionGenerators, ExpressionGenerators};
use super::statement::StatementGenerators;
use crate::ir::iter::FunctionIdentifier;

bitflags::bitflags! {
    pub struct CodeContext: u8 {
        const LOOP_BODY = 0b00000001;
        const LOOP_CONTINUING = 0b00000010;
        const SWITCH_CASE = 0b00000100;
        const DEAD = 0b00001000;
    }
}

struct EntryPointSpec;
impl EntryPointSpec {
    const COMPUTE_OPT_INPUTS: [(TypeInner, Binding); 5] = [
        (
            TypeInner::Scalar {
                kind: ScalarKind::Uint,
                width: 4,
            },
            Binding::BuiltIn(BuiltIn::LocalInvocationIndex),
        ),
        (
            TypeInner::Vector {
                size: VectorSize::Tri,
                kind: ScalarKind::Uint,
                width: 4,
            },
            Binding::BuiltIn(BuiltIn::LocalInvocationId),
        ),
        (
            TypeInner::Vector {
                size: VectorSize::Tri,
                kind: ScalarKind::Uint,
                width: 4,
            },
            Binding::BuiltIn(BuiltIn::GlobalInvocationId),
        ),
        (
            TypeInner::Vector {
                size: VectorSize::Tri,
                kind: ScalarKind::Uint,
                width: 4,
            },
            Binding::BuiltIn(BuiltIn::WorkGroupId),
        ),
        (
            TypeInner::Vector {
                size: VectorSize::Tri,
                kind: ScalarKind::Uint,
                width: 4,
            },
            Binding::BuiltIn(BuiltIn::NumWorkGroups),
        ),
    ];

    const VERTEX_OPT_INPUTS: [(TypeInner, Binding); 2] = [
        (
            TypeInner::Scalar {
                kind: ScalarKind::Uint,
                width: 4,
            },
            Binding::BuiltIn(BuiltIn::VertexIndex),
        ),
        (
            TypeInner::Scalar {
                kind: ScalarKind::Uint,
                width: 4,
            },
            Binding::BuiltIn(BuiltIn::InstanceIndex),
        ),
    ];

    const FRAGMENT_OPT_INPUTS: [(TypeInner, Binding); 4] = [
        (
            TypeInner::Scalar {
                kind: ScalarKind::Uint,
                width: 4,
            },
            Binding::BuiltIn(BuiltIn::SampleIndex),
        ),
        (
            TypeInner::Scalar {
                kind: ScalarKind::Uint,
                width: 4,
            },
            Binding::BuiltIn(BuiltIn::SampleMask),
        ),
        (
            TypeInner::Scalar {
                kind: ScalarKind::Bool,
                width: 1,
            },
            Binding::BuiltIn(BuiltIn::FrontFacing),
        ),
        (
            TypeInner::Vector {
                size: VectorSize::Quad,
                kind: ScalarKind::Float,
                width: 4,
            },
            Binding::BuiltIn(BuiltIn::Position { invariant: false }),
        ),
    ];

    const FRAGMENT_OPT_OUTPUTS: [(TypeInner, Binding); 2] = [
        (
            TypeInner::Scalar {
                kind: ScalarKind::Float,
                width: 4,
            },
            Binding::BuiltIn(BuiltIn::FragDepth),
        ),
        (
            TypeInner::Scalar {
                kind: ScalarKind::Uint,
                width: 4,
            },
            Binding::BuiltIn(BuiltIn::SampleMask),
        ),
    ];

    fn required_output(stage: ShaderStage) -> Option<(TypeInner, Binding)> {
        match stage {
            ShaderStage::Vertex => {
                let inner = TypeInner::Vector {
                    size: VectorSize::Quad,
                    kind: ScalarKind::Float,
                    width: 4,
                };
                let binding = Binding::BuiltIn(BuiltIn::Position { invariant: false });
                Some((inner, binding))
            }
            ShaderStage::Fragment => None,
            ShaderStage::Compute => unreachable!(),
        }
    }

    fn optional_inputs(stage: ShaderStage) -> &'static [(TypeInner, Binding)] {
        match stage {
            ShaderStage::Vertex => &Self::VERTEX_OPT_INPUTS,
            ShaderStage::Fragment => &Self::FRAGMENT_OPT_INPUTS,
            ShaderStage::Compute => &Self::COMPUTE_OPT_INPUTS,
        }
    }

    fn optional_outputs(stage: ShaderStage) -> &'static [(TypeInner, Binding)] {
        match stage {
            ShaderStage::Vertex => &[],
            ShaderStage::Fragment => &Self::FRAGMENT_OPT_OUTPUTS,
            ShaderStage::Compute => unreachable!(),
        }
    }

    fn random_io(location: u32, rng: &mut StdRand) -> (TypeInner, Binding) {
        let (kind, width) = rng.choose([
            (ScalarKind::Sint, 4),
            (ScalarKind::Uint, 4),
            (ScalarKind::Float, 4),
        ]);
        let inner = if rng.probability(0.5) {
            TypeInner::Scalar { kind, width }
        } else {
            let size = rng.choose([VectorSize::Bi, VectorSize::Tri, VectorSize::Quad]);
            TypeInner::Vector { size, kind, width }
        };

        let interpolation = {
            if matches!(
                inner,
                TypeInner::Scalar {
                    kind: ScalarKind::Float,
                    ..
                } | TypeInner::Vector {
                    kind: ScalarKind::Float,
                    ..
                }
            ) {
                rng.choose([
                    Interpolation::Flat,
                    Interpolation::Linear,
                    Interpolation::Perspective,
                ])
            } else {
                Interpolation::Flat
            }
        };
        (
            inner,
            Binding::Location {
                location,
                second_blend_source: false,
                interpolation: Some(interpolation),
                sampling: None,
            },
        )
    }
}

pub(super) struct GlobalGenCtx<'a> {
    pub rng: StdRand,
    pub module: &'a mut Module,
    pub global_exprs: ExprScope,
    const_expr_generators: Vec<(ConstExpressionGenerators, u32)>,
}

impl<'a> GlobalGenCtx<'a> {
    fn new(config: &'a GeneratorConfig, module: &'a mut Module, rng: StdRand) -> GlobalGenCtx<'a> {
        let mut global_exprs = ExprScope::new(None, true);
        for (handle, expr) in module.const_expressions.iter() {
            global_exprs.add_available(module, handle);
            global_exprs.add_use(expr);
        }

        Self {
            rng,
            module,
            global_exprs,
            const_expr_generators: Self::setup_const_expr_generators(config),
        }
    }

    fn setup_const_expr_generators(
        config: &GeneratorConfig,
    ) -> Vec<(ConstExpressionGenerators, u32)> {
        ConstExpressionGenerators::iter()
            .map(|gen| {
                (
                    gen,
                    config.const_expression_weight_map.weights[gen as usize],
                )
            })
            .collect()
    }

    fn emit_basic_types(&mut self) {
        use ScalarKind as S;
        use TypeInner as TI;
        for (kind, width) in [(S::Bool, 1), (S::Sint, 4), (S::Uint, 4), (S::Float, 4)].into_iter() {
            let inner = TI::Scalar { kind, width };
            let typ = Type { name: None, inner };
            self.module.types.insert(typ, Span::UNDEFINED);
        }

        self.module.types.insert(
            Type {
                name: None,
                inner: TI::Atomic {
                    kind: S::Sint,
                    width: 4,
                },
            },
            Span::UNDEFINED,
        );
        self.module.types.insert(
            Type {
                name: None,
                inner: TI::Atomic {
                    kind: S::Uint,
                    width: 4,
                },
            },
            Span::UNDEFINED,
        );
    }

    fn create_entrypoint_struct(&mut self, io_parameter: &[(TypeInner, Binding)]) -> Type {
        let mut members = Vec::with_capacity(io_parameter.len());
        let mut offset: u32 = 0;
        let mut max_align = 0;
        for (idx, (inner, binding)) in io_parameter.iter().enumerate() {
            let size = inner.size(self.module.to_ctx());
            let align = size.next_power_of_two();
            max_align = std::cmp::max(max_align, align);
            offset = offset.next_multiple_of(align);
            let ty = Type {
                name: None,
                inner: inner.clone(),
            };
            let handle = self.module.types.insert(ty, Span::UNDEFINED);
            let member = StructMember {
                name: Some(format!("m{}", idx)),
                ty: handle,
                binding: Some(binding.clone()),
                offset,
            };
            offset += size;
            members.push(member);
        }
        Type {
            name: Some(format!("S{}", self.rng.next() as u32)),
            inner: TypeInner::Struct {
                members,
                span: offset.next_multiple_of(max_align),
            },
        }
    }

    fn create_entrypoint_arguments(&mut self, stage: ShaderStage) -> Vec<FunctionArgument> {
        let mut inputs = EntryPointSpec::optional_inputs(stage).to_owned();
        inputs.shuffle(&mut self.rng);
        inputs.truncate(self.rng.below(inputs.len() as u64 + 1) as usize);
        if self.rng.probability(0.5) {
            let num_scalars = self.rng.between(1, 5) as u32;
            for idx in 0..num_scalars {
                inputs.push(EntryPointSpec::random_io(idx, &mut self.rng));
            }
        }
        inputs.shuffle(&mut self.rng);
        let min_inputs = if inputs.is_empty() { 0 } else { 1 };
        let num_args = self.rng.between(min_inputs, inputs.len() as u64) as usize;
        let mut args = Vec::with_capacity(num_args);
        let mut remaining_inputs = inputs.len();
        for idx in 0..num_args {
            let min_params = if idx == num_args - 1 {
                remaining_inputs
            } else {
                1
            };
            let max_params = remaining_inputs - (num_args - idx - 1);
            let params = self.rng.between(min_params as u64, max_params as u64) as usize;
            remaining_inputs -= params;
            let (ty, binding) = {
                if params == 1 && self.rng.probability(0.8) {
                    let (inner, binding) = inputs[remaining_inputs].clone();
                    let ty = Type { name: None, inner };
                    (ty, Some(binding))
                } else {
                    let ty = self.create_entrypoint_struct(
                        &inputs[remaining_inputs..(remaining_inputs + params)],
                    );
                    (ty, None)
                }
            };
            let handle = self.module.types.insert(ty, Span::UNDEFINED);
            args.push(FunctionArgument {
                name: Some(format!("arg{}", idx)),
                ty: handle,
                binding,
            });
        }
        args
    }

    fn create_entrypoint_result(&mut self, stage: ShaderStage) -> Option<FunctionResult> {
        match stage {
            ShaderStage::Compute => None,
            ShaderStage::Vertex | ShaderStage::Fragment => {
                let mut optional_outputs = EntryPointSpec::optional_outputs(stage).to_owned();
                optional_outputs.shuffle(&mut self.rng);
                optional_outputs
                    .truncate(self.rng.below(optional_outputs.len() as u64 + 1) as usize);
                let mut outputs = optional_outputs;
                if let Some(required_output) = EntryPointSpec::required_output(stage) {
                    outputs.push(required_output);
                }
                if self.rng.probability(0.5) {
                    let num_scalars = self.rng.between(1, 5) as u32;
                    for idx in 0..num_scalars {
                        outputs.push(EntryPointSpec::random_io(idx, &mut self.rng));
                    }
                }
                outputs.shuffle(&mut self.rng);
                match (outputs.len(), self.rng.probability(0.2)) {
                    (0, _) => None,
                    (1, true) => {
                        let (inner, binding) = outputs[0].clone();
                        let ty = Type { name: None, inner };
                        let ty = self.module.types.insert(ty, Span::UNDEFINED);
                        Some(FunctionResult {
                            ty,
                            binding: Some(binding),
                        })
                    }
                    _ => {
                        let ty = self.create_entrypoint_struct(&outputs);
                        let ty = self.module.types.insert(ty, Span::UNDEFINED);
                        Some(FunctionResult { ty, binding: None })
                    }
                }
            }
        }
    }

    fn create_struct_type(&mut self) -> Option<Type> {
        let require_constructible = self.rng.probability(0.5);

        let info = {
            let module = Module {
                types: self.module.types.clone(),
                ..Default::default()
            };
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .ok()?
        };

        let filter = |handle, ty: &TypeInner, last: bool| match ty {
            TypeInner::Scalar { .. } | TypeInner::Vector { .. } | TypeInner::Matrix { .. } => true,
            TypeInner::Atomic { .. } => !require_constructible,
            TypeInner::Array { .. } => {
                let mut required_flags = TypeFlags::empty();
                if require_constructible {
                    required_flags |= TypeFlags::CONSTRUCTIBLE;
                }
                if !last {
                    required_flags |= TypeFlags::SIZED;
                }
                let flags: TypeFlags = info[handle];
                flags.contains(required_flags)
            }
            TypeInner::Struct { .. } => {
                let mut required_flags = TypeFlags::SIZED;
                if require_constructible {
                    required_flags |= TypeFlags::CONSTRUCTIBLE;
                }
                let flags: TypeFlags = info[handle];
                flags.contains(required_flags)
            }
            _ => false,
        };

        let num_members = self.rng.between(1, 5);
        let mut members = Vec::with_capacity(num_members as usize);
        let mut offset = 0;
        for idx in 0..num_members {
            let last = idx + 1 == num_members;
            let (handle, ty) = self
                .type_matching(|handle, ty| filter(handle, ty, last))
                .unwrap();
            let member = StructMember {
                name: Some(format!("m{}", idx)),
                ty: handle,
                binding: None,
                offset,
            };
            offset += ty.size(self.module.to_ctx());
            offset = (offset + 3) / 4 * 4;
            members.push(member);
        }
        Some(Type {
            name: Some(format!("S{}", self.rng.next() as u32)),
            inner: TypeInner::Struct {
                members,
                span: offset,
            },
        })
    }

    fn create_matrix_type(&mut self) -> Option<Type> {
        let vector_sizes = [VectorSize::Bi, VectorSize::Tri, VectorSize::Quad];
        let columns = *self.rng.choose(&vector_sizes);
        let rows = *self.rng.choose(&vector_sizes);
        let inner = TypeInner::Matrix {
            columns,
            rows,
            width: 4,
        };
        Some(Type { name: None, inner })
    }

    fn create_array_type(&mut self) -> Option<Type> {
        let info = {
            let module = Module {
                types: self.module.types.clone(),
                ..Default::default()
            };
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .ok()?
        };

        let filter = |handle, _: &TypeInner| {
            let flags: TypeFlags = info[handle];
            flags.contains(TypeFlags::DATA | TypeFlags::SIZED)
        };
        let (base, _) = self.type_matching(filter).unwrap();

        let size = {
            if self.rng.probability(0.7) {
                let size = self.rng.between(1, 10) as u32;
                ArraySize::Constant(NonZeroU32::try_from(size).unwrap())
            } else {
                ArraySize::Dynamic
            }
        };
        let inner = TypeInner::Array {
            base,
            size,
            stride: 4,
        };
        Some(Type { name: None, inner })
    }

    fn create_vector_type(&mut self) -> Option<Type> {
        let vector_sizes = [VectorSize::Bi, VectorSize::Tri, VectorSize::Quad];
        let size = *self.rng.choose(&vector_sizes);

        let scalar_def = [
            (ScalarKind::Bool, 1),
            (ScalarKind::Sint, 4),
            (ScalarKind::Uint, 4),
            (ScalarKind::Float, 4),
        ];
        let (kind, width) = *self.rng.choose(&scalar_def);
        let inner = TypeInner::Vector { size, kind, width };
        Some(Type { name: None, inner })
    }

    fn create_pointer_type(&mut self) -> Option<Type> {
        let info = {
            let module = Module {
                types: self.module.types.clone(),
                ..Default::default()
            };
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .ok()?
        };

        let filter = |handle, _: &TypeInner| {
            let flags: TypeFlags = info[handle];
            flags.contains(TypeFlags::DATA)
        };
        let (base, _) = self.type_matching(filter).unwrap();
        let space = AddressSpace::Private;

        let inner = TypeInner::Pointer { base, space };
        Some(Type { name: None, inner })
    }

    fn create_type(&mut self) -> Option<Handle<Type>> {
        let ty = match self.rng.below(5) {
            0 => self.create_struct_type(),
            1 => self.create_vector_type(),
            2 => self.create_matrix_type(),
            3 => self.create_array_type(),
            4 => self.create_pointer_type(),
            _ => unreachable!(),
        }?;
        Some(self.module.types.insert(ty, Span::UNDEFINED))
    }

    fn create_global(&mut self) -> Option<Handle<GlobalVariable>> {
        let info = {
            let module = Module {
                types: self.module.types.clone(),
                ..Default::default()
            };
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .ok()?
        };

        let name = Some(format!("G{}", self.rng.next() as u32));
        let accesses = [
            StorageAccess::LOAD,
            StorageAccess::LOAD | StorageAccess::STORE,
        ];
        let access = *self.rng.choose(&accesses);
        let spaces = [
            AddressSpace::Private,
            AddressSpace::Uniform,
            AddressSpace::WorkGroup,
            AddressSpace::Storage { access },
        ];
        let space = *self.rng.choose(&spaces);
        let (required_type_flags, is_resource) = match space {
            AddressSpace::Private => (TypeFlags::CONSTRUCTIBLE, false),
            AddressSpace::WorkGroup => (TypeFlags::DATA | TypeFlags::SIZED, false),
            AddressSpace::Uniform => (
                TypeFlags::DATA | TypeFlags::COPY | TypeFlags::SIZED | TypeFlags::HOST_SHAREABLE,
                true,
            ),
            AddressSpace::Storage { .. } => (TypeFlags::DATA | TypeFlags::HOST_SHAREABLE, true),
            AddressSpace::PushConstant => (
                TypeFlags::DATA | TypeFlags::COPY | TypeFlags::HOST_SHAREABLE | TypeFlags::SIZED,
                false,
            ),
            _ => unreachable!(),
        };

        let binding = is_resource.then(|| ResourceBinding {
            group: self.rng.between(0, 255) as u32,
            binding: self.rng.between(0, 255) as u32,
        });

        let filter = |handle, inner: &TypeInner| {
            let type_flags: TypeFlags = info[handle];
            if matches!(inner, TypeInner::Atomic { .. })
                && matches!(
                    space,
                    AddressSpace::Storage {
                        access: StorageAccess::LOAD
                    }
                )
            {
                false
            } else {
                type_flags.contains(required_type_flags)
            }
        };
        let (ty, inner) = self.type_matching(filter)?;

        let init = match space {
            AddressSpace::Private => self.global_exprs.of_type(inner, &self.module.types),
            _ => None,
        };
        let gvar = GlobalVariable {
            name,
            space,
            binding,
            ty,
            init,
        };
        Some(self.module.global_variables.append(gvar, Span::UNDEFINED))
    }

    fn create_const_expr(&mut self) -> Option<Handle<Expression>> {
        let (gen, _) = *self
            .const_expr_generators
            .choose_weighted(&mut self.rng, |(_, weight)| *weight)
            .ok()?;

        let expr = gen.generate(self)?;
        self.global_exprs.add_use(&expr);
        let handle = self.module.const_expressions.append(expr, Span::UNDEFINED);
        self.global_exprs.add_available(self.module, handle);
        Some(handle)
    }

    fn create_function(&mut self) -> Handle<Function> {
        let mut func = Function {
            name: Some(format!("f{}", self.rng.next() as u32)),
            ..Default::default()
        };

        let info = {
            let module = Module {
                types: self.module.types.clone(),
                ..Default::default()
            };
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .ok()
        };
        let Some(info) = info else {
            return self.module.functions.append(func, Span::UNDEFINED);
        };

        let num_locals = self.rng.below(5);
        let num_args = self.rng.below(3);
        let filter_constructible = |handle, _: &TypeInner| {
            let type_flags: TypeFlags = info[handle];
            type_flags.contains(TypeFlags::CONSTRUCTIBLE)
        };
        let filter_args = |handle, ty: &TypeInner| {
            filter_constructible(handle, ty)
                || matches!(ty, TypeInner::Pointer { .. } | TypeInner::Sampler { .. })
        };

        for idx in 0..num_args {
            let Some((ty, _)) = self.type_matching(filter_args) else {
                break;
            };
            let name = Some(format!("arg{}", idx));
            func.arguments.push(FunctionArgument {
                name,
                ty,
                binding: None,
            });
        }
        if self.rng.probability(0.5) {
            if let Some((ty, _)) = self.type_matching(filter_constructible) {
                func.result = Some(FunctionResult { ty, binding: None });
                let loc = LocalVariable {
                    name: Some(format!("loc{}", self.rng.next_u32())),
                    ty,
                    init: None,
                };
                let local_var = func.local_variables.append(loc, Span::UNDEFINED);
                let pointer = func
                    .expressions
                    .append(Expression::LocalVariable(local_var), Span::UNDEFINED);
                let handle = func
                    .expressions
                    .append(Expression::Load { pointer }, Span::UNDEFINED);
                func.body = Block::from_vec(
                    [Statement::Return {
                        value: Some(handle),
                    }]
                    .into(),
                );
            }
        }

        for _ in 0..num_locals {
            let Some((ty, _)) = self.type_matching(filter_constructible) else {
                break;
            };
            let loc = LocalVariable {
                name: Some(format!("loc{}", self.rng.next_u32())),
                ty,
                init: None,
            };
            let handle = func.local_variables.append(loc, Span::UNDEFINED);
            func.expressions
                .append(Expression::LocalVariable(handle), Span::UNDEFINED);
        }

        self.module.functions.append(func, Span::UNDEFINED)
    }

    fn create_entrypoint(&mut self) -> usize {
        let stages = [
            ShaderStage::Compute,
            ShaderStage::Fragment,
            ShaderStage::Vertex,
        ];
        assert_eq!(std::mem::variant_count::<ShaderStage>(), stages.len());
        let stage = *self.rng.choose(&stages);

        let mut func = Function::default();
        let num_locals = self.rng.below(5);

        func.arguments = self.create_entrypoint_arguments(stage);
        if let Some(result) = self.create_entrypoint_result(stage) {
            let ty = result.ty;
            func.result = Some(result);
            let loc = LocalVariable {
                name: Some(format!("loc{}", self.rng.next_u32())),
                ty,
                init: None,
            };
            let local_var = func.local_variables.append(loc, Span::UNDEFINED);
            let pointer = func
                .expressions
                .append(Expression::LocalVariable(local_var), Span::UNDEFINED);
            let handle = func
                .expressions
                .append(Expression::Load { pointer }, Span::UNDEFINED);
            func.body = Block::from_vec(
                [Statement::Return {
                    value: Some(handle),
                }]
                .into(),
            );
        }

        let info = {
            let module = Module {
                types: self.module.types.clone(),
                ..Default::default()
            };
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .ok()
        };
        if let Some(info) = info {
            let filter_constructible = |handle, _: &TypeInner| {
                let type_flags: TypeFlags = info[handle];
                type_flags.contains(TypeFlags::CONSTRUCTIBLE)
            };
            for _ in 0..num_locals {
                let Some((ty, _)) = self.type_matching(filter_constructible) else {
                    break;
                };
                let loc = LocalVariable {
                    name: Some(format!("loc{}", self.rng.next_u32())),
                    ty,
                    init: None,
                };
                let handle = func.local_variables.append(loc, Span::UNDEFINED);
                func.expressions
                    .append(Expression::LocalVariable(handle), Span::UNDEFINED);
            }
        }

        let workgroup_size = if stage == ShaderStage::Compute {
            [1, 1, 1]
        } else {
            [0, 0, 0]
        };
        let ep = EntryPoint {
            name: format!("ep{}", self.rng.next_u32()),
            stage,
            early_depth_test: None,
            workgroup_size,
            function: func,
        };

        self.module.entry_points.push(ep);
        self.module.entry_points.len() - 1
    }

    fn type_matching<'b, F>(&'b self, filter: F) -> Option<(Handle<Type>, &'b TypeInner)>
    where
        F: Fn(Handle<Type>, &TypeInner) -> bool,
    {
        let mut rng = self.rng;
        let f = |(handle, ty): (Handle<Type>, &'b Type)| {
            filter(handle, &ty.inner).then_some((handle, &ty.inner))
        };

        self.module.types.iter().filter_map(f).choose(&mut rng)
    }
}

pub(crate) struct FunctionGenCtx<'a> {
    pub config: &'a GeneratorConfig,
    pub rng: StdRand,
    pub module: &'a mut Module,
    pub expr_scope: ExprScope,
    pub code_ctx: CodeContext,
    pub for_shader_stages: ShaderStages,
    pub available_funcs: Vec<Handle<Function>>,
    pub recursive_budget_required: u32,
    function: FunctionIdentifier,
    expr_generators: Vec<(ExpressionGenerators, u32)>,
    stmt_generators: Vec<(StatementGenerators, u32)>,
    expr_vs_stmt_prob: f64,
}

impl<'a> FunctionGenCtx<'a> {
    pub fn new(
        config: &'a GeneratorConfig,
        module: &'a mut Module,
        function: FunctionIdentifier,
        seed: u64,
        strict_typing: bool,
    ) -> FunctionGenCtx<'a> {
        let available_funcs: Vec<Handle<Function>> = {
            match function {
                FunctionIdentifier::Function(function) => module
                    .functions
                    .iter()
                    .filter_map(|(handle, _)| (handle.index() < function.index()).then_some(handle))
                    .collect(),
                FunctionIdentifier::EntryPoint(_) => {
                    module.functions.iter().map(|(handle, _)| handle).collect()
                }
            }
        };

        let for_shader_stages = {
            match function {
                FunctionIdentifier::Function(_) => ShaderStages::all(),
                FunctionIdentifier::EntryPoint(ep) => match module.entry_points[ep].stage {
                    ShaderStage::Vertex => ShaderStages::VERTEX,
                    ShaderStage::Fragment => ShaderStages::FRAGMENT,
                    ShaderStage::Compute => ShaderStages::COMPUTE,
                },
            }
        };

        FunctionGenCtx {
            config,
            rng: StdRand::with_seed(seed),
            module,
            function,
            expr_generators: Self::setup_expr_generators(config),
            stmt_generators: Self::setup_stmt_generators(config),
            expr_scope: ExprScope::new(Some(function), strict_typing),
            available_funcs,
            code_ctx: CodeContext::empty(),
            for_shader_stages,
            expr_vs_stmt_prob: 0.7,
            recursive_budget_required: 5,
        }
    }

    fn setup_expr_generators(config: &GeneratorConfig) -> Vec<(ExpressionGenerators, u32)> {
        ExpressionGenerators::iter()
            .map(|gen| (gen, config.expression_weight_map.weights[gen as usize]))
            .collect()
    }

    fn setup_stmt_generators(config: &GeneratorConfig) -> Vec<(StatementGenerators, u32)> {
        StatementGenerators::iter()
            .map(|gen| (gen, config.statement_weight_map.weights[gen as usize]))
            .collect()
    }

    pub fn get_function_mut(&mut self) -> &mut Function {
        match self.function {
            FunctionIdentifier::Function(handle) => &mut self.module.functions[handle],
            FunctionIdentifier::EntryPoint(idx) => &mut self.module.entry_points[idx].function,
        }
    }

    pub fn get_function(&self) -> &Function {
        match self.function {
            FunctionIdentifier::Function(handle) => &self.module.functions[handle],
            FunctionIdentifier::EntryPoint(idx) => &self.module.entry_points[idx].function,
        }
    }

    fn generate_expr(&mut self) -> Option<Expression> {
        let (gen, _) = *self
            .expr_generators
            .choose_weighted(&mut self.rng, |(gen, weight)| {
                if !gen.allowed_shader_stages().contains(self.for_shader_stages) {
                    return 0;
                }
                *weight
            })
            .ok()?;
        gen.generate(self)
    }

    fn generate_stmt(&mut self, budget: u32) -> Option<(Statement, u32)> {
        let (gen, _) = *self
            .stmt_generators
            .choose_weighted(&mut self.rng, |(gen, weight)| {
                if budget <= self.recursive_budget_required && gen.may_recurse() {
                    return 0;
                }
                if !gen.allowed_code_context(self.code_ctx) {
                    return 0;
                }
                if !gen.allowed_shader_stages().contains(self.for_shader_stages) {
                    return 0;
                }
                *weight
            })
            .ok()?;
        gen.generate(self, budget)
    }

    pub(super) fn recursive_generate(&mut self, budget: u32) -> (Block, u32) {
        let entry_expressions = self.expr_scope.scope_available.len();
        let mut body = Vec::new();
        let mut spent_budget = 0;

        'outer: loop {
            if spent_budget >= budget {
                break;
            }
            let remaining_budget = budget - spent_budget;

            if remaining_budget > 1 && self.rng.probability(self.expr_vs_stmt_prob) {
                for _ in 0..20 {
                    let Some(expr) = self.generate_expr() else {
                        continue;
                    };

                    let exprs = &self.get_function().expressions;
                    let always_available = ExprScope::is_always_available(&expr);
                    if always_available
                        && self
                            .expr_scope
                            .always_available
                            .iter()
                            .any(|eh| exprs[*eh] == expr)
                    {
                        spent_budget += 1;
                        continue;
                    }
                    self.expr_scope.add_use(&expr);
                    let handle = self
                        .get_function_mut()
                        .expressions
                        .append(expr, Span::UNDEFINED);
                    self.expr_scope.add_available(self.module, handle);

                    if !always_available {
                        body.push(Statement::Emit(Range::new_from_bounds(handle, handle)));
                    }
                    spent_budget += 1;
                    continue 'outer;
                }
                break 'outer;
            } else {
                for _ in 0..10 {
                    let Some((stmt, cost)) = self.generate_stmt(budget) else {
                        continue;
                    };
                    spent_budget += cost;
                    body.push(stmt);
                    if self.code_ctx.contains(CodeContext::DEAD) {
                        break 'outer;
                    }
                    continue 'outer;
                }
                break 'outer;
            }
        }
        assert!(entry_expressions <= self.expr_scope.scope_available.len());
        self.expr_scope.scope_available.truncate(entry_expressions);
        (Block::from_vec(body), spent_budget)
    }

    pub fn generate_at(&mut self, block: *const Block, budget: u32) -> u32 {
        let (initial_block, available_exprs) = 'outer: {
            let mut scope_exprs: Vec<Vec<_>> = Vec::new();
            for item in DfsFuncIter::new(&self.get_function().body) {
                match item {
                    DfsItem::BlockOpen(b) => {
                        if b as *const Block == block {
                            let initial_block = b.clone();
                            let available_exprs: Vec<_> =
                                scope_exprs.into_iter().flatten().collect();
                            break 'outer (initial_block, available_exprs);
                        } else {
                            scope_exprs.push(Vec::new());
                        }
                    }
                    DfsItem::BlockClose(_) => {
                        scope_exprs.pop();
                    }
                    DfsItem::Statement(Statement::Emit(exprs)) => {
                        scope_exprs.last_mut().unwrap().extend(exprs.clone());
                    }
                    DfsItem::Statement(_) => {}
                }
            }
            unreachable!("block not part of function");
        };

        for handle in available_exprs.into_iter() {
            self.expr_scope.add_available(self.module, handle);
        }

        let (mut new_block, cost) = self.recursive_generate(budget);
        if !self.code_ctx.contains(CodeContext::DEAD) {
            new_block.extend_block(initial_block);
        }

        let mut updated_block = false;
        let block_updater = |b: &mut Block| {
            if b as *const Block == block {
                updated_block = true;
                *b = std::mem::take(&mut new_block);
                return false;
            }
            true
        };
        self.get_function_mut().visit_blocks_mut(block_updater);
        assert!(updated_block);

        cost
    }

    pub fn recursive_budget(&mut self, initial_budget: u32, num_cases: u32) -> u32 {
        let base_budget = initial_budget / num_cases;
        let min_budget = (base_budget as f32 * self.config.recursive_budget_rate.min) as u64;
        let max_budget = (base_budget as f32 * self.config.recursive_budget_rate.max) as u64;
        let budget = self.rng.between(min_budget, max_budget);
        std::cmp::max(1, budget) as u32
    }

    pub fn expr_matching<F>(&self, filter: F) -> Option<(Handle<Expression>, &TypeInner)>
    where
        F: Fn(Handle<Expression>, &TypeInner) -> bool,
    {
        self.expr_scope.matching(filter, &self.module.types)
    }

    pub fn expr_of_type(&self, ty: &TypeInner) -> Option<Handle<Expression>> {
        let filter = |_, other_ty: &TypeInner| other_ty == ty;
        self.expr_matching(filter).map(|(expr, _)| expr)
    }
}

#[derive(Debug)]
pub struct IRGenerator {
    config: GeneratorConfig,
}

impl IRGenerator {
    /// Creates a new [`IRGenerator`].
    #[must_use]
    pub const fn new(config: GeneratorConfig) -> Self {
        Self { config }
    }
}

impl Named for IRGenerator {
    fn name(&self) -> &str {
        "IRGenerator"
    }
}

impl<S> Generator<LayeredInput, S> for IRGenerator
where
    S: HasRand,
{
    fn generate(&mut self, state: &mut S) -> Result<LayeredInput, Error> {
        let mut module = Module::default();
        let mut rng = StdRand::with_seed(state.rand_mut().next());
        let mut gen = GlobalGenCtx::new(&self.config, &mut module, rng);

        gen.emit_basic_types();
        for _ in 0..self.config.complex_types_count.choose(&mut rng) {
            gen.create_type();
        }
        let mut eps = Vec::new();
        for _ in 0..self.config.entrypoints_count.choose(&mut rng) {
            eps.push(gen.create_entrypoint());
        }

        for _ in 0..self.config.global_const_exprs_count.choose(&mut rng) {
            gen.create_const_expr();
        }
        for _ in 0..self.config.global_variables_count.choose(&mut rng) {
            gen.create_global();
        }

        let mut functions = Vec::new();
        for _ in 0..self.config.functions_count.choose(&mut rng) {
            functions.push(gen.create_function());
        }

        for handle in functions {
            let entry: *const Block = &module.functions[handle].body;
            let budget = self.config.function_budget.choose(&mut rng);
            let mut gen = FunctionGenCtx::new(
                &self.config,
                &mut module,
                FunctionIdentifier::Function(handle),
                rng.next(),
                true,
            );
            gen.generate_at(entry, budget);
        }

        for ep in eps {
            let entry: *const Block = &module.entry_points[ep].function.body;
            let budget = self.config.function_budget.choose(&mut rng);
            let mut gen = FunctionGenCtx::new(
                &self.config,
                &mut module,
                FunctionIdentifier::EntryPoint(ep),
                rng.next(),
                true,
            );
            gen.generate_at(entry, budget);
        }

        let ir = IR::new(module);

        let text = match ir.try_get_text() {
            Ok(text) => text,
            Err(err) => {
                println!("Generator built invalid file: {}", err);
                let ast = Ast::try_from_wgsl("".as_bytes()).unwrap();
                return Ok(LayeredInput::Ast(ast));
            }
        };

        match IR::try_from(text.as_str()) {
            Ok(ir) => Ok(LayeredInput::IR(ir)),
            Err(_) => {
                let ast = Ast::try_from_wgsl("".as_bytes()).unwrap();
                Ok(LayeredInput::Ast(ast))
            }
        }
    }
}
