use std::borrow::Cow;
use std::num::NonZeroU32;

use libafl::{generators::Generator, state::HasRand};
use libafl_bolts::{
    rands::{Rand, StdRand},
    Error, Named,
};
use log::error;
use naga::{
    valid::{ShaderStages, TypeFlags},
    AddressSpace, ArraySize, Binding, Block, BuiltIn, EntryPoint, Expression, Function,
    FunctionArgument, FunctionResult, GlobalVariable, Handle, Interpolation, LocalVariable, Module,
    Range, ResourceBinding, Scalar, ScalarKind, ShaderStage, Span, Statement, StorageAccess,
    StructMember, Type, TypeInner, VectorSize,
};
use rand::{
    seq::{IndexedRandom, IteratorRandom, SliceRandom},
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
use crate::ir::naga_enums::NagaEnum;

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
    const COMPUTE_OPT_INPUTS: &'static [(TypeInner, Binding)] = &[
        (
            TypeInner::Scalar(Scalar::U32),
            Binding::BuiltIn(BuiltIn::LocalInvocationIndex),
        ),
        (
            TypeInner::Vector {
                size: VectorSize::Tri,
                scalar: Scalar::U32,
            },
            Binding::BuiltIn(BuiltIn::LocalInvocationId),
        ),
        (
            TypeInner::Vector {
                size: VectorSize::Tri,
                scalar: Scalar::U32,
            },
            Binding::BuiltIn(BuiltIn::GlobalInvocationId),
        ),
        (
            TypeInner::Vector {
                size: VectorSize::Tri,
                scalar: Scalar::U32,
            },
            Binding::BuiltIn(BuiltIn::WorkGroupId),
        ),
        (
            TypeInner::Vector {
                size: VectorSize::Tri,
                scalar: Scalar::U32,
            },
            Binding::BuiltIn(BuiltIn::NumWorkGroups),
        ),
    ];

    const VERTEX_OPT_INPUTS: &'static [(TypeInner, Binding)] = &[
        (
            TypeInner::Scalar(Scalar::U32),
            Binding::BuiltIn(BuiltIn::VertexIndex),
        ),
        (
            TypeInner::Scalar(Scalar::U32),
            Binding::BuiltIn(BuiltIn::InstanceIndex),
        ),
    ];

    const FRAGMENT_OPT_INPUTS: &'static [(TypeInner, Binding)] = &[
        (
            TypeInner::Scalar(Scalar::U32),
            Binding::BuiltIn(BuiltIn::SampleIndex),
        ),
        (
            TypeInner::Scalar(Scalar::U32),
            Binding::BuiltIn(BuiltIn::SampleMask),
        ),
        (
            TypeInner::Scalar(Scalar::BOOL),
            Binding::BuiltIn(BuiltIn::FrontFacing),
        ),
        (
            TypeInner::Vector {
                size: VectorSize::Quad,
                scalar: Scalar::F32,
            },
            Binding::BuiltIn(BuiltIn::Position { invariant: false }),
        ),
    ];

    const FRAGMENT_OPT_OUTPUTS: &'static [(TypeInner, Binding)] = &[
        (
            TypeInner::Scalar(Scalar::F32),
            Binding::BuiltIn(BuiltIn::FragDepth),
        ),
        (
            TypeInner::Scalar(Scalar::U32),
            Binding::BuiltIn(BuiltIn::SampleMask),
        ),
    ];

    fn required_output(stage: ShaderStage) -> Option<(TypeInner, Binding)> {
        match stage {
            ShaderStage::Vertex => {
                let inner = TypeInner::Vector {
                    size: VectorSize::Quad,
                    scalar: Scalar::F32,
                };
                let binding = Binding::BuiltIn(BuiltIn::Position { invariant: false });
                Some((inner, binding))
            }
            ShaderStage::Fragment => None,
            ShaderStage::Compute | ShaderStage::Task | ShaderStage::Mesh => unreachable!(),
        }
    }

    fn optional_inputs(stage: ShaderStage) -> &'static [(TypeInner, Binding)] {
        match stage {
            ShaderStage::Vertex => Self::VERTEX_OPT_INPUTS,
            ShaderStage::Fragment => Self::FRAGMENT_OPT_INPUTS,
            ShaderStage::Compute => Self::COMPUTE_OPT_INPUTS,
            ShaderStage::Task | ShaderStage::Mesh => &[],
        }
    }

    fn optional_outputs(stage: ShaderStage) -> &'static [(TypeInner, Binding)] {
        match stage {
            ShaderStage::Vertex => &[],
            ShaderStage::Fragment => Self::FRAGMENT_OPT_OUTPUTS,
            ShaderStage::Compute | ShaderStage::Task | ShaderStage::Mesh => unreachable!(),
        }
    }

    fn random_io(location: u32, rng: &mut StdRand) -> (TypeInner, Binding) {
        let scalar = rng.choose([Scalar::I32, Scalar::U32, Scalar::F32]).unwrap();
        let inner = if rng.probability(0.5) {
            TypeInner::Scalar(scalar)
        } else {
            let size = rng
                .choose([VectorSize::Bi, VectorSize::Tri, VectorSize::Quad])
                .unwrap();
            TypeInner::Vector { size, scalar }
        };

        let interpolation = {
            if matches!(
                inner,
                TypeInner::Scalar(naga::Scalar {
                    kind: ScalarKind::Float,
                    ..
                }) | TypeInner::Vector {
                    scalar: naga::Scalar {
                        kind: ScalarKind::Float,
                        ..
                    },
                    ..
                }
            ) {
                rng.choose([
                    Interpolation::Flat,
                    Interpolation::Linear,
                    Interpolation::Perspective,
                ])
                .unwrap()
            } else {
                Interpolation::Flat
            }
        };
        (
            inner,
            Binding::Location {
                location,
                interpolation: Some(interpolation),
                sampling: None,
                blend_src: None,
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
        for (handle, expr) in module.global_expressions.iter() {
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
        use TypeInner as TI;
        for scalar in [Scalar::BOOL, Scalar::I32, Scalar::U32, Scalar::F32].into_iter() {
            let inner = TI::Scalar(scalar);
            let typ = Type { name: None, inner };
            self.module.types.insert(typ, Span::UNDEFINED);
        }

        self.module.types.insert(
            Type {
                name: None,
                inner: TI::Atomic(Scalar::I32),
            },
            Span::UNDEFINED,
        );
        self.module.types.insert(
            Type {
                name: None,
                inner: TI::Atomic(Scalar::U32),
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
        inputs.truncate(self.rng.below_or_zero(inputs.len() + 1));
        if self.rng.probability(0.5) {
            let num_scalars = self.rng.between(1, 5) as u32;
            for idx in 0..num_scalars {
                inputs.push(EntryPointSpec::random_io(idx, &mut self.rng));
            }
        }
        inputs.shuffle(&mut self.rng);
        let min_inputs = if inputs.is_empty() { 0 } else { 1 };
        let num_args = self.rng.between(min_inputs, inputs.len());
        let mut args = Vec::with_capacity(num_args);
        let mut remaining_inputs = inputs.len();
        for idx in 0..num_args {
            let min_params = if idx == num_args - 1 {
                remaining_inputs
            } else {
                1
            };
            let max_params = remaining_inputs - (num_args - idx - 1);
            let params = self.rng.between(min_params, max_params);
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
            ShaderStage::Compute | ShaderStage::Task | ShaderStage::Mesh => None,
            ShaderStage::Vertex | ShaderStage::Fragment => {
                let mut optional_outputs = EntryPointSpec::optional_outputs(stage).to_owned();
                optional_outputs.shuffle(&mut self.rng);
                optional_outputs.truncate(self.rng.below_or_zero(optional_outputs.len() + 1));
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
            TypeInner::Scalar(_) | TypeInner::Vector { .. } | TypeInner::Matrix { .. } => true,
            TypeInner::Atomic(_) => !require_constructible,
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
        let columns = *self.rng.choose(&vector_sizes).unwrap();
        let rows = *self.rng.choose(&vector_sizes).unwrap();
        let inner = TypeInner::Matrix {
            columns,
            rows,
            scalar: Scalar::F32,
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
        let size = *self.rng.choose(&vector_sizes).unwrap();

        let scalar = *self
            .rng
            .choose(&[Scalar::BOOL, Scalar::I32, Scalar::U32, Scalar::F32])
            .unwrap();
        let inner = TypeInner::Vector { size, scalar };
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
        let ty = match self.rng.below_or_zero(5) {
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
        let access = *self.rng.choose(&accesses).unwrap();
        let spaces = [
            AddressSpace::Private,
            AddressSpace::Uniform,
            AddressSpace::WorkGroup,
            AddressSpace::Storage { access },
        ];
        let space = *self.rng.choose(&spaces).unwrap();
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
            if matches!(inner, TypeInner::Atomic(_))
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
        let handle = self.module.global_expressions.append(expr, Span::UNDEFINED);
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

        let num_locals = self.rng.below_or_zero(5);
        let num_args = self.rng.below_or_zero(3);
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
                    [
                        Statement::Emit(Range::new_from_bounds(handle, handle)),
                        Statement::Return {
                            value: Some(handle),
                        },
                    ]
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
            let expr = Expression::LocalVariable(handle);
            func.expressions.append(expr, Span::UNDEFINED);
        }

        self.module.functions.append(func, Span::UNDEFINED)
    }

    fn create_entrypoint(&mut self) -> usize {
        let stage = *self.rng.choose(ShaderStage::VALUES).unwrap();

        let mut func = Function::default();
        let num_locals = self.rng.below_or_zero(5);

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
                [
                    Statement::Emit(Range::new_from_bounds(handle, handle)),
                    Statement::Return {
                        value: Some(handle),
                    },
                ]
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
            workgroup_size_overrides: None,
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
                    ShaderStage::Task | ShaderStage::Mesh => unreachable!(),
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
        let min_budget = (base_budget as f32 * self.config.recursive_budget_rate.min) as usize;
        let max_budget = (base_budget as f32 * self.config.recursive_budget_rate.max) as usize;
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
    fn name(&self) -> &Cow<'static, str> {
        const NAME: Cow<'static, str> = Cow::Borrowed("IRGenerator");
        &NAME
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
                error!("Generator built invalid file: {}", err);
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

#[cfg(test)]
mod tests {
    use super::*;

    struct SeededState(StdRand);
    impl HasRand for SeededState {
        type Rand = StdRand;
        fn rand(&self) -> &Self::Rand {
            &self.0
        }
        fn rand_mut(&mut self) -> &mut Self::Rand {
            &mut self.0
        }
    }

    const GENERATION_COUNT: u64 = 500;

    struct MutatorTestState {
        rand: StdRand,
        corpus: libafl::corpus::InMemoryCorpus<LayeredInput>,
        current_corpus_id: Option<libafl::corpus::CorpusId>,
    }

    impl HasRand for MutatorTestState {
        type Rand = StdRand;

        fn rand(&self) -> &Self::Rand {
            &self.rand
        }

        fn rand_mut(&mut self) -> &mut Self::Rand {
            &mut self.rand
        }
    }

    impl libafl::state::HasCorpus<LayeredInput> for MutatorTestState {
        type Corpus = libafl::corpus::InMemoryCorpus<LayeredInput>;

        fn corpus(&self) -> &Self::Corpus {
            &self.corpus
        }

        fn corpus_mut(&mut self) -> &mut Self::Corpus {
            &mut self.corpus
        }
    }

    impl libafl::corpus::HasCurrentCorpusId for MutatorTestState {
        fn current_corpus_id(&self) -> Result<Option<libafl::corpus::CorpusId>, libafl::Error> {
            Ok(self.current_corpus_id)
        }

        fn set_corpus_id(&mut self, id: libafl::corpus::CorpusId) -> Result<(), libafl::Error> {
            self.current_corpus_id = Some(id);
            Ok(())
        }

        fn clear_corpus_id(&mut self) -> Result<(), libafl::Error> {
            self.current_corpus_id = None;
            Ok(())
        }
    }

    // Baseline measurements taken on 2026-09-17 with naga 0.14.2 and GeneratorConfig::default():
    // Ten consecutive runs of 100 iterations yielded success counts of:
    //   87, 84, 86, 84, 87, 87, 87, 86, 93, 87 (mean: 86.8%).
    // Note that this baseline was measured with module validation effectively disabled,
    // because `ValidationFlags::all()` evaluates to 0 on naga 0.14.2 — the flag constants
    // are gated behind a `validate` feature this crate does not enable.
    //
    // Re-baselined on 2026-09-18 for naga 27.0.3: 12 pooled runs of n=500 gave a mean of
    // 63.6%, range 59.6%–66.2%, sigma 2.2. Validation is still disabled, now explicitly
    // (see `LayeredInput::try_get_text`), so this remains a like-for-like comparison.
    //
    // The drop from 86.8% is dominated by naga 27's WGSL front end rejecting its own back
    // end's output: under zero validation flags every re-parse failure is a parse error,
    // and most are constant-evaluation rejections (non-finite float literals, shift counts
    // >= 32, division and remainder by zero) that naga 0.14's front end did not perform.
    //
    // The threshold is a safety floor, not an estimate of the true rate: it is set at
    // roughly mean - 5 sigma so that normal run-to-run variation cannot trip it, while a
    // real regression still will.
    const MIN_SUCCESS_PERCENT: u64 = 50;

    /// Verifies that the IR generator produces modules that successfully emit and re-parse as WGSL.
    ///
    /// Why this test exists:
    /// `IRGenerator::generate` silently swallows generation, validation, and re-parsing failures,
    /// returning `Ok(LayeredInput::Ast(<empty>))` on error. A broken generator producing only
    /// empty AST fallbacks is therefore indistinguishable from a healthy one at all call sites.
    /// This test turns that silent degradation into an explicit, monitored health measurement.
    ///
    /// Run with `cargo +nightly test -p darthshader -- --nocapture` to inspect the emitted success rate.
    #[test]
    fn generator_emits_reparsable_wgsl() {
        let mut generator = IRGenerator::new(GeneratorConfig::default());
        let mut successes = 0u64;
        let mut failing_seeds = Vec::new();

        // Note: the seed narrows a failure down but does not fully guarantee reproducibility,
        // because `ExprScope::matching` and `ExprScope::any` (`ir/exprscope.rs:214`, `:255`)
        // clock-seed a fresh RNG per call.
        for seed in 0..GENERATION_COUNT {
            let mut state = SeededState(StdRand::with_seed(seed));
            match generator
                .generate(&mut state)
                .expect("IRGenerator::generate should never return Err")
            {
                LayeredInput::IR(_) => successes += 1,
                LayeredInput::Ast(_) => failing_seeds.push(seed),
            }
        }

        let pct = (successes * 100) / GENERATION_COUNT;
        println!(
            "generated {successes}/{GENERATION_COUNT} modules that emit re-parsable WGSL ({pct}%)"
        );

        let capped_failing: Vec<_> = failing_seeds.iter().copied().take(20).collect();
        let truncated = if failing_seeds.len() > 20 {
            format!(" (truncated, showing first 20 of {})", failing_seeds.len())
        } else {
            String::new()
        };

        assert!(
            successes * 100 >= GENERATION_COUNT * MIN_SUCCESS_PERCENT,
            "Generator success rate below threshold: {successes}/{GENERATION_COUNT} ({pct}% < {MIN_SUCCESS_PERCENT}%). Failing seeds{truncated}: {capped_failing:?}"
        );
    }

    /// Baseline convergence floor for multi-generation round trips.
    ///
    /// Measured on 2026-09-18 with naga 0.14.2 and `GeneratorConfig::default()` over 3 runs of n=500:
    /// - Convergence histogram: k=1 25.8%, k=2 40.2%, k=3 27.0%, k=4 4.8%, unconverged 2.1%
    ///   (cumulative: 93.1% by k=3, 97.9% by k=4).
    /// - Zero oscillations observed across 1,143 evaluated modules.
    ///
    /// This threshold is a conservative floor with margin to catch degradation regressions,
    /// not an estimate of the true convergence rate.
    ///
    /// Re-baselined on 2026-09-18 for naga 27.0.3: 12 pooled runs gave a mean of 93.3%,
    /// range 90.9%–96.7%, sigma 1.5, with 6–15 modules degraded per run.
    ///
    /// Note that the previous floor of 90 had to move irrespective of this upgrade — the
    /// lowest of those 12 runs was 90.9%, so it was already close enough to trip on
    /// ordinary variation. The new value is set at roughly mean - 5 sigma.
    const MIN_CONVERGENCE_PERCENT: u64 = 85;
    const ROUND_TRIP_COUNT: usize = 4;

    /// Tests that repeated WGSL emit -> parse -> emit round trips converge to a textual fixpoint.
    ///
    /// The round trip is **not** a one-step fixpoint — only ~25% of modules are byte-identical
    /// after a single round trip — because `front::wgsl` canonicalises input during parsing:
    /// it folds constant expressions, canonicalises vector index accesses (e.g. `v[0]` -> `v.x`),
    /// and renumbers expression handles. Because those canonicalisations are idempotent, the
    /// meaningful semantic and syntactic property across round trips is **convergence**.
    ///
    /// Unlike a golden-file test, this test compares naga against itself, making it resilient
    /// against harmless backend formatting changes. However, its coverage envelope is bounded
    /// by the generator, so this test detects changes and regressions, not absolute correctness.
    ///
    /// The small non-converging remainder (~2%) is dominantly caused by naga's namer appending
    /// a disambiguating underscore on successive passes (observed: `_e24` -> `_e24_`, and
    /// `_e113` -> `_e113_1`).
    ///
    /// Shaders that parse once but fail a subsequent round-trip parse are counted as `degraded`
    /// and treated as evaluated but non-converged, ensuring mid-chain parse regressions directly
    /// trip the convergence threshold.
    #[test]
    fn generator_round_trip_converges() {
        let mut generator = IRGenerator::new(GeneratorConfig::default());
        let mut generated = 0u64;
        let mut evaluated = 0u64;
        let mut converged = 0u64;
        let mut degraded = 0u64;
        let mut non_converged_seeds = Vec::new();

        for seed in 0..GENERATION_COUNT {
            let mut state = SeededState(StdRand::with_seed(seed));
            let ir = match generator
                .generate(&mut state)
                .expect("IRGenerator::generate should never return Err")
            {
                LayeredInput::IR(ir) => {
                    generated += 1;
                    ir
                }
                LayeredInput::Ast(_) => continue,
            };

            let mut text = match ir.try_get_text() {
                Ok(t) => t.clone(),
                Err(_) => continue,
            };

            let mut was_evaluated = false;
            let mut is_converged = false;

            for round in 0..ROUND_TRIP_COUNT {
                let next_ir = match IR::try_from(text.as_str()) {
                    Ok(ir) => ir,
                    Err(_) => {
                        if round > 0 {
                            degraded += 1;
                        }
                        break;
                    }
                };
                let next_text = match next_ir.try_get_text() {
                    Ok(t) => t.clone(),
                    Err(_) => {
                        if round > 0 {
                            degraded += 1;
                        }
                        break;
                    }
                };

                if round == 0 {
                    evaluated += 1;
                    was_evaluated = true;
                }

                if next_text == text {
                    converged += 1;
                    is_converged = true;
                    break;
                }
                text = next_text;
            }

            if was_evaluated && !is_converged {
                non_converged_seeds.push(seed);
            }
        }

        let pct = if evaluated > 0 {
            (converged * 100) / evaluated
        } else {
            0
        };

        println!(
            "round-trip convergence: generated {generated}, evaluated {evaluated}, converged {converged} ({pct}%), degraded {degraded}"
        );

        let capped_non_converged: Vec<_> = non_converged_seeds.iter().copied().take(10).collect();
        let truncated = if non_converged_seeds.len() > 10 {
            format!(
                " (truncated, showing first 10 of {})",
                non_converged_seeds.len()
            )
        } else {
            String::new()
        };

        assert!(
            evaluated > 0 && converged * 100 >= evaluated * MIN_CONVERGENCE_PERCENT,
            "Round-trip convergence below threshold: {converged}/{evaluated} ({pct}% < {MIN_CONVERGENCE_PERCENT}%). Generated: {generated}, degraded: {degraded}. Non-converged seeds{truncated}: {capped_non_converged:?}"
        );
    }

    /// Number of generated modules that every mutator is applied to.
    ///
    /// This test applies all ten mutators to each seed, so it is ten times the work per seed
    /// that the generation tests are. 250 keeps it near two seconds while still giving each
    /// mutator roughly 190 mutations per run.
    const MUTATION_SEED_COUNT: u64 = 250;

    const AGGREGATE_MUTATION_FLOOR: u64 = 80;

    /// Per-mutator floors for the fraction of mutated modules that still emit WGSL.
    ///
    /// The keys are the names the mutators report at runtime, which are not their type names.
    /// The mapping is: `IRStatementInputMutator (untyped)` is `RewireStatementMutator`,
    /// `IRRewireExpressionMutator (untyped)` is `RewireExpressionMutator`, and `IRBinOPMutator`
    /// is `BinOpMutator`; the rest correspond by inspection. A mutator whose reported name is
    /// absent from this table fails the test, so adding one to `ir_mutations()` cannot leave it
    /// silently unmeasured.
    ///
    /// Measured 2026-09-18 with naga 0.14.2, then re-baselined the same day for naga 27.0.3
    /// over 12 pooled runs. Floors are set at roughly mean - 5 sigma, so that ordinary
    /// run-to-run variation cannot trip them while a real regression still will.
    ///
    /// | mutator                     | 0.14   | 27.0.3 mean (range)  | floor    |
    /// |-----------------------------|--------|----------------------|----------|
    /// | `UnaryOpMutator`            | 100.0% | 100.0% (no variance) | 90       |
    /// | `LiteralMutator`            | 100.0% | 100.0% (no variance) | 90       |
    /// | `TypeMutator`               | 100.0% | 100.0% (no variance) | 90       |
    /// | `CodeGenerationMutation`    | 100.0% | 100.0% (no variance) | 90       |
    /// | `FullGenerationMutation`    | 100.0% | 100.0% (no variance) | 90       |
    /// | `RewireStatementMutator`    | 100.0% | 99.7% (98.1–100)     | 90       |
    /// | `StatementMutator`          | 97.8%  | 96.7% (94.1–98.4)    | 85       |
    /// | `BinOpMutator`              | 100.0% | 88.5% (83.6–96.3)    | 90 -> 70 |
    /// | `MathFuncMutator`           | 80.6%  | 35.5% (29.5–41.9)    | 60 -> 20 |
    /// | `RewireExpressionMutator`   | 54.3%  | 31.2% (22.0–37.8)    | 35 -> 15 |
    /// | aggregate                   | 94.0%  | 87.5% (86.3–88.4)    | 85 -> 80 |
    ///
    /// `RewireExpressionMutator` is wired as `new(false)`, untyped, in production. A large
    /// invalid rate is the mutator working as designed — it deliberately rewires operands
    /// without regard to type compatibility. Not a defect.
    ///
    /// Observed failure mechanisms:
    /// - `StatementMutator`: deleting a `break` to leave a fall-through switch case, which WGSL forbids.
    /// - `MathFuncMutator`: it overwrites `Expression::Math`'s `fun` in place and leaves `arg`,
    ///   `arg1`, `arg2` and `arg3` untouched, so it produces calls of the wrong arity —
    ///   `Clamp(a, b, c)` becomes `Min(a, b, c)`, `Abs(a)` becomes `Clamp(a)` — as well as
    ///   swapping between functions with incompatible operand types. This is a pre-existing
    ///   defect, not an effect of the naga upgrade; naga 0.14 validated nothing and its WGSL
    ///   writer was laxer, which is why it measured 80.6% there. The fix is arity- and
    ///   type-aware replacement selection, which changes mutator semantics and is tracked
    ///   separately rather than being folded into a dependency bump.
    ///
    ///   Measured over 782 mutations: 540 failed, of which 290 (54%) had mismatched arity and
    ///   250 (46%) did not. Since a wrong-arity call essentially never emits, the arity-matched
    ///   population succeeds at 242/492 = 49%, so making selection arity-aware would lift this
    ///   mutator to roughly 49% — a real gain, but nowhere near the old 80.6%. Of the failures
    ///   that already had matching arity, 96% are operand-type mismatches (matrix-only builtins
    ///   such as `Transpose` applied to vectors, `Pack`/`Unpack` applied to scalars, integer
    ///   builtins applied to floats). Restoring the old rate therefore needs type-aware
    ///   selection, which is the substantially larger change.
    /// - `RewireExpressionMutator`: beyond the intended type incompatibility, roughly half its
    ///   failures (353 of 670 measured) are *forward references* — it rewires an operand to an
    ///   expression with a **higher** handle index than the expression using it. naga's handle
    ///   validation requires `depends_on < self` (`Handle::check_dep` in `valid/handles.rs`) and
    ///   runs regardless of `ValidationFlags`, so this is rejected even with validation otherwise
    ///   switched off. The remainder are type mismatches, chiefly pointer-shaped (148) and
    ///   indexing or scalar/vector shape errors (127).
    const MUTATOR_FLOORS: &[(&str, u64)] = &[
        ("IRUnaryOpMutator", 90),
        ("IRBinOPMutator", 70),
        ("IRMathFuncMutator", 20),
        ("IRLiteralMutator", 90),
        ("IRRewireExpressionMutator (untyped)", 15),
        ("IRStatementInputMutator (untyped)", 90),
        ("IRStatementMutator", 85),
        ("IRTypeMutator", 90),
        ("IRFullGenerationMutation", 90),
        ("IRCodeGenerationMutation", 90),
    ];

    fn lookup_mutator_floor(name: &str) -> Option<u64> {
        for &(n, floor) in MUTATOR_FLOORS {
            if n == name {
                return Some(floor);
            }
        }
        None
    }

    /// Tests that production IR mutators produce valid, emittable WGSL at expected rates.
    ///
    /// While `generator_emits_reparsable_wgsl` (T2b) establishes that the generator produces
    /// valid and re-parsable WGSL, this test covers the mutation pipeline (T2a). Generation
    /// and mutation fail for different reasons and would be broken independently by a naga upgrade.
    ///
    /// Per-mutator floors are enforced because an aggregate metric alone would hide a collapse
    /// in an individual mutator: a 100% -> 60% regression in one mutator shifts the aggregate
    /// rate by only about four percentage points.
    #[test]
    fn mutators_emit_valid_wgsl() {
        use crate::ir::mutate::ir_mutations;
        use libafl::corpus::{CorpusId, InMemoryCorpus};
        use libafl::mutators::{MutationId, MutationResult, MutatorsTuple};
        use libafl_bolts::tuples::NamedTuple;
        use libafl_bolts::HasLen;
        use std::collections::BTreeMap;

        let mut generator = IRGenerator::new(GeneratorConfig::default());
        let mut mutators = ir_mutations();
        let num_mutators = mutators.len();
        let mutator_names = mutators.names();

        for name in &mutator_names {
            assert!(
                lookup_mutator_floor(name).is_some(),
                "Mutator '{name}' present in ir_mutations() is missing from MUTATOR_FLOORS table. A floor must be defined for any new mutator."
            );
        }

        #[derive(Default)]
        struct MutStats {
            skipped: u64,
            errored: u64,
            mutated: u64,
            emits_wgsl: u64,
            errors: BTreeMap<String, u64>,
        }

        let mut stats: Vec<MutStats> = (0..num_mutators).map(|_| MutStats::default()).collect();

        for seed in 0..MUTATION_SEED_COUNT {
            let mut state = MutatorTestState {
                rand: StdRand::with_seed(seed),
                corpus: InMemoryCorpus::new(),
                current_corpus_id: Some(CorpusId::from(seed as usize)),
            };

            let ir = match generator
                .generate(&mut state)
                .expect("IRGenerator::generate should never return Err")
            {
                LayeredInput::IR(ir) => ir,
                LayeredInput::Ast(_) => continue,
            };

            for (idx, stat) in stats.iter_mut().enumerate().take(num_mutators) {
                let mut input = LayeredInput::IR(ir.clone());
                match mutators.get_and_mutate(MutationId::from(idx), &mut state, &mut input) {
                    Ok(MutationResult::Skipped) => {
                        stat.skipped += 1;
                    }
                    Ok(MutationResult::Mutated) => {
                        // An IR mutator is not expected to change the input's layer. If one
                        // ever does, count it once as an error rather than as a mutation, and
                        // name it distinctly so it is not mistaken for a WGSL emission failure.
                        let LayeredInput::IR(ref mut_ir) = input else {
                            stat.errored += 1;
                            *stat
                                .errors
                                .entry("mutator returned a non-IR input".to_string())
                                .or_insert(0) += 1;
                            continue;
                        };
                        stat.mutated += 1;
                        match mut_ir.try_get_text() {
                            Ok(_) => {
                                stat.emits_wgsl += 1;
                            }
                            Err(e) => {
                                let mut first_line = e.clone();
                                if let Some(pos) = first_line.find('\n') {
                                    first_line.truncate(pos);
                                }
                                *stat.errors.entry(first_line).or_insert(0) += 1;
                            }
                        }
                    }
                    Err(e) => {
                        stat.errored += 1;
                        let err_str = format!("Mutator error: {e}");
                        *stat.errors.entry(err_str).or_insert(0) += 1;
                    }
                }
            }
        }

        println!(
            "{:<36} | {:>7} | {:>7} | {:>7} | {:>10} | {:>10} | {:>9}",
            "Mutator", "Skipped", "Errored", "Mutated", "Emits WGSL", "Rate (%)", "Floor (%)"
        );
        println!(
            "{:-<36}-+-{:-<7}-+-{:-<7}-+-{:-<7}-+-{:-<10}-+-{:-<10}-+-{:-<9}",
            "", "", "", "", "", "", ""
        );

        let mut agg_skipped = 0u64;
        let mut agg_errored = 0u64;
        let mut agg_mutated = 0u64;
        let mut agg_emits = 0u64;

        for (idx, name) in mutator_names.iter().enumerate() {
            let s = &stats[idx];
            let total_attempts = s.mutated + s.errored;
            let rate = (s.emits_wgsl * 100)
                .checked_div(total_attempts)
                .unwrap_or(0);
            let floor = lookup_mutator_floor(name).unwrap_or(0);
            println!(
                "{:<36} | {:>7} | {:>7} | {:>7} | {:>10} | {:>9}% | {:>8}%",
                name, s.skipped, s.errored, s.mutated, s.emits_wgsl, rate, floor
            );
            agg_skipped += s.skipped;
            agg_errored += s.errored;
            agg_mutated += s.mutated;
            agg_emits += s.emits_wgsl;
        }

        let total_agg_attempts = agg_mutated + agg_errored;
        let agg_rate = (agg_emits * 100)
            .checked_div(total_agg_attempts)
            .unwrap_or(0);
        println!(
            "{:-<36}-+-{:-<7}-+-{:-<7}-+-{:-<7}-+-{:-<10}-+-{:-<10}-+-{:-<9}",
            "", "", "", "", "", "", ""
        );
        println!(
            "{:<36} | {:>7} | {:>7} | {:>7} | {:>10} | {:>9}% | {:>8}%",
            "AGGREGATE",
            agg_skipped,
            agg_errored,
            agg_mutated,
            agg_emits,
            agg_rate,
            AGGREGATE_MUTATION_FLOOR
        );

        let mut failures = Vec::new();

        // 1. Assert every mutator produced at least one Mutated
        for (idx, name) in mutator_names.iter().enumerate() {
            if stats[idx].mutated == 0 {
                failures.push(format!(
                    "Mutator '{name}' produced zero mutations across {MUTATION_SEED_COUNT} seeds (always skipped)"
                ));
            }
        }

        // 2. Assert each mutator meets its floor
        for (idx, name) in mutator_names.iter().enumerate() {
            let s = &stats[idx];
            let total_attempts = s.mutated + s.errored;
            let rate = (s.emits_wgsl * 100)
                .checked_div(total_attempts)
                .unwrap_or(0);
            let floor = lookup_mutator_floor(name).unwrap_or(0);
            if rate < floor {
                let err_context = if s.errors.is_empty() {
                    String::new()
                } else {
                    let mut top_errs: Vec<_> = s.errors.iter().collect();
                    top_errs.sort_by_key(|&(_, c)| std::cmp::Reverse(*c));
                    let top: Vec<_> = top_errs
                        .iter()
                        .take(3)
                        .map(|(msg, c)| format!("      [{c}] {msg}"))
                        .collect();
                    format!("\n    top failure reasons:\n{}", top.join("\n"))
                };
                failures.push(format!(
                    "Mutator '{name}' below floor: {}/{} ({rate}% < floor {floor}%, skipped: {}, errored: {}){err_context}",
                    s.emits_wgsl, total_attempts, s.skipped, s.errored
                ));
            }
        }

        // 3. Assert aggregate meets floor
        if agg_rate < AGGREGATE_MUTATION_FLOOR {
            failures.push(format!(
                "AGGREGATE mutation rate below floor: {agg_emits}/{total_agg_attempts} ({agg_rate}% < floor {AGGREGATE_MUTATION_FLOOR}%, skipped: {agg_skipped}, errored: {agg_errored})"
            ));
        }

        assert!(
            failures.is_empty(),
            "Mutation validity assertions failed ({} failure(s)):\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
}
