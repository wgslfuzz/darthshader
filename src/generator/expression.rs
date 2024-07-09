use libafl_bolts::rands::{Rand, StdRand};
use naga::{
    valid::{ShaderStages, TypeFlags},
    AddressSpace, Arena, ArraySize, Constant, Expression, GlobalVariable, Handle, Literal, Module,
    ScalarKind, Type, TypeInner, UniqueArena, VectorSize,
};
use rand::seq::IteratorRandom;
use serde::Deserialize;

use crate::{ir::exprscope::ExprScope, randomext::RandExt};

use super::generateir::{FunctionGenCtx, GlobalGenCtx};

trait ExpressionGenerator {
    fn generate(ctx: &mut FunctionGenCtx) -> Option<Expression>;
}

trait ConstExpressionGenerator {
    fn generate(ctx: &mut GlobalGenCtx) -> Option<Expression>;
}

#[derive(Clone, Copy, Debug, Deserialize, Hash, Eq, PartialEq)]
pub(super) enum ConstExpressionGenerators {
    Compose,
    Constant,
    Literal,
    Splat,
    ZeroVal,
}

impl ConstExpressionGenerators {
    pub(super) fn generate(&self, ctx: &mut GlobalGenCtx) -> Option<Expression> {
        match self {
            Self::Compose => <ComposeGenerator as ConstExpressionGenerator>::generate(ctx),
            Self::Constant => <ConstantGenerator as ConstExpressionGenerator>::generate(ctx),
            Self::Literal => <LiteralGenerator as ConstExpressionGenerator>::generate(ctx),
            Self::Splat => <SplatGenerator as ConstExpressionGenerator>::generate(ctx),
            Self::ZeroVal => <ZeroValGenerator as ConstExpressionGenerator>::generate(ctx),
        }
    }

    pub(super) fn iter() -> impl Iterator<Item = ConstExpressionGenerators> {
        use ConstExpressionGenerators as CEG;
        static VALUES: [CEG; std::mem::variant_count::<CEG>()] = [
            CEG::Compose,
            CEG::Constant,
            CEG::Literal,
            CEG::Splat,
            CEG::ZeroVal,
        ];
        VALUES.iter().cloned()
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Hash, Eq, PartialEq)]
pub(super) enum ExpressionGenerators {
    Access,
    AccessIndex,
    ArrayLength,
    Derivative,
    FunctionArgument,
    GlobalVar,
    Load,
    LocalVar,
    Relational,
    As,
    Binary,
    Select,
    Math,
    Unary,
    Swizzle,
    Literal,
    Compose,
    Constant,
    Splat,
    ZeroVal,
}

impl ExpressionGenerators {
    pub fn allowed_shader_stages(&self) -> ShaderStages {
        match self {
            Self::Derivative => ShaderStages::FRAGMENT,
            _ => ShaderStages::all(),
        }
    }

    pub fn generate(&self, ctx: &mut FunctionGenCtx) -> Option<Expression> {
        match self {
            Self::Access => AccessGenerator::generate(ctx),
            Self::AccessIndex => AccessIndexGenerator::generate(ctx),
            Self::ArrayLength => ArrayLengthGenerator::generate(ctx),
            Self::Derivative => DerivativeGenerator::generate(ctx),
            Self::FunctionArgument => FunctionArgumentGenerator::generate(ctx),
            Self::GlobalVar => GlobalVarGenerator::generate(ctx),
            Self::Load => LoadGenerator::generate(ctx),
            Self::LocalVar => LocalVarGenerator::generate(ctx),
            Self::Relational => RelationalGenerator::generate(ctx),
            Self::As => AsGenerator::generate(ctx),
            Self::Binary => BinaryGenerator::generate(ctx),
            Self::Select => SelectGenerator::generate(ctx),
            Self::Math => MathGenerator::generate(ctx),
            Self::Unary => UnaryGenerator::generate(ctx),
            Self::Swizzle => SwizzleGenerator::generate(ctx),
            Self::Literal => <LiteralGenerator as ExpressionGenerator>::generate(ctx),
            Self::Compose => <ComposeGenerator as ExpressionGenerator>::generate(ctx),
            Self::Constant => <ConstantGenerator as ExpressionGenerator>::generate(ctx),
            Self::Splat => <SplatGenerator as ExpressionGenerator>::generate(ctx),
            Self::ZeroVal => <ZeroValGenerator as ExpressionGenerator>::generate(ctx),
        }
    }

    pub fn iter() -> impl Iterator<Item = ExpressionGenerators> {
        use ExpressionGenerators as EG;
        static VALUES: [EG; std::mem::variant_count::<EG>()] = [
            EG::Access,
            EG::AccessIndex,
            EG::ArrayLength,
            EG::Derivative,
            EG::FunctionArgument,
            EG::GlobalVar,
            EG::Load,
            EG::LocalVar,
            EG::Relational,
            EG::As,
            EG::Binary,
            EG::Select,
            EG::Math,
            EG::Unary,
            EG::Swizzle,
            EG::Literal,
            EG::Compose,
            EG::Constant,
            EG::Splat,
            EG::ZeroVal,
        ];
        VALUES.iter().cloned()
    }
}

struct ConstantGenerator;
impl ConstantGenerator {
    fn internal_generate(constants: &Arena<Constant>, rng: &mut StdRand) -> Option<Expression> {
        let (handle, _) = constants.iter().choose(rng)?;
        Some(Expression::Constant(handle))
    }
}

impl ExpressionGenerator for ConstantGenerator {
    fn generate(ctx: &mut FunctionGenCtx) -> Option<Expression> {
        Self::internal_generate(&ctx.module.constants, &mut ctx.rng)
    }
}

impl ConstExpressionGenerator for ConstantGenerator {
    fn generate(ctx: &mut GlobalGenCtx) -> Option<Expression> {
        Self::internal_generate(&ctx.module.constants, &mut ctx.rng)
    }
}

struct LiteralGenerator;
impl LiteralGenerator {
    fn generate(rng: &mut StdRand) -> Option<Expression> {
        let literal = match rng.below(4) {
            0 => Literal::F32(rng.random_f32()),
            1 => Literal::U32(rng.random_u32()),
            2 => Literal::I32(rng.random_i32()),
            3 => Literal::Bool(rng.random_bool()),
            _ => unreachable!(),
        };
        Some(Expression::Literal(literal))
    }
}

impl ExpressionGenerator for LiteralGenerator {
    fn generate(ctx: &mut FunctionGenCtx) -> Option<Expression> {
        Self::generate(&mut ctx.rng)
    }
}

impl ConstExpressionGenerator for LiteralGenerator {
    fn generate(ctx: &mut GlobalGenCtx) -> Option<Expression> {
        Self::generate(&mut ctx.rng)
    }
}

struct UnaryGenerator;
impl ExpressionGenerator for UnaryGenerator {
    fn generate(ctx: &mut FunctionGenCtx) -> Option<Expression> {
        use naga::UnaryOperator as Uo;
        use ScalarKind as S;
        use TypeInner as TI;
        let unary_ops = [Uo::BitwiseNot, Uo::LogicalNot, Uo::Negate];
        assert_eq!(std::mem::variant_count::<Uo>(), unary_ops.len());
        let op = *ctx.rng.choose(&unary_ops);

        let filter = match op {
            Uo::LogicalNot => |_, ty: &TypeInner| {
                matches!(
                    ty,
                    TI::Scalar { kind: S::Bool, .. } | TI::Vector { kind: S::Bool, .. }
                )
            },
            Uo::Negate => |_, ty: &TypeInner| {
                matches!(
                    ty,
                    TI::Scalar { kind: S::Sint, .. } | TI::Vector { kind: S::Sint, .. }
                )
            },
            Uo::BitwiseNot => |_, ty: &TypeInner| {
                matches!(
                    ty,
                    TI::Scalar {
                        kind: S::Sint | S::Uint,
                        ..
                    } | TI::Vector {
                        kind: S::Sint | S::Uint,
                        ..
                    }
                )
            },
        };
        let (expr, _) = ctx.expr_matching(filter)?;
        Some(Expression::Unary { op, expr })
    }
}

struct BinaryGenerator;
impl ExpressionGenerator for BinaryGenerator {
    fn generate(ctx: &mut FunctionGenCtx) -> Option<Expression> {
        use naga::BinaryOperator as Bi;
        use ScalarKind as S;
        use TypeInner as TI;
        let binary_ops = [
            Bi::Add,
            Bi::And,
            Bi::Divide,
            Bi::Equal,
            Bi::ExclusiveOr,
            Bi::Greater,
            Bi::GreaterEqual,
            Bi::InclusiveOr,
            Bi::Less,
            Bi::LessEqual,
            Bi::LogicalAnd,
            Bi::LogicalOr,
            Bi::Modulo,
            Bi::Multiply,
            Bi::NotEqual,
            Bi::ShiftLeft,
            Bi::ShiftRight,
            Bi::Subtract,
        ];
        assert_eq!(std::mem::variant_count::<Bi>(), binary_ops.len());

        let op = *ctx.rng.choose(&binary_ops);
        let (left, right) = match op {
            Bi::ShiftLeft | Bi::ShiftRight => {
                let filter = |_, ty: &TypeInner| {
                    matches!(
                        ty,
                        TI::Scalar {
                            kind: S::Sint | S::Uint,
                            ..
                        } | TI::Vector {
                            kind: S::Sint | S::Uint,
                            ..
                        }
                    )
                };

                let (left, ty) = ctx.expr_matching(filter)?;
                let (right, _) = match ty {
                    TI::Scalar { .. } => {
                        let filter =
                            |_, ty: &TypeInner| matches!(ty, TI::Scalar { kind: S::Uint, .. });
                        ctx.expr_matching(filter)?
                    }
                    TI::Vector {
                        size: left_size, ..
                    } => {
                        let filter = |_, ty: &TypeInner| match ty {
                            TI::Vector {
                                kind: S::Uint,
                                size: right_size,
                                ..
                            } => left_size == right_size,
                            _ => false,
                        };
                        ctx.expr_matching(filter)?
                    }
                    _ => unreachable!(),
                };
                (left, right)
            }
            Bi::Add | Bi::Subtract => {
                let filter = |_, ty: &TypeInner| {
                    matches!(
                        ty,
                        TI::Scalar {
                            kind: S::Sint | S::Uint | S::Float,
                            ..
                        } | TI::Vector {
                            kind: S::Sint | S::Uint | S::Float,
                            ..
                        } | TI::Matrix { .. }
                    )
                };
                let (left, ty) = ctx.expr_matching(filter)?;
                (left, ctx.expr_of_type(ty).unwrap())
            }
            Bi::Divide | Bi::Modulo => {
                let filter = |_, ty: &TypeInner| {
                    matches!(
                        ty,
                        TI::Scalar {
                            kind: S::Sint | S::Uint | S::Float,
                            ..
                        } | TI::Vector {
                            kind: S::Sint | S::Uint | S::Float,
                            ..
                        }
                    )
                };
                let (left, ty) = ctx.expr_matching(filter)?;
                (left, ctx.expr_of_type(ty).unwrap())
            }
            Bi::Multiply => {
                let filter = |_, ty: &TypeInner| {
                    matches!(
                        ty,
                        TI::Scalar {
                            kind: S::Sint | S::Uint | S::Float,
                            ..
                        } | TI::Vector {
                            kind: S::Sint | S::Uint | S::Float,
                            ..
                        }
                    )
                };
                let (left, ty) = ctx.expr_matching(filter)?;
                (left, ctx.expr_of_type(ty).unwrap())
            }
            other => {
                let filter = match other {
                    Bi::LogicalAnd | Bi::LogicalOr => {
                        |_, ty: &TypeInner| matches!(ty, TI::Scalar { kind: S::Bool, .. })
                    }
                    Bi::And => |_, ty: &TypeInner| {
                        matches!(
                            ty,
                            TI::Scalar { kind: S::Bool, .. } | TI::Vector { kind: S::Bool, .. }
                        )
                    },
                    Bi::Equal | Bi::NotEqual => {
                        |_, ty: &TypeInner| matches!(ty, TI::Scalar { .. } | TI::Vector { .. })
                    }
                    Bi::Less | Bi::LessEqual | Bi::Greater | Bi::GreaterEqual => {
                        |_, ty: &TypeInner| {
                            matches!(
                                ty,
                                TI::Scalar {
                                    kind: S::Sint | S::Uint | S::Float,
                                    ..
                                } | TI::Vector {
                                    kind: S::Sint | S::Uint | S::Float,
                                    ..
                                }
                            )
                        }
                    }
                    Bi::ExclusiveOr | Bi::InclusiveOr => |_, ty: &TypeInner| {
                        matches!(
                            ty,
                            TI::Scalar {
                                kind: S::Sint | S::Uint,
                                ..
                            } | TI::Vector {
                                kind: S::Sint | S::Uint,
                                ..
                            }
                        )
                    },
                    _ => unreachable!(),
                };
                let (left, ty) = ctx.expr_matching(filter)?;
                (left, ctx.expr_of_type(ty).unwrap())
            }
        };
        Some(Expression::Binary { op, left, right })
    }
}

struct ZeroValGenerator;
impl ZeroValGenerator {
    fn generate(types: &UniqueArena<Type>, rng: &mut StdRand) -> Option<Expression> {
        let info = {
            let module = Module {
                types: types.clone(),
                ..Default::default()
            };
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .ok()?
        };

        let filter = |(handle, _)| {
            let flags: TypeFlags = info[handle];
            flags.contains(TypeFlags::CONSTRUCTIBLE).then_some(handle)
        };

        let ty = types.iter().filter_map(filter).choose(rng)?;
        Some(Expression::ZeroValue(ty))
    }
}

impl ExpressionGenerator for ZeroValGenerator {
    fn generate(ctx: &mut FunctionGenCtx) -> Option<Expression> {
        Self::generate(&ctx.module.types, &mut ctx.rng)
    }
}

impl ConstExpressionGenerator for ZeroValGenerator {
    fn generate(ctx: &mut GlobalGenCtx) -> Option<Expression> {
        Self::generate(&ctx.module.types, &mut ctx.rng)
    }
}

struct SplatGenerator;
impl SplatGenerator {
    fn generate(
        scope: &ExprScope,
        types: &UniqueArena<Type>,
        rng: &mut StdRand,
    ) -> Option<Expression> {
        use naga::VectorSize as Vs;
        let sizes = [Vs::Bi, Vs::Tri, Vs::Quad];
        assert_eq!(std::mem::variant_count::<Vs>(), sizes.len());
        let size = *rng.choose(&sizes);

        let is_scalar = |_, ty: &TypeInner| matches!(ty, TypeInner::Scalar { .. });
        let (value, _) = scope.matching(is_scalar, types)?;
        Some(Expression::Splat { value, size })
    }
}

impl ExpressionGenerator for SplatGenerator {
    fn generate(ctx: &mut FunctionGenCtx) -> Option<Expression> {
        Self::generate(&ctx.expr_scope, &ctx.module.types, &mut ctx.rng)
    }
}

impl ConstExpressionGenerator for SplatGenerator {
    fn generate(ctx: &mut GlobalGenCtx) -> Option<Expression> {
        Self::generate(&ctx.global_exprs, &ctx.module.types, &mut ctx.rng)
    }
}

struct SwizzleGenerator;
impl ExpressionGenerator for SwizzleGenerator {
    fn generate(ctx: &mut FunctionGenCtx) -> Option<Expression> {
        use naga::SwizzleComponent as Sc;
        use naga::VectorSize as Vs;
        let is_vector = |_, ty: &TypeInner| matches!(ty, TypeInner::Vector { .. });
        let (vector, ty) = ctx.expr_matching(is_vector)?;
        let TypeInner::Vector { size: src_size, .. } = ty else {
            unreachable!();
        };
        let components = match src_size {
            Vs::Bi => [Sc::X, Sc::Y].as_slice(),
            Vs::Tri => [Sc::X, Sc::Y, Sc::Z].as_slice(),
            Vs::Quad => [Sc::X, Sc::Y, Sc::Z, Sc::W].as_slice(),
        };

        let pattern = [
            *ctx.rng.choose(components),
            *ctx.rng.choose(components),
            *ctx.rng.choose(components),
            *ctx.rng.choose(components),
        ];

        let sizes = [Vs::Bi, Vs::Tri, Vs::Quad];
        assert_eq!(std::mem::variant_count::<Vs>(), sizes.len());
        let size = *ctx.rng.choose(&sizes);

        Some(Expression::Swizzle {
            size,
            vector,
            pattern,
        })
    }
}

struct SelectGenerator;
impl ExpressionGenerator for SelectGenerator {
    fn generate(ctx: &mut FunctionGenCtx) -> Option<Expression> {
        let is_bool_or_vec_bool = |_, ty: &TypeInner| {
            matches!(
                ty,
                TypeInner::Scalar {
                    kind: ScalarKind::Bool,
                    ..
                } | TypeInner::Vector {
                    kind: ScalarKind::Bool,
                    ..
                }
            )
        };
        let (condition, cond_ty) = ctx.expr_matching(is_bool_or_vec_bool)?;
        let (accept, ty) = match cond_ty {
            TypeInner::Scalar {
                kind: ScalarKind::Bool,
                ..
            } => {
                let is_scalar_or_vec = |_, ty: &TypeInner| {
                    matches!(ty, TypeInner::Scalar { .. } | TypeInner::Vector { .. })
                };
                ctx.expr_matching(is_scalar_or_vec).unwrap()
            }
            TypeInner::Vector {
                kind: ScalarKind::Bool,
                size,
                ..
            } => {
                let filter = |_, ty: &TypeInner| match ty {
                    TypeInner::Vector {
                        size: inner_size, ..
                    } => inner_size == size,
                    _ => false,
                };
                ctx.expr_matching(filter).unwrap()
            }
            _ => unreachable!(),
        };
        let reject = ctx.expr_of_type(ty).unwrap();
        Some(Expression::Select {
            condition,
            accept,
            reject,
        })
    }
}

struct MathGenerator;
impl ExpressionGenerator for MathGenerator {
    fn generate(ctx: &mut FunctionGenCtx) -> Option<Expression> {
        use naga::MathFunction as Mf;
        use ScalarKind as S;
        use TypeInner as TI;
        let mathfuncs = &[
            Mf::Abs,
            Mf::Min,
            Mf::Max,
            Mf::Clamp,
            Mf::Saturate,
            Mf::Cos,
            Mf::Cosh,
            Mf::Sin,
            Mf::Sinh,
            Mf::Tan,
            Mf::Tanh,
            Mf::Acos,
            Mf::Asin,
            Mf::Atan,
            Mf::Atan2,
            Mf::Asinh,
            Mf::Acosh,
            Mf::Atanh,
            Mf::Radians,
            Mf::Degrees,
            Mf::Ceil,
            Mf::Floor,
            Mf::Round,
            Mf::Fract,
            Mf::Trunc,
            Mf::Modf,
            Mf::Frexp,
            Mf::Ldexp,
            Mf::Exp,
            Mf::Exp2,
            Mf::Log,
            Mf::Log2,
            Mf::Pow,
            Mf::Dot,
            Mf::Cross,
            Mf::Distance,
            Mf::Length,
            Mf::Normalize,
            Mf::FaceForward,
            Mf::Reflect,
            Mf::Refract,
            Mf::Sign,
            Mf::Fma,
            Mf::Mix,
            Mf::Step,
            Mf::SmoothStep,
            Mf::Sqrt,
            Mf::InverseSqrt,
            Mf::Transpose,
            Mf::Determinant,
            Mf::CountTrailingZeros,
            Mf::CountLeadingZeros,
            Mf::CountOneBits,
            Mf::ReverseBits,
            Mf::ExtractBits,
            Mf::InsertBits,
            Mf::FindLsb,
            Mf::FindMsb,
            Mf::Pack4x8snorm,
            Mf::Pack4x8unorm,
            Mf::Pack2x16snorm,
            Mf::Pack2x16unorm,
            Mf::Pack2x16float,
            Mf::Unpack4x8snorm,
            Mf::Unpack4x8unorm,
            Mf::Unpack2x16snorm,
            Mf::Unpack2x16unorm,
            Mf::Unpack2x16float,
        ];
        assert_eq!(std::mem::variant_count::<Mf>(), mathfuncs.len() + 2);
        let fun = *ctx.rng.choose(mathfuncs);

        let expr = match fun {
            Mf::Abs => {
                let filter = |_, ty: &TypeInner| {
                    matches!(
                        ty,
                        TI::Scalar {
                            kind: S::Sint | S::Uint | S::Float,
                            ..
                        } | TI::Vector {
                            kind: S::Sint | S::Uint | S::Float,
                            ..
                        }
                    )
                };
                let (arg, _) = ctx.expr_matching(filter)?;
                Expression::Math {
                    fun,
                    arg,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                }
            }
            Mf::Acos
            | Mf::Acosh
            | Mf::Asin
            | Mf::Asinh
            | Mf::Atan
            | Mf::Atanh
            | Mf::Ceil
            | Mf::Cos
            | Mf::Cosh
            | Mf::Degrees
            | Mf::Exp
            | Mf::Exp2
            | Mf::Floor
            | Mf::Fract
            | Mf::InverseSqrt
            | Mf::Length
            | Mf::Log
            | Mf::Log2
            | Mf::Radians
            | Mf::Round
            | Mf::Saturate
            | Mf::Sin
            | Mf::Sinh
            | Mf::Sqrt
            | Mf::Tan
            | Mf::Tanh
            | Mf::Trunc => {
                let filter = |_, ty: &TypeInner| {
                    matches!(
                        ty,
                        TI::Scalar { kind: S::Float, .. } | TI::Vector { kind: S::Float, .. }
                    )
                };
                let (arg, _) = ctx.expr_matching(filter)?;
                Expression::Math {
                    fun,
                    arg,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                }
            }
            Mf::Atan2 => {
                let filter = |_, ty: &TypeInner| {
                    matches!(
                        ty,
                        TI::Scalar { kind: S::Float, .. } | TI::Vector { kind: S::Float, .. }
                    )
                };
                let (arg, ty) = ctx.expr_matching(filter)?;
                let arg1 = Some(ctx.expr_of_type(ty).unwrap());
                Expression::Math {
                    fun,
                    arg,
                    arg1,
                    arg2: None,
                    arg3: None,
                }
            }
            Mf::Clamp => {
                let filter = |_, ty: &TypeInner| {
                    matches!(
                        ty,
                        TI::Scalar {
                            kind: S::Float | S::Sint | S::Uint,
                            ..
                        } | TI::Vector {
                            kind: S::Float | S::Sint | S::Uint,
                            ..
                        }
                    )
                };
                let (arg, ty) = ctx.expr_matching(filter)?;
                let arg1 = Some(ctx.expr_of_type(ty).unwrap());
                let arg2 = Some(ctx.expr_of_type(ty).unwrap());
                Expression::Math {
                    fun,
                    arg,
                    arg1,
                    arg2,
                    arg3: None,
                }
            }
            Mf::CountLeadingZeros | Mf::CountOneBits | Mf::CountTrailingZeros => {
                let filter = |_, ty: &TypeInner| {
                    matches!(
                        ty,
                        TI::Scalar {
                            kind: S::Sint | S::Uint,
                            ..
                        } | TI::Vector {
                            kind: S::Sint | S::Uint,
                            ..
                        }
                    )
                };
                let (arg, _) = ctx.expr_matching(filter)?;
                Expression::Math {
                    fun,
                    arg,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                }
            }
            Mf::Cross => {
                let filter = |_, ty: &TypeInner| {
                    matches!(
                        ty,
                        TI::Vector {
                            size: VectorSize::Tri,
                            kind: S::Float,
                            ..
                        }
                    )
                };
                let (arg, ty) = ctx.expr_matching(filter)?;
                let arg1 = Some(ctx.expr_of_type(ty).unwrap());
                Expression::Math {
                    fun,
                    arg,
                    arg1,
                    arg2: None,
                    arg3: None,
                }
            }
            Mf::Determinant => {
                let filter = |_, ty: &TypeInner| match ty {
                    TI::Matrix { columns, rows, .. } => columns == rows,
                    _ => false,
                };
                let (arg, _) = ctx.expr_matching(filter)?;
                Expression::Math {
                    fun,
                    arg,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                }
            }
            Mf::Distance | Mf::Pow | Mf::Step => {
                let filter = |_, ty: &TypeInner| {
                    matches!(
                        ty,
                        TI::Scalar { kind: S::Float, .. } | TI::Vector { kind: S::Float, .. }
                    )
                };
                let (arg, ty) = ctx.expr_matching(filter)?;
                let arg1 = Some(ctx.expr_of_type(ty).unwrap());
                Expression::Math {
                    fun,
                    arg,
                    arg1,
                    arg2: None,
                    arg3: None,
                }
            }
            Mf::Dot => {
                let filter = |_, ty: &TypeInner| {
                    matches!(
                        ty,
                        TI::Vector {
                            kind: S::Float | S::Sint | S::Uint,
                            ..
                        }
                    )
                };
                let (arg, ty) = ctx.expr_matching(filter)?;
                let arg1 = Some(ctx.expr_of_type(ty).unwrap());
                Expression::Math {
                    fun,
                    arg,
                    arg1,
                    arg2: None,
                    arg3: None,
                }
            }
            Mf::FaceForward => {
                let filter = |_, ty: &TypeInner| matches!(ty, TI::Vector { kind: S::Float, .. });
                let (arg, ty) = ctx.expr_matching(filter)?;
                let arg1 = Some(ctx.expr_of_type(ty).unwrap());
                let arg2 = Some(ctx.expr_of_type(ty).unwrap());
                Expression::Math {
                    fun,
                    arg,
                    arg1,
                    arg2,
                    arg3: None,
                }
            }
            Mf::FindLsb | Mf::FindMsb | Mf::ReverseBits => {
                let filter = |_, ty: &TypeInner| {
                    matches!(
                        ty,
                        TI::Scalar {
                            kind: S::Sint | S::Uint,
                            ..
                        } | TI::Vector {
                            kind: S::Sint | S::Uint,
                            ..
                        }
                    )
                };
                let (arg, _) = ctx.expr_matching(filter)?;
                Expression::Math {
                    fun,
                    arg,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                }
            }
            Mf::Fma | Mf::SmoothStep => {
                let filter = |_, ty: &TypeInner| {
                    matches!(
                        ty,
                        TI::Scalar { kind: S::Float, .. } | TI::Vector { kind: S::Float, .. }
                    )
                };
                let (arg, ty) = ctx.expr_matching(filter)?;
                let arg1 = Some(ctx.expr_of_type(ty).unwrap());
                let arg2 = Some(ctx.expr_of_type(ty).unwrap());
                Expression::Math {
                    fun,
                    arg,
                    arg1,
                    arg2,
                    arg3: None,
                }
            }
            Mf::Max | Mf::Min => {
                let filter = |_, ty: &TypeInner| {
                    matches!(
                        ty,
                        TI::Scalar {
                            kind: S::Float | S::Sint | S::Uint,
                            ..
                        } | TI::Vector {
                            kind: S::Float | S::Sint | S::Uint,
                            ..
                        }
                    )
                };
                let (arg, ty) = ctx.expr_matching(filter)?;
                let arg1 = Some(ctx.expr_of_type(ty).unwrap());
                Expression::Math {
                    fun,
                    arg,
                    arg1,
                    arg2: None,
                    arg3: None,
                }
            }
            Mf::Mix => {
                if ctx.rng.probability(0.5) {
                    let filter = |_, ty: &TypeInner| {
                        matches!(
                            ty,
                            TI::Scalar { kind: S::Float, .. } | TI::Vector { kind: S::Float, .. }
                        )
                    };
                    let (arg, ty) = ctx.expr_matching(filter)?;
                    let arg1 = Some(ctx.expr_of_type(ty).unwrap());
                    let arg2 = Some(ctx.expr_of_type(ty).unwrap());
                    Expression::Math {
                        fun,
                        arg,
                        arg1,
                        arg2,
                        arg3: None,
                    }
                } else {
                    let filter =
                        |_, ty: &TypeInner| matches!(ty, TI::Scalar { kind: S::Float, .. });
                    let (arg2, ty) = ctx.expr_matching(filter)?;
                    let TI::Scalar { kind, .. } = ty else {
                        unreachable!()
                    };

                    let filter = |_, ty: &TypeInner| match ty {
                        TI::Vector { kind: vec_kind, .. } => vec_kind == kind,
                        _ => false,
                    };
                    let (arg, ty) = ctx.expr_matching(filter)?;
                    let arg1 = ctx.expr_of_type(ty).unwrap();

                    Expression::Math {
                        fun,
                        arg,
                        arg1: Some(arg1),
                        arg2: Some(arg2),
                        arg3: None,
                    }
                }
            }
            Mf::Normalize => {
                let filter = |_, ty: &TypeInner| matches!(ty, TI::Vector { kind: S::Float, .. });
                let (arg, _) = ctx.expr_matching(filter)?;
                Expression::Math {
                    fun,
                    arg,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                }
            }
            Mf::Sign => {
                let filter = |_, ty: &TypeInner| {
                    matches!(
                        ty,
                        TI::Vector {
                            kind: S::Float | S::Sint,
                            ..
                        }
                    )
                };
                let (arg, _) = ctx.expr_matching(filter)?;
                Expression::Math {
                    fun,
                    arg,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                }
            }
            Mf::Pack4x8snorm | Mf::Pack4x8unorm => {
                let filter = |_, ty: &TypeInner| {
                    matches!(
                        ty,
                        TI::Vector {
                            kind: S::Float,
                            size: VectorSize::Quad,
                            width: 4
                        }
                    )
                };
                let (arg, _) = ctx.expr_matching(filter)?;
                Expression::Math {
                    fun,
                    arg,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                }
            }
            Mf::Pack2x16float | Mf::Pack2x16snorm | Mf::Pack2x16unorm => {
                let filter = |_, ty: &TypeInner| {
                    matches!(
                        ty,
                        TI::Vector {
                            kind: S::Float,
                            size: VectorSize::Bi,
                            width: 4
                        }
                    )
                };
                let (arg, _) = ctx.expr_matching(filter)?;
                Expression::Math {
                    fun,
                    arg,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                }
            }
            Mf::Unpack4x8snorm
            | Mf::Unpack4x8unorm
            | Mf::Unpack2x16float
            | Mf::Unpack2x16snorm
            | Mf::Unpack2x16unorm => {
                let filter = |_, ty: &TypeInner| {
                    matches!(
                        ty,
                        TI::Scalar {
                            kind: S::Uint,
                            width: 4
                        }
                    )
                };
                let (arg, _) = ctx.expr_matching(filter)?;
                Expression::Math {
                    fun,
                    arg,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                }
            }
            Mf::Reflect => {
                let filter = |_, ty: &TypeInner| matches!(ty, TI::Vector { kind: S::Float, .. });
                let (arg, ty) = ctx.expr_matching(filter)?;
                let arg1 = Some(ctx.expr_of_type(ty).unwrap());
                Expression::Math {
                    fun,
                    arg,
                    arg1,
                    arg2: None,
                    arg3: None,
                }
            }
            Mf::Refract => {
                let filter = |_, ty: &TypeInner| matches!(ty, TI::Vector { kind: S::Float, .. });
                let (arg, ty) = ctx.expr_matching(filter)?;
                let arg1 = ctx.expr_of_type(ty).unwrap();

                let TI::Vector { kind, width, .. } = ty else {
                    unreachable!()
                };
                let arg2 = ctx.expr_of_type(&TypeInner::Scalar {
                    kind: *kind,
                    width: *width,
                })?;
                Expression::Math {
                    fun,
                    arg,
                    arg1: Some(arg1),
                    arg2: Some(arg2),
                    arg3: None,
                }
            }
            Mf::Frexp | Mf::Modf => {
                use naga::PredeclaredType;

                let filter = |_, ty: &TypeInner| {
                    matches!(
                        ty,
                        TI::Scalar { kind: S::Float, .. } | TI::Vector { kind: S::Float, .. }
                    )
                };
                let (arg, ty) = ctx.expr_matching(filter)?;
                let (size, width) = match ty {
                    TI::Scalar { width, .. } => (None, *width),
                    TI::Vector { size, width, .. } => (Some(*size), *width),
                    _ => unreachable!(),
                };
                ctx.module
                    .generate_predeclared_type(PredeclaredType::FrexpResult { size, width });
                ctx.module
                    .generate_predeclared_type(PredeclaredType::ModfResult { size, width });

                Expression::Math {
                    fun,
                    arg,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                }
            }
            Mf::Transpose => {
                let filter = |_, ty: &TypeInner| matches!(ty, TI::Matrix { .. });
                let (arg, _) = ctx.expr_matching(filter)?;
                Expression::Math {
                    fun,
                    arg,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                }
            }
            Mf::ExtractBits => {
                return None;
            }
            Mf::InsertBits => {
                return None;
            }
            Mf::Ldexp => {
                return None;
            }
            Mf::Inverse | Mf::Outer => unreachable!(),
        };
        Some(expr)
    }
}

struct AsGenerator;
impl ExpressionGenerator for AsGenerator {
    fn generate(ctx: &mut FunctionGenCtx) -> Option<Expression> {
        use ScalarKind as S;
        let filter = |_, ty: &TypeInner| {
            matches!(
                ty,
                TypeInner::Scalar {
                    kind: ScalarKind::Float | ScalarKind::Sint | ScalarKind::Uint,
                    ..
                } | TypeInner::Vector {
                    kind: ScalarKind::Float | ScalarKind::Sint | ScalarKind::Uint,
                    ..
                }
            )
        };
        let (expr, ty) = ctx.expr_matching(filter)?;
        let _base_width = match ty {
            TypeInner::Scalar { width, .. } | TypeInner::Vector { width, .. } => *width,
            _ => unreachable!(),
        };
        let kind = ctx.rng.choose([S::Float, S::Sint, S::Uint]);
        let kind_width = match kind {
            S::Sint | S::Uint => 4u8,
            S::Float => 4,
            S::Bool => unreachable!(),
        };
        Some(Expression::As {
            expr,
            kind,
            convert: Some(kind_width),
        })
    }
}

struct ComposeGenerator;
impl ComposeGenerator {
    fn generate(
        scope: &ExprScope,
        types: &UniqueArena<Type>,
        rng: &mut StdRand,
    ) -> Option<Expression> {
        use TypeInner as TI;
        let filter = |(_, ty): &(_, &Type)| {
            matches!(
                ty.inner,
                TI::Vector { .. }
                    | TI::Matrix { .. }
                    | TI::Struct { .. }
                    | TI::Array {
                        size: ArraySize::Constant(..),
                        ..
                    }
            )
        };
        const SIZE_LIMIT: u32 = 2048;
        let (handle, ty) = types.iter().filter(filter).choose(rng)?;
        let components: Option<Vec<Handle<Expression>>> = match &ty.inner {
            TI::Vector { size, kind, width } => {
                let mut remaining = *size as u32;
                if remaining > SIZE_LIMIT {
                    return None;
                }
                let filter = |ty: &TypeInner, remaining: u32| match *ty {
                    TI::Scalar {
                        kind: comp_kind,
                        width: comp_width,
                    } => comp_kind == *kind && comp_width == *width,
                    TI::Vector {
                        kind: comp_kind,
                        width: comp_width,
                        size: comp_size,
                    } => {
                        comp_kind == *kind && comp_width == *width && comp_size as u32 <= remaining
                    }
                    _ => false,
                };
                let mut components = Vec::new();
                while remaining > 0 {
                    let (expr, ty) = scope.matching(|_, ty| filter(ty, remaining), types)?;
                    let cur_size = match ty {
                        TI::Scalar { .. } => 1,
                        TI::Vector { size, .. } => *size as u32,
                        _ => unreachable!(),
                    };
                    remaining -= cur_size;
                    components.push(expr);
                }
                Some(components)
            }
            TI::Matrix {
                columns,
                rows,
                width,
            } => {
                let inner = TI::Vector {
                    size: *rows,
                    kind: ScalarKind::Float,
                    width: *width,
                };
                let columns = *columns as u32;
                if columns > SIZE_LIMIT {
                    return None;
                }
                (0..columns).map(|_| scope.of_type(&inner, types)).collect()
            }
            TI::Array {
                base,
                size: ArraySize::Constant(count),
                ..
            } => {
                let base = &types[*base].inner;
                let count = count.get();
                if count > SIZE_LIMIT {
                    return None;
                }
                (0..count).map(|_| scope.of_type(base, types)).collect()
            }
            TI::Struct { members, .. } => members
                .iter()
                .map(|m| {
                    let inner = &types[m.ty].inner;
                    scope.of_type(inner, types)
                })
                .collect(),
            _ => unreachable!(),
        };
        let components = components?;
        Some(Expression::Compose {
            ty: handle,
            components,
        })
    }
}

impl ExpressionGenerator for ComposeGenerator {
    fn generate(ctx: &mut FunctionGenCtx) -> Option<Expression> {
        Self::generate(&ctx.expr_scope, &ctx.module.types, &mut ctx.rng)
    }
}

impl ConstExpressionGenerator for ComposeGenerator {
    fn generate(ctx: &mut GlobalGenCtx) -> Option<Expression> {
        Self::generate(&ctx.global_exprs, &ctx.module.types, &mut ctx.rng)
    }
}

struct AccessIndexGenerator;
impl ExpressionGenerator for AccessIndexGenerator {
    fn generate(ctx: &mut FunctionGenCtx) -> Option<Expression> {
        use TypeInner as TI;
        fn resolve_index_limit(
            module: &Module,
            ty: &TypeInner,
            top_level: bool,
        ) -> Result<u32, ()> {
            let limit = match *ty {
                TI::Vector { size, .. }
                | TI::ValuePointer {
                    size: Some(size), ..
                } => size as u32,
                TI::Matrix { columns, .. } => columns as u32,
                TI::Array {
                    size: ArraySize::Constant(len),
                    ..
                } => len.get(),
                TI::Array { .. } | TI::BindingArray { .. } => u32::MAX,
                TI::Pointer { base, .. } if top_level => {
                    resolve_index_limit(module, &module.types[base].inner, false)?
                }
                TI::Struct { ref members, .. } => members.len() as u32,
                _ => return Err(()),
            };
            Ok(limit)
        }

        let filter = |_, ty: &TypeInner| resolve_index_limit(ctx.module, ty, true).is_ok();

        let (base, ty) = ctx.expr_matching(filter)?;
        let limit = resolve_index_limit(ctx.module, ty, true).unwrap();
        if limit == 0 {
            None
        } else {
            let index = ctx.rng.below(limit as u64) as u32;
            Some(Expression::AccessIndex { base, index })
        }
    }
}

struct AccessGenerator;
impl ExpressionGenerator for AccessGenerator {
    fn generate(ctx: &mut FunctionGenCtx) -> Option<Expression> {
        use TypeInner as TI;
        let filter = |_, ty: &TypeInner| ty.indexable_length(ctx.module).is_ok();

        let (base, ty) = ctx.expr_matching(filter)?;
        let dynamic_indexing_restricted = match ty {
            TI::Matrix { .. } | TI::Array { .. } => true,
            TI::Vector { .. }
            | TI::Pointer { .. }
            | TI::ValuePointer { size: Some(_), .. }
            | TI::BindingArray { .. } => false,
            _ => unreachable!(),
        };

        let (index, _) = match ty.indexable_length(ctx.module).unwrap() {
            naga::proc::IndexableLength::Known(lenght) => {
                let filter = |handle: Handle<Expression>, ty: &TypeInner| {
                    if !matches!(
                        ty,
                        TI::Scalar {
                            kind: ScalarKind::Sint | ScalarKind::Uint,
                            ..
                        }
                    ) {
                        return false;
                    }
                    let expr = &ctx.get_function().expressions[handle];
                    if let Expression::Literal(Literal::I32(l)) = expr {
                        if *l < 0 || *l as u32 >= lenght {
                            return false;
                        }
                    }
                    if let Expression::Literal(Literal::U32(l)) = expr {
                        if *l >= lenght {
                            return false;
                        }
                    }
                    if dynamic_indexing_restricted && expr.is_dynamic_index(ctx.module) {
                        return false;
                    }
                    true
                };
                ctx.expr_matching(filter)?
            }
            naga::proc::IndexableLength::Dynamic => {
                let filter = |handle: Handle<Expression>, ty: &TypeInner| {
                    if !matches!(
                        ty,
                        TI::Scalar {
                            kind: ScalarKind::Sint | ScalarKind::Uint,
                            ..
                        }
                    ) {
                        return false;
                    }
                    if dynamic_indexing_restricted
                        && ctx.get_function().expressions[handle].is_dynamic_index(ctx.module)
                    {
                        return false;
                    }
                    true
                };
                ctx.expr_matching(filter)?
            }
        };
        Some(Expression::Access { base, index })
    }
}

struct ArrayLengthGenerator;
impl ExpressionGenerator for ArrayLengthGenerator {
    fn generate(ctx: &mut FunctionGenCtx) -> Option<Expression> {
        let dyn_array = |_, ty: &TypeInner| match ty {
            TypeInner::Pointer { base, .. } => {
                let inner = &ctx.module.types[*base].inner;
                matches!(
                    inner,
                    TypeInner::Array {
                        size: ArraySize::Dynamic,
                        ..
                    }
                )
            }
            _ => false,
        };
        let (dyn_array, _) = ctx.expr_matching(dyn_array)?;
        Some(Expression::ArrayLength(dyn_array))
    }
}

struct DerivativeGenerator;
impl ExpressionGenerator for DerivativeGenerator {
    fn generate(ctx: &mut FunctionGenCtx) -> Option<Expression> {
        use naga::DerivativeAxis as RA;
        use naga::DerivativeControl as RC;
        let filter = |_, ty: &TypeInner| {
            matches!(
                ty,
                TypeInner::Scalar {
                    kind: ScalarKind::Float,
                    ..
                } | TypeInner::Vector {
                    kind: ScalarKind::Float,
                    ..
                }
            )
        };
        let (expr, _) = ctx.expr_matching(filter)?;

        let axes = [RA::X, RA::Y, RA::Width];
        assert_eq!(std::mem::variant_count::<RA>(), axes.len());
        let axis = *ctx.rng.choose(&axes);

        let controls = [RC::None, RC::Fine, RC::Coarse];
        assert_eq!(std::mem::variant_count::<RC>(), controls.len());
        let ctrl = *ctx.rng.choose(&controls);

        Some(Expression::Derivative { axis, ctrl, expr })
    }
}

struct FunctionArgumentGenerator;
impl ExpressionGenerator for FunctionArgumentGenerator {
    fn generate(ctx: &mut FunctionGenCtx) -> Option<Expression> {
        if ctx.get_function().arguments.is_empty() {
            return None;
        }
        let arg = ctx.rng.below(ctx.get_function().arguments.len() as u64);
        Some(Expression::FunctionArgument(arg as u32))
    }
}

struct GlobalVarGenerator;
impl ExpressionGenerator for GlobalVarGenerator {
    fn generate(ctx: &mut FunctionGenCtx) -> Option<Expression> {
        let filter_map = |(handle, gvar): (_, &GlobalVariable)| {
            if ctx
                .for_shader_stages
                .intersects(ShaderStages::FRAGMENT | ShaderStages::VERTEX)
                && matches!(gvar.space, AddressSpace::WorkGroup)
            {
                None
            } else {
                Some(Expression::GlobalVariable(handle))
            }
        };

        ctx.module
            .global_variables
            .iter()
            .filter_map(filter_map)
            .choose(&mut ctx.rng)
    }
}

struct LoadGenerator;
impl ExpressionGenerator for LoadGenerator {
    fn generate(ctx: &mut FunctionGenCtx) -> Option<Expression> {
        let filter = |_, ty: &TypeInner| {
            matches!(
                ty,
                TypeInner::Pointer { .. } | TypeInner::ValuePointer { .. }
            )
        };

        let (pointer, _) = ctx.expr_matching(filter)?;
        Some(Expression::Load { pointer })
    }
}

struct LocalVarGenerator;
impl ExpressionGenerator for LocalVarGenerator {
    fn generate(ctx: &mut FunctionGenCtx) -> Option<Expression> {
        let mut rng = ctx.rng;
        ctx.get_function()
            .local_variables
            .iter()
            .choose(&mut rng)
            .map(|(local_var, _)| Expression::LocalVariable(local_var))
    }
}

struct RelationalGenerator;
impl ExpressionGenerator for RelationalGenerator {
    fn generate(ctx: &mut FunctionGenCtx) -> Option<Expression> {
        use naga::RelationalFunction as Rf;
        let rfs = [Rf::All, Rf::Any];
        let fun = *ctx.rng.choose(&rfs);
        let argument = match fun {
            Rf::All | Rf::Any => {
                let is_vec_of_bools = |_, ty: &TypeInner| {
                    matches!(
                        ty,
                        TypeInner::Vector {
                            kind: ScalarKind::Bool,
                            ..
                        }
                    )
                };
                let (expr, _) = ctx.expr_matching(is_vec_of_bools)?;
                expr
            }
            _ => unreachable!(),
        };
        Some(Expression::Relational { fun, argument })
    }
}
