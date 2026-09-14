//! The `naga` enums that darthshader enumerates exhaustively, re-exported
//! together with the list of their variants and how many there are.
//!
//! `std::mem::variant_count` would give the counts in one line, but it is a
//! nightly-only intrinsic (rust-lang/rust#73662) and darthshader builds on
//! stable, so the variants are listed by hand in [`NagaEnum::VALUES`] and the
//! count is derived from that list.
//!
//! A hand-written list would rot silently, so every impl is paired with an
//! exhaustive `match` over the enum. naga gaining, losing or renaming a
//! variant is then a build error pointing straight at the list to update.
//!
//! Prefer [`NagaEnum::VALUES`] over writing out a literal array. Where only a
//! subset is wanted, size the array with [`NagaEnum::VARIANT_COUNT`] so that
//! the subset also stops compiling when naga changes, rather than tripping an
//! assertion at run time:
//!
//! ```text
//! // Every math function except `Inverse`.
//! const MATHFUNCS: [MathFunction; MathFunction::VARIANT_COUNT - 1] = [..];
//! ```

pub(crate) use naga::{
    AtomicFunction, BinaryOperator, DerivativeAxis, DerivativeControl, MathFunction, ShaderStage,
    UnaryOperator, VectorSize,
};

/// A `naga` enum whose variants darthshader enumerates.
pub(crate) trait NagaEnum: Sized + 'static {
    /// The variants, in naga's declaration order.
    ///
    /// Every impl but [`AtomicFunction`] lists all of them; see there for why
    /// it is the exception.
    const VALUES: &'static [Self];

    /// How many variants the enum has.
    ///
    /// Spelled out rather than derived from [`Self::VALUES`], so that dropping
    /// a variant from that list is caught by the assertion below each impl
    /// instead of silently shrinking the count along with it.
    const VARIANT_COUNT: usize;
}

impl NagaEnum for AtomicFunction {
    /// Every variant except [`AtomicFunction::Exchange`], which carries a
    /// payload and so has no single value that could stand for the variant.
    const VALUES: &'static [Self] = &[
        Self::Add,
        Self::Subtract,
        Self::And,
        Self::ExclusiveOr,
        Self::InclusiveOr,
        Self::Min,
        Self::Max,
    ];

    const VARIANT_COUNT: usize = 8;
}

const _: () = {
    // Catches a variant accidentally dropped from `VALUES`.
    assert!(AtomicFunction::VALUES.len() + 1 == AtomicFunction::VARIANT_COUNT);

    /// Fails to compile if [`AtomicFunction`] gains, loses or renames a variant, which
    /// is the signal to update the impl above.
    #[allow(dead_code)]
    fn assert_exhaustive(value: AtomicFunction) {
        use AtomicFunction::*;
        match value {
            Add => {}
            Subtract => {}
            And => {}
            ExclusiveOr => {}
            InclusiveOr => {}
            Min => {}
            Max => {}
            Exchange { .. } => {}
        }
    }
};

impl NagaEnum for BinaryOperator {
    /// Every variant.
    const VALUES: &'static [Self] = &[
        Self::Add,
        Self::Subtract,
        Self::Multiply,
        Self::Divide,
        Self::Modulo,
        Self::Equal,
        Self::NotEqual,
        Self::Less,
        Self::LessEqual,
        Self::Greater,
        Self::GreaterEqual,
        Self::And,
        Self::ExclusiveOr,
        Self::InclusiveOr,
        Self::LogicalAnd,
        Self::LogicalOr,
        Self::ShiftLeft,
        Self::ShiftRight,
    ];

    const VARIANT_COUNT: usize = 18;
}

const _: () = {
    // Catches a variant accidentally dropped from `VALUES`.
    assert!(BinaryOperator::VALUES.len() == BinaryOperator::VARIANT_COUNT);

    /// Fails to compile if [`BinaryOperator`] gains, loses or renames a variant, which
    /// is the signal to update the impl above.
    #[allow(dead_code)]
    fn assert_exhaustive(value: BinaryOperator) {
        use BinaryOperator::*;
        match value {
            Add => {}
            Subtract => {}
            Multiply => {}
            Divide => {}
            Modulo => {}
            Equal => {}
            NotEqual => {}
            Less => {}
            LessEqual => {}
            Greater => {}
            GreaterEqual => {}
            And => {}
            ExclusiveOr => {}
            InclusiveOr => {}
            LogicalAnd => {}
            LogicalOr => {}
            ShiftLeft => {}
            ShiftRight => {}
        }
    }
};

impl NagaEnum for DerivativeAxis {
    /// Every variant.
    const VALUES: &'static [Self] = &[Self::X, Self::Y, Self::Width];

    const VARIANT_COUNT: usize = 3;
}

const _: () = {
    // Catches a variant accidentally dropped from `VALUES`.
    assert!(DerivativeAxis::VALUES.len() == DerivativeAxis::VARIANT_COUNT);

    /// Fails to compile if [`DerivativeAxis`] gains, loses or renames a variant, which
    /// is the signal to update the impl above.
    #[allow(dead_code)]
    fn assert_exhaustive(value: DerivativeAxis) {
        use DerivativeAxis::*;
        match value {
            X => {}
            Y => {}
            Width => {}
        }
    }
};

impl NagaEnum for DerivativeControl {
    /// Every variant.
    const VALUES: &'static [Self] = &[Self::Coarse, Self::Fine, Self::None];

    const VARIANT_COUNT: usize = 3;
}

const _: () = {
    // Catches a variant accidentally dropped from `VALUES`.
    assert!(DerivativeControl::VALUES.len() == DerivativeControl::VARIANT_COUNT);

    /// Fails to compile if [`DerivativeControl`] gains, loses or renames a variant, which
    /// is the signal to update the impl above.
    #[allow(dead_code)]
    fn assert_exhaustive(value: DerivativeControl) {
        use DerivativeControl::*;
        match value {
            Coarse => {}
            Fine => {}
            None => {}
        }
    }
};

impl NagaEnum for MathFunction {
    /// Every variant.
    const VALUES: &'static [Self] = &[
        Self::Abs,
        Self::Min,
        Self::Max,
        Self::Clamp,
        Self::Saturate,
        Self::Cos,
        Self::Cosh,
        Self::Sin,
        Self::Sinh,
        Self::Tan,
        Self::Tanh,
        Self::Acos,
        Self::Asin,
        Self::Atan,
        Self::Atan2,
        Self::Asinh,
        Self::Acosh,
        Self::Atanh,
        Self::Radians,
        Self::Degrees,
        Self::Ceil,
        Self::Floor,
        Self::Round,
        Self::Fract,
        Self::Trunc,
        Self::Modf,
        Self::Frexp,
        Self::Ldexp,
        Self::Exp,
        Self::Exp2,
        Self::Log,
        Self::Log2,
        Self::Pow,
        Self::Dot,
        Self::Outer,
        Self::Cross,
        Self::Distance,
        Self::Length,
        Self::Normalize,
        Self::FaceForward,
        Self::Reflect,
        Self::Refract,
        Self::Sign,
        Self::Fma,
        Self::Mix,
        Self::Step,
        Self::SmoothStep,
        Self::Sqrt,
        Self::InverseSqrt,
        Self::Inverse,
        Self::Transpose,
        Self::Determinant,
        Self::CountTrailingZeros,
        Self::CountLeadingZeros,
        Self::CountOneBits,
        Self::ReverseBits,
        Self::ExtractBits,
        Self::InsertBits,
        Self::FindLsb,
        Self::FindMsb,
        Self::Pack4x8snorm,
        Self::Pack4x8unorm,
        Self::Pack2x16snorm,
        Self::Pack2x16unorm,
        Self::Pack2x16float,
        Self::Unpack4x8snorm,
        Self::Unpack4x8unorm,
        Self::Unpack2x16snorm,
        Self::Unpack2x16unorm,
        Self::Unpack2x16float,
    ];

    const VARIANT_COUNT: usize = 70;
}

const _: () = {
    // Catches a variant accidentally dropped from `VALUES`.
    assert!(MathFunction::VALUES.len() == MathFunction::VARIANT_COUNT);

    /// Fails to compile if [`MathFunction`] gains, loses or renames a variant, which
    /// is the signal to update the impl above.
    #[allow(dead_code)]
    fn assert_exhaustive(value: MathFunction) {
        use MathFunction::*;
        match value {
            Abs => {}
            Min => {}
            Max => {}
            Clamp => {}
            Saturate => {}
            Cos => {}
            Cosh => {}
            Sin => {}
            Sinh => {}
            Tan => {}
            Tanh => {}
            Acos => {}
            Asin => {}
            Atan => {}
            Atan2 => {}
            Asinh => {}
            Acosh => {}
            Atanh => {}
            Radians => {}
            Degrees => {}
            Ceil => {}
            Floor => {}
            Round => {}
            Fract => {}
            Trunc => {}
            Modf => {}
            Frexp => {}
            Ldexp => {}
            Exp => {}
            Exp2 => {}
            Log => {}
            Log2 => {}
            Pow => {}
            Dot => {}
            Outer => {}
            Cross => {}
            Distance => {}
            Length => {}
            Normalize => {}
            FaceForward => {}
            Reflect => {}
            Refract => {}
            Sign => {}
            Fma => {}
            Mix => {}
            Step => {}
            SmoothStep => {}
            Sqrt => {}
            InverseSqrt => {}
            Inverse => {}
            Transpose => {}
            Determinant => {}
            CountTrailingZeros => {}
            CountLeadingZeros => {}
            CountOneBits => {}
            ReverseBits => {}
            ExtractBits => {}
            InsertBits => {}
            FindLsb => {}
            FindMsb => {}
            Pack4x8snorm => {}
            Pack4x8unorm => {}
            Pack2x16snorm => {}
            Pack2x16unorm => {}
            Pack2x16float => {}
            Unpack4x8snorm => {}
            Unpack4x8unorm => {}
            Unpack2x16snorm => {}
            Unpack2x16unorm => {}
            Unpack2x16float => {}
        }
    }
};

impl NagaEnum for ShaderStage {
    /// Every variant.
    const VALUES: &'static [Self] = &[Self::Vertex, Self::Fragment, Self::Compute];

    const VARIANT_COUNT: usize = 3;
}

const _: () = {
    // Catches a variant accidentally dropped from `VALUES`.
    assert!(ShaderStage::VALUES.len() == ShaderStage::VARIANT_COUNT);

    /// Fails to compile if [`ShaderStage`] gains, loses or renames a variant, which
    /// is the signal to update the impl above.
    #[allow(dead_code)]
    fn assert_exhaustive(value: ShaderStage) {
        use ShaderStage::*;
        match value {
            Vertex => {}
            Fragment => {}
            Compute => {}
        }
    }
};

impl NagaEnum for UnaryOperator {
    /// Every variant.
    const VALUES: &'static [Self] = &[Self::Negate, Self::LogicalNot, Self::BitwiseNot];

    const VARIANT_COUNT: usize = 3;
}

const _: () = {
    // Catches a variant accidentally dropped from `VALUES`.
    assert!(UnaryOperator::VALUES.len() == UnaryOperator::VARIANT_COUNT);

    /// Fails to compile if [`UnaryOperator`] gains, loses or renames a variant, which
    /// is the signal to update the impl above.
    #[allow(dead_code)]
    fn assert_exhaustive(value: UnaryOperator) {
        use UnaryOperator::*;
        match value {
            Negate => {}
            LogicalNot => {}
            BitwiseNot => {}
        }
    }
};

impl NagaEnum for VectorSize {
    /// Every variant.
    const VALUES: &'static [Self] = &[Self::Bi, Self::Tri, Self::Quad];

    const VARIANT_COUNT: usize = 3;
}

const _: () = {
    // Catches a variant accidentally dropped from `VALUES`.
    assert!(VectorSize::VALUES.len() == VectorSize::VARIANT_COUNT);

    /// Fails to compile if [`VectorSize`] gains, loses or renames a variant, which
    /// is the signal to update the impl above.
    #[allow(dead_code)]
    fn assert_exhaustive(value: VectorSize) {
        use VectorSize::*;
        match value {
            Bi => {}
            Tri => {}
            Quad => {}
        }
    }
};
