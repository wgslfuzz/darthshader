use super::expression::{ConstExpressionGenerators, ExpressionGenerators};
use super::statement::StatementGenerators;
use libafl_bolts::rands::{Rand, StdRand};
use serde::Deserialize;

#[derive(Deserialize, Debug)]
struct MinMaxUnchecked<T>
where
    T: PartialOrd + std::fmt::Debug,
{
    min: T,
    max: T,
}

impl<T> TryFrom<MinMaxUnchecked<T>> for MinMax<T>
where
    T: PartialOrd + std::fmt::Debug,
{
    type Error = String;

    fn try_from(value: MinMaxUnchecked<T>) -> Result<Self, Self::Error> {
        if value.min <= value.max {
            Ok(Self {
                min: value.min,
                max: value.max,
            })
        } else {
            Err(format!(
                "Invalid min/max values {:?} {:?}",
                value.min, value.max
            ))
        }
    }
}

#[derive(Deserialize, Debug)]
#[serde(try_from = "MinMaxUnchecked<T>")]
pub struct MinMax<T>
where
    T: PartialOrd + std::fmt::Debug,
{
    pub min: T,
    pub max: T,
}

impl MinMax<u32> {
    pub fn choose(&self, rng: &mut StdRand) -> u32 {
        rng.between(self.min as u64, self.max as u64) as u32
    }
}

#[derive(Deserialize, Debug)]
#[serde(rename_all(deserialize = "camelCase"))]
pub(crate) struct GeneratorConfig {
    pub global_const_exprs_count: MinMax<u32>,
    pub global_variables_count: MinMax<u32>,
    pub functions_count: MinMax<u32>,
    pub entrypoints_count: MinMax<u32>,
    pub complex_types_count: MinMax<u32>,
    pub function_budget: MinMax<u32>,
    pub switch_cases: MinMax<u32>,
    pub recursive_budget_rate: MinMax<f32>,
    pub expression_weight_map: ExpressionWeightMap,
    pub const_expression_weight_map: ConstExpressionWeightMap,
    pub statement_weight_map: StatementWeightMap,
}

#[derive(Debug)]
pub(crate) struct ExpressionWeightMap {
    pub weights: [u32; std::mem::variant_count::<ExpressionGenerators>()],
}

#[derive(Debug)]
pub(crate) struct ConstExpressionWeightMap {
    pub weights: [u32; std::mem::variant_count::<ConstExpressionGenerators>()],
}

#[derive(Debug)]
pub(crate) struct StatementWeightMap {
    pub weights: [u32; std::mem::variant_count::<StatementGenerators>()],
}

impl<'de> Deserialize<'de> for StatementWeightMap {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct WeightMapVisitor;
        impl<'de> serde::de::Visitor<'de> for WeightMapVisitor {
            type Value = StatementWeightMap;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a statement generator weight map")
            }

            fn visit_map<V>(self, mut map: V) -> Result<StatementWeightMap, V::Error>
            where
                V: serde::de::MapAccess<'de>,
            {
                let mut weights = [None; std::mem::variant_count::<StatementGenerators>()];
                while let Some(key) = map.next_key::<StatementGenerators>()? {
                    weights[key as usize] = Some(map.next_value()?);
                }
                let weights: Option<Vec<u32>> = weights.into_iter().collect();
                let weights = weights.ok_or_else(|| {
                    serde::de::Error::missing_field("Statement generator weight missing")
                })?;
                let weights = weights.try_into().unwrap();
                Ok(StatementWeightMap { weights })
            }
        }

        deserializer.deserialize_map(WeightMapVisitor)
    }
}

impl<'de> Deserialize<'de> for ConstExpressionWeightMap {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct WeightMapVisitor;
        impl<'de> serde::de::Visitor<'de> for WeightMapVisitor {
            type Value = ConstExpressionWeightMap;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("an expression generator weight map")
            }

            fn visit_map<V>(self, mut map: V) -> Result<ConstExpressionWeightMap, V::Error>
            where
                V: serde::de::MapAccess<'de>,
            {
                let mut weights = [None; std::mem::variant_count::<ConstExpressionGenerators>()];
                while let Some(key) = map.next_key::<ConstExpressionGenerators>()? {
                    weights[key as usize] = Some(map.next_value()?);
                }
                let weights: Option<Vec<u32>> = weights.into_iter().collect();
                let weights = weights.ok_or_else(|| {
                    serde::de::Error::missing_field("Constexpr generator weight missing")
                })?;
                let weights = weights.try_into().unwrap();
                Ok(ConstExpressionWeightMap { weights })
            }
        }

        deserializer.deserialize_map(WeightMapVisitor)
    }
}

impl<'de> Deserialize<'de> for ExpressionWeightMap {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct WeightMapVisitor;
        impl<'de> serde::de::Visitor<'de> for WeightMapVisitor {
            type Value = ExpressionWeightMap;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("an expression generator weight map")
            }

            fn visit_map<V>(self, mut map: V) -> Result<ExpressionWeightMap, V::Error>
            where
                V: serde::de::MapAccess<'de>,
            {
                let mut weights = [None; std::mem::variant_count::<ExpressionGenerators>()];
                while let Some(key) = map.next_key::<ExpressionGenerators>()? {
                    weights[key as usize] = Some(map.next_value()?);
                }
                let weights: Option<Vec<u32>> = weights.into_iter().collect();
                let weights = weights.ok_or_else(|| {
                    serde::de::Error::missing_field("Expr generator weight missing")
                })?;
                let weights = weights.try_into().unwrap();
                Ok(ExpressionWeightMap { weights })
            }
        }

        deserializer.deserialize_map(WeightMapVisitor)
    }
}

impl Default for ExpressionWeightMap {
    fn default() -> Self {
        let mut weights = [None; std::mem::variant_count::<ExpressionGenerators>()];
        weights[ExpressionGenerators::Access as usize] = Some(1);
        weights[ExpressionGenerators::AccessIndex as usize] = Some(1);
        weights[ExpressionGenerators::ArrayLength as usize] = Some(1);
        weights[ExpressionGenerators::Derivative as usize] = Some(1);
        weights[ExpressionGenerators::FunctionArgument as usize] = Some(1);
        weights[ExpressionGenerators::GlobalVar as usize] = Some(1);
        weights[ExpressionGenerators::Load as usize] = Some(1);
        weights[ExpressionGenerators::LocalVar as usize] = Some(1);
        weights[ExpressionGenerators::Relational as usize] = Some(1);
        weights[ExpressionGenerators::As as usize] = Some(1);
        weights[ExpressionGenerators::Binary as usize] = Some(1);
        weights[ExpressionGenerators::Select as usize] = Some(1);
        weights[ExpressionGenerators::Math as usize] = Some(1);
        weights[ExpressionGenerators::Unary as usize] = Some(1);
        weights[ExpressionGenerators::Swizzle as usize] = Some(1);
        weights[ExpressionGenerators::Literal as usize] = Some(1);
        weights[ExpressionGenerators::Compose as usize] = Some(1);
        weights[ExpressionGenerators::Constant as usize] = Some(1);
        weights[ExpressionGenerators::Splat as usize] = Some(1);
        weights[ExpressionGenerators::ZeroVal as usize] = Some(1);
        let weights: Option<Vec<u32>> = weights.into_iter().collect();
        let weights = weights.unwrap().try_into().unwrap();
        Self { weights }
    }
}

impl Default for ConstExpressionWeightMap {
    fn default() -> Self {
        let mut weights = [None; std::mem::variant_count::<ConstExpressionGenerators>()];
        weights[ConstExpressionGenerators::Compose as usize] = Some(1);
        weights[ConstExpressionGenerators::Constant as usize] = Some(1);
        weights[ConstExpressionGenerators::Literal as usize] = Some(1);
        weights[ConstExpressionGenerators::Splat as usize] = Some(1);
        weights[ConstExpressionGenerators::ZeroVal as usize] = Some(1);
        let weights: Option<Vec<u32>> = weights.into_iter().collect();
        let weights = weights.unwrap().try_into().unwrap();
        Self { weights }
    }
}

impl Default for StatementWeightMap {
    fn default() -> Self {
        let mut weights = [None; std::mem::variant_count::<StatementGenerators>()];
        weights[StatementGenerators::Atomic as usize] = Some(1);
        weights[StatementGenerators::Barrier as usize] = Some(1);
        weights[StatementGenerators::Block as usize] = Some(1);
        weights[StatementGenerators::Break as usize] = Some(1);
        weights[StatementGenerators::Call as usize] = Some(1);
        weights[StatementGenerators::Continue as usize] = Some(1);
        weights[StatementGenerators::If as usize] = Some(1);
        weights[StatementGenerators::Kill as usize] = Some(1);
        weights[StatementGenerators::Loop as usize] = Some(1);
        weights[StatementGenerators::Return as usize] = Some(1);
        weights[StatementGenerators::Store as usize] = Some(1);
        weights[StatementGenerators::Switch as usize] = Some(1);
        weights[StatementGenerators::WorkgroupLoad as usize] = Some(1);
        let weights: Option<Vec<u32>> = weights.into_iter().collect();
        let weights = weights.unwrap().try_into().unwrap();
        Self { weights }
    }
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            global_const_exprs_count: MinMax { min: 0, max: 10 },
            global_variables_count: MinMax { min: 0, max: 10 },
            functions_count: MinMax { min: 0, max: 5 },
            entrypoints_count: MinMax { min: 0, max: 2 },
            complex_types_count: MinMax { min: 3, max: 10 },
            function_budget: MinMax { min: 20, max: 200 },
            switch_cases: MinMax { min: 0, max: 8 },
            recursive_budget_rate: MinMax {
                min: 0.05,
                max: 0.5,
            },
            expression_weight_map: Default::default(),
            const_expression_weight_map: Default::default(),
            statement_weight_map: Default::default(),
        }
    }
}
