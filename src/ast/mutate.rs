use libafl::{
    prelude::{Corpus, MutationResult, Mutator},
    state::{HasCorpus, HasRand},
};
use libafl_bolts::{
    rands::Rand,
    tuples::{tuple_list, tuple_list_type},
    Error, Named,
};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand::{seq::IteratorRandom, Rng};
use std::cmp;

use crate::randomext::RandExt;
use crate::{
    ast::tree::{ASTHandle, Cursor, NodeKind},
    layeredinput::LayeredInput,
};

/// Tuple type of the mutations that compose the AST mutator
pub type ASTMutationsType = tuple_list_type!(
    ASTDeleteMutator,
    ASTReplaceTokenMutator,
    ASTSpliceMutator,
    ASTSwapChildrenMutator,
    ASTIdentifierMutator,
    ASTDeepenMutator,
);

/// Get the mutations that compose the AST mutator
#[must_use]
pub fn ast_mutations() -> ASTMutationsType {
    tuple_list!(
        ASTDeleteMutator::new(),
        ASTReplaceTokenMutator::new(),
        ASTSpliceMutator::new(),
        ASTSwapChildrenMutator::new(),
        ASTIdentifierMutator::new(),
        ASTDeepenMutator::new(),
    )
}

#[derive(Default, Debug)]
pub struct ASTSpliceMutator;

impl ASTSpliceMutator {
    /// Creates a new [`ASTSpliceMutator`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Named for ASTSpliceMutator {
    fn name(&self) -> &str {
        "ASTSpliceMutator"
    }
}

impl<S> Mutator<S::Input, S> for ASTSpliceMutator
where
    S: HasRand + HasCorpus<Input = LayeredInput>,
{
    fn mutate(
        &mut self,
        state: &mut S,
        input: &mut LayeredInput,
        _stage_idx: i32,
    ) -> Result<MutationResult, Error> {
        let LayeredInput::Ast(ast) = input else {
            return Ok(MutationResult::Skipped);
        };

        let idx = 'outer: {
            for _ in 0..20 {
                let idx = libafl::random_corpus_id!(state.corpus(), state.rand_mut());
                let other_testcase = state.corpus().get(idx)?.borrow();
                let Some(other) = other_testcase.input().as_ref() else {
                    continue;
                };
                if matches!(other, LayeredInput::Ast(_)) {
                    break 'outer idx;
                }
            }
            return Ok(MutationResult::Skipped);
        };

        let mut rng = SmallRng::seed_from_u64(state.rand_mut().next());
        let other_testcase = state.corpus().get(idx)?.borrow();
        let Some(other) = other_testcase.input().as_ref() else {
            unreachable!("No longer as_ref()");
        };
        let LayeredInput::Ast(aux) = other else {
            unreachable!("No longer an AST");
        };

        let Some(handle_aux) = aux.iter().map(|(handle, _)| handle).choose(&mut rng) else {
            return Ok(MutationResult::Skipped);
        };
        let kind_aux = aux.get_node(handle_aux).kind;

        let insert_point = {
            if rng.gen_bool(0.5) {
                let (insert_point, _) = ast.iter().choose(&mut rng).unwrap();
                insert_point
            } else if let Some(insert_point) = ast
                .iter()
                .filter_map(|(handle, node)| {
                    if node.kind == kind_aux {
                        Some(handle)
                    } else {
                        None
                    }
                })
                .choose(&mut rng)
            {
                insert_point
            } else {
                let (insert_point, _) = ast.iter().choose(&mut rng).unwrap();
                insert_point
            }
        };

        ast.splice(insert_point, aux, handle_aux);

        if rng.gen_bool(0.5) {
            return Ok(MutationResult::Mutated);
        }

        Ok(MutationResult::Mutated)
    }
}

#[derive(Default, Debug)]
pub struct ASTDeleteMutator;

impl ASTDeleteMutator {
    /// Creates a new [`ASTDeleteMutator`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Named for ASTDeleteMutator {
    fn name(&self) -> &str {
        "ASTDeleteMutator"
    }
}

impl<S> Mutator<S::Input, S> for ASTDeleteMutator
where
    S: HasRand + HasCorpus<Input = LayeredInput>,
{
    fn mutate(
        &mut self,
        state: &mut S,
        input: &mut LayeredInput,
        _stage_idx: i32,
    ) -> Result<MutationResult, Error> {
        let LayeredInput::Ast(ast) = input else {
            return Ok(MutationResult::Skipped);
        };

        let max_del = std::cmp::min(50, ast.len() / 10);
        let max_del = std::cmp::max(10, max_del);
        let max_del = 1 + state.rand_mut().below(max_del as u64) as usize;
        let mut rng = SmallRng::seed_from_u64(state.rand_mut().next());

        let mut handle = ast
            .iter()
            .filter_map(
                |(handle, node)| {
                    if node.is_leaf() {
                        Some(handle)
                    } else {
                        None
                    }
                },
            )
            .choose(&mut rng)
            .unwrap();

        if handle == ast.get_root() {
            return Ok(MutationResult::Skipped);
        }

        let mut deleted = 0usize;
        while handle != ast.get_root() && deleted < max_del {
            let (parent, _) = Cursor::new(ast, handle).goto_parent().unwrap();
            deleted += ast.purge(handle);
            handle = parent;
        }
        assert!(deleted > 0);
        Ok(MutationResult::Mutated)
    }
}
#[derive(Default, Debug)]

pub struct ASTIdentifierMutator;

impl ASTIdentifierMutator {
    /// Creates a new [`ASTIdentifierMutator`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Named for ASTIdentifierMutator {
    fn name(&self) -> &str {
        "ASTIdentifierMutator"
    }
}

impl<S> Mutator<S::Input, S> for ASTIdentifierMutator
where
    S: HasRand + HasCorpus<Input = LayeredInput>,
{
    fn mutate(
        &mut self,
        state: &mut S,
        input: &mut LayeredInput,
        _stage_idx: i32,
    ) -> Result<MutationResult, Error> {
        let LayeredInput::Ast(ast) = input else {
            return Ok(MutationResult::Skipped);
        };

        let mut rng = SmallRng::seed_from_u64(state.rand_mut().next());
        let handle = ast
            .iter()
            .filter_map(|(handle, node)| {
                if node.is_identifier() {
                    Some(handle)
                } else {
                    None
                }
            })
            .choose(&mut rng);
        let Some(handle) = handle else {
            return Ok(MutationResult::Skipped);
        };

        let mut cursor = Cursor::new(ast, handle);
        let func_handle = loop {
            let Some((parent_handle, parent_node)) = cursor.goto_parent() else {
                return Ok(MutationResult::Skipped);
            };
            if parent_node.is_function() {
                break parent_handle;
            }
        };
        let other_node = ast
            .iter_dfs(func_handle)
            .filter_map(|(_, node)| {
                if node.is_identifier() {
                    Some(node)
                } else {
                    None
                }
            })
            .choose(&mut rng)
            .unwrap();
        let identifier = other_node.get_text().to_owned();
        ast.get_node_mut(handle).set_text(&identifier);

        Ok(MutationResult::Mutated)
    }
}

#[derive(Default, Debug)]
pub struct ASTSwapChildrenMutator;

impl ASTSwapChildrenMutator {
    /// Creates a new [`ASTSwapChildrenMutator`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Named for ASTSwapChildrenMutator {
    fn name(&self) -> &str {
        "ASTSwapChildrenMutator"
    }
}

impl<S> Mutator<S::Input, S> for ASTSwapChildrenMutator
where
    S: HasRand + HasCorpus<Input = LayeredInput>,
{
    fn mutate(
        &mut self,
        state: &mut S,
        input: &mut LayeredInput,
        _stage_idx: i32,
    ) -> Result<MutationResult, Error> {
        let LayeredInput::Ast(ast) = input else {
            return Ok(MutationResult::Skipped);
        };

        let mut rng = SmallRng::seed_from_u64(state.rand_mut().next());
        let handle = ast
            .iter()
            .filter_map(|(handle, node)| {
                if node.children().len() > 1 {
                    Some(handle)
                } else {
                    None
                }
            })
            .choose(&mut rng);
        let Some(handle) = handle else {
            return Ok(MutationResult::Skipped);
        };

        let node = ast.get_node_mut(handle);

        let selected = node.children().iter().choose_multiple(&mut rng, 2);
        assert!(selected.len() == 2);
        let a = *selected[0];
        let b = *selected[1];
        ast.swap(a, b);

        Ok(MutationResult::Mutated)
    }
}

#[derive(Default, Debug)]
pub struct ASTReplaceTokenMutator {
    token: Vec<String>,
}

impl ASTReplaceTokenMutator {
    /// Creates a new [`ASTReplaceTokenMutator`].
    #[must_use]
    pub fn new() -> Self {
        let s = include_str!("../dictionary.txt");
        let token: Vec<String> = s.lines().map(|l| l.to_owned()).collect();
        Self { token }
    }
}

impl Named for ASTReplaceTokenMutator {
    fn name(&self) -> &str {
        "ASTReplaceTokenMutator"
    }
}

impl<S> Mutator<S::Input, S> for ASTReplaceTokenMutator
where
    S: HasRand + HasCorpus<Input = LayeredInput>,
{
    fn mutate(
        &mut self,
        state: &mut S,
        input: &mut LayeredInput,
        _stage_idx: i32,
    ) -> Result<MutationResult, Error> {
        let LayeredInput::Ast(ast) = input else {
            return Ok(MutationResult::Skipped);
        };

        let mut rng = SmallRng::seed_from_u64(state.rand_mut().next());

        let handle = ast
            .iter()
            .filter_map(|(handle, node)| {
                if node.get_text().is_empty() {
                    None
                } else {
                    Some(handle)
                }
            })
            .choose(&mut rng);
        let Some(handle) = handle else {
            return Ok(MutationResult::Skipped);
        };

        let node = ast.get_node_mut(handle);
        node.set_text(state.rand_mut().choose(&self.token));

        Ok(MutationResult::Mutated)
    }
}

#[derive(Default, Debug)]
pub struct ASTDeepenMutator;

impl ASTDeepenMutator {
    /// Creates a new [`ASTDeepenMutator`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Named for ASTDeepenMutator {
    fn name(&self) -> &str {
        "ASTDeepenMutator"
    }
}

impl<S> Mutator<S::Input, S> for ASTDeepenMutator
where
    S: HasRand + HasCorpus<Input = LayeredInput>,
{
    fn mutate(
        &mut self,
        state: &mut S,
        input: &mut LayeredInput,
        _stage_idx: i32,
    ) -> Result<MutationResult, Error> {
        let LayeredInput::Ast(ast) = input else {
            return Ok(MutationResult::Skipped);
        };

        if ast.len() > 2_000 {
            return Ok(MutationResult::Skipped);
        }
        let handles = {
            let mut handles = Vec::new();
            let root = ast.get_node(ast.get_root());
            let mut handle_by_kind: [Option<ASTHandle>; NodeKind::MAX] = [None; NodeKind::MAX];
            handle_by_kind[root.kind.id() as usize] = Some(ast.get_root());
            let mut stack_size = vec![root.children().len()];
            let mut stack_kind = vec![root.kind.id() as usize];
            let mut queue = Vec::from_iter(root.children().iter().cloned());
            while let Some(handle) = queue.pop() {
                assert!(stack_kind.len() == stack_size.len());
                while *stack_size.last().unwrap() == 0 {
                    stack_size.pop();
                    let kind = stack_kind.pop().unwrap();
                    handle_by_kind[kind] = None;
                }
                *stack_size.last_mut().unwrap() -= 1;

                let node = ast.get_node(handle);
                let kind = node.kind.id() as usize;
                if let Some(dom) = handle_by_kind[kind] {
                    handle_by_kind[kind] = None;
                    handles.push(dom);
                }
                if !node.children().is_empty() {
                    handle_by_kind[kind] = Some(handle);
                    stack_kind.push(kind);
                    stack_size.push(node.children().len());
                    queue.extend(node.children().iter());
                }
            }
            handles
        };

        if handles.is_empty() {
            return Ok(MutationResult::Skipped);
        }

        let handle = *state.rand_mut().choose(&handles);
        let node = ast.get_node(handle);

        let (subtree, handle_translate) = ast.extract_subtree(handle);
        let mut handles_insert: Vec<_> = ast
            .iter_dfs(handle)
            .filter_map(|(child_handle, child_node)| {
                if child_handle == handle || child_node.kind != node.kind {
                    None
                } else {
                    Some(child_handle)
                }
            })
            .collect();

        if handles_insert.is_empty() {
            return Ok(MutationResult::Skipped);
        }

        let mut handle_insert = 'outer: {
            let subtree_len = subtree.get_text().len();
            state.rand_mut().shuffle(&mut handles_insert);
            for handle in handles_insert.into_iter().take(20) {
                if subtree_len > ast.extract_subtree(handle).0.get_text().len() {
                    break 'outer handle;
                }
            }
            return Ok(MutationResult::Skipped);
        };

        let subtree_insert_point = handle_translate[handle_insert].unwrap();
        let mut extension = subtree.clone();
        let children: Vec<_> = extension
            .get_node(subtree_insert_point)
            .children()
            .iter()
            .cloned()
            .collect();
        for c in children.into_iter() {
            extension.purge(c);
        }
        let subtree_to_extension = extension.deflate();
        let extension_insert_point = subtree_to_extension[subtree_insert_point].unwrap();

        let char_increase = extension.get_text().len();
        assert!(char_increase > 0);
        let max_dup_fac = 4_096 / char_increase;
        let max_dup_fac = cmp::min(max_dup_fac, 4_096 / extension.len());
        let max_dup_fac = cmp::max(1, max_dup_fac);

        let dup_magnitude = usize::ilog2(max_dup_fac) as u64;
        let dup_fac = 1 << state.rand_mut().below(dup_magnitude + 1);

        ast.reserve(dup_fac * extension.len() + subtree.len());
        for _ in 0..dup_fac {
            let handle_translate = ast.merge_ast(handle_insert, &extension);
            handle_insert = handle_translate[extension_insert_point].unwrap();
        }
        ast.merge_ast(handle_insert, &subtree);

        Ok(MutationResult::Mutated)
    }
}
