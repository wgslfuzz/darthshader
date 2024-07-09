use std::{
    borrow::Cow,
    cell::OnceCell,
    collections::{hash_map::DefaultHasher, HashSet},
    hash::{Hash, Hasher},
    ops::Index,
    sync::LazyLock,
};

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tinystr::TinyAsciiStr;
use tinyvec::ArrayVec;
use tree_sitter::{Language, Parser, TreeCursor};

extern "C" {
    fn tree_sitter_wgsl() -> Language;
}

static ATOMS: LazyLock<[&'static str; 301]> = LazyLock::new(|| {
    let s = include_str!("../dictionary.txt");
    let mut token: Vec<&'static str> = s.lines().collect();
    token.sort();
    token
        .try_into()
        .expect("Lenght of array must match length of dictionary.txt")
});

static WGSLLANGUAGE: LazyLock<Language> = LazyLock::new(|| unsafe { tree_sitter_wgsl() });

#[derive(Serialize, Deserialize, Clone, Debug, Copy, PartialEq, Eq, Hash)]
pub struct ASTHandle(u32);

impl ASTHandle {
    fn index(&self) -> usize {
        self.0 as usize
    }
}

impl Default for ASTHandle {
    fn default() -> Self {
        Self(u32::MAX)
    }
}

impl From<usize> for ASTHandle {
    fn from(value: usize) -> Self {
        Self(value.try_into().expect("Handle to large"))
    }
}

#[derive(Debug)]
pub struct HandleTransition {
    vec: Vec<Option<ASTHandle>>,
}

impl Index<ASTHandle> for HandleTransition {
    type Output = Option<ASTHandle>;

    fn index(&self, index: ASTHandle) -> &Self::Output {
        &self.vec[index.0 as usize]
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Copy, Hash)]
pub struct NodeKind(u16);

impl Default for NodeKind {
    fn default() -> Self {
        Self(u16::MAX)
    }
}

impl NodeKind {
    pub const MAX: usize = 314;

    pub fn id(&self) -> u16 {
        assert!((self.0 as usize) < Self::MAX);
        self.0
    }
}

impl From<(&str, u16)> for NodeKind {
    fn from(t: (&str, u16)) -> Self {
        assert!((t.1 as usize) < Self::MAX);
        Self(t.1)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Copy)]
enum NodeText {
    Atom(&'static str),
    Tiny(TinyAsciiStr<{ NodeText::MAX_STR }>),
}

impl NodeText {
    const MAX_STR: usize = 16;

    pub fn as_str(&self) -> &str {
        match self {
            Self::Atom(a) => a,
            Self::Tiny(t) => t.as_str(),
        }
    }
}

impl From<&str> for NodeText {
    fn from(value: &str) -> Self {
        if value.len() <= NodeText::MAX_STR {
            NodeText::Tiny(TinyAsciiStr::<{ NodeText::MAX_STR }>::from_str(value).unwrap())
        } else if let Ok(index) = ATOMS.binary_search(&value) {
            NodeText::Atom(ATOMS[index])
        } else {
            println!("hashing: {}", value);
            let mut hasher = DefaultHasher::new();
            value.hash(&mut hasher);
            let trunc = format!("trunc_{}", hasher.finish() as u32);
            NodeText::Tiny(TinyAsciiStr::<{ NodeText::MAX_STR }>::from_str(trunc.as_str()).unwrap())
        }
    }
}

fn deserialize_text<'de, D>(deserializer: D) -> Result<NodeText, D::Error>
where
    D: Deserializer<'de>,
{
    let s: Cow<'de, str> = Deserialize::deserialize(deserializer)?;
    Ok(s.as_ref().into())
}

fn serialize_text<S>(t: &NodeText, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    s.serialize_str(t.as_str())
}
#[derive(Serialize, Deserialize, Clone, Debug, Copy, Default)]
pub struct ASTChildren {
    v: ArrayVec<[ASTHandle; 9]>,
}

impl ASTChildren {
    pub fn len(&self) -> usize {
        self.v.len()
    }

    pub fn is_empty(&self) -> bool {
        self.v.is_empty()
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &ASTHandle> {
        self.v.iter()
    }

    fn iter_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut ASTHandle> {
        self.v.iter_mut()
    }

    fn push(&mut self, child: ASTHandle) {
        self.v.push(child)
    }

    fn remove(&mut self, nth: usize) -> ASTHandle {
        self.v.remove(nth)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Copy)]
pub struct ASTNode {
    pub kind: NodeKind,
    children: ASTChildren,
    parent: Option<(ASTHandle, u8)>,
    #[serde(
        serialize_with = "serialize_text",
        deserialize_with = "deserialize_text"
    )]
    text: NodeText,
    is_function: bool,
    is_identifier: bool,
    is_builtin: bool,
}

impl ASTNode {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn get_text(&self) -> &str {
        self.text.as_str()
    }

    pub fn set_text(&mut self, s: &str) {
        self.text = s.into();
    }

    pub fn is_function(&self) -> bool {
        self.is_function
    }

    pub fn is_identifier(&self) -> bool {
        self.is_identifier
    }

    pub fn children(&self) -> &ASTChildren {
        &self.children
    }

    pub fn children_mut(&mut self) -> &mut ASTChildren {
        &mut self.children
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

impl Default for ASTNode {
    fn default() -> Self {
        Self {
            kind: Default::default(),
            parent: None,
            text: NodeText::Tiny(TinyAsciiStr::<{ NodeText::MAX_STR }>::from_str("").unwrap()),
            children: Default::default(),
            is_function: false,
            is_identifier: false,
            is_builtin: false,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Copy)]
enum ASTNodeOrHole {
    ASTNode(ASTNode),
    Hole,
}

impl Default for ASTNodeOrHole {
    fn default() -> Self {
        Self::Hole
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Ast {
    nodes: Vec<ASTNodeOrHole>,
    free_nodes: Vec<ASTHandle>,
    root: ASTHandle,
    #[serde(skip)]
    text: OnceCell<String>,
}

#[derive(Debug)]
enum AstIterItem {
    Children(ASTHandle),
    Process(ASTHandle),
}

#[derive(Debug)]
pub struct Cursor<'a> {
    ast: &'a Ast,
    itr: &'a ASTNode,
    handle: ASTHandle,
    iter_end: bool,
}

impl<'a> Cursor<'a> {
    pub fn new(ast: &'a Ast, start: ASTHandle) -> Self {
        Self {
            ast,
            handle: start,
            itr: ast.get_node(start),
            iter_end: false,
        }
    }

    pub fn goto_parent(&mut self) -> Option<(ASTHandle, &'a ASTNode)> {
        if let Some((handle, _)) = self.itr.parent {
            self.handle = handle;
            self.itr = self.ast.get_node(handle);
            self.iter_end = false;
            Some((handle, self.itr))
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub struct PostOrderIter<'a> {
    ast: &'a Ast,
    queue: Vec<AstIterItem>,
}

impl<'a> PostOrderIter<'a> {
    fn new(ast: &'a Ast, start: ASTHandle) -> Self {
        Self {
            ast,
            queue: vec![AstIterItem::Children(start)],
        }
    }
}

impl<'a> Iterator for PostOrderIter<'a> {
    type Item = (ASTHandle, &'a ASTNode);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(item) = self.queue.pop() {
            match item {
                AstIterItem::Children(handle) => {
                    self.queue.push(AstIterItem::Process(handle));
                    let node = self.ast.get_node(handle);
                    self.queue.extend(
                        node.children
                            .iter()
                            .rev()
                            .map(|c| AstIterItem::Children(*c)),
                    );
                }
                AstIterItem::Process(handle) => {
                    let node = self.ast.get_node(handle);
                    return Some((handle, node));
                }
            }
        }
        None
    }
}

struct DfsIter<'a> {
    ast: &'a Ast,
    queue: Vec<ASTHandle>,
}

impl<'a> DfsIter<'a> {
    fn new(ast: &'a Ast, start: ASTHandle) -> Self {
        Self {
            ast,
            queue: vec![start],
        }
    }
}

impl<'a> Iterator for DfsIter<'a> {
    type Item = (ASTHandle, &'a ASTNode);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(handle) = self.queue.pop() {
            let node = self.ast.get_node(handle);
            if !node.children.is_empty() {
                self.queue.extend(node.children.iter().rev().cloned());
            }
            Some((handle, node))
        } else {
            None
        }
    }
}

impl Ast {
    pub fn try_from_wgsl(slice: &[u8]) -> Result<Self, ()> {
        fn traverse(
            walker: &mut TreeCursor,
            nodes: &mut Vec<ASTNode>,
            at: ASTHandle,
            slice: &[u8],
        ) -> Result<(), ()> {
            let id = walker.node().id();
            let node = nodes.get_mut(at.index()).unwrap();
            if walker.node().is_error() {
                return Err(());
            }
            node.kind = (walker.node().kind(), walker.node().kind_id()).into();
            if walker.node().kind() == "ident_pattern_token" {
                node.is_identifier = true;
            }
            if walker.node().kind() == "function_decl" {
                node.is_function = true;
            }

            if !walker.goto_first_child() {
                node.set_text(walker.node().utf8_text(slice.as_ref()).map_err(|_| ())?);
                if ATOMS.binary_search(&node.get_text()).is_ok() {
                    node.is_builtin = true;
                }
            } else {
                let mut child_count = 0u8;
                let mut child = ASTNode::new();
                child.parent = Some((at, child_count));
                child_count += 1;
                nodes.push(child);
                let c = ASTHandle::from(nodes.len() - 1);
                nodes.get_mut(at.index()).unwrap().children.push(c);

                traverse(walker, nodes, c, slice)?;

                while walker.goto_next_sibling() {
                    let mut child = ASTNode::new();
                    child.parent = Some((at, child_count));
                    child_count += 1;
                    nodes.push(child);
                    let c = ASTHandle::from(nodes.len() - 1);
                    nodes[at.index()].children.push(c);
                    traverse(walker, nodes, c, slice)?;
                }

                walker.goto_parent();
            }
            assert_eq!(walker.node().id(), id);
            Ok(())
        }

        let mut parser = Parser::new();
        parser.set_language(&WGSLLANGUAGE).unwrap();
        let tree = parser.parse(slice, None).ok_or(())?;
        let mut walker = tree.walk();
        let mut nodes = vec![Default::default()];

        traverse(&mut walker, &mut nodes, ASTHandle(0), slice)?;

        let nodes = nodes.into_iter().map(ASTNodeOrHole::ASTNode).collect();
        Ok(Ast {
            root: ASTHandle(0),
            nodes,
            free_nodes: Vec::new(),
            text: OnceCell::<String>::new(),
        })
    }

    pub fn get_text(&self) -> &String {
        self.text.get_or_init(|| self.into())
    }

    pub fn get_root(&self) -> ASTHandle {
        self.root
    }

    pub fn get_node(&self, handle: ASTHandle) -> &ASTNode {
        let ASTNodeOrHole::ASTNode(node) = &self.nodes[handle.index()] else {
            panic!("Node is gone");
        };
        node
    }

    pub fn get_node_mut(&mut self, handle: ASTHandle) -> &mut ASTNode {
        self.text.take();
        let ASTNodeOrHole::ASTNode(node) = &mut self.nodes[handle.index()] else {
            panic!("Node is gone");
        };
        node
    }

    pub fn swap(&mut self, first: ASTHandle, second: ASTHandle) {
        self.text.take();
        let first_parent = self.get_node(first).parent.expect("Cannot swap root");
        let second_parent = self.get_node(second).parent.expect("Cannot swap root");
        if first_parent.0 != second_parent.0 {
            unimplemented!("Currently, only swapping two children of same parent are supported");
        }
        self.get_node_mut(first).parent = Some(second_parent);
        self.get_node_mut(second).parent = Some(first_parent);
        let same_parent = self.get_node_mut(first_parent.0);
        same_parent.children.v[first_parent.1 as usize] = second;
        same_parent.children.v[second_parent.1 as usize] = first;
    }

    pub fn len(&self) -> usize {
        self.nodes.len() - self.free_nodes.len()
    }

    pub fn assert_dense(&self) {
        assert_eq!(self.free_nodes.len(), 0);
    }

    pub fn reserve(&mut self, additional: usize) {
        self.nodes.reserve(additional);
    }

    pub fn iter_post_order(&self, start: ASTHandle) -> impl Iterator<Item = (ASTHandle, &ASTNode)> {
        PostOrderIter::new(self, start)
    }

    pub fn iter_dfs(&self, start: ASTHandle) -> impl Iterator<Item = (ASTHandle, &ASTNode)> {
        DfsIter::new(self, start)
    }

    pub fn iter(&self) -> impl Iterator<Item = (ASTHandle, &ASTNode)> {
        self.nodes.iter().enumerate().filter_map(|(handle, entry)| {
            if let ASTNodeOrHole::ASTNode(node) = entry {
                Some((ASTHandle::from(handle), node))
            } else {
                None
            }
        })
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (ASTHandle, &mut ASTNode)> {
        self.text.take();
        self.nodes
            .iter_mut()
            .enumerate()
            .filter_map(|(handle, entry)| {
                if let ASTNodeOrHole::ASTNode(node) = entry {
                    Some((ASTHandle::from(handle), node))
                } else {
                    None
                }
            })
    }

    pub fn purge(&mut self, handle: ASTHandle) -> usize {
        let node = self.get_node_mut(handle);
        let parent = node.parent;

        let mut purged = 0;
        let mut pivot = self.free_nodes.len();
        self.free_nodes.push(handle);

        while pivot < self.free_nodes.len() {
            let handle = self.free_nodes[pivot];
            let ASTNodeOrHole::ASTNode(node) = &self.nodes[handle.index()] else {
                panic!("Live node has dead child");
            };
            self.free_nodes.extend(node.children.iter());
            self.nodes[handle.index()] = ASTNodeOrHole::Hole;
            purged += 1;
            pivot += 1;
        }

        if let Some((handle, nth)) = parent {
            let node = self.get_node_mut(handle);
            node.children.remove(nth as usize);
            let children = node.children;
            for sibling_handle in children.iter().skip(nth as usize) {
                let sibling = self.get_node_mut(*sibling_handle);
                let (handle, nth) = sibling.parent.unwrap();
                sibling.parent = Some((handle, nth - 1));
            }
        }

        debug_assert!(self.verify());
        purged
    }

    pub fn extract_subtree(&self, start: ASTHandle) -> (Ast, HandleTransition) {
        let mut subtree_nodes = Vec::new();
        let mut handle_translate = vec![None; self.nodes.len()];
        for (new_handle, (old_handle, node)) in self.iter_dfs(start).enumerate() {
            handle_translate[old_handle.index()] = Some(ASTHandle::from(new_handle));
            subtree_nodes.push(ASTNodeOrHole::ASTNode(*node));
        }

        for (idx, node) in subtree_nodes.iter_mut().enumerate() {
            let ASTNodeOrHole::ASTNode(node) = node else {
                panic!("No longer a node");
            };
            for c in node.children.iter_mut() {
                *c = handle_translate[c.index()].unwrap();
            }
            if idx == 0 {
                node.parent = None;
            } else {
                let (handle, nth) = node.parent.unwrap();
                node.parent = Some((handle_translate[handle.index()].unwrap(), nth));
            }
        }

        let mut subtree = Ast {
            root: handle_translate[start.index()].unwrap(),
            nodes: subtree_nodes,
            free_nodes: Vec::new(),
            text: OnceCell::<String>::new(),
        };
        subtree.get_node_mut(subtree.get_root()).parent = None;
        debug_assert!(subtree.verify());

        (
            subtree,
            HandleTransition {
                vec: handle_translate,
            },
        )
    }

    pub fn splice(&mut self, at: ASTHandle, other: &Self, from: ASTHandle) {
        self.text.take();

        let mut pivot = self.free_nodes.len();
        let children = self.get_node(at).children;
        self.free_nodes.extend(children.iter());
        while pivot < self.free_nodes.len() {
            let handle = self.free_nodes[pivot];
            let children = self.get_node(handle).children;
            self.free_nodes.extend(children.iter());
            self.nodes[handle.index()] = ASTNodeOrHole::Hole;
            pivot += 1;
        }

        let mut node = *other.get_node(from);
        node.parent = self.get_node(at).parent;
        self.nodes[at.index()] = ASTNodeOrHole::ASTNode(node);
        let mut worklist = vec![at];
        while let Some(handle) = worklist.pop() {
            let mut children = self.get_node(handle).children;
            for (nth, old_handle) in children.iter_mut().enumerate() {
                let new_handle = {
                    if let Some(new_handle) = self.free_nodes.pop() {
                        new_handle
                    } else {
                        self.nodes.push(ASTNodeOrHole::Hole);
                        ASTHandle::from(self.nodes.len() - 1)
                    }
                };
                let mut new_node = *other.get_node(*old_handle);
                new_node.parent = Some((handle, nth as u8));
                self.nodes[new_handle.index()] = ASTNodeOrHole::ASTNode(new_node);
                *old_handle = new_handle;
                worklist.push(new_handle);
            }
            self.get_node_mut(handle).children = children;
        }

        debug_assert!(self.verify());
    }

    pub fn merge_ast(&mut self, at: ASTHandle, other: &Self) -> HandleTransition {
        self.text.take();

        let parent = self.get_node(at).parent;
        let mut tree_iter = self.iter_dfs(at).map(|(handle, _)| handle);
        let mut handle_translate = Vec::new();
        handle_translate.resize_with(other.nodes.len(), || tree_iter.next());

        let free_handles: Vec<_> = tree_iter.collect();
        for handle in free_handles.into_iter() {
            self.nodes[handle.index()] = ASTNodeOrHole::Hole;
            self.free_nodes.push(handle);
        }

        for (node, new_handle) in other.nodes.iter().zip(handle_translate.iter_mut()) {
            let new_handle = match new_handle {
                Some(handle) => *handle,
                None => {
                    if let Some(handle) = self.free_nodes.pop() {
                        *new_handle = Some(handle);
                        handle
                    } else {
                        self.nodes.push(ASTNodeOrHole::Hole);
                        let handle = ASTHandle::from(self.nodes.len() - 1);
                        *new_handle = Some(handle);
                        handle
                    }
                }
            };
            self.nodes[new_handle.index()] = *node;
        }

        for new_handle in handle_translate.iter() {
            let Some(new_handle) = new_handle else {
                continue;
            };
            let ASTNodeOrHole::ASTNode(node) = &mut self.nodes[new_handle.index()] else {
                panic!("Unexpected, must be a node");
            };
            for c in node.children.iter_mut() {
                *c = handle_translate[c.index()].unwrap();
            }
            if let Some((handle, nth)) = node.parent {
                node.parent = Some((handle_translate[handle.index()].unwrap(), nth));
            } else {
                node.parent = parent;
            }
        }
        debug_assert!(self.verify());

        HandleTransition {
            vec: handle_translate,
        }
    }

    pub fn deflate(&mut self) -> HandleTransition {
        let mut handle_translate = vec![None; self.nodes.len()];
        let mut new_nodes = Vec::new();
        for (old_handle, node) in self.iter() {
            new_nodes.push(ASTNodeOrHole::ASTNode(*node));
            handle_translate[old_handle.index()] = Some(ASTHandle::from(new_nodes.len() - 1));
        }
        for node in new_nodes.iter_mut() {
            let ASTNodeOrHole::ASTNode(node) = node else {
                unreachable!("No longer a node");
            };
            for c in node.children.iter_mut() {
                *c = handle_translate[c.index()].unwrap();
            }
            if let Some((handle, nth)) = node.parent {
                node.parent = Some((handle_translate[handle.index()].unwrap(), nth));
            }
        }
        self.nodes = new_nodes;
        self.free_nodes.clear();
        debug_assert_eq!(self.nodes.len(), self.iter_dfs(self.get_root()).count());
        debug_assert!(self.verify());

        HandleTransition {
            vec: handle_translate,
        }
    }

    fn verify(&self) -> bool {
        assert_eq!(self.get_node(self.get_root()).parent, None);
        for (handle, node) in self.iter() {
            for (nth, c) in node.children().iter().enumerate() {
                let child = self.get_node(*c);
                let Some((c_parent, c_nth)) = child.parent else {
                    panic!("Child has no parent");
                };
                assert_eq!(c_nth as usize, nth);
                assert_eq!(c_parent, handle);
            }
        }

        assert_eq!(self.free_nodes.len() + self.len(), self.nodes.len());
        let handles_dfs: HashSet<ASTHandle> =
            HashSet::from_iter(self.iter_dfs(self.get_root()).map(|(handle, _)| handle));
        let handles_pot: HashSet<ASTHandle> = HashSet::from_iter(
            self.iter_post_order(self.get_root())
                .map(|(handle, _)| handle),
        );
        assert_eq!(self.len(), handles_dfs.len());
        assert_eq!(self.len(), handles_pot.len());
        true
    }
}

impl From<&Ast> for String {
    fn from(ast: &Ast) -> Self {
        let mut s = String::new();

        for (_, node) in ast.iter_post_order(ast.get_root()) {
            if !node.get_text().is_empty() {
                assert!(node.children.is_empty());
                s.push_str(node.get_text());
                s.push(' ');
            }
        }
        s
    }
}
