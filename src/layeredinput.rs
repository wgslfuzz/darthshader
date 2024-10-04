use libafl::prelude::{Error, HasBytesVec, HasTargetBytes, Input};
use libafl_bolts::{ownedref::OwnedSlice, HasLen};
use naga::{
    back::wgsl::Writer, Block, Expression, Function, Handle, Range, ScalarKind, Statement, Type,
    TypeInner,
};
use serde::{Deserialize, Serialize};
use std::fmt::Write;
use std::time::SystemTime;
use std::{cell::OnceCell, fs, path::Path};

use crate::ast::Ast;
use crate::ir::iter::{BlockVisitorMut, FunctionIdentifier, IterFuncs};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub(crate) struct IR {
    module: naga::Module,
    #[serde(skip)]
    text: OnceCell<Result<String, String>>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum LayeredInput {
    IR(IR),
    Ast(Ast),
}

struct BoundedString {
    inner: String,
    max: usize,
}

impl BoundedString {
    pub fn new(max: usize) -> Self {
        Self {
            inner: String::new(),
            max,
        }
    }

    pub fn into_inner(self) -> String {
        return self.inner;
    }
}

impl Write for BoundedString {
    fn write_str(&mut self, s: &str) -> std::result::Result<(), std::fmt::Error> {
        if s.len() + self.inner.len() > self.max {
            return Err(std::fmt::Error);
        }
        self.inner.write_str(s)
    }
}

impl IR {
    pub fn try_get_text(&self) -> Result<&String, &String> {
        use naga::back::wgsl;
        self.text
            .get_or_init(|| {
                let info = naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::all(),
                )
                .validate(self.get_module())
                .map_err(|e| format!("Validation failed: {}", e))?;

                let mut w = Writer::new(BoundedString::new(131072), wgsl::WriterFlags::empty());
                w.write(self.get_module(), &info)
                    .map_err(|e| format!("Lifting failed: {}", e))?;
                let output = w.finish();
                Ok(output.into_inner())
            })
            .as_ref()
    }

    pub fn get_func(&self, fid: FunctionIdentifier) -> &Function {
        match fid {
            FunctionIdentifier::Function(handle) => &self.get_module().functions[handle],
            FunctionIdentifier::EntryPoint(idx) => &self.get_module().entry_points[idx].function,
        }
    }

    pub fn get_func_mut(&mut self, fid: FunctionIdentifier) -> &mut Function {
        match fid {
            FunctionIdentifier::Function(handle) => &mut self.get_module_mut().functions[handle],
            FunctionIdentifier::EntryPoint(idx) => {
                &mut self.get_module_mut().entry_points[idx].function
            }
        }
    }

    pub fn get_module(&self) -> &naga::Module {
        &self.module
    }

    pub fn get_module_mut(&mut self) -> &mut naga::Module {
        self.text.take();
        &mut self.module
    }

    pub fn new(module: naga::Module) -> Self {
        let mut s = Self {
            module,
            text: OnceCell::new(),
        };

        let expand_emits = |block: &mut Block| {
            while let Some(idx) = block.iter().position(|stmt| match stmt {
                Statement::Emit(range) => range.zero_based_index_range().len() != 1,
                _ => false,
            }) {
                let Statement::Emit(range) = &block[idx] else {
                    unreachable!();
                };
                let expanded = Block::from(
                    range
                        .clone()
                        .map(|handle| Statement::Emit(Range::new_from_bounds(handle, handle)))
                        .collect::<Vec<Statement>>(),
                );
                block.splice(idx..=idx, expanded);
            }
            true
        };
        for (_, func) in s.iter_funcs_mut() {
            func.visit_blocks_mut(expand_emits);
            for (.., expr) in func.expressions.iter_mut() {
                let Expression::As { kind, convert, .. } = expr else {
                    continue;
                };
                if convert.is_none() {
                    let size = match kind {
                        ScalarKind::Sint => 4,
                        ScalarKind::Uint => 4,
                        ScalarKind::Float => 4,
                        ScalarKind::Bool => 1,
                    };
                    *convert = Some(size);
                }
            }
        }
        s
    }

    pub fn shorten_identifiers(&mut self) {
        fn renumber_locals(f: &mut Function) {
            for arg in f.arguments.iter_mut() {
                let Some(ref mut name) = arg.name else {
                    continue;
                };
                name.clear();
                name.push('l');
            }
            for (_, l) in f.local_variables.iter_mut() {
                let Some(ref mut name) = l.name else {
                    continue;
                };
                name.clear();
                name.push('l');
            }
        }

        let module = self.get_module_mut();
        for (_, g) in module.global_variables.iter_mut() {
            let Some(ref mut name) = g.name else {
                continue;
            };
            name.clear();
            name.push('g')
        }

        for (_, c) in module.constants.iter_mut() {
            let Some(ref mut name) = c.name else {
                continue;
            };
            name.clear();
            name.push('c');
        }

        for (_, f) in module.functions.iter_mut() {
            renumber_locals(f);
        }
        for e in module.entry_points.iter_mut() {
            renumber_locals(&mut e.function);
        }

        for (_, f) in module.functions.iter_mut() {
            let Some(ref mut name) = f.name else {
                continue;
            };
            name.clear();
            name.push('f');
        }

        let t_handles: Vec<Handle<Type>> = module.types.iter().map(|(h, _)| h).collect();
        for h in t_handles.into_iter() {
            let mut t = module.types.get_handle(h).unwrap().clone();
            if let Some(ref mut name) = t.name {
                name.clear();
                name.push('t');
                if let TypeInner::Struct {
                    ref mut members, ..
                } = t.inner
                {
                    for m in members.iter_mut() {
                        if let Some(ref mut name) = m.name {
                            name.clear();
                            name.push('m');
                        }
                    }
                }
                if module.types.get(&t).is_none() {
                    module.types.replace(h, t);
                }
            };
        }
    }
}

impl TryFrom<&str> for IR {
    type Error = ();

    fn try_from(input: &str) -> Result<Self, Self::Error> {
        let module = naga::front::wgsl::parse_str(input).map_err(|_| ())?;

        naga::valid::Validator::new(
            naga::valid::ValidationFlags::from_bits(0).unwrap(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .map_err(|_| ())?;

        Ok(IR::new(module))
    }
}

impl Input for LayeredInput {
    fn generate_name(&self, idx: usize) -> String {
        let delta = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap();
        format!("{:010}_{:06}.ron", delta.as_secs(), idx)
    }

    fn from_file<P>(path: P) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        println!("from file {:?}", path.as_ref());

        let input = fs::read(path.as_ref())?;
        let input = String::from_utf8(input)?;
        let module = match path
            .as_ref()
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| {
                Error::illegal_argument(format!("illegal extension: {:?}", path.as_ref()))
            })? {
            "spv" => {
                let options = naga::front::spv::Options {
                    adjust_coordinate_space: true,
                    strict_capabilities: false,
                    block_ctx_dump_prefix: None,
                };
                naga::front::spv::parse_u8_slice(input.as_bytes(), &options)
                    .map_err(|e| Error::illegal_argument(e.to_string()))?
            }
            ext @ ("vert" | "frag" | "comp") => {
                let mut parser = naga::front::glsl::Frontend::default();
                parser
                    .parse(
                        &naga::front::glsl::Options {
                            stage: match ext {
                                "vert" => naga::ShaderStage::Vertex,
                                "frag" => naga::ShaderStage::Fragment,
                                "comp" => naga::ShaderStage::Compute,
                                _ => unreachable!(),
                            },
                            defines: Default::default(),
                        },
                        &input,
                    )
                    .map_err(|_| Error::illegal_argument("cannot parse glsl"))?
            }
            "wgsl" => {
                let res = std::panic::catch_unwind(|| {
                    naga::front::wgsl::parse_str(&input)
                        .map_err(|e| Error::illegal_argument(e.to_string()))
                });

                let Ok(Ok(module)) = res else {
                    println!("Attempting to parse as AST: {:?}", path.as_ref());
                    let ast = match Ast::try_from_wgsl(input.as_bytes()) {
                        Ok(ast) => ast,
                        Err(_) => Ast::try_from_wgsl("".as_bytes()).unwrap(),
                    };
                    return Ok(LayeredInput::Ast(ast));
                };
                module
            }
            _ => {
                println!("Offending file: {:?}", path.as_ref());
                panic!("file import error");
            }
        };

        naga::valid::Validator::new(
            naga::valid::ValidationFlags::from_bits(0).unwrap(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .map_err(|e| Error::illegal_argument(e.to_string()))?;

        let mut ir = IR::new(module);
        ir.shorten_identifiers();

        Ok(LayeredInput::IR(ir))
    }

    fn to_file<P>(&self, path: P) -> Result<(), Error>
    where
        P: AsRef<Path>,
    {
        let mut opts = ron::Options::default();
        opts.recursion_limit = None;
        let string = opts
            .to_string(&self)
            .map_err(|e| Error::illegal_argument(e.to_string()))?;

        fs::write(path, string)?;
        Ok(())
    }
}

impl HasLen for LayeredInput {
    fn len(&self) -> usize {
        match self {
            Self::IR(module) => match module.try_get_text() {
                Ok(text) => text.as_bytes().len(),
                Err(_) => 0,
            },
            Self::Ast(ast) => ast.get_text().as_bytes().len(),
        }
    }
}

impl HasTargetBytes for LayeredInput {
    fn target_bytes(&self) -> OwnedSlice<u8> {
        let slice = match self {
            Self::IR(module) => match module.try_get_text() {
                Ok(text) => text.as_bytes(),
                Err(_) => &[],
            },
            Self::Ast(ast) => ast.get_text().as_bytes(),
        };
        slice.into()
    }
}

impl HasBytesVec for LayeredInput {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::IR(module) => match module.try_get_text() {
                Ok(text) => text.as_bytes(),
                Err(_) => &[],
            },
            Self::Ast(ast) => ast.get_text().as_bytes(),
        }
    }

    fn bytes_mut(&mut self) -> &mut Vec<u8> {
        match self {
            _ => panic!("Must not be called on AST/IR"),
        }
    }
}
