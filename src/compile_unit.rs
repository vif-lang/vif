use std::{
	cell::RefCell,
	fmt::Write,
	fs::File,
	io::Read,
	marker::PhantomData,
	path::{
		Path,
		PathBuf,
	},
	sync::{
		Arc,
		nonpoison::{
			Mutex,
			RwLock,
		},
	},
	time::Duration,
};

use bumpalo::Bump;
use internment::Intern;
use relative_path::{
	PathExt,
	RelativePathBuf,
};
use rustc_hash::{
	FxBuildHasher,
	FxHashMap,
};
use target_lexicon::Triple;

use crate::{
	Args,
	Build,
	BuildError,
	codegen::{
		self,
	},
	common::{
		COMMON_INTERNS,
		IndexVec,
		Span,
		diagnostic::{
			DiagSpan,
			Diagnostic,
			DiagnosticWriter,
			Label,
		},
	},
	compile_unit::{
		module::{
			ArcModule,
			Module,
			ModuleAnalyzeState,
			ModuleId,
		},
		sema::Sema,
	},
	frontend::{
		self,
		ast,
	},
	ir::{
		self,
		vuir::Vuir,
	},
	value::{
		self,
		ValueMap,
		ValueStore,
	},
};

pub mod module;
mod sema;

// =============================================================================
//                                 Decl type
// =============================================================================

#[derive(Debug)]
pub enum DeclAnalysisState {
	Unanalysed {
		module: ModuleId,
		vuir_id: ir::vuir::InstructionId,
	},
	Failed(sema::AnalyzeError),
	TypeKnown(value::Index),
	Analysed {
		value: value::Index,
	},
}

#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Default, bytemuck::NoUninit)]
pub struct DeclId(usize);
impl From<DeclId> for usize {
	#[inline(always)]
	fn from(value: DeclId) -> Self {
		value.0
	}
}
impl From<usize> for DeclId {
	#[inline(always)]
	fn from(value: usize) -> Self {
		Self(value)
	}
}

/// Represent a named, typed declaration
#[derive(Debug)]
pub struct Decl {
	pub name: Intern<str>,
	pub module: ModuleId,
	pub namespace: NamespaceId,
	pub analysis_state: DeclAnalysisState,
}

/// Namespace composed of declarations
#[derive(Debug)]
pub struct Namespace {
	pub parent: Option<NamespaceId>,
	pub decls: FxHashMap<Intern<str>, DeclId>,
	/// the type owning this namespace
	pub owner_type: value::Index,
}

impl Namespace {
	#[inline(always)]
	pub fn with_owner_type(owner_type: value::Index) -> Self {
		Self {
			parent: None,
			decls: Default::default(),
			owner_type,
		}
	}

	#[inline(always)]
	pub fn with_parent(
		parent: NamespaceId,
		owner_type: value::Index,
	) -> Self {
		Self {
			parent: Some(parent),
			decls: Default::default(),
			owner_type,
		}
	}
}

#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Default)]
pub struct NamespaceId(usize);
impl From<NamespaceId> for usize {
	#[inline(always)]
	fn from(value: NamespaceId) -> Self {
		value.0
	}
}
impl From<usize> for NamespaceId {
	#[inline(always)]
	fn from(value: usize) -> Self {
		Self(value)
	}
}

/// A analysis unit is the unit of semantic analysis.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum SemaUnit {
	DeclValue(DeclId),
	RuntimeFunc(value::Index),
}

/// State of a sema unit in the job system
#[derive(Debug)]
pub enum SemaJobState {
	/// Job is in-flight
	Queued,
	/// Job completed (successfully or not).
	Done,
}

/// A deferred effect check for a call where the callee's effects weren't available at analysis time.
pub struct DeferredEffectCheck {
	pub callee_fn: value::Index,
	pub handled_effects: Vec<value::Index>,
	pub span: Span,
	pub module: ModuleId,
}

#[derive(Clone, Debug)]
pub struct ResolvedTargetInfo {
	pub triple: Triple,
	pub builtin_arch_tag: &'static str,
	pub builtin_os_tag: &'static str,
	pub winapi_uses_stdcall: bool,
	pub ptr_width_in_bits: u8,
}

impl ResolvedTargetInfo {
	pub fn resolve_host_default() -> Self {
		let triple = Triple::host();

		let arch_name = triple.architecture.to_string().to_ascii_lowercase();
		let os_name = triple.operating_system.to_string().to_ascii_lowercase();

		let builtin_arch_tag = Self::map_arch_to_builtin_tag(&arch_name);
		let builtin_os_tag = Self::map_os_to_builtin_tag(&os_name);
		let winapi_uses_stdcall = builtin_os_tag == "windows" && matches!(builtin_arch_tag, "x86");
		let ptr_width_in_bits = triple.pointer_width().expect("cannot fetch target ptr size").bits();

		Self {
			triple,
			builtin_arch_tag,
			builtin_os_tag,
			winapi_uses_stdcall,
			ptr_width_in_bits,
		}
	}

	fn map_arch_to_builtin_tag(arch: &str) -> &'static str {
		match arch {
			"aarch64" => "aarch64",
			"aarch64_be" => "aarch64_be",
			"amdgcn" => "amdgcn",
			"arm" => "arm",
			"armeb" => "armeb",
			"avr" => "avr",
			"hexagon" => "hexagon",
			"loongarch64" => "loongarch64",
			"m68k" => "m68k",
			"mips" => "mips",
			"mipsel" => "mipsel",
			"mips64" => "mips64",
			"mips64el" => "mips64el",
			"nvptx64" => "nvptx64",
			"powerpc" => "powerpc",
			"powerpc64" => "powerpc64",
			"powerpc64le" => "powerpc64le",
			"riscv32" => "riscv32",
			"riscv64" => "riscv64",
			"s390x" => "s390x",
			"sparc" => "sparc",
			"sparc64" => "sparc64",
			"thumb" => "thumb",
			"thumbeb" => "thumbeb",
			"wasm32" => "wasm32",
			"wasm64" => "wasm64",
			"xcore" => "xcore",
			"xtensa" => "xtensa",
			"xtensaeb" => "xtensaeb",
			"x86_64" => "x86_64",
			_ if arch.starts_with("i") && arch.ends_with("86") => "x86",
			_ => "x86_64",
		}
	}

	fn map_os_to_builtin_tag(os: &str) -> &'static str {
		match os {
			"windows" => "windows",
			"linux" => "linux",
			"macos" | "darwin" => "macos",
			"ios" => "ios",
			"tvos" => "tvos",
			"watchos" => "watchos",
			"visionos" => "visionos",
			"freebsd" => "freebsd",
			"openbsd" => "openbsd",
			"netbsd" => "netbsd",
			"dragonfly" => "dragonfly",
			"haiku" => "haiku",
			"illumos" => "illumos",
			"solaris" => "illumos",
			"emscripten" => "emscripten",
			"wasi" => "wasi",
			"uefi" => "uefi",
			"fuchsia" => "fuchsia",
			"hurd" => "hurd",
			"rtems" => "rtems",
			"hermit" => "hermit",
			"plan9" => "plan9",
			"serenity" => "serenity",
			"contiki" => "contiki",
			"managarm" => "managarm",
			"none" | "unknown" => "freestanding",
			_ => "other",
		}
	}
}

pub struct CompilationUnit {
	/// The root path where all paths stored are relative to
	cwd: PathBuf,
	pub root_module: ModuleId,
	pub std_module: ModuleId,
	pub builtin_module: ModuleId,
	builtin_prelude_module: ModuleId,
	std_rt_module: ModuleId,
	// builtin_prelude_module: ModuleId,
	build_args: &'static Build,
	pub resolved_target: ResolvedTargetInfo,

	// mutable state
	pub modules: RwLock<IndexVec<ModuleId, ArcModule>>,
	pub namespaces: RwLock<IndexVec<NamespaceId, Namespace>>,
	pub values: ValueStore,
	/// modules that failed parsing or vuir generation
	pub failed_modules: Mutex<Vec<ModuleId>>,
	pub module_path_to_id: Mutex<FxHashMap<RelativePathBuf, ModuleId>>,
	pub decls: Mutex<IndexVec<DeclId, Decl>>,
	pub sema_jobs: Mutex<FxHashMap<SemaUnit, SemaJobState>>,

	// TODO(zino): no error per module, sema errors should be independent of module
	pub sema_errors: Mutex<FxHashMap<ModuleId, Vec<Diagnostic>>>,
	pub deferred_effect_checks: Mutex<Vec<DeferredEffectCheck>>,
	pub codegen_tasks: crossbeam::queue::SegQueue<(value::Index, ir::vtir::Vtir)>,
}

enum CodegenLowerer<'ctx> {
	#[cfg(feature = "llvm")]
	Llvm(codegen::llvm::Lowerer<'ctx>),
}

impl<'ctx> CodegenLowerer<'ctx> {
	fn lower_function(
		&mut self,
		compilation_unit: &CompilationUnit,
		fun: value::Index,
		vtir: &ir::vtir::Vtir,
		build_opts: &Build,
	) {
		match self {
			#[cfg(feature = "llvm")]
			Self::Llvm(lowerer) => {
				lowerer.lower_function(compilation_unit, fun, vtir, build_opts);
			},
		}
	}

	fn finish(
		self,
		build_opts: &Build,
	) -> Result<Vec<u8>, ()> {
		match self {
			#[cfg(feature = "llvm")]
			Self::Llvm(lowerer) => lowerer.finish(build_opts).map(|obj| obj.as_slice().to_vec()).map_err(|_err| ()),
		}
	}
}

impl CompilationUnit {
	#[inline]
	pub fn is_std_rt_module(
		&self,
		module: ModuleId,
	) -> bool {
		module == self.std_rt_module
	}

	pub fn new(
		build_args: &'static Build,
		cwd: PathBuf,
		root_module_path: &RelativePathBuf,
	) -> Arc<Self> {
		// TODO(ldubos): add a way to specify target info instead of always resolving host default
		let resolved_target = ResolvedTargetInfo::resolve_host_default();

		let compiler_exe_path = std::env::current_exe().unwrap();
		let std_path = compiler_exe_path.parent().unwrap().join("std").relative_to(&cwd).unwrap();

		let (modules, root_module, builtin_prelude_module, std_module, builtin_module, std_rt_module) = {
			let mut modules = IndexVec::default();
			let root_module: ModuleId = modules.push(ArcModule::new(Module::new(root_module_path.clone(), None)));
			let builtin_prelude_module = modules.push(ArcModule::new(Module::new(std_path.join("builtin_prelude.vif"), None)));
			let std_module = modules.push(ArcModule::new(Module::new(std_path.join("std.vif"), None)));
			let builtin_module = modules.push(ArcModule::new(Module::new(std_path.join("builtin.vif"), None)));
			let std_rt_module = modules.push(ArcModule::new(Module::new(std_path.join("rt.vif"), None)));
			(
				modules,
				root_module,
				builtin_prelude_module,
				std_module,
				builtin_module,
				std_rt_module,
			)
		};

		let module_path_to_id = {
			let mut module_path_to_id = FxHashMap::default();
			module_path_to_id.insert(modules[root_module].path.clone(), root_module);
			module_path_to_id.insert(modules[std_rt_module].path.clone(), std_rt_module);
			// builtin_prelude_module has no path to access it
			module_path_to_id
		};

		let cu = CompilationUnit {
			build_args,
			resolved_target,
			cwd,
			root_module,
			std_module,
			builtin_module,
			builtin_prelude_module,
			std_rt_module,
			values: ValueStore::new(
				std::thread::available_parallelism()
					.map(|i| i.get())
					.unwrap_or(1)
					.next_power_of_two() as _,
			),
			modules: RwLock::new(modules),
			namespaces: RwLock::default(),
			failed_modules: Mutex::new(Vec::default()),
			module_path_to_id: Mutex::new(module_path_to_id),
			decls: Default::default(),
			sema_jobs: Default::default(),
			sema_errors: Default::default(),
			deferred_effect_checks: Default::default(),
			codegen_tasks: Default::default(),
		};
		Arc::new(cu)
	}

	fn inject_builtin_declarations(&self) -> String {
		format!(
			r#"
pub const target: Target = Target {{
	.cpu = Target.CPU {{
			.arch = .{}
	}},
	.os = .{}
}};
"#,
			self.resolved_target.builtin_arch_tag, self.resolved_target.builtin_os_tag,
		)
	}

	pub fn compile(
		self: Arc<CompilationUnit>,
		launch_args: &Args,
		stderr: &mut dyn std::io::Write,
	) -> Result<Vec<u8>, BuildError> {
		profiling::scope!("compilation");

		let (modules_to_scan, root_module) = {
			let module_path_to_id = self.module_path_to_id.lock();
			let mut modules_to_scan = module_path_to_id.values().copied().collect::<Vec<_>>();
			modules_to_scan.push(self.builtin_prelude_module);
			// parse std module for @import("std")
			// NOTE(ldubs): we probably want to parse std only if it is actually imported,
			// both for performance and for a `no_std`-like compilation target.
			modules_to_scan.push(self.std_module);
			// parse builtin module for @import("builtin")
			modules_to_scan.push(self.builtin_module);
			(modules_to_scan, self.root_module)
		};

		// launch parsing for every modules we have knowledge of right now
		rayon::scope(|scope| {
			profiling::scope!("parse modules");

			for module in modules_to_scan {
				let compilation_unit = self.clone();
				scope.spawn(move |scope| compilation_unit.job_parse_module(scope, module));
			}
		});

		// launch semantic analysis of the builtin_prelude and builtin module before any root
		if self
			.get_or_analyze_module_sync(self.builtin_prelude_module)
			.and_then(|_| self.get_or_analyze_module_sync(self.builtin_module))
			.is_ok()
		{
			// great, start analysis of the runtime module, it contains the main() of the compilation unit
			// that will recursively trigger analysis if needed
			let compilation_unit = self.clone();
			rayon::spawn_fifo(move || {
				compilation_unit.get_or_analyze_module_sync(compilation_unit.std_rt_module).ok();
			})
		};

		// prepare codegen ctx
		let mut codegen = match self.build_args.backend {
			crate::Backend::Llvm => {
				cfg_select! {
					feature = "llvm" => {
						let llvm_ctx = Box::leak(Box::new(codegen::llvm::initialize(self.build_args)));
						CodegenLowerer::Llvm(codegen::llvm::Lowerer::new(llvm_ctx, self.as_ref(), self.build_args))
					}, _ => {
						unreachable!()
					}
				}
			},
		};

		loop {
			// while any task is referencing us loops
			if Arc::strong_count(&self) == 1 && self.codegen_tasks.is_empty() {
				break;
			}

			// prioritize codegen tasks
			while let Some((fun, vtir)) = self.codegen_tasks.pop() {
				profiling::scope!("codegen");
				codegen.lower_function(self.as_ref(), fun, &vtir, self.build_args);
			}

			match rayon::yield_now() {
				Some(_) => {},
				None => std::thread::yield_now(),
			}
		}

		// no more tasks...
		let modules = self.modules.read();

		// print any module errors if any
		if let Some(module) = self.failed_modules.lock().iter().next() {
			let module = &modules[module];
			let diagnostics = module.diagnostics.lock().clone();
			for diag in &diagnostics {
				let _ = DiagnosticWriter::new(diag, &modules, stderr, !launch_args.ui_testing).write();
			}

			return Err(BuildError::ParsingError);
		}

		// then sema errors
		let sema_errors = self.sema_errors.lock();
		if !sema_errors.is_empty() {
			for (module, errors) in sema_errors.iter() {
				let module = &modules[module];
				for diag in errors {
					let _ = DiagnosticWriter::new(diag, &modules, stderr, !launch_args.ui_testing).write();
				}
			}

			return Err(BuildError::SemanticError);
		}

		profiling::scope!("emit object");
		codegen.finish(self.build_args).map_err(|_err| BuildError::CodegenError)
	}

	/// Try to analyze the decl if needed. Blocks (cooperatively via rayon) until done.
	pub fn get_or_analyze_decl_value(
		self: &Arc<CompilationUnit>,
		decl_id: DeclId,
	) -> Result<Option<value::Index>, sema::AnalyzeError> {
		// already analysed
		if let Some(result) = self.decls.with_mut(|decls| {
			decls.get(decl_id).and_then(|decl| match &decl.analysis_state {
				DeclAnalysisState::Analysed { value } => Some(Ok(Some(*value))),
				DeclAnalysisState::Unanalysed { .. } | DeclAnalysisState::TypeKnown(..) => None,
				DeclAnalysisState::Failed(err) => Some(Err(*err)),
			})
		}) {
			return result;
		}

		let unit = SemaUnit::DeclValue(decl_id);
		self.analyze_sema_unit_if_needed(SemaUnit::DeclValue(decl_id));
		loop {
			if let Some(result) = self.decls.with_mut(|decls| {
				decls.get(decl_id).and_then(|decl| match &decl.analysis_state {
					DeclAnalysisState::Analysed { value } => Some(Ok(Some(*value))),
					DeclAnalysisState::Failed(err) => Some(Err(*err)),
					_ => None,
				})
			}) {
				return result;
			}
			rayon::yield_now();
		}
	}

	/// Try to analyze the value if needed..
	pub fn get_or_analyze_module_sync(
		self: &Arc<CompilationUnit>,
		module_id: ModuleId,
	) -> Result<value::Index, sema::AnalyzeError> {
		let module = self.modules.with(|modules| modules[module_id].clone());

		if module.vuir.get().is_none() {
			return Err(sema::AnalyzeError::AnalysisFailed);
		}

		loop {
			let claimed = module.sema_state.with_mut(|state| match state {
				ModuleAnalyzeState::Pending => {
					*state = ModuleAnalyzeState::InProgress;
					true
				},
				_ => false,
			});
			if claimed {
				match Self::job_analyze_module(self, module_id) {
					Ok(value) => {
						module.sema_state.with_mut(|state| *state = ModuleAnalyzeState::Done(value));
						return Ok(value);
					},
					Err(e) => {
						module.sema_state.with_mut(|state| *state = ModuleAnalyzeState::Failed);
						return Err(e);
					},
				}
			}
			match module.sema_state.lock().clone() {
				ModuleAnalyzeState::Pending => unreachable!(),
				ModuleAnalyzeState::InProgress => {
					match rayon::yield_now() {
						Some(_) => {},
						None => std::thread::yield_now(),
					}
					continue;
				},
				ModuleAnalyzeState::Done(v) => return Ok(v),
				ModuleAnalyzeState::Failed => return Err(sema::AnalyzeError::AnalysisFailed),
			}
		}
	}

	pub fn queue_runtime_function_analysis_if_needed(
		self: &Arc<CompilationUnit>,
		fun: value::Index,
	) {
		self.analyze_sema_unit_if_needed(SemaUnit::RuntimeFunc(fun));
	}

	/// Atomically look up or spawn the job for `unit`
	fn analyze_sema_unit_if_needed(
		self: &Arc<CompilationUnit>,
		unit: SemaUnit,
	) {
		let mut guard = self.sema_jobs.lock();
		match guard.get(&unit) {
			Some(_) => {},
			None => {
				guard.insert(unit, SemaJobState::Queued);
				drop(guard); // no need to lock the mutex further

				let compilation_unit = self.clone();
				rayon::spawn_fifo(move || {
					let value = compilation_unit.job_sema_analyze_unit(unit);
					compilation_unit.sema_jobs.with_mut(|jobs| {
						jobs.insert(unit, SemaJobState::Done);
					});
				});
			},
		}
	}

	// ================== Jobs functions

	fn job_parse_module(
		self: &Arc<CompilationUnit>,
		scope: &rayon::Scope,
		module_id: ModuleId,
	) {
		let module = self.modules.with(|modules| modules[module_id].clone());
		profiling::scope!("parse module", format!("{:?}", module.path).as_str());

		let source = {
			let absolute_path = &module.path.to_path(&self.cwd);
			let mut file = match File::open(absolute_path) {
				Ok(file) => file,
				Err(_) => {
					let mut diag = Diagnostic::error().with_message(format!("cannot open module `{}`", module.path));
					if let Some(imported_from) = module.first_imported_by {
						diag = diag.with_label(Label::primary().with_span(imported_from).with_message("imported here"));
					}
					module.diagnostics.with_mut(|diagnostics| diagnostics.push(diag));
					self.failed_modules.with_mut(|failed_modules| {
						failed_modules.push(module_id);
					});
					return;
				},
			};

			let mut buffer = String::new();
			if let Err(err) = file.read_to_string(&mut buffer) {
				let mut diag = Diagnostic::error().with_message(format!("cannot read module `{}`: {}", module.path, err));
				if let Some(imported_from) = module.first_imported_by {
					diag = diag.with_label(Label::primary().with_span(imported_from).with_message("imported here"));
				}
				module.diagnostics.with_mut(|diagnostics| diagnostics.push(diag));
				self.failed_modules.with_mut(|failed_modules| {
					failed_modules.push(module_id);
				});
				return;
			}

			if module_id == self.builtin_module {
				buffer.push_str(&self.inject_builtin_declarations());
			}

			// lexer expect NUL terminator to identify end of the file as it does not check length
			buffer.push('\0');
			buffer
		};

		// ast
		let ast = frontend::parser::Parser::new(&source, module_id).parse_module();

		// now to vuir
		let vuir = if let Ok(ast) = &ast {
			Some(ir::vuir::from_ast::to_vuir(self, &source, module_id, ast))
		} else {
			None
		};

		// have failed ? :(
		if ast.is_err() || vuir.as_ref().is_some_and(|vuir| vuir.is_err()) {
			self.failed_modules.with_mut(|failed_modules| {
				failed_modules.push(module_id);
			});
		}

		let parsed_vuir = match (ast, vuir) {
			(Ok(_ast), Some(Ok(vuir))) => {
				module.source.set(source).unwrap();
				// vuir is set later
				Some(vuir)
			},
			(Err(diags), _) | (_, Some(Err(diags))) => {
				module.source.set(source).unwrap();
				module.diagnostics.with_mut(|diagnostics| *diagnostics = diags);
				None
			},
			_ => unreachable!("module parse failed without diagnostics"),
		};

		if let Some(vuir) = parsed_vuir {
			// VUIR parsed, now parse imports and create their modules.
			for import in vuir
				.imports
				.iter()
				.filter_map(|import| match str::from_utf8(&import.path).unwrap() {
					// exclude imports that aren't files
					"root" => None,
					"std" => None,
					"builtin" => None,
					path => Some((path, import.span)),
				}) {
				let (import, import_span) = import;
				let import_module_path = module.path.parent().unwrap().join(import).normalize();

				let maybe_new_module = {
					let mut module_path_to_id = self.module_path_to_id.lock();
					if module_path_to_id.contains_key(&import_module_path) {
						None
					} else {
						let module = self.modules.with_mut(|modules| {
							let module = ArcModule::new(Module::new(
								import_module_path.clone(),
								Some(DiagSpan {
									module: module_id,
									span: import_span,
								}),
							));
							modules.push(module)
						});
						module_path_to_id.insert(import_module_path.clone(), module);
						Some(module)
					}
				};

				if let Some(module) = maybe_new_module {
					let compilation_unit = self.clone();
					scope.spawn(move |scope| compilation_unit.job_parse_module(scope, module));
				}
			}

			if self
				.build_args
				.dump_vuir
				.as_ref()
				.is_some_and(|regex| regex.is_match(module.path.as_str()))
			{
				println!("--- START OF VUIR DUMP OF {:?} ---", module.path);
				let stdout = std::io::stdout();
				let mut handle = stdout.lock();
				let _ = vuir.pretty_print(&mut handle);
				println!("--- END OF VUIR DUMP OF {:?} ---", module.path);
			}

			module.vuir.set(vuir).unwrap();
		}
	}

	fn job_analyze_module(
		self: &Arc<CompilationUnit>,
		module_id: ModuleId,
	) -> Result<value::Index, sema::AnalyzeError> {
		use ir::vuir;

		let mut main_fn_decl_id = None;
		let value = {
			let modules = self.modules.read();
			let module = &modules[module_id];
			profiling::scope!("analyze module", format!("{:?}", module.path).as_str());

			let Some(vuir) = module.vuir.get() else {
				return Err(sema::AnalyzeError::AnalysisFailed);
			};

			let module_namespace = self.namespaces.with_mut(|namespaces| {
				namespaces.push(Namespace::with_owner_type(self.values.common.generic_poison_t)) // TODO
			});

			// module has a decl
			let (module_decl_name, module_decl_id) = self.decls.with_mut(|decls| {
				let name = Intern::from(module.path.file_name().unwrap());
				(
					name,
					decls.push(Decl {
						name,
						module: module_id,
						namespace: module_namespace,
						analysis_state: DeclAnalysisState::Unanalysed {
							module: module_id,
							vuir_id: vuir::InstructionId::FILE_MODULE,
						},
					}),
				)
			});

			let vuir::Opcode::DeclStruct { decls, fields: _, .. } = &vuir.instructions[vuir::InstructionId::FILE_MODULE] else {
				unreachable!("module root must be a struct, other types are not supported")
			};

			{
				let mut sema = Sema::new(self, vuir, module_id, module_decl_id, None);
				let block = {
					sema.blocks.push(sema::Block {
						parent: None,
						namespace: module_namespace,
						instructions: bumpalo::collections::Vec::new_in(sema.instructions_payload_alloc),
						vuir_block: None,
						comptime: true,
						inlined: true,
						base_type_name: module_decl_name,
						decl_fn_params: Default::default(),
						handler_stack: vec![],
						capture_context: Default::default(),
					});
					sema::BlockId(sema.blocks.len() - 1)
				};
				sema.analyze_comptime_block(block, &[vuir::InstructionId::FILE_MODULE]).unwrap();

				// TODO(zino): this lookup path should be cleaned up.
				let value = sema.vuir_map[&vuir::InstructionId::FILE_MODULE].as_interned();

				// append builtin_prelude to namespace and also search for the main decl id
				self.namespaces.with_mut(|namespaces| {
					if module_id == self.builtin_prelude_module {
						return; // we are the builtin_prelude module !
					}

					let builtin_prelude_module = &modules[self.builtin_prelude_module];
					let builtin_prelude_module_ns = *builtin_prelude_module.namespace.get().unwrap();

					// SAFETY: tkt frere
					let [builtin_prelude_module_ns, module_namespace] =
						unsafe { namespaces.get_disjoint_unchecked_mut([builtin_prelude_module_ns, module_namespace]) };

					module_namespace.owner_type = value;

					for (k, v) in &builtin_prelude_module_ns.decls {
						module_namespace.decls.insert(*k, *v);
					}

					module_namespace.decls.insert(COMMON_INTERNS.self_ty_symbol, module_decl_id);

					// plus search main decl for std rt
					if module_id == self.std_rt_module {
						for (name, decl) in &module_namespace.decls {
							if *name == COMMON_INTERNS.main_symbol {
								main_fn_decl_id = Some(*decl);
								break;
							}
						}

						assert!(main_fn_decl_id.is_some());
					}
				});

				module
					.namespace
					.set(module_namespace)
					.unwrap_or_else(|_| unreachable!("module namespace was already published"));

				// store analysis
				self.decls.with_mut(|decls| {
					let mut decl = &mut decls[module_decl_id];
					decl.analysis_state = DeclAnalysisState::Analysed { value };
				});
				value
			}
		};

		// module has a main decl id, for now be lazy and analyse it even if its not the root module
		if let Some(main_decl_id) = main_fn_decl_id {
			let compilation_unit = self.clone();
			rayon::spawn_fifo(move || {
				let decl_value = compilation_unit.get_or_analyze_decl_value(main_decl_id).unwrap();
				let fun_decl = decl_value.unwrap();
				let fun_decl_key = compilation_unit.values.index_to_key(fun_decl).as_fn_decl();
				let fun_ty = fun_decl_key.ty;
				let fun = compilation_unit.values.intern_non_trivial(
					&value::Key::Fn(value::FnKey {
						ty: fun_ty,
						decl: fun_decl,
						comptime_args: &[],
						owner_decl: main_decl_id,
					}),
					value::Value::none(),
				);

				compilation_unit.queue_runtime_function_analysis_if_needed(fun);
			});
		}

		Ok(value)
	}

	fn job_sema_analyze_unit(
		self: &Arc<CompilationUnit>,
		unit: SemaUnit,
	) -> Result<Option<value::Index>, sema::AnalyzeError> {
		profiling::scope!("analyze sema unit", format!("{unit:?}").as_str());

		let value = match unit {
			SemaUnit::DeclValue(decl_id) => {
				let (name, module_id, vuir_id, namespace) = self.decls.with_mut(|decls| {
					let mut decl = &mut decls[decl_id];
					let DeclAnalysisState::Unanalysed {
						module: module_id,
						vuir_id,
					} = decl.analysis_state
					else {
						unreachable!("job_analyze_decl queued with already analysed decl {decl:?}")
					};
					(decl.name, module_id, vuir_id, decl.namespace)
				});

				let module = self.modules.with(|modules| modules[module_id].clone());
				let vuir = module.vuir.get().unwrap();

				// get the decl instruction to resolve its body
				let ir::vuir::Opcode::Declaration(decl_inst) = &vuir.instructions[vuir_id] else {
					unreachable!("{} must be a declaration", vuir_id);
				};

				let mut sema = Sema::new(self, vuir, module_id, decl_id, None);
				let block = {
					sema.blocks.push(sema::Block {
						parent: None,
						namespace,
						instructions: bumpalo::collections::Vec::new_in(sema.instructions_payload_alloc),
						vuir_block: None,
						comptime: true,
						inlined: false,
						base_type_name: name,
						decl_fn_params: Default::default(),
						handler_stack: vec![],
						capture_context: Default::default(),
					});
					sema::BlockId(sema.blocks.len() - 1)
				};
				let decl_val = match sema.analyze_comptime_block(block, decl_inst.value) {
					Ok(val) => val
						.unwrap_or_else(|| ir::vtir::InstructionRef::Interned(self.values.common.generic_poison_t))
						.as_interned(),
					Err(err) => {
						self.decls.with_mut(|decls| {
							decls[decl_id].analysis_state = DeclAnalysisState::Failed(err);
						});
						return Err(err);
					},
				};

				// store analysis
				self.decls.with_mut(|decls| {
					let mut decl = &mut decls[decl_id];
					decl.analysis_state = DeclAnalysisState::Analysed { value: decl_val };
				});
				Some(decl_val)
			},
			SemaUnit::RuntimeFunc(interned_fun) => {
				let fun = self.values.index_to_key(interned_fun).as_fn();
				let fun_decl = self.values.index_to_key(fun.decl).as_fn_decl();
				let fun_ty = self.values.index_to_key(fun.ty).as_type_fn();

				let module = self.modules.with(|modules| modules[fun_decl.func_decl_inst.module].clone());
				let vuir = module.vuir.get().unwrap();

				let ir::vuir::Opcode::DeclFn { body, params, builtin, .. } = &vuir.instructions[fun_decl.func_decl_inst.inst] else {
					unreachable!();
				};

				let (fn_decl_name, namespace) = self.decls.with_mut(|decls| {
					let decl = &decls[fun_decl.owner_decl];
					(decl.name, decl.namespace)
				});

				let body = {
					// collect fn params
					let mut sema = Sema::new(self, vuir, fun_decl.func_decl_inst.module, fun_decl.owner_decl, Some(interned_fun));
					let block = sema.blocks.push(sema::Block {
						namespace,
						parent: None,
						instructions: bumpalo::collections::Vec::new_in(sema.instructions_payload_alloc),
						vuir_block: None,
						comptime: false,
						inlined: false,
						base_type_name: fn_decl_name,
						decl_fn_params: Default::default(),
						handler_stack: vec![],
						capture_context: Default::default(),
					});

					// TODO(zino): don't use analyze comptime block...
					let _ = sema.analyze_comptime_block(block, params).unwrap();

					let owner_type = self.namespaces.with(|namespaces| namespaces[namespace].owner_type);
					let source_param_offset = 0;

					// instantiate fn params, only for runtime params
					let regular_param_count = sema.blocks[block].decl_fn_params.len();
					for (i, param) in sema.blocks[block].decl_fn_params.clone().iter().enumerate() {
						let physical_i = source_param_offset + i;
						if fun_ty.comptime_params[physical_i] {
							sema.vuir_map.insert(
								param.vuir_id,
								ir::vtir::InstructionRef::Interned(fun.comptime_args[physical_i].unwrap()),
							);
						} else {
							let param_ty = fun_ty.params[physical_i];
							let vtir_inst = sema.inst(block, ir::vtir::Opcode::FnParam {
								name: param.name,
								ty: param_ty, /* don't take fn block param type but
								               * instantiated fn param type */
							});
							sema.vuir_map.insert(param.vuir_id, vtir_inst);

							// Register linear params for tracking
							if self.values.type_is_linear(param_ty) {
								sema.register_linear_param(vtir_inst, param.name, param_ty, param.span);
							}
						}
					}

					if let Some(builtin) = builtin {
						unreachable!(
							"a builtin call must not be analysed with a SemaUnit as we doesn't track the caller span, they are made to be \
							 always inline"
						);
					} else if !fun_ty.external {
						sema.analyze_fn_body(block, body, fun_ty.ret_ty)?;
					}

					sema.finish(block)
				};

				if self.build_args.dump_vtir {
					println!("--- START OF VTIR DUMP OF {fn_decl_name} ---");
					let stdout = std::io::stdout();
					let mut handle = stdout.lock();
					let _ = body.pretty_print(&mut handle);
					println!("--- END OF VTIR DUMP OF {fn_decl_name} ---");
				}

				// kickoff codegen for this function
				self.codegen_tasks.push((interned_fun, body));

				None
			},
		};

		Ok(value)
	}
}
