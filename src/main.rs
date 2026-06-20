#![feature(allocator_api)]
#![feature(btreemap_alloc)]
#![feature(core_intrinsics)]
#![feature(duration_millis_float)]
#![feature(exact_div)]
#![feature(exclusive_wrapper)]
#![feature(explicit_tail_calls)]
#![feature(f128)]
#![feature(f16)]
#![feature(get_disjoint_mut_helpers)]
#![feature(int_from_ascii)]
#![feature(iter_collect_into)]
#![feature(iterator_try_collect)]
#![feature(likely_unlikely)]
#![feature(lock_value_accessors)]
#![feature(loop_match)]
#![feature(nonpoison_mutex)]
#![feature(nonpoison_rwlock)]
#![feature(pointer_is_aligned_to)]
#![feature(rustc_attrs)]
#![feature(slice_ptr_get)]
#![feature(sync_nonpoison)]
#![feature(unsafe_fields)]
#![allow(arithmetic_overflow, incomplete_features, internal_features, unused, unsafe_op_in_unsafe_fn)]
#![deny(clippy::disallowed_methods)]

use std::{
	fs::File,
	io::{
		Read,
		Write,
	},
	path::{
		Path,
		PathBuf,
	},
	process::Output,
	sync::Arc,
	time::{
		Duration,
		Instant,
	},
};

use argh::FromArgs;
use cfg_if::cfg_if;
use regex_lite::Regex;
use relative_path::{
	PathExt,
	RelativePathBuf,
};

use crate::{
	common::diagnostic::DiagnosticWriter,
	compile_unit::CompilationUnit,
	ir::vtir,
};

mod codegen;
mod common;
mod compile_unit;
mod frontend;
mod int;
mod ir;
mod value;

#[cfg(test)]
mod tests;

use mimalloc::MiMalloc;

#[global_allocator]
#[cfg(not(miri))]
static GLOBAL: MiMalloc = MiMalloc;

/// vifc command line.
#[derive(Debug, FromArgs)]
pub struct Args {
	/// print verbose compilation timings
	#[argh(switch)]
	verbose_timings: bool,

	/// disable colors and use stable diagnostics for UI tests
	#[argh(switch)]
	ui_testing: bool,

	#[argh(subcommand)]
	command: Command,
}

/// compiler command.
#[derive(Debug, FromArgs)]
#[argh(subcommand)]
pub enum Command {
	Build(Build),
	Run(Run),
}

/// build/run options.
#[derive(Clone, Debug, Default, FromArgs)]
#[argh(subcommand, name = "build")]
pub struct Build {
	/// file to build
	#[argh(positional)]
	file: String,

	/// code generation backend: llvm or cranelift
	#[argh(option, default = "Backend::Llvm", from_str_fn(parse_backend))]
	backend: Backend,

	/// optimization level, from 0 to 3
	#[argh(option, short = 'O', default = "0")]
	opt: u8,

	/// emit debug information
	#[argh(switch, short = 'D')]
	debug_info: bool,

	/// dump the VUIR of all modules, optionally filtered by module path regex
	#[argh(option)]
	dump_vuir: Option<Regex>,

	/// dump VTIR
	#[argh(switch)]
	dump_vtir: bool,

	/// dump LLVM IR
	#[argh(switch)]
	dump_llvm_ir: bool,

	/// dump LLVM timings
	#[argh(switch)]
	dump_llvm_timings: bool,

	/// dump assembly
	#[argh(switch)]
	dump_asm: bool,

	/// library search path passed to the linker
	#[argh(option, short = 'L')]
	link_lib_paths: Vec<String>,

	/// library to link against
	#[argh(option, short = 'l')]
	link_libs: Vec<String>,

	/// on Windows, use WINDOWS subsystem instead of CONSOLE
	#[argh(switch)]
	windows_subsystem: bool,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub enum Backend {
	#[default]
	Llvm,
}

fn parse_backend(value: &str) -> Result<Backend, String> {
	match value {
		"llvm" => Ok(Backend::Llvm),
		_ => Err(format!("unknown backend `{value}`, expected `llvm`")),
	}
}

#[derive(Clone, Debug, Default)]
pub struct Run {
	build: Build,
}

impl argh::SubCommand for Run {
	const COMMAND: &'static argh::CommandInfo = &argh::CommandInfo {
		name: "run",
		short: &'\0',
		description: "build and run a source file",
	};
}

impl FromArgs for Run {
	fn from_args(
		command_name: &[&str],
		args: &[&str],
	) -> Result<Self, argh::EarlyExit> {
		Build::from_args(command_name, args).map(|build| Self { build })
	}

	fn redact_arg_values(
		command_name: &[&str],
		args: &[&str],
	) -> Result<Vec<String>, argh::EarlyExit> {
		Build::redact_arg_values(command_name, args)
	}
}

#[derive(Debug)]
pub enum Error {
	BuildError(BuildError),
	RunError(std::io::Error),
}

#[derive(Debug)]
pub enum BuildError {
	InvalidStdLib,
	LexingError,
	ParsingError,
	SemanticError,
	CodegenError,
	InternalCompilerError,
	LinkError,
}

fn build(
	args: &Args,
	stdout: &mut dyn Write,
	stderr: &mut dyn Write,
	build: &'static Build,
) -> Result<(), BuildError> {
	let total_duration = Instant::now();

	let compilation_duration = Instant::now();

	let cwd = std::env::current_dir().unwrap();

	let root_module_path = if let file = PathBuf::from(&build.file)
		&& file.is_absolute()
	{
		let cwd = cwd.canonicalize().unwrap();
		file.relative_to(&cwd).unwrap()
	} else {
		cwd.join(&build.file).relative_to(&cwd).unwrap()
	};
	let compilation_unit = CompilationUnit::new(build, cwd, &root_module_path);

	let obj = compilation_unit.compile(args, stderr)?;

	let compilation_duration = compilation_duration.elapsed();

	let out_stem = Path::new(&build.file).file_stem().and_then(|s| s.to_str()).unwrap_or("vifc-out");
	let out_dir = PathBuf::from("target-vifc");
	let obj_path = out_dir.join(format!("{out_stem}.obj"));
	let out_exe_path = out_dir.join(format!("{out_stem}.exe"));

	// Write object file
	{
		profiling::scope!("write object file");

		std::fs::create_dir_all(obj_path.parent().unwrap());

		let mut obj_file = File::options().create(true).write(true).truncate(true).open(&obj_path).unwrap();
		obj_file.write_all(obj.as_slice()).unwrap();
	}

	// Link the object file with the runtime and extra libraries
	let link_duration = Instant::now();
	{
		profiling::scope!("linking");

		#[cfg(target_os = "windows")]
		fn link_windows_lld_link(
			obj_path: &Path,
			out_exe_path: &Path,
			extra_lib_paths: &[String],
			extra_libs: &[String],
			windows_subsystem: bool,
		) -> std::io::Result<Output> {
			let vc_and_windows_sdk = thound::find_vc_and_windows_sdk().expect("failed to find Windows SDK and/or VC Runtime");

			let winsdk = vc_and_windows_sdk.sdk.unwrap();
			let toolchain = vc_and_windows_sdk.toolchain.unwrap();
			let subsystem = if windows_subsystem {
				"/SUBSYSTEM:windows"
			} else {
				"/SUBSYSTEM:console"
			};
			let entry = if windows_subsystem {
				"/ENTRY:WinMainCRTStartup"
			} else {
				"/ENTRY:mainCRTStartup"
			};
			let mut cmd = std::process::Command::new(toolchain.exe_path.join("link.exe"));
			cmd.arg("/NOLOGO");
			cmd.arg(format!("/OUT:{}", out_exe_path.display()));
			cmd.arg(format!("/PDB:{}", out_exe_path.with_extension("pdb").display()));
			cmd.args([
				entry,
				subsystem,
				"/DEBUG:FULL",
				&format!("/LIBPATH:{}", winsdk.ucrt_lib_path.to_string_lossy()),
				&format!("/LIBPATH:{}", winsdk.um_lib_path.to_string_lossy()),
				&format!("/LIBPATH:{}", toolchain.lib_path.to_string_lossy()),
			]);
			for path in extra_lib_paths {
				cmd.arg(format!("/LIBPATH:{path}"));
			}
			cmd.args(["kernel32.lib", "libcmt.lib", "libvcruntime.lib", "libucrt.lib"]);
			for lib in extra_libs {
				cmd.arg(lib);
			}
			cmd.arg(obj_path.to_string_lossy().as_ref());
			cmd.args(["/OPT:REF", "/OPT:ICF", "/NODEFAULTLIB"]);
			cmd.output()
		}

		#[cfg(target_os = "macos")]
		fn link_macos(
			out_exe_path: &Path,
			obj_path: &Path,
		) -> std::io::Result<Output> {
			std::process::Command::new("cc")
				.args(["-o", &out_exe_path.to_string_lossy(), obj_path.to_string_lossy().as_ref()])
				.output()
		}

		#[cfg(target_os = "linux")]
		fn link_linux(
			out_exe_path: &Path,
			obj_path: &Path,
		) -> std::io::Result<Output> {
			std::process::Command::new("cc")
				.args(["-o", &out_exe_path.to_string_lossy(), obj_path.to_string_lossy().as_ref()])
				.output()
		}

		let output = {
			cfg_if! {
				if #[cfg(target_os = "windows")] {
					link_windows_lld_link(
						&obj_path,
						&out_exe_path,
						&build.link_lib_paths,
						&build.link_libs,
						build.windows_subsystem,
					)
				} else if #[cfg(target_os = "macos")] {
					link_macos(&out_exe_path, &obj_path)
				} else if #[cfg(target_os = "linux")] {
					link_linux(&out_exe_path, &obj_path)
				} else {
					unreachable!("unsupported target_os")
				}
			}
		}
		.unwrap();

		if !output.stdout.is_empty() {
			println!("{}", String::from_utf8(output.stdout).unwrap());
		}
		if !output.stderr.is_empty() {
			eprintln!("{}", String::from_utf8(output.stderr).unwrap());
		}
		if !output.status.success() {
			return Err(BuildError::LinkError);
		}
	}
	let link_duration = link_duration.elapsed();

	println!(
		"
STAGE        ┃    DURATION
━━━━━━━━━━━━━╋━━━━━━━━━━━━
COMPILE      ┃ {:>8.2} ms
LINKING      ┃ {:>8.2} ms
TOTAL        ┃ {:>8.2} ms
",
		compilation_duration.as_millis_f32(),
		link_duration.as_millis_f32(),
		(compilation_duration + link_duration).as_millis_f32(),
	);

	Ok(())
}

fn main() -> Result<(), Error> {
	// SAFETY: Enabling the diagnostic handler is process-global and done once at startup.
	#[cfg(all(debug_assertions, target_os = "windows", feature = "windows-stackoverflow-backtrace"))]
	unsafe {
		w_boson::enable()
	};

	let args = Box::leak(Box::new(parse_args_from_env()));
	run_from_args(args, &mut std::io::stdout(), &mut std::io::stderr())
}

fn parse_args_from_env() -> Args {
	match parse_args_from_iter(std::env::args()) {
		Ok(args) => args,
		Err(err) => {
			if err.status.is_ok() {
				print!("{}", err.output);
				std::process::exit(0);
			} else {
				eprint!("{}", err.output);
				std::process::exit(1);
			}
		},
	}
}

pub fn parse_args_from_iter<I, S>(args: I) -> Result<Args, argh::EarlyExit>
where
	I: IntoIterator<Item = S>,
	S: Into<String>,
{
	let args = args.into_iter().map(Into::into).collect::<Vec<String>>();
	let command_name = [args[0].as_str()];
	let rest = args[1..].iter().map(String::as_str).collect::<Vec<_>>();
	Args::from_args(&command_name, &rest)
}

pub fn run_from_args(
	args: &'static Args,
	stdout: &mut (dyn Write + Send),
	stderr: &mut (dyn Write + Send),
) -> Result<(), Error> {
	let pool = rayon::ThreadPoolBuilder::new()
		.num_threads(
				std::thread::available_parallelism()
				.unwrap_or(1.try_into().unwrap())
				.get(),
		)
		.use_current_thread() // we want the main thread to participate in the pool
		.build()
		.unwrap();

	pool.install(|| match &args.command {
		Command::Build(b) => build(args, stdout, stderr, b).map_err(Error::BuildError),
		Command::Run(run) => {
			let b = &run.build;
			build(args, stdout, stderr, b).map_err(Error::BuildError)?;

			let out_stem = Path::new(&b.file).file_stem().and_then(|s| s.to_str()).unwrap_or("vifc-out");
			let out_exe = PathBuf::from("target-vifc").join(format!("{out_stem}.exe"));

			let exit_status = if args.ui_testing {
				// In test mode capture stdout/stderr so the test framework can match them
				let output = std::process::Command::new(&out_exe).output().map_err(|e| {
					writeln!(stderr, "failed to run built executable (error: {e})");
					Error::RunError(e)
				})?;
				stdout.write_all(&output.stdout).ok();
				stderr.write_all(&output.stderr).ok();
				output.status
			} else {
				// Normal mode: inherit stdio so output is live
				std::process::Command::new(&out_exe).status().map_err(|e| {
					writeln!(stderr, "failed to run built executable (error: {e})");
					Error::RunError(e)
				})?
			};

			if exit_status.success() {
				Ok(())
			} else {
				writeln!(
					stderr,
					"process didn't exit successfully: {out_stem}.exe (error code: {:?})",
					exit_status.code()
				);
				Err(Error::RunError(std::io::ErrorKind::Other.into()))
			}
		},
	})
}
