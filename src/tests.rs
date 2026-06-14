use std::path::PathBuf;

use pretty_assertions::assert_eq;
use rstest::rstest;

use crate::{
	parse_args_from_iter,
	run_from_args,
};

// NOTE: rstest expands the fixture file list when this module is compiled.
#[rstest]
fn for_each_file(
	#[files("**/*.vif")]
	#[base_dir = "src/tests"]
	path: PathBuf
) {
	let content = std::fs::read_to_string(&path).unwrap();
	let fixture = TestFixture::parse(&content);
	if fixture.arch.as_deref().is_some_and(|arch| arch != std::env::consts::ARCH) {
		return;
	}

	let mut stdout = Vec::new();
	let mut stderr = Vec::new();
	let _ = {
		let mut args = vec![];
		args.push(""); // arg[0] = program name
		args.push("--ui-testing");
		for arg in &fixture.args {
			args.push(arg);
		}
		let path = path.to_string_lossy();
		args.push(&path);

		let args = Box::leak(Box::new(parse_args_from_iter(args).unwrap()));
		run_from_args(args, &mut stdout, &mut stderr)
	};

	let stdout_output = String::from_utf8(stdout).unwrap().replace("\r\n", "\n");
	if let Some(expected_stdout) = fixture.expected_stdout {
		assert_eq!(
			expected_stdout.trim_end(),
			stdout_output.trim_end(),
			"stdout mismatch for {}",
			path.to_string_lossy(),
		);
	} else {
		assert!(
			stdout_output.is_empty(),
			"expected no stdout but one is present:\n{stdout_output:?}"
		);
	}

	let stderr_output = String::from_utf8(stderr).unwrap();
	if let Some(expected_stderr) = fixture.expected_stderr {
		assert_eq!(expected_stderr, stderr_output, "stderr mismatch for {}", path.to_string_lossy(),);
	} else {
		assert!(
			stderr_output.is_empty(),
			"expected no stderr but one is present:\n{stderr_output:?}"
		);
	}
}

#[derive(PartialEq, Debug, Default)]
pub struct TestFixture {
	pub args: Vec<String>,
	pub arch: Option<String>,
	pub expected_stdout: Option<String>,
	pub expected_stderr: Option<String>,
}

impl TestFixture {
	pub fn parse(content: &str) -> Self {
		let mut fixture = TestFixture::default();
		let mut current_section: Option<String> = None;
		let mut section_lines: Vec<String> = Vec::new();

		for line in content.lines() {
			let Some(line) = line.strip_prefix("//") else {
				// close current section if any
				if let Some(section) = current_section.take() {
					fixture.set_section(&section, &section_lines.join("\n"));
					section_lines.clear();
				}
				continue;
			};

			// check if this is a directive comment
			if let Some(directive) = line.strip_prefix("@") {
				// if we were collecting a section, finalize it
				if let Some(section) = current_section.take() {
					fixture.set_section(&section, &section_lines.join("\n"));
					section_lines.clear();
				}

				// parse the new directive
				if let Some((key, value)) = directive.trim_start().split_once(':') {
					let key = key.trim();
					let value = value.trim();

					match key {
						"args" => {
							fixture.args = value.split_whitespace().map(String::from).collect();
						},
						"arch" => fixture.arch = Some(value.to_string()),
						"stdout" | "stderr" => {
							current_section = Some(key.to_string());
						},
						_ => {},
					}
				}
			} else if current_section.is_some() {
				let line = line.strip_prefix(" ").unwrap_or(line);
				section_lines.push(line.to_string());
			}
		}

		// finalize any remaining section
		if let Some(section) = current_section {
			fixture.set_section(&section, &section_lines.join("\n"));
		}

		fixture
	}

	fn set_section(
		&mut self,
		section: &str,
		content: &str,
	) {
		match section {
			"stdout" => self.expected_stdout = Some(content.to_string()),
			"stderr" => self.expected_stderr = Some(content.to_string()),
			_ => {},
		}
	}
}

#[test]
fn test_parse_basic_fixture() {
	let source = r#"fn main() void {
    if false {
        // noop
    } else {
        var a: i32 = false;
    }
}
//@ args: build
//@ stderr:
// error: expected type `i32`, found `bool`
// 5 |         var a: i32 = false;
//   |                      ^^^^^
"#;

	let fixture = TestFixture::parse(source);

	assert_eq!(fixture.args, vec!["build"]);
	assert_eq!(
		fixture.expected_stderr,
		Some("error: expected type `i32`, found `bool`\n5 |         var a: i32 = false;\n  |                      ^^^^^".to_string())
	);
	assert_eq!(fixture.expected_stdout, None);
}

#[test]
fn test_parse_with_stdout() {
	let source = r#"//@ args: run
//@ stdout:
// Hello, World!
// Program finished
fn main() void {
    print("Hello, World!");
}
"#;

	let fixture = TestFixture::parse(source);

	assert_eq!(fixture.args, vec!["run"]);
	assert_eq!(fixture.expected_stdout, Some("Hello, World!\nProgram finished".to_string()));
}

#[test]
fn test_empty_fixture() {
	let source = r#"fn main() void {
    // This is just a regular comment
    var x = 42;
}
"#;

	let fixture = TestFixture::parse(source);

	assert_eq!(fixture.args, Vec::<String>::new());
	assert_eq!(fixture.expected_stdout, None);
	assert_eq!(fixture.expected_stderr, None);
}

#[test]
fn test_multiple_args() {
	let source = r#"//@ args: build --release --verbose
fn main() void {}
"#;

	let fixture = TestFixture::parse(source);

	assert_eq!(fixture.args, vec!["build", "--release", "--verbose"]);
}
