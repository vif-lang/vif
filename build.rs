use std::{
	env,
	path::{
		Path,
		PathBuf,
	},
};

fn main() {
	println!("cargo:rerun-if-changed=build.rs");

	#[cfg(test)]
	{
		println!("cargo:rerun-if-changed=src/tests");
	}

	let profile = env::var("PROFILE").unwrap_or_default();
	let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set for build scripts");

	// Find the top-level output directory.
	let output_dir = {
		let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set for build scripts"));
		let mut sub_path = out_dir.as_path();
		let mut output = None;

		while let Some(parent) = sub_path.parent() {
			if parent.ends_with(&profile) {
				output = Some(parent.to_owned());
				break;
			}
			sub_path = parent;
		}

		output.expect("Unable to determine build output directory from OUT_DIR")
	};

	let std_dir = PathBuf::from(manifest_dir).join("std");
	let std_dir_out = output_dir.join("std");
	let std_dir_deps = output_dir.join("deps").join("std");

	symlink_or_copy(&std_dir, std_dir_out);
	symlink_or_copy(&std_dir, std_dir_deps);
}

fn symlink_or_copy(
	src: impl AsRef<Path>,
	dst: impl AsRef<Path>,
) {
	let src = src.as_ref();
	let dst = dst.as_ref();

	#[cfg(windows)]
	{
		if dst.exists() {
			// If the directory already exists and is a symlink, we can assume it's correct and do nothing.
			if dst.is_symlink() {
				return;
			} else {
				// Otherwise, remove the existing directory to replace it with a symlink.
				std::fs::remove_dir_all(dst).expect("Failed to remove existing std directory in output directory");
			}
		}

		use std::os::windows::fs::symlink_dir;

		// First try to symlink the std directory to avoid copying files.
		if symlink_dir(src, dst).is_err() {
			println!("cargo:rerun-if-changed=std/");

			for entry in walk_dir(&src).expect("Failed to read std directory") {
				let relative_path = entry.strip_prefix(src).unwrap();
				let target_path = dst.join(relative_path);

				if let Some(parent) = target_path.parent()
					&& !parent.exists()
				{
					std::fs::create_dir_all(parent).expect("Failed to create parent directory for std file in output directory");
				}

				std::fs::copy(&entry, &target_path).expect("Failed to copy std file to output directory");
			}
		}
	}

	#[cfg(not(windows))]
	{
		use std::os::unix::fs::symlink;

		if dst.exists() {
			// If the directory already exists and is a symlink, we can assume it's correct and do nothing.
			if dst.is_symlink() {
				return;
			} else {
				// Otherwise, remove the existing directory to replace it with a symlink.
				std::fs::remove_dir_all(&dst).expect("Failed to remove existing std directory in output directory");
			}
		}

		symlink(&src, &dst).expect("Failed to create symlink for std directory in output directory");
	}
}

fn walk_dir(root: impl AsRef<Path>) -> Result<impl Iterator<Item = PathBuf>, std::io::Error> {
	let root_path = root.as_ref();

	let entries = match std::fs::read_dir(root_path) {
		Ok(mut dir) => dir.try_fold(Vec::with_capacity(32), |mut acc, entry| match entry {
			Ok(entry) => {
				let path = entry.path();

				if path.is_dir() {
					acc.extend(walk_dir(path)?);
				} else {
					acc.push(path);
				}

				Ok(acc)
			},
			Err(e) => Err(std::io::Error::new(
				e.kind(),
				format!("Error reading directory entry in '{}': {e}", root_path.display()),
			)),
		}),
		Err(e) => Err(std::io::Error::new(
			e.kind(),
			format!("Error reading directory '{}': {e}", root_path.display()),
		)),
	}?;

	Ok(entries.into_iter())
}
