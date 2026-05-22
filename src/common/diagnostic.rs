use std::{
	cmp::Ordering,
	collections::BTreeMap,
	fmt::{
		Display,
		Formatter,
	},
	io::Write,
};

use owo_colors::{
	OwoColorize,
	Style,
};
use sorted_insert::SortedInsertBy;

use crate::{
	common::{
		IndexVec,
		span::Span,
	},
	compile_unit::module::{
		ArcModule,
		ModuleId,
	},
};

/// Wraps a span alongside its originating module to support inter-module diagnostics.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct DiagSpan {
	pub module: ModuleId,
	pub span: Span,
}

impl DiagSpan {
	pub fn module_content<'a>(
		&self,
		modules: &'a IndexVec<ModuleId, ArcModule>,
	) -> &'a str {
		modules[self.module].source.get().unwrap().as_str()
	}

	pub fn start_line_col(
		&self,
		modules: &IndexVec<ModuleId, ArcModule>,
	) -> (usize, usize) {
		self.span.start_line_col(self.module_content(modules))
	}

	pub fn end_line_col(
		&self,
		modules: &IndexVec<ModuleId, ArcModule>,
	) -> (usize, usize) {
		self.span.end_line_col(self.module_content(modules))
	}
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Severity {
	Error,
	Warning,
	Note,
}

impl Display for Severity {
	fn fmt(
		&self,
		f: &mut Formatter<'_>,
	) -> std::fmt::Result {
		match self {
			Self::Error => write!(f, "error"),
			Self::Warning => write!(f, "warning"),
			Self::Note => write!(f, "note"),
		}
	}
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub enum LabelType {
	Primary,
	Secondary,
}

#[derive(Clone, PartialEq, Debug)]
pub struct Label {
	pub ty: LabelType,
	pub message: String,
	pub span: Option<DiagSpan>,
}

impl Label {
	pub fn new(ty: LabelType) -> Self {
		Self {
			ty,
			message: String::new(),
			span: None,
		}
	}

	pub fn primary() -> Self {
		Self::new(LabelType::Primary)
	}

	pub fn secondary() -> Self {
		Self::new(LabelType::Secondary)
	}

	pub fn with_message<S: AsRef<str>>(
		mut self,
		message: S,
	) -> Self {
		self.message = message.as_ref().to_string();
		self
	}

	pub fn with_span(
		mut self,
		span: DiagSpan,
	) -> Self {
		self.span = Some(span);
		self
	}
}

#[derive(Clone, PartialEq, Debug)]
pub struct Diagnostic {
	pub severity: Severity,
	pub message: String,
	pub code: Option<String>,
	pub labels: Vec<Label>,
	pub notes: Vec<String>,
	#[cfg(all(not(test), debug_assertions))]
	pub location: &'static std::panic::Location<'static>,
}

impl Diagnostic {
	#[inline(always)]
	#[cfg_attr(debug_assertions, track_caller)]
	pub fn new(severity: Severity) -> Self {
		Self {
			severity,
			message: String::new(),
			code: None,
			labels: Vec::new(),
			notes: vec![],
			#[cfg(all(not(test), debug_assertions))]
			location: std::panic::Location::caller(),
		}
	}

	#[inline(always)]
	#[cfg_attr(debug_assertions, track_caller)]
	pub fn error() -> Self {
		Self::new(Severity::Error)
	}

	#[inline(always)]
	#[cfg_attr(debug_assertions, track_caller)]
	pub fn warning() -> Self {
		Self::new(Severity::Warning)
	}

	#[inline(always)]
	pub fn with_code(
		mut self,
		code: &str,
	) -> Self {
		self.code = Some(code.to_string());
		self
	}

	#[inline(always)]
	pub fn with_message<S: AsRef<str>>(
		mut self,
		message: S,
	) -> Self {
		self.message = message.as_ref().to_string();
		self
	}

	#[inline(always)]
	pub fn with_label(
		mut self,
		label: Label,
	) -> Self {
		self.labels.sorted_insert_by(label, |a, b| a.span.le(&b.span));
		self
	}

	#[inline(always)]
	pub fn with_note<S: AsRef<str>>(
		mut self,
		note: S,
	) -> Self {
		self.notes.push(note.as_ref().to_string());
		self
	}
}

pub struct DiagnosticFormatterStyles {
	header_error: Style,
	header_warning: Style,
	header_note: Style,
	header_message: Style,
	primary_label: Style,
	secondary_label: Style,
	border: Style,
	#[cfg(all(not(test), debug_assertions))]
	debug_info: Style,
}

impl Default for DiagnosticFormatterStyles {
	fn default() -> Self {
		Self {
			header_error: Style::new().red().bold(),
			header_warning: Style::new().yellow().bold(),
			header_note: Style::new().cyan().bold(),
			header_message: Style::new().white().bold(),
			primary_label: Style::new().red().bold(),
			secondary_label: Style::new().cyan().bold(),
			border: Style::new().cyan().bold(),
			#[cfg(all(not(test), debug_assertions))]
			debug_info: Style::new().magenta().bold(),
		}
	}
}

impl DiagnosticFormatterStyles {
	fn header(
		&self,
		severity: Severity,
	) -> Style {
		match severity {
			Severity::Error => self.header_error,
			Severity::Warning => self.header_warning,
			Severity::Note => self.header_note,
		}
	}
}

fn line_number_width(mut value: usize) -> usize {
	(value.checked_ilog10().unwrap_or(0) + 1) as usize
}

fn cmp_labels_for_render(
	a: &Label,
	b: &Label,
) -> Ordering {
	let a_span = a.span.expect("label span should be filtered before sorting");
	let b_span = b.span.expect("label span should be filtered before sorting");

	let a_secondary = a.ty == LabelType::Secondary;
	let b_secondary = b.ty == LabelType::Secondary;

	match (a_secondary, b_secondary) {
		(true, true) | (false, false) => a_span.cmp(&b_span),
		(false, true) => {
			if a_span <= b_span {
				Ordering::Less
			} else {
				Ordering::Greater
			}
		},
		(true, false) => {
			if a_span < b_span {
				Ordering::Less
			} else {
				Ordering::Greater
			}
		},
	}
}

struct RenderLabel<'a> {
	label: &'a Label,
	span: DiagSpan,
	start_line: usize,
	start_col: usize,
	end_line: usize,
	end_col: usize,
}

pub struct DiagnosticWriter<'a> {
	diagnostic: &'a Diagnostic,
	modules: &'a IndexVec<ModuleId, ArcModule>,
	writer: &'a mut dyn Write,
	styles: DiagnosticFormatterStyles,
	use_color: bool,
}

impl<'a> DiagnosticWriter<'a> {
	pub fn new(
		diagnostic: &'a Diagnostic,
		modules: &'a IndexVec<ModuleId, ArcModule>,
		writer: &'a mut dyn Write,
		use_color: bool,
	) -> Self {
		Self {
			diagnostic,
			modules,
			writer,
			styles: DiagnosticFormatterStyles::default(),
			use_color,
		}
	}

	fn write_styled(
		&mut self,
		text: impl Display,
		style: Style,
	) -> std::io::Result<()> {
		if self.use_color {
			write!(self.writer, "{}", text.style(style))
		} else {
			write!(self.writer, "{text}")
		}
	}

	fn write_line_number(
		&mut self,
		line: usize,
	) -> std::io::Result<()> {
		let border = self.styles.border;
		self.write_styled(line, border)
	}

	fn write_gutter(&mut self) -> std::io::Result<()> {
		let border = self.styles.border;
		self.write_styled(" | ", border)
	}

	fn write_label(
		&mut self,
		label: &Label,
		from_col: usize,
		len_col: usize,
	) -> std::io::Result<()> {
		let caret = match label.ty {
			LabelType::Primary => "^",
			LabelType::Secondary => "-",
		};

		let style = match label.ty {
			LabelType::Primary => self.styles.primary_label,
			LabelType::Secondary => self.styles.secondary_label,
		};

		if label.message.is_empty() {
			write!(self.writer, "{}", " ".repeat(from_col))?;
			self.write_styled(caret.repeat(len_col), style)?;
			writeln!(self.writer)?;
		} else {
			write!(self.writer, "{}", " ".repeat(from_col))?;
			self.write_styled(caret.repeat(len_col), style)?;
			writeln!(self.writer, " {}", label.message)?;
		}
		Ok(())
	}

	fn write_padding(
		&mut self,
		len: usize,
	) -> std::io::Result<()> {
		write!(self.writer, "{}", " ".repeat(len))
	}

	fn write_break(&mut self) -> std::io::Result<()> {
		let border = self.styles.border;
		self.write_styled("...", border)?;
		writeln!(self.writer)
	}

	fn write_empty_gutter(
		&mut self,
		padding: usize,
	) -> std::io::Result<()> {
		self.write_padding(padding)?;
		let border = self.styles.border;
		self.write_styled(" |", border)?;
		writeln!(self.writer)
	}

	fn write_location_line(
		&mut self,
		line: usize,
		col: usize,
		padding: usize,
		filename: &str,
	) -> std::io::Result<()> {
		self.write_padding(padding)?;
		let border = self.styles.border;
		self.write_styled(format_args!(" --> {filename}:{line}:{col}"), border)?;
		writeln!(self.writer)
	}

	fn write_header(&mut self) -> std::io::Result<()> {
		let header_style = self.styles.header(self.diagnostic.severity);

		match self.diagnostic.code.as_deref() {
			None => self.write_styled(self.diagnostic.severity, header_style)?,
			Some(code) => self.write_styled(format_args!("{}[{code}]", self.diagnostic.severity), header_style)?,
		}

		let header_message = self.styles.header_message;
		self.write_styled(format_args!(": {}", self.diagnostic.message), header_message)?;
		writeln!(self.writer)
	}

	pub fn write(mut self) -> std::io::Result<()> {
		self.write_header()?;

		let mut sorted_labels: Vec<&Label> = self.diagnostic.labels.iter().filter(|label| label.span.is_some()).collect();
		sorted_labels.sort_by(|a, b| cmp_labels_for_render(a, b));

		if sorted_labels.is_empty() {
			return Ok(());
		}

		let mut rendered_labels = Vec::with_capacity(sorted_labels.len());
		let mut max_line_number = 1;
		for label in sorted_labels {
			let span = label.span.expect("label span should be filtered before render");
			let content = span.module_content(self.modules);
			let (start_line, start_col) = span.span.start_line_col(content);
			let (end_line, end_col) = span.span.end_line_col(content);
			max_line_number = max_line_number.max(start_line.max(end_line));

			rendered_labels.push(RenderLabel {
				label,
				span,
				start_line,
				start_col,
				end_line,
				end_col,
			});
		}

		let padding = line_number_width(max_line_number);

		let loc_label = rendered_labels
			.iter()
			.find(|l| l.label.ty == LabelType::Primary)
			.unwrap_or(&rendered_labels[0]);
		let loc_filename = self.modules[loc_label.span.module].path.as_str();
		self.write_location_line(loc_label.start_line, loc_label.start_col + 1, padding, loc_filename)?;

		// Opening separator gutter.
		self.write_empty_gutter(padding)?;

		let mut current_line = usize::MAX;
		let mut current_module: Option<ModuleId> = None;
		let mut module_lines: BTreeMap<ModuleId, Vec<&str>> = BTreeMap::new();

		let mut i = 0;
		while i < rendered_labels.len() {
			let rl = &rendered_labels[i];
			let module = rl.span.module;
			let content = rl.span.module_content(self.modules);
			module_lines.entry(module).or_insert_with(|| content.lines().collect());

			// If the module changed, emit a new file location header.
			if Some(module) != current_module {
				if current_module.is_some() {
					self.write_empty_gutter(padding)?;
					let filename = self.modules[module].path.as_str();
					self.write_location_line(rl.start_line, rl.start_col + 1, padding, filename)?;
					self.write_empty_gutter(padding)?;
				}
				current_module = Some(module);
				current_line = usize::MAX;
			}

			// Detect how many consecutive labels share the same single source line.
			let group_end = if rl.start_line == rl.end_line {
				let line = rl.start_line;
				let mut end = i + 1;
				while end < rendered_labels.len() {
					let next = &rendered_labels[end];
					if next.span.module == module && next.start_line == line && next.end_line == line {
						end += 1;
					} else {
						break;
					}
				}
				end
			} else {
				i + 1
			};

			let lines = &module_lines[&module];

			if group_end > i + 1 {
				// Multiple single-line labels on the same source line: render their
				// underlines combined on one annotation line, rightmost message inline,
				// remaining messages below.
				let group = &rendered_labels[i..group_end];
				let line = rl.start_line;

				if current_line != usize::MAX && line.saturating_sub(current_line) > 5 {
					self.write_break()?;
				}
				if line != current_line {
					let line_text = lines.get(line.saturating_sub(1)).copied().unwrap_or("");
					self.write_line_number(line)?;
					self.write_padding(padding - line_number_width(line))?;
					self.write_gutter()?;
					writeln!(self.writer, "{line_text}")?;
					current_line = line;
				}

				// Combined underline row.
				self.write_padding(padding)?;
				self.write_gutter()?;
				let mut cur_col = 0;
				for (idx, label) in group.iter().enumerate() {
					let from_col = label.start_col;
					let len = label.end_col.saturating_sub(label.start_col).max(1);
					if from_col > cur_col {
						write!(self.writer, "{}", " ".repeat(from_col - cur_col))?;
					}
					let style = match label.label.ty {
						LabelType::Primary => self.styles.primary_label,
						LabelType::Secondary => self.styles.secondary_label,
					};
					let caret = match label.label.ty {
						LabelType::Primary => "^",
						LabelType::Secondary => "-",
					};
					self.write_styled(caret.repeat(len), style)?;
					cur_col = from_col + len;
					// Rightmost label gets its message appended inline.
					if idx == group.len() - 1 && !label.label.message.is_empty() {
						write!(self.writer, " {}", label.label.message)?;
					}
				}
				writeln!(self.writer)?;

				// Messages for all non-rightmost labels, each on its own row.
				for label in group[..group.len() - 1].iter() {
					if label.label.message.is_empty() {
						continue;
					}
					self.write_padding(padding)?;
					self.write_gutter()?;
					let style = match label.label.ty {
						LabelType::Primary => self.styles.primary_label,
						LabelType::Secondary => self.styles.secondary_label,
					};
					write!(self.writer, "{}", " ".repeat(label.start_col))?;
					self.write_styled(&label.label.message, style)?;
					writeln!(self.writer)?;
				}

				i = group_end;
			} else {
				// Single label: existing rendering logic.
				for line in rl.start_line..=rl.end_line {
					if current_line != usize::MAX && line.saturating_sub(current_line) > 5 {
						self.write_break()?;
					}

					if line != current_line {
						let line_text = lines.get(line.saturating_sub(1)).copied().unwrap_or("");
						self.write_line_number(line)?;
						self.write_padding(padding - line_number_width(line))?;
						self.write_gutter()?;
						writeln!(self.writer, "{line_text}")?;
						current_line = line;
					}

					let line_text = lines.get(line.saturating_sub(1)).copied().unwrap_or("");
					let line_len = line_text.len();

					let (from_col, len) = if rl.start_line == rl.end_line {
						// single-line span: [start_col, end_col)
						let len = rl.end_col.saturating_sub(rl.start_col);
						(rl.start_col, len)
					} else if line == rl.start_line {
						// first line: [start_col, EOL)
						(rl.start_col, line_len.saturating_sub(rl.start_col))
					} else if line == rl.end_line {
						// last line: [SOL, end_col)
						(0, rl.end_col.min(line_len))
					} else {
						// middle line: entire line
						(0, line_len)
					};

					self.write_padding(padding)?;
					self.write_gutter()?;
					self.write_label(rl.label, from_col, len)?;
				}
				i += 1;
			}
		}

		if !self.diagnostic.notes.is_empty() {
			self.write_empty_gutter(padding)?;

			for note in &self.diagnostic.notes {
				self.write_padding(padding)?;
				let border = self.styles.border;
				self.write_styled(" = ", border)?;
				let header_message = self.styles.header_message;
				self.write_styled("note", header_message)?;
				writeln!(self.writer, ": {}", note)?;
			}
		}

		#[cfg(all(not(test), debug_assertions))]
		{
			self.write_padding(padding)?;
			self.write_gutter()?;
			let debug_info = self.styles.debug_info;
			self.write_styled(
				format_args!(
					" created at {}:{}:{}",
					self.diagnostic.location.file(),
					self.diagnostic.location.line(),
					self.diagnostic.location.column()
				),
				debug_info,
			)?;
			writeln!(self.writer)?;
		}

		Ok(())
	}
}
