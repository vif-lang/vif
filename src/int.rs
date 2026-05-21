use std::{
	cmp::Ordering,
	fmt,
	hash::{
		Hash,
		Hasher,
	},
	ops::{
		Add,
		BitAnd,
		BitOr,
		BitXor,
		Div,
		Mul,
		Neg,
		Not,
		Rem,
		Shl,
		Shr,
		Sub,
	},
};

#[derive(Clone, Debug)]
pub struct Anyint {
	negative: bool,
	limbs: Vec<u64>,
}

impl Anyint {
	pub const ZERO: Self = Self {
		negative: false,
		limbs: Vec::new(),
	};

	pub fn parse_radix(
		radix: u32,
		digits: &str,
	) -> Option<Self> {
		assert!((2..=36).contains(&radix));
		let mut result = Self::ZERO;
		for b in digits.bytes() {
			let digit = match b {
				b'0'..=b'9' => b - b'0',
				b'a'..=b'z' => b - b'a' + 10,
				b'A'..=b'Z' => b - b'A' + 10,
				b'_' => continue,
				_ => return None,
			};
			if digit as u32 >= radix {
				return None;
			}
			result.mul_small_assign(radix as u64);
			result.add_small_assign(digit as u64);
		}
		Some(result)
	}

	pub fn is_negative(&self) -> bool {
		self.negative
	}

	pub fn to_u64(&self) -> Option<u64> {
		if self.negative || self.limbs.len() > 1 {
			return None;
		}
		Some(self.limbs.first().copied().unwrap_or(0))
	}

	pub fn significant_bits(&self) -> u64 {
		let Some(last) = self.limbs.last() else {
			return 0;
		};
		((self.limbs.len() as u64 - 1) * 64) + (64 - last.leading_zeros() as u64)
	}

	pub fn fits_unsigned_bits(
		&self,
		bits: u16,
	) -> bool {
		!self.negative && self.significant_bits() <= bits as u64
	}

	pub fn fits_signed_bits(
		&self,
		bits: u16,
	) -> bool {
		assert!(bits > 0);
		if self.negative {
			self.abs_cmp_pow2((bits - 1) as u64) != Ordering::Greater
		} else {
			self.significant_bits() < bits as u64
		}
	}

	fn is_zero(&self) -> bool {
		self.limbs.is_empty()
	}

	fn normalize(&mut self) {
		while self.limbs.last() == Some(&0) {
			self.limbs.pop();
		}
		if self.limbs.is_empty() {
			self.negative = false;
		}
	}

	fn abs_cmp(
		&self,
		other: &Self,
	) -> Ordering {
		match self.limbs.len().cmp(&other.limbs.len()) {
			Ordering::Equal => self.limbs.iter().rev().cmp(other.limbs.iter().rev()),
			order => order,
		}
	}

	fn abs_cmp_pow2(
		&self,
		bit: u64,
	) -> Ordering {
		let limb = (bit / 64) as usize;
		let bit_in_limb = bit % 64;
		let needed_len = limb + 1;
		match self.limbs.len().cmp(&needed_len) {
			Ordering::Less => Ordering::Less,
			Ordering::Greater => Ordering::Greater,
			Ordering::Equal => {
				let top = 1u64 << bit_in_limb;
				match self.limbs[limb].cmp(&top) {
					Ordering::Equal => {
						if self.limbs[..limb].iter().any(|&x| x != 0) {
							Ordering::Greater
						} else {
							Ordering::Equal
						}
					},
					order => order,
				}
			},
		}
	}

	fn add_abs(
		lhs: &[u64],
		rhs: &[u64],
	) -> Vec<u64> {
		let len = lhs.len().max(rhs.len());
		let mut out = Vec::with_capacity(len + 1);
		let mut carry = 0u128;
		for i in 0..len {
			let a = lhs.get(i).copied().unwrap_or(0) as u128;
			let b = rhs.get(i).copied().unwrap_or(0) as u128;
			let sum = a + b + carry;
			out.push(sum as u64);
			carry = sum >> 64;
		}
		if carry != 0 {
			out.push(carry as u64);
		}
		out
	}

	fn sub_abs(
		lhs: &[u64],
		rhs: &[u64],
	) -> Vec<u64> {
		let mut out = Vec::with_capacity(lhs.len());
		let mut borrow = 0u128;
		for (i, &a) in lhs.iter().enumerate() {
			let b = rhs.get(i).copied().unwrap_or(0) as u128;
			let sub = b + borrow;
			if (a as u128) >= sub {
				out.push((a as u128 - sub) as u64);
				borrow = 0;
			} else {
				out.push(((1u128 << 64) + a as u128 - sub) as u64);
				borrow = 1;
			}
		}
		out
	}

	fn add_small_assign(
		&mut self,
		rhs: u64,
	) {
		if rhs == 0 {
			return;
		}
		let mut carry = rhs as u128;
		let mut i = 0;
		while carry != 0 {
			if i == self.limbs.len() {
				self.limbs.push(0);
			}
			let sum = self.limbs[i] as u128 + carry;
			self.limbs[i] = sum as u64;
			carry = sum >> 64;
			i += 1;
		}
	}

	fn mul_small_assign(
		&mut self,
		rhs: u64,
	) {
		if rhs == 0 || self.is_zero() {
			*self = Self::ZERO;
			return;
		}
		let mut carry = 0u128;
		for limb in &mut self.limbs {
			let prod = *limb as u128 * rhs as u128 + carry;
			*limb = prod as u64;
			carry = prod >> 64;
		}
		if carry != 0 {
			self.limbs.push(carry as u64);
		}
	}

	fn div_rem_small(
		&self,
		rhs: u64,
	) -> (Self, u64) {
		assert!(rhs != 0);
		let mut out = vec![0; self.limbs.len()];
		let mut rem = 0u128;
		for i in (0..self.limbs.len()).rev() {
			let cur = (rem << 64) | self.limbs[i] as u128;
			out[i] = (cur / rhs as u128) as u64;
			rem = cur % rhs as u128;
		}
		let mut q = Self {
			negative: false,
			limbs: out,
		};
		q.normalize();
		(q, rem as u64)
	}

	fn set_bit(
		&mut self,
		bit: u64,
	) {
		let limb = (bit / 64) as usize;
		let bit = bit % 64;
		if self.limbs.len() <= limb {
			self.limbs.resize(limb + 1, 0);
		}
		self.limbs[limb] |= 1u64 << bit;
	}

	fn abs_shl(
		&self,
		shift: u64,
	) -> Self {
		if self.is_zero() {
			return Self::ZERO;
		}
		let limb_shift = (shift / 64) as usize;
		let bit_shift = shift % 64;
		let mut out = vec![0; limb_shift + self.limbs.len() + 1];
		for (i, &limb) in self.limbs.iter().enumerate() {
			let idx = i + limb_shift;
			out[idx] |= limb << bit_shift;
			if bit_shift != 0 {
				out[idx + 1] |= limb >> (64 - bit_shift);
			}
		}
		let mut res = Self {
			negative: self.negative,
			limbs: out,
		};
		res.normalize();
		res
	}

	fn abs_shr(
		&self,
		shift: u64,
	) -> Self {
		let limb_shift = (shift / 64) as usize;
		let bit_shift = shift % 64;
		if limb_shift >= self.limbs.len() {
			return Self::ZERO;
		}
		let mut out = vec![0; self.limbs.len() - limb_shift];
		for i in limb_shift..self.limbs.len() {
			let dst = i - limb_shift;
			out[dst] |= self.limbs[i] >> bit_shift;
			if bit_shift != 0 && i + 1 < self.limbs.len() {
				out[dst] |= self.limbs[i + 1] << (64 - bit_shift);
			}
		}
		let mut res = Self {
			negative: false,
			limbs: out,
		};
		res.normalize();
		res
	}

	fn div_rem_abs(
		lhs: &Self,
		rhs: &Self,
	) -> (Self, Self) {
		assert!(!rhs.is_zero());
		if lhs.abs_cmp(rhs) == Ordering::Less {
			return (Self::ZERO, lhs.clone());
		}
		if rhs.limbs.len() == 1 {
			let (q, r) = lhs.div_rem_small(rhs.limbs[0]);
			return (q, Self::from(r));
		}

		let mut quotient = Self::ZERO;
		let mut remainder = Self::ZERO;
		for bit in (0..lhs.significant_bits()).rev() {
			remainder = remainder.abs_shl(1);
			let limb = (bit / 64) as usize;
			let bit_in_limb = bit % 64;
			if ((lhs.limbs[limb] >> bit_in_limb) & 1) != 0 {
				remainder.add_small_assign(1);
			}
			if remainder.abs_cmp(rhs) != Ordering::Less {
				remainder = &remainder - rhs;
				quotient.set_bit(bit);
			}
		}
		(quotient, remainder)
	}

	fn twos_complement_limbs(
		&self,
		len: usize,
	) -> Vec<u64> {
		let mut out = vec![0; len];
		for (i, &limb) in self.limbs.iter().enumerate().take(len) {
			out[i] = limb;
		}
		if self.negative {
			for limb in &mut out {
				*limb = !*limb;
			}
			let mut carry = 1u128;
			for limb in &mut out {
				let sum = *limb as u128 + carry;
				*limb = sum as u64;
				carry = sum >> 64;
				if carry == 0 {
					break;
				}
			}
		}
		out
	}

	fn from_twos_complement_limbs(mut limbs: Vec<u64>) -> Self {
		let negative = limbs.last().is_some_and(|limb| (limb >> 63) != 0);
		if negative {
			for limb in &mut limbs {
				*limb = !*limb;
			}
			let mut carry = 1u128;
			for limb in &mut limbs {
				let sum = *limb as u128 + carry;
				*limb = sum as u64;
				carry = sum >> 64;
				if carry == 0 {
					break;
				}
			}
		}
		let mut res = Self { negative, limbs };
		res.normalize();
		res
	}

	fn bitwise(
		lhs: &Self,
		rhs: &Self,
		op: impl Fn(u64, u64) -> u64,
	) -> Self {
		let bits = lhs.significant_bits().max(rhs.significant_bits()) + 2;
		let len = (bits.div_ceil(64) as usize).max(1);
		let lhs = lhs.twos_complement_limbs(len);
		let rhs = rhs.twos_complement_limbs(len);
		let out = lhs.into_iter().zip(rhs).map(|(l, r)| op(l, r)).collect();
		Self::from_twos_complement_limbs(out)
	}
}

impl Default for Anyint {
	fn default() -> Self {
		Self::ZERO
	}
}

impl PartialEq for Anyint {
	fn eq(
		&self,
		other: &Self,
	) -> bool {
		self.negative == other.negative && self.limbs == other.limbs
	}
}
impl Eq for Anyint {}

impl Hash for Anyint {
	fn hash<H: Hasher>(
		&self,
		state: &mut H,
	) {
		self.negative.hash(state);
		self.limbs.hash(state);
	}
}

impl Ord for Anyint {
	fn cmp(
		&self,
		other: &Self,
	) -> Ordering {
		match (self.negative, other.negative) {
			(true, false) => Ordering::Less,
			(false, true) => Ordering::Greater,
			(false, false) => self.abs_cmp(other),
			(true, true) => other.abs_cmp(self),
		}
	}
}
impl PartialOrd for Anyint {
	fn partial_cmp(
		&self,
		other: &Self,
	) -> Option<Ordering> {
		Some(self.cmp(other))
	}
}

impl fmt::Display for Anyint {
	fn fmt(
		&self,
		f: &mut fmt::Formatter<'_>,
	) -> fmt::Result {
		if self.is_zero() {
			return f.write_str("0");
		}
		let mut n = self.clone();
		n.negative = false;
		let mut digits = Vec::new();
		while !n.is_zero() {
			let (q, r) = n.div_rem_small(10);
			digits.push((b'0' + r as u8) as char);
			n = q;
		}
		if self.negative {
			f.write_str("-")?;
		}
		for c in digits.iter().rev() {
			write!(f, "{c}")?;
		}
		Ok(())
	}
}

impl Neg for Anyint {
	type Output = Self;

	fn neg(mut self) -> Self::Output {
		if !self.is_zero() {
			self.negative = !self.negative;
		}
		self
	}
}

impl Neg for &Anyint {
	type Output = Anyint;

	fn neg(self) -> Self::Output {
		-self.clone()
	}
}

impl Add for &Anyint {
	type Output = Anyint;

	fn add(
		self,
		rhs: Self,
	) -> Self::Output {
		let mut res = if self.negative == rhs.negative {
			Anyint {
				negative: self.negative,
				limbs: Anyint::add_abs(&self.limbs, &rhs.limbs),
			}
		} else {
			match self.abs_cmp(rhs) {
				Ordering::Greater | Ordering::Equal => Anyint {
					negative: self.negative,
					limbs: Anyint::sub_abs(&self.limbs, &rhs.limbs),
				},
				Ordering::Less => Anyint {
					negative: rhs.negative,
					limbs: Anyint::sub_abs(&rhs.limbs, &self.limbs),
				},
			}
		};
		res.normalize();
		res
	}
}

impl Sub for &Anyint {
	type Output = Anyint;

	fn sub(
		self,
		rhs: Self,
	) -> Self::Output {
		self + &(-rhs)
	}
}

impl Mul for &Anyint {
	type Output = Anyint;

	fn mul(
		self,
		rhs: Self,
	) -> Self::Output {
		if self.is_zero() || rhs.is_zero() {
			return Anyint::ZERO;
		}
		let mut out = vec![0u64; self.limbs.len() + rhs.limbs.len()];
		for (i, &a) in self.limbs.iter().enumerate() {
			let mut carry = 0u128;
			for (j, &b) in rhs.limbs.iter().enumerate() {
				let idx = i + j;
				let cur = out[idx] as u128 + a as u128 * b as u128 + carry;
				out[idx] = cur as u64;
				carry = cur >> 64;
			}
			let mut idx = i + rhs.limbs.len();
			while carry != 0 {
				let cur = out[idx] as u128 + carry;
				out[idx] = cur as u64;
				carry = cur >> 64;
				idx += 1;
			}
		}
		let mut res = Anyint {
			negative: self.negative ^ rhs.negative,
			limbs: out,
		};
		res.normalize();
		res
	}
}

impl Div for &Anyint {
	type Output = Anyint;

	fn div(
		self,
		rhs: Self,
	) -> Self::Output {
		assert!(!rhs.is_zero());
		let mut lhs_abs = self.clone();
		lhs_abs.negative = false;
		let mut rhs_abs = rhs.clone();
		rhs_abs.negative = false;
		let (mut q, _) = Anyint::div_rem_abs(&lhs_abs, &rhs_abs);
		q.negative = !q.is_zero() && (self.negative ^ rhs.negative);
		q
	}
}

impl Rem for &Anyint {
	type Output = Anyint;

	fn rem(
		self,
		rhs: Self,
	) -> Self::Output {
		assert!(!rhs.is_zero());
		let mut lhs_abs = self.clone();
		lhs_abs.negative = false;
		let mut rhs_abs = rhs.clone();
		rhs_abs.negative = false;
		let (_, mut r) = Anyint::div_rem_abs(&lhs_abs, &rhs_abs);
		r.negative = !r.is_zero() && self.negative;
		r
	}
}

impl BitAnd for &Anyint {
	type Output = Anyint;

	fn bitand(
		self,
		rhs: Self,
	) -> Self::Output {
		Anyint::bitwise(self, rhs, |l, r| l & r)
	}
}

impl BitOr for &Anyint {
	type Output = Anyint;

	fn bitor(
		self,
		rhs: Self,
	) -> Self::Output {
		Anyint::bitwise(self, rhs, |l, r| l | r)
	}
}

impl BitXor for &Anyint {
	type Output = Anyint;

	fn bitxor(
		self,
		rhs: Self,
	) -> Self::Output {
		Anyint::bitwise(self, rhs, |l, r| l ^ r)
	}
}

impl Not for &Anyint {
	type Output = Anyint;

	fn not(self) -> Self::Output {
		-&(self + &Anyint::from(1u64))
	}
}

impl Shl<u64> for &Anyint {
	type Output = Anyint;

	fn shl(
		self,
		rhs: u64,
	) -> Self::Output {
		self.abs_shl(rhs)
	}
}

impl Shr<u64> for &Anyint {
	type Output = Anyint;

	fn shr(
		self,
		rhs: u64,
	) -> Self::Output {
		if !self.negative {
			return self.abs_shr(rhs);
		}
		// Arithmetic right shift: x >> n == -(((-x) + 2^n - 1) >> n)
		let mut abs = -self;
		if rhs > 0 {
			let one = Anyint::from(1u64);
			let mask = &(&one << rhs) - &one;
			abs = &abs + &mask;
		}
		-(&abs >> rhs)
	}
}

macro_rules! from_unsigned {
	($($ty:ty),* $(,)?) => {
		$(
			impl From<$ty> for Anyint {
				fn from(value: $ty) -> Self {
					let mut res = Self {
						negative: false,
						limbs: vec![value as u64],
					};
					if core::mem::size_of::<$ty>() > 8 {
						let high = ((value as u128) >> 64) as u64;
						if high != 0 {
							res.limbs.push(high);
						}
					}
					res.normalize();
					res
				}
			}
		)*
	};
}

macro_rules! from_signed {
	($($ty:ty),* $(,)?) => {
		$(
			impl From<$ty> for Anyint {
				fn from(value: $ty) -> Self {
					let negative = value < 0;
					let abs = if negative {
						(value as i128).wrapping_neg() as u128
					} else {
						value as u128
					};
					let mut res = Self {
						negative,
						limbs: vec![abs as u64],
					};
					let high = (abs >> 64) as u64;
					if high != 0 {
						res.limbs.push(high);
					}
					res.normalize();
					res
				}
			}
		)*
	};
}

from_unsigned!(u8, u16, u32, u64, usize);
from_signed!(i8, i16, i32, i64, i128, isize);

impl From<u128> for Anyint {
	fn from(value: u128) -> Self {
		let mut res = Self {
			negative: false,
			limbs: vec![value as u64, (value >> 64) as u64],
		};
		res.normalize();
		res
	}
}

#[cfg(test)]
mod tests {
	use super::Anyint;

	fn n(s: &str) -> Anyint {
		Anyint::parse_radix(10, s).unwrap()
	}

	#[test]
	fn parse_and_format_large_decimal() {
		let s = "340282366920938463463374607431768211456";
		assert_eq!(n(s).to_string(), s);
	}

	#[test]
	fn parse_hex_and_bits() {
		let value = Anyint::parse_radix(16, "100000000000000000000000000000000").unwrap();
		assert_eq!(value.significant_bits(), 129);
	}

	#[test]
	fn signed_and_unsigned_fits() {
		assert!(n("127").fits_signed_bits(8));
		assert!(!n("128").fits_signed_bits(8));
		assert!((-n("128")).fits_signed_bits(8));
		assert!(!(-n("129")).fits_signed_bits(8));
		assert!(n("255").fits_unsigned_bits(8));
		assert!(!n("256").fits_unsigned_bits(8));
	}

	#[test]
	fn arithmetic_over_u128() {
		let a = n("340282366920938463463374607431768211456");
		assert_eq!((&a + &a).to_string(), "680564733841876926926749214863536422912");
		assert_eq!((&a * &Anyint::from(3u64)).to_string(), "1020847100762815390390123822295304634368");
	}

	#[test]
	fn div_and_rem() {
		let a = n("100000000000000000000000000000000000000");
		let b = Anyint::from(97u64);
		assert_eq!((&a / &b).to_string(), "1030927835051546391752577319587628865");
		assert_eq!((&a % &b).to_string(), "95");
		assert_eq!((&(-a.clone()) / &b).to_string(), "-1030927835051546391752577319587628865");
		assert_eq!((&(-a) % &b).to_string(), "-95");
	}

	#[test]
	fn shifts() {
		assert_eq!((&Anyint::from(1u64) << 130).to_string(), "1361129467683753853853498429727072845824");
		assert_eq!((&(-Anyint::from(8u64)) >> 1).to_string(), "-4");
		assert_eq!((&(-Anyint::from(7u64)) >> 1).to_string(), "-4");
	}

	#[test]
	fn bitwise_negative_values() {
		assert_eq!((!&Anyint::from(0u64)).to_string(), "-1");
		assert_eq!((&(-Anyint::from(1u64)) & &Anyint::from(3u64)).to_string(), "3");
		assert_eq!((&(-Anyint::from(4u64)) | &Anyint::from(1u64)).to_string(), "-3");
		assert_eq!((&(-Anyint::from(1u64)) ^ &Anyint::from(1u64)).to_string(), "-2");
	}
}
