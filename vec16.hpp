#ifndef VECLIB_VEC16_HPP
#define VECLIB_VEC16_HPP

// SSE-based vector types - named for their width, 16 bytes
// veclib is written and maintained by Gregory Meyer.

#include "traits.hpp"

#include <xmmintrin.h>
#include <emmintrin.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <stdexcept>
#include <iostream>

namespace vlib {

template <typename E>
class DVec16Expr {
public:
	auto operator[](const std::size_t pos) const -> double {
		return as_base()[pos];
	}

	auto at(const std::size_t pos) const -> double {
		check_bounds(pos);
		return (*this)[pos];
	}

	auto size() const noexcept -> std::size_t {
		return as_base().size();
	}

	auto front() const noexcept -> double {
		return as_base().front();
	}

	auto back() const noexcept -> double {
		return as_base().back();
	}

	auto get() const -> __m128d {
		return static_cast<const E&>(*this).get();
	}

	explicit operator __m128d() const {
		return get();
	}

private:
	auto as_base() const -> const E& {
		return static_cast<const E&>(*this);
	}

	operator const E&() const {
		return as_base();
	}

	auto check_bounds(const std::size_t pos) const -> void {
		if (pos >= size()) {
			throw std::out_of_range{ "DVec16Expr::check_bounds" };
		}
	}
};

class DVec16 : public DVec16Expr<DVec16> {
public:
	DVec16() noexcept : vector_{ _mm_setzero_pd() } { }

	explicit DVec16(const double value) noexcept :
		vector_{ _mm_set1_pd(value) }
	{ }

	DVec16(std::initializer_list<double> init) {
		if (init.size() != 2) {
			throw std::length_error{ "DVec16::DVec16(initializer_list)" };
		}

		const double first = *init.begin();
		const double second = *(init.begin() + 1);

		vector_ = _mm_set_pd(first, second);
	}

	explicit DVec16(const double *const address) {
		const auto as_int = reinterpret_cast<std::uintptr_t>(address);
		const bool is_aligned = not (as_int & 15);

		if (is_aligned) {
			vector_ = _mm_load_pd(address);
		} else {
			vector_ = _mm_loadu_pd(address);
		}
	}

	template <typename E>
	DVec16(const DVec16Expr<E> &expression) : vector_{ expression.get() } { }

	auto operator[](const std::size_t pos) -> double& {
		return array_[pos];
	}

	auto operator[](const std::size_t pos) const -> double {
		return array_[pos];
	}

	auto at(const std::size_t pos) -> double& {
		check_bounds(pos);
		return (*this)[pos];
	}

	auto at(const std::size_t pos) const -> double {
		check_bounds(pos);
		return (*this)[pos];
	}

	auto size() const noexcept -> std::size_t {
		return 2;
	}

	auto front() noexcept -> double& {
		return array_[0];
	}

	auto front() const noexcept -> double {
		return array_[0];
	}

	auto back() noexcept -> double& {
		return array_[1];
	}

	auto back() const noexcept -> double {
		return array_[1];
	}

	auto begin() noexcept -> double* {
		return static_cast<double*>(array_);
	}

	auto begin() const noexcept -> const double* {
		return cbegin();
	}

	auto cbegin() const noexcept -> const double* {
		return static_cast<const double*>(array_);
	}

	auto end() noexcept -> double* {
		return static_cast<double*>(array_) + 2;
	}

	auto end() const noexcept -> const double* {
		return cend();
	}

	auto cend() const noexcept -> const double* {
		return static_cast<const double*>(array_) + 2;
	}

	auto get() const -> __m128d {
		return vector_;
	}

private:
	auto check_bounds(const std::size_t pos) const -> void {
		if (pos >= size()) {
			throw std::out_of_range{ "DVec16::check_bounds" };
		}
	}

	union {
		double array_[2];
		__m128d vector_;
	};
};

template <typename E>
auto operator<<(std::ostream &os, const DVec16Expr<E> &vec) -> std::ostream& {
	return os << "{ " << vec.front() << ", " << vec.back() << " }";
}

template <typename E1, typename E2>
class DVec16Add : public DVec16Expr<DVec16Add<E1, E2>> {
public:
	DVec16Add(const E1 &lhs, const E2 &rhs) : lhs_{ lhs }, rhs_{ rhs } { }

	auto operator[](const std::size_t pos) const -> double {
		return lhs_[pos] + rhs_[pos];
	}

	auto at(const std::size_t pos) const -> double {
		check_bounds(pos);
		return (*this)[pos];
	}

	auto size() const noexcept -> std::size_t {
		return lhs_.size();
	}

	auto front() const noexcept -> double {
		return lhs_.front() + rhs_.front();
	}

	auto back() const noexcept -> double {
		return lhs_.back() + rhs_.back();
	}

	auto get() const -> __m128d {
		return _mm_add_pd(lhs_.get(), rhs_.get());
	}

private:
	auto check_bounds(const std::size_t pos) const -> void {
		if (pos >= size()) {
			throw std::out_of_range{ "DVec16Expr::check_bounds" };
		}
	}

private:
	const E1 &lhs_;
	const E2 &rhs_;
};

template <typename E1, typename E2>
class DVec16Sub : public DVec16Expr<DVec16Sub<E1, E2>> {
public:
	DVec16Sub(const E1 &lhs, const E2 &rhs) : lhs_{ lhs }, rhs_{ rhs } { }

	auto operator[](const std::size_t pos) const -> double {
		return lhs_[pos] - rhs_[pos];
	}

	auto at(const std::size_t pos) const -> double {
		check_bounds(pos);
		return (*this)[pos];
	}

	auto size() const noexcept -> std::size_t {
		return lhs_.size();
	}

	auto front() const noexcept -> double {
		return lhs_.front() - rhs_.front();
	}

	auto back() const noexcept -> double {
		return lhs_.back() - rhs_.back();
	}

	auto get() const -> __m128d {
		return _mm_sub_pd(lhs_.get(), rhs_.get());
	}

private:
	auto check_bounds(const std::size_t pos) const -> void {
		if (pos >= size()) {
			throw std::out_of_range{ "DVec16Expr::check_bounds" };
		}
	}

private:
	const E1 &lhs_;
	const E2 &rhs_;
};

template <typename E1, typename E2>
class DVec16Mul : public DVec16Expr<DVec16Mul<E1, E2>> {
public:
	DVec16Mul(const E1 &lhs, const E2 &rhs) : lhs_{ lhs }, rhs_{ rhs } { }

	auto operator[](const std::size_t pos) const -> double {
		return lhs_[pos] * rhs_[pos];
	}

	auto at(const std::size_t pos) const -> double {
		check_bounds(pos);
		return (*this)[pos];
	}

	auto size() const noexcept -> std::size_t {
		return lhs_.size();
	}

	auto front() const noexcept -> double {
		return lhs_.front() * rhs_.front();
	}

	auto back() const noexcept -> double {
		return lhs_.back() * rhs_.back();
	}

	auto get() const -> __m128d {
		return _mm_mul_pd(lhs_.get(), rhs_.get());
	}

private:
	auto check_bounds(const std::size_t pos) const -> void {
		if (pos >= size()) {
			throw std::out_of_range{ "DVec16Expr::check_bounds" };
		}
	}

	const E1 &lhs_;
	const E2 &rhs_;
};

template <typename E1, typename E2>
class DVec16Div : public DVec16Expr<DVec16Div<E1, E2>> {
public:
	DVec16Div(const E1 &lhs, const E2 &rhs) : lhs_{ lhs }, rhs_{ rhs } { }

	auto operator[](const std::size_t pos) const -> double {
		return lhs_[pos] / rhs_[pos];
	}

	auto at(const std::size_t pos) const -> double {
		check_bounds(pos);
		return (*this)[pos];
	}

	auto size() const noexcept -> std::size_t {
		return lhs_.size();
	}

	auto front() const noexcept -> double {
		return lhs_.front() / rhs_.front();
	}

	auto back() const noexcept -> double {
		return lhs_.back() / rhs_.back();
	}

	auto get() const -> __m128d {
		return _mm_div_pd(lhs_.get(), rhs_.get());
	}

private:
	auto check_bounds(const std::size_t pos) const -> void {
		if (pos >= size()) {
			throw std::out_of_range{ "DVec16Expr::check_bounds" };
		}
	}

private:
	const E1 &lhs_;
	const E2 &rhs_;
};

template <typename E1, typename E2>
auto operator+(const E1 &lhs, const E2 &rhs) -> DVec16Add<E1, E2> {
	return { lhs, rhs };
}

template <typename E1, typename E2>
auto operator-(const E1 &lhs, const E2 &rhs) -> DVec16Sub<E1, E2> {
	return { lhs, rhs };
}

template <typename E1, typename E2>
auto operator*(const E1 &lhs, const E2 &rhs) -> DVec16Mul<E1, E2> {
	return { lhs, rhs };
}

template <typename E1, typename E2>
auto operator/(const E1 &lhs, const E2 &rhs) -> DVec16Div<E1, E2> {
	return { lhs, rhs };
}

} // namespace vlib

#endif
