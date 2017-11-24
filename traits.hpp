#ifndef VECLIB_TRAITS_HPP
#define VECLIB_TRAITS_HPP

#include <type_traits>

namespace vlib {

template <typename T>
struct is_vector_type : public std::false_type { };

template <>
struct is_vector_type<float> : public std::true_type { };

template <>
struct is_vector_type<double> : public std::true_type { };

template <>
struct is_vector_type<char> : public std::true_type { };

template <>
struct is_vector_type<short> : public std::true_type { };

template <>
struct is_vector_type<int> : public std::true_type { };

template <>
struct is_vector_type<long> : public std::true_type { };

template <typename T>
inline constexpr bool is_vector_type_v = is_vector_type<T>::value;

} // namespace vlib

#endif
