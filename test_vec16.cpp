#include "vec16.hpp"

#include <iostream>

struct A {
	int data;
};

auto main() -> int {
	const vlib::DVec16 a{ 15.0, 20.0 }; 
	const vlib::DVec16 b{ 1.0, -1.0 };
	const vlib::DVec16 c{ 32.0, 64.0 };

	const vlib::DVec16 x = a + b + c;

	std::cout << x << std::endl;
	std::cout << (a * b * c) << std::endl;
}
