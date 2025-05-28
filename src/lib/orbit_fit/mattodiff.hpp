// Matthew J. Holman 2025 May 28
//
// This is a lightweight, header-only library that implements
// dual numbers for automatic differentiation.  There is
// no fancy templating.  And there are no helper routines for
// things like Jacobians.  But there are no dependencies.
//
// I got started with this by googling "dual numbers c++
// automatic differention".  The AI result gave me enough
// to get started.
//
// Dual numbers are defined as
// d = a + epsilon b, 
// where a and b are both real numbers and epsilon is small.
// The axiom is that epsilon^2 = 0.  The b component can
// be considered the derivative.  (Here a and b are implemented
// as doubles.)
//
// The Dual class uses operator overloading to implement
// the mathematical operators and functions that I currently
// need.  It's straightforward to add others.

#ifndef MATTODIFF_H
#define MATTODIFF_H
class Dual {
public:
    double real;
    double dual;
    
    // Constructors
    Dual(double real, double dual = 0) : real(real), dual(dual) {}
    Dual() : real(0.0), dual(0.0) {}    

    // Overload arithmetic operators
    Dual operator+(const Dual& other) const {
        return Dual(real + other.real, dual + other.dual);
    }

    friend Dual operator+(const Dual& v0, double v1) {
	return Dual(v0.real + v1, v0.dual);
    }

    friend Dual operator+(double v0, const Dual& v1) {
	return Dual(v0 + v1.real, v1.dual);
    }

    Dual operator-(const Dual& other) const {
        return Dual(real - other.real, dual - other.dual);
    }

    friend Dual operator-(const Dual& v0, double v1) {
	return Dual(v0.real - v1, v0.dual);
    }

    friend Dual operator-(double v0, const Dual& v1) {
	return Dual(v0 - v1.real, -v1.dual);
    }

    Dual operator-() const {
        return Dual(-real, -dual);
    }

    Dual operator*(const Dual& other) const {
        return Dual(real * other.real, dual * other.real + real * other.dual);
    }

    friend Dual operator*(const Dual& v0, double v1) {
	return Dual(v0.real*v1, v0.dual*v1);
    }

    friend Dual operator*(double v0, const Dual& v1) {
	return Dual(v0*v1.real, v0*v1.dual);
    }

    Dual operator/(const Dual& other) const {
       return Dual(real / other.real, (dual * other.real - real * other.dual) / (other.real * other.real));
    }

    friend Dual operator/(const Dual& v0, double v1) {
	return Dual(v0.real / v1, v0.dual / v1);
    }

    friend Dual operator/(double v0, const Dual& v1) {
	return Dual(v0 / v1.real, (- v0 * v1.dual)/(v1.real * v1.real));
    }

    friend bool operator>(double v0, const Dual& v1) {
	return v0 > v1.real;
    }

    friend bool operator>(const Dual& v0, double v1) {
	return v0.real > v1;
    }
    
    friend bool operator>(const Dual& v0, const Dual& v1) {
	return v0.real > v1.real;
    }
    
    friend bool operator<(double v0, const Dual& v1) {
	return v0 < v1.real;
    }

    friend bool operator<(const Dual& v0, double v1) {
	return v0.real < v1;
    }
    
    friend bool operator<(const Dual& v0, const Dual& v1) {
	return v0.real < v1.real;
    }
    
    friend bool operator<=(double v0, const Dual& v1) {
	return v0 <= v1.real;
    }

    friend bool operator<=(const Dual& v0, double v1) {
	return v0.real <= v1;
    }
    
    friend bool operator<=(const Dual& v0, const Dual& v1) {
	return v0.real <= v1.real;
    }

    friend bool operator>=(double v0, const Dual& v1) {
	return v0 >= v1.real;
    }

    friend bool operator>=(const Dual& v0, double v1) {
	return v0.real >= v1;
    }
    
    friend bool operator>=(const Dual& v0, const Dual& v1) {
	return v0.real >= v1.real;
    }
    
    friend bool operator==(double v0, const Dual& v1) {
	return v0 == v1.real;
    }
    
    friend bool operator==(const Dual& v0, double v1) {
	return v0.real == v1;
    }
    
    friend bool operator==(const Dual& v0, const Dual& v1) {
	return v0.real == v1.real;
    }
    
    friend bool operator!=(double v0, const Dual& v1) {
	return v0 != v1.real;
    }
    
    friend bool operator!=(const Dual& v0, double v1) {
	return v0.real != v1;
    }
    
    friend bool operator!=(const Dual& v0, const Dual& v1) {
	return v0.real != v1.real;
    }
    
    // Overload math functions
    Dual sin() const {
        return Dual(std::sin(real), dual * std::cos(real));
    }

    friend Dual sin(const Dual& v0) {
        return Dual(std::sin(v0.real), v0.dual * std::cos(v0.real));
    }
    
    Dual cos() const {
        return Dual(std::cos(real), -dual * std::sin(real));
    }

    friend Dual cos(const Dual& v0) {
        return Dual(std::cos(v0.real), -v0.dual * std::sin(v0.real));
    }
    
    Dual acos() const {
        return Dual(std::acos(real), -dual / sqrt(1.0 - real * real));
    }

    friend Dual acos(const Dual& v0) {
        return Dual(std::acos(v0.real), -v0.dual / sqrt(1.0 - v0.real * v0.real));
    }
    
    Dual pow(double exponent) const {
        return Dual(std::pow(real, exponent), dual * exponent * std::pow(real, exponent - 1));
    }

    friend Dual pow(const Dual& v0, double exponent) {
        return Dual(std::pow(v0.real, exponent), v0.dual * exponent * std::pow(v0.real, exponent - 1));
    }

    friend Dual sqrt(const Dual& v0) {
        return Dual(std::sqrt(v0.real), 0.5 * v0.dual / std::sqrt(v0.real));
    }

    Dual tan() const {
        return Dual(std::tan(real), dual / (std::cos(real) * std::cos(real)));
    }

    friend Dual tan(const Dual& v0) {
        return Dual(std::tan(v0.real), v0.dual /(std::cos(v0.real) * std::cos(v0.real)));
    }
    
    friend Dual atan(Dual v0) {
        return Dual(std::atan(v0.real), v0.dual / (1.0 + v0.real * v0.real));
    }    

    friend Dual atan2(Dual y, Dual x) {
	Dual res;
	res.real = std::atan2(y.real, x.real);
	res.dual = (y.dual * x.real - y.real * x.dual) / (x.real * x.real + y.real * y.real);
	return res;
    }    

    friend Dual atanh(Dual v0) {
        return Dual(std::atanh(v0.real), v0.dual / (1.0 - v0.real * v0.real));
    }    

    friend Dual sinh(Dual v0) {
        return Dual(std::sinh(v0.real), v0.dual * cosh(v0.real));
    }    
    
};
#endif // MATTODIFF_H
