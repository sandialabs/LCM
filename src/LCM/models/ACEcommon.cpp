// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.

#include "ACEcommon.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

std::vector<RealType>
LCM::vectorFromFile(std::string const& filename)
{
  std::ifstream file(filename);
  ALBANY_ASSERT(file.good() == true, "**** ERROR opening file " + filename);
  std::stringstream buffer;
  buffer << file.rdbuf();
  file.close();
  std::istringstream       iss(buffer.str());
  Teuchos::Array<RealType> values;
  iss >> values;
  return values.toVector();
}

std::istream&
LCM::operator>>(std::istream& is, std::vector<RealType>& vec)
{
  // #1 - check if it starts from '['
  char c;
  is >> c;
  if (c != '[') throw std::runtime_error(std::string("Invalid character : ") + c + " when parsing vector of double");
  // #2 - get the line till ']'
  std::string line;
  if (!std::getline(is, line, ']')) throw std::runtime_error("Error parsing vector of double");

  // #3 - parse values inside '[' and ']'
  std::istringstream lstr(line);
  std::string        value;
  while (std::getline(lstr, value, ',')) vec.push_back(stod(value));
  return is;
}

std::istream&
LCM::operator>>(std::istream& is, std::vector<std::vector<RealType>>& m)
{
  // #1 - check if it starts from '['
  char c;
  is >> c;
  if (c != '[') throw std::runtime_error(std::string("Invalid character : ") + c + " when parsing matrix of double");

  // parse matrix line-by-line
  while (true) {
    std::vector<double> tmp;
    is >> tmp;
    m.push_back(std::move(tmp));
    // if matrix finihed, c should contain ']', else - ','
    is >> c;
    if (c == ']') return is;
    if (c != ',') throw std::runtime_error(std::string("Invalid character : ") + c + " when parsing matrix of double");
  }
}
// display vector
// template<typename T>
std::ostream&
LCM::operator<<(std::ostream& os, const std::vector<RealType>& t)
{
  os << "[";
  for (auto it = t.begin(); it != t.end(); ++it) os << (it != t.begin() ? ", " : "") << *it;
  os << "]";
  return os;
}

std::vector<std::vector<RealType>>
LCM::twoDvectorFromFile(std::string const& filename)
{
  std::ifstream file(filename);
  ALBANY_ASSERT(file.good() == true, "**** ERROR opening file " + filename);
  std::vector<std::vector<RealType>> m;
  file >> m;
  // std::cout << "Matrix: " << m << std::endl;
  return m;
}

RealType
LCM::interpolateVectors(std::vector<RealType> const& xv, std::vector<RealType> const& yv, RealType const x)
{
  auto const n = xv.size();
  ALBANY_ASSERT(n == yv.size(), "Vectors must have same size.\n");

  if (n == 0) return 0.0;
  if (x <= xv[0]) return yv[0];
  if (x >= xv[n - 1]) return yv[n - 1];

  // Binary search for the interval containing x
  auto const it = std::lower_bound(xv.begin(), xv.end(), x);
  auto const i  = static_cast<size_t>(it - xv.begin());

  if (i == 0) return yv[0];

  RealType const dx    = xv[i] - xv[i - 1];
  RealType const slope = (yv[i] - yv[i - 1]) / dx;
  return yv[i - 1] + slope * (x - xv[i - 1]);
}
