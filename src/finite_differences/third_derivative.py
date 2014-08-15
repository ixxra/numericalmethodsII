# coding: utf-8
import sympy as sp
a, h = sp.symbols('ah')
a, h = sp.symbol('ah')
a, h = sp.symbols('a', 'h')
a, h = sp.symbols('a, h')
f = sp.Function('f')
f(h)
f.taylor_term(1, h)
f(h).taylor_term(1, h)
f, f1, f2, f3, f4 = sp.symbols('f f1 f2 f3 f4')
pt = lambda h : f + f1 * h + f2 * h**2 / 2 + f3 * h**3 / 6 + f4 * h**4 / 24
for k in range(-2, 3):
    print pt(k*h)
    
a, a1, a2, a3, a4 = sp.symbols('a a1 a2 a3 a4')
sum([x*f(k*h) for x, k in zip([a, a1, a2, a3, a4], range(-2, 3))])
f(-2*h)
f
sum([x*pt(k*h) for x, k in zip([a, a1, a2, a3, a4], range(-2, 3))])
s = sum([x*pt(k*h) for x, k in zip([a, a1, a2, a3, a4], range(-2, 3))])
s.as_poly(f)
s.as_poly(f, f1, f2, f3, f4)
s.as_poly(f, f1, f2, f3, f4)
p = s.as_poly(f, f1, f2, f3, f4)
p
p.coeffs()
p
p.coeffs()
c = p.coeffs()

[x.as_poly().coeffs() for x in c]
M ) [x.as_poly().coeffs() for x in c]
M = [x.as_poly().coeffs() for x in c]
sp.Matrix(M[1:])
M= sp.Matrix(M[1:])
M.col_insert(3, [0, 0,1, 0])
rhs = sp.zeros(4,1)
M.col_insert (3, rhs)
M.col_insert (4, rhs)
M = M.col_insert (4, rhs)
sp.solve_linear_system(M, a, a1, a3, a4)
M.col_del(4)
rhs[2] = 1 / h**3
M.solve(rhs)
