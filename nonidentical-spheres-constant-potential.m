#!/usr/bin/octave -q
#
# Numerical solution of the nonlinear Poisson-Boltzmann equation in bispherical 
# coordinates, in the case of two nonidentical spheres. The nonlinear 
# Poisson-Boltzmann equation and boundary conditions read,
#
# dlen^2 Laplacian f(xi,eta) = sinh f(xi,eta), 
#
# df/dxi |_xi=0 = 0     (the line of symmetry between the spheres),
# df/dxi |_xi=pi = 0    (the line of symmetry behind each of the spheres),
# f(xi,eta1) = fSphere1 (the constant potential at the first sphere surface),
# f(xi,eta2) = fSphere2 (the constant potential at the second sphere surface),
#
# where dlen is the Debye length. Note that bispherical coordinates are 
# connected to cylindrical coordinates rho and z via,
#
# rho = const * sin(xi) / ( cosh eta - cos xi ),
# z = const * sinh(eta) / ( cosh eta - cos xi ),
#
# where const is determined by the distance between spheres and their radius 
# (see Eqs. 3 in Ref. 1). Also note that nothing depends on phi due to 
# cylindrical symmetry of the problem at hand. 
#
# Bibliography,
#
# 1. S. L. Carnie, D. Y. C. Chan, and J. Stankovich, 
# "Computation of forces between spherical colloidal particles, nonlinear 
# Poisson-Boltzmann theory", J. Col. Int. Sci. 165, 116 (1994).
#
# 2. P. Warszynski, and Z. Adamczyk, 
# "Calculations of double-layer electrostatic interactions for the sphere-plane 
# geometry", J. Col. Int. Sci. 187, 283 (1997).
#
# 3. J. Stankovich, and S. L. Carnie,  
# "Electrical double layer interaction between dissimilar spherical colloidal 
# particles and between a sphere and a plate: nonlinear Poisson-Boltzmann theory", 
# Langmuir 12, 1453 (1997).
#

argList = argv();

# Problem parameters,
dlen     = str2num( argList{1} ); # Debye length,
r2r1     = str2num( argList{2} ); # ratio of radii of the spheres,
sr1      = 1.;                    # closest approach between the speres divided by the first sphere radius,
fSphere1 = str2num( argList{3} ); # potential on the first sphere surface,
fSphere2 = str2num( argList{4} ); # potential on the second sphere surface;

#for sr1 = [0.05:0.05:1] 
for sr1 = [0.02:0.02:0.28, 0.3:0.05:1.5]

bsCoef = ( sr1 * ( 2 + sr1 ) * ( 2 * r2r1 + sr1 ) * ( 2 + 2 * r2r1 + sr1 ) )^0.5 / ( 2 * ( 1 + r2r1 + sr1 ) ); 
# coefficient from the expression for bispherical coordinates SCALED BY dlen!
# (given by the soluton of Eqs. (5)-(7) from Ref. 3);

eta1 = asinh( bsCoef );          # see Eq. (5) in Ref. 3.
eta2 = asinh( - bsCoef / r2r1 ); # see Eq. (6) in Ref. 3.

# Grid parameters,
n1   = str2num( argList{5} );    # number of grid points along xi or eta axis.
n2   = n1 * n1;
dXi  = pi / (n1+1);
dEta = ( eta1 - eta2 ) / (n1+1);
xis  = dXi * [1:n1]';
etas = eta2 + dEta * [1:n1]';

# Differentiation matirces,
dMatXi = spdiags( kron( ones(n2,1), [-1,1] ), [-n1,n1], n2, n2 ) / ( 2 * dXi );
dMatXi2 = spdiags( kron( ones(n2,1), [1,-2,1] ), [-n1,0,n1], n2, n2 ) / dXi^2;

dMatEta = spdiags( [ -( mod( [1:n2]', n1 ) != 0 ), mod( [1:n2]'-1, n1 ) != 0 ], [-1,1], n2, n2 ) / ( 2 * dEta );
dMatEta2 = spdiags( [ mod( [1:n2]', n1 ) != 0, -2 * ones(n2,1), mod( [1:n2]'-1, n1 ) != 0 ], ...
  [-1,0,1], n2, n2 ) / dEta^2;

# Boundary conditions:
# 1) the line of symmetry between the sphere and the wall;
# note the coefficients 4/3 and 2/3, they are necessary to reach second-order precision,
dMatXi( 1:n1, 1:n1 ) = - 4./3. * speye(n1) / ( 2. * dXi );
dMatXi( 1:n1, n1+1:2*n1 ) *= 4./3.;
dMatXi2( 1:n1, 1:n1 ) /= 3.;
dMatXi2( 1:n1, n1+1:2*n1 ) *= 2./3.;

# 2) the line of symmetry behind the sphere,
# note the coefficients 4/3 and 2/3, they are necessary to reach second-order precision,
dMatXi( n2-n1+1:n2, n2-n1+1:n2 ) = 4./3. * speye(n1) / ( 2 * dXi );
dMatXi( n2-n1+1:n2, n2-2*n1+1:n2-n1 ) *= 4./3.;
dMatXi2( n2-n1+1:n2, n2-n1+1:n2 ) /= 3.;
dMatXi2( n2-n1+1:n2, n2-2*n1+1:n2-n1 ) *= 2./3.;

# 3) the constant potential at the second sphere surface,
bcs = zeros( n1 );
bcs(1,:) = bcs(1,:) - fSphere2 * ( dEta^(-2) + sinh( eta2 ) ./ ( cosh( eta2 ) - cos( xis' ) ) / ( 2 * dEta ) );
# ^-- f(xi,0) = fPlate,
# MIND THE STRUCTURE OF LAPLACES OPERATOR IN BISPHERICAL COORDINATES!
# "+" in front of sinh is due to "-" in finite difference approxiamtion of the first derivative.

# 4) the constant potential at the first sphere surface,
bcs(n1,:) = bcs(n1,:) - fSphere1 * ( dEta^(-2) - sinh( eta1 ) ./ ( cosh( eta1 ) - cos( xis' ) ) / ( 2 * dEta ) ); 
# ^-- f(xi,eta0) = fSphere,
# MIND THE STRUCTURE OF LAPLACES OPERATOR IN BISPHERICAL COORDINATES!
bcsR = reshape( bcs, n2, 1 );

# Laplacian in bispherical coordinates (see Eq. (A4) in Ref. 2), 
for i = 1:n1
  for j = 1:n1
    coef1Mat(i,j) = ( cosh( etas(i) ) * cos( xis(j) ) - 1 ) ...
      / ( sin( xis(j) ) * ( cosh( etas(i) ) - cos( xis(j) ) ) );
    coef2Mat(i,j) = -sinh( etas(i) ) / ( cosh( etas(i) ) - cos( xis(j) ) );
    coef3Mat(i,j) = ( bsCoef / dlen )^2 / ( cosh( etas(i) ) - cos( xis(j) ) )^2;
  end
end
coef1 = spdiags( reshape( coef1Mat, n2, 1 ), [0], n2, n2 );
coef2 = spdiags( reshape( coef2Mat, n2, 1 ), [0], n2, n2 );
coef3 = spdiags( reshape( coef3Mat, n2, 1 ), [0], n2, n2 );
delta = dMatXi2 + coef1 * dMatXi + dMatEta2 + coef2 * dMatEta; 

#
# Solution via Newton iterations (see Eq. (23) in Ref. 1),
#
# Laplacian f+ = sinh f + d (sinh f) / df * ( f+ - f ),
#
# where f+ is the solution at the next step of iteration process.
#
fi = zeros( n2, 1 );
intResNew = sum( abs( ( delta * fi - coef3 * sinh(fi) - bcsR ) ./ ( delta * fi ) ) ) * dXi * dEta;
do
  fi = ( delta - coef3 * spdiags( cosh(fi), [0], n2, n2 ) ) ...
    \ ( coef3 * ( sinh(fi) - cosh(fi) .* fi ) + bcsR );
  intResOld = intResNew;
  intResNew = sum( abs( ( delta * fi - coef3 * sinh(fi) - bcsR ) ./ ( delta * fi ) ) ) * dXi * dEta;

  #'---'
  #sum( abs( ( delta * fi - coef3 * sinh(fi) - bcsR ) ./ ( delta * fi ) ) ) * dXi * dEta
  #max( abs( ( delta * fi - coef3 * sinh(fi) - bcsR ) ./ ( delta * fi ) ) )
until intResOld < intResNew

#surf( xis, etas, reshape( fi, n1, n1 ) );

#
# Computation of the dimensionless force (see Eq. (12) in Ref. 3),
#
fFin = reshape( fi, n1, n1 );
dFdXiFin = reshape( dMatXi * fi, n1, n1 );
dFdEtaFin = reshape( dMatEta * fi, n1, n1 );
etaF = etas( n1/2 );
force = 2 * pi * sum( ( ...
    ( ( bsCoef / dlen )^2 * ( cosh( fFin(n1/2,:)' ) - 1 ) ./ ( cosh( etaF ) - cos( xis ) ).^2 ...
      + ( ( dFdXiFin(n1/2,:)' ).^2 - ( dFdEtaFin(n1/2,:)' ).^2 ) / 2 ) ...
        .* ( 1. - cosh( etaF ) * cos( xis ) ) ...
    + sinh( etaF ) * sin( xis ) .* dFdXiFin(n1/2,:)' .* dFdEtaFin(n1/2,:)' ...
  ) .* sin( xis ) ./ ( cosh( etaF ) - cos( xis ) ) ) * dXi;

relErr = sum( abs( ( delta * fi - coef3 * sinh(fi) - bcsR ) ./ ( delta * fi ) ) ) * dXi * dEta;

# NOTE THE SCALING, OK?
# IT MATCHES THE FORCE:
# F=-64*\pi*tanh(zeta_1/4)*tanh(zeta_2/4)*(rho_1*rho_2)/(rho_1+rho_2+D)/delta*exp(-D/delta).
# Ory writes: Here zeta is the "surface potential" normalised by phi_t, D=h/a 
# (a is the particle radius), and delta is the dimensionless Debye length (kappa*a)^(-1). 
printf( "%f, %e, %e, %e\n", sr1, force * dlen / exp( -sr1 / dlen ), force, relErr );
# Also note that rad1 is taken as a unit of length here!

#plot( xis, fFin(1,:).^2 );
#plot( xis, dFdXiFin(1,:).^2 );
#waitforbuttonpress();

end
