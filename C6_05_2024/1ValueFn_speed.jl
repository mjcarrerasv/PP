# January 2021
# STEP ONE
# Value functions -- iteration

# Note: Timing is changed
    # Final firms decide on new orders from foreign good and location
    # Shocks are realized
    # Consumption and production take place


# Step 0: Create additional functions
# Linear Interpolation function for the value function
# Note vinit is a matrix with sgrid rows and lamgrid columns
# Then for each shock, [i], it interpolates among the sgrid
#vfun(i)(j) gives you from the shock i, the interpolated value of
    # the value function for s point j.
#vfuncmat(sfgrid::Array{Float64}, sdgrid::Array{Float64}, vinit::Array{Float64}) =
#interpolate( (sfgrid,sdgrid), vinit, Gridded(Linear()) )

vfuncmat(sfgrid::Array{Float64}, sdgrid::Array{Float64}, vinit::Array{Float64}) =
interpolate( (sfgrid,sdgrid), vinit, Gridded(Linear()) )

import Base.+
+(f::Function, g::Function) = (x...) -> f(x...) + g(x...)

function unconstrained_fn(
                            lamgrid0::Array{Float64},
                            # other parameters
                            pa0::Float64,
                            ya0::Float64,
                            pd0::Float64,
                            pfa0::Float64,
                            tau0::Float64,
                            th0::Float64,
                            w0::Float64,
                            parameters0::Array{Float64})

    parameters = ones(11)
    parameters::Array{Float64} .= parameters0
    sig::Float64 = parameters[5]
    beta::Float64 = parameters[1]
    epsi::Float64 = parameters[2]
    al::Float64 = parameters[3]
    delta::Float64 = parameters[4]
    ome::Float64 = parameters[10]
    #Xguess::Float64 = Xguess0
    lamgridsize::Int64 = Int(parameters[7])
    lamgrid = zeros(lamgridsize, 3)
    lamgrid::Array{Float64} .= lamgrid0

    w::Float64 = w0
    pa::Float64 = pa0       # price of sector a
    ya::Float64 = ya0       # output of sector a
    pd::Float64 = pd0       # domestic input price
    pfa::Float64 = pfa0    # foreign input price from country i
    tau::Float64 = tau0     # tariff country i
    th::Float64 = th0       # sector specific

    AL::Float64 = ((al)^al)*((1-al)^(1-al))
    EP::Float64 = epsi/(epsi-1)
    EP_::Float64 = (epsi-1)/epsi
    SI::Float64 = sig-1
    SIG::Float64 = (sig-1)/(sig)
    SIG_::Float64 = (sig)/(sig-1)
    sig_::Float64 = 1/sig

    p::Float64 = 0.0
    y::Float64 = 0.0
    x::Float64 = 0.0
    xf::Float64 = 0.0
    LS::Float64 = 0.0
    lam::Float64 = 0.0
    nu::Float64 = 0.0
    XF::Array{Float64} = zeros(lamgridsize)
    X::Array{Float64} = zeros(lamgridsize)
    L::Array{Float64} = zeros(lamgridsize)
    Y::Array{Float64} = zeros(lamgridsize)
    P::Array{Float64} = zeros(lamgridsize)
    XD::Array{Float64} = zeros(lamgridsize)
    Xguess::Array{Float64} = zeros(6)

    # vec = x1:xf, x2:x, x3:l, x4:y, x5:p, x6:xd
    for i = 1:lamgridsize
    lamf = lamgrid[i, 1]
	lamd = lamgrid[i, 2]
    nu = lamgrid[i, 3]
    LSf = (1-lamf*delta)/(1-delta)
	LSd = (1-lamd*delta)/(1-delta)

    p = (EP)*(w^(1-al))*(( (th*((LSd/pd)^SI)) + (1-th)*((LSf/(tau*pfa))^SI) )^(al/(1-sig)) )*(1/AL)
    y = (pa^epsi)*(p^(-epsi))*ya*nu
    x = (EP_)*al*p*y*( ( (th*((LSd/pd)^SI)) + (1-th)*((LSf/(tau*pfa))^SI) )^(1/(sig-1)) )
    xf = ((LSf)^sig)*(1-th)*(((EP_*al*p*y)/(tau*pfa))^sig)*(x^(1-sig))
	xd = ((LSd)^sig)*(th)*(((EP_*al*p*y)/(pd))^sig)*(x^(1-sig))
    l = ((EP_*(1-al)*p*y)/(w))

    XF[i] = xf
    X[i] = x
    L[i] = l
    Y[i] = y
    P[i] = p
    XD[i] = xd
    end
    Xguess[1] = mean(XF)
    Xguess[2] = mean(X)
    Xguess[3] = mean(L)
    Xguess[4] = mean(Y)
    Xguess[5] = mean(P)
    Xguess[6] = mean(XD)

    return Xguess
end



function cases_fn(			nf0, # optimize over n
							nd0, # optimize over n
                            sf0::Float64, # state variable
							sd0::Float64, # state variable
                            lamf0::Float64, # state variable
							lamd0::Float64, # state variable
                            nu0::Float64, # state var -- demand shock
                            # other parameters
                            pa0::Float64,
                            ya0::Float64,
                            pd0::Float64,
                            pfa0::Float64,
                            tau0::Float64,
                            th0::Float64,
                            w0::Float64,
                            parameters0::Array{Float64},
							lamgrid0::Array{Float64},
							pass::Int64
                            #Xguess0::Float64
                            )

parameters = ones(11)
  parameters::Array{Float64} .= parameters0
  sig::Float64 = parameters[5]
  epsi::Float64 = parameters[2]
  al::Float64 = parameters[3]
  delta::Float64 = parameters[4]

lamgridsize::Int64 = Int(parameters[7])
lamgrid = zeros(lamgridsize, 3)
lamgrid::Array{Float64} .= lamgrid0

  w::Float64 = w0
  pa::Float64 = pa0       # price of sector a
  ya::Float64 = ya0       # output of sector a
  pd::Float64 = pd0       # domestic input price
  pfa::Float64 = pfa0    # foreign input price from country i
  tau::Float64 = tau0     # tariff country i
  th::Float64 = th0       # sector specific
  sf::Float64 = sf0
  sd::Float64 = sd0
  nf::Float64 = nf0
  nd::Float64 = nd0
  lamf::Float64 = lamf0
  lamd::Float64 = lamd0
  nu::Float64 = nu0
  xfss = 0.0
  xdss = 0.0
	X::Array{Float64} = ones(6)
	X_1::Array{Float64} = ones(6)
	X_2::Array{Float64} = ones(6)
	X_3::Array{Float64} = ones(6)
	X_4::Array{Float64} = ones(6)

	X_1 = case_xfss_xdss(nf, nd, sf, sd, lamf, lamd, nu,
								pa, ya, pd, pfa, tau, th, w, parameters, lamgrid)
	#X_2 = case_xfc_xdc(nf, nd, sf, sd, lamf, lamd, nu,
	#							pa, ya, pd, pfa, tau, th, w, parameters, lamgrid)
	xfss = X_1[1]
	xdss = X_1[6]
#=
  # vec = x1:xf, x2:x, x3:l, x4:y, x5:p, x6:xd
  if xfss < sf + lamf*nf && xdss < sd + lamd*nd
		X = X_1
  elseif xfss > sf + lamf*nf && xdss > sd + lamd*nd
		X = X_2
  elseif xfss < sf + lamf*nf && xdss > sd + lamd*nd
		X_3 = case_xfs_xdc(nf, nd, sf, sd, lamf, lamd, nu,
									pa, ya, pd, pfa, tau, th, w, parameters, lamgrid)
    if X_3[1] < sf + lamf*nf
			X = X_3
    elseif  X_3[1] > sf + lamf*nf
			X = X_2
    end
  elseif xfss > sf + lamf*nf && xdss < sd + lamd*nd
		X_4 = case_xfc_xds(nf, nd, sf, sd, lamf, lamd, nu,
									pa, ya, pd, pfa, tau, th, w, parameters, lamgrid)
    if X_4[6] < sd + lamd*nd
			X = X_4
    elseif  X_4[6] > sd + lamd*nd
			X = X_2
    end
end
=#
# vec = x1:xf, x2:x, x3:l, x4:y, x5:p, x6:xd
Y = 0
if xfss <= sf + lamf*nf && xdss <= sd + lamd*nd
	  X = X_1
	  Y = 1
	  #@show "C1", X_1
elseif xfss <= sf + lamf*nf && xdss > sd + lamd*nd
	  X_3 = case_xfs_xdc(nf, nd, sf, sd, lamf, lamd, nu,
								  pa, ya, pd, pfa, tau, th, w, parameters, lamgrid)
  if X_3[1] <= sf + lamf*nf #&& X_3[6] <= sd + lamd*nd
		  X = X_3
		  Y = 3
		  #@show "C3", X_3
  elseif  X_3[1] > sf + lamf*nf
	  	  X_4 = case_xfc_xds(nf, nd, sf, sd, lamf, lamd, nu,
									  pa, ya, pd, pfa, tau, th, w, parameters, lamgrid)
		 if X_4[6] <= sd + lamd*nd
			 X = X_4
			 Y = 4
			 #@show "C4", X_4
		 else
		  X = case_xfc_xdc(nf, nd, sf, sd, lamf, lamd, nu,
									  pa, ya, pd, pfa, tau, th, w, parameters, lamgrid)
			Y = 2
			#@show "C2", X
		end
  end
elseif xfss > sf + lamf*nf && xdss <= sd + lamd*nd
	  X_4 = case_xfc_xds(nf, nd, sf, sd, lamf, lamd, nu,
								  pa, ya, pd, pfa, tau, th, w, parameters, lamgrid)
  if X_4[6] <= sd + lamd*nd
		  X = X_4
		  Y = 4
		  #@show "C4", X_4
  elseif  X_4[6] > sd + lamd*nd
	  	X_3 = case_xfs_xdc(nf, nd, sf, sd, lamf, lamd, nu,
								  pa, ya, pd, pfa, tau, th, w, parameters, lamgrid)
		if X_3[1] <= sf + lamf*nf
			X = X_3
			Y = 3
			#@show "C3", X_3
		else
		  X = case_xfc_xdc(nf, nd, sf, sd, lamf, lamd, nu,
									  pa, ya, pd, pfa, tau, th, w, parameters, lamgrid)
		Y = 2
		#@show "C2", X
		end
  end
elseif xfss > sf + lamf*nf && xdss > sd + lamd*nd
	X = case_xfc_xdc(nf, nd, sf, sd, lamf, lamd, nu,
								pa, ya, pd, pfa, tau, th, w, parameters, lamgrid)
	Y = 2
	#@show "C2", X
end
if pass == 1
	return X
else
	return X, Y, X_1
end

return X
end



function case_xfss_xdss(
                            nf0, # optimize over n
							nd0, # optimize over n
                            sf0::Float64, # state variable
							sd0::Float64, # state variable
                            lamf0::Float64, # state variable
							lamd0::Float64, # state variable
                            nu0::Float64, # state var -- demand shock
                            # other parameters
                            pa0::Float64,
                            ya0::Float64,
                            pd0::Float64,
                            pfa0::Float64,
                            tau0::Float64,
                            th0::Float64,
                            w0::Float64,
                            parameters0::Array{Float64},
							lamgrid0::Array{Float64}
                            #Xguess0::Float64
                            )

    parameters = ones(11)
    parameters::Array{Float64} .= parameters0
    sig::Float64 = parameters[5]
    epsi::Float64 = parameters[2]
    al::Float64 = parameters[3]
	delta::Float64 = parameters[4]

    w::Float64 = w0
    pa::Float64 = pa0       # price of sector a
    ya::Float64 = ya0       # output of sector a
    pd::Float64 = pd0       # domestic input price
    pfa::Float64 = pfa0    # foreign input price from country i
    tau::Float64 = tau0     # tariff country i
    th::Float64 = th0       # sector specific
    sf::Float64 = sf0
    sd::Float64 = sd0
    nf::Float64 = nf0
    nd::Float64 = nd0
    lamf::Float64 = lamf0
	lamd::Float64 = lamd0
    nu::Float64 = nu0

	  AL::Float64 = ((al)^al)*((1-al)^(1-al))
	  EP::Float64 = epsi/(epsi-1)
	  EP_::Float64 = (epsi-1)/epsi
	  SI::Float64 = sig-1
	  SIG::Float64 = (sig-1)/(sig)
	  SIG_::Float64 = (sig)/(sig-1)
	  sig_::Float64 = 1/sig

	Xss = ones(6)
	  #C1: Unsconstrained_ss -- x < s+lamn
	  LSf = (1-lamf*delta)/(1-delta)
	  LSd = (1-lamd*delta)/(1-delta)
	  Xss[5] = (EP)*(w^(1-al))*(( (th*((LSd/pd)^SI)) + (1-th)*((LSf/(tau*pfa))^SI) )^(al/(1-sig)))*(1/AL)
	  Xss[4] = (pa^epsi)*(Xss[5]^(-epsi))*ya*nu
	  Xss[2] = (EP_)*al*Xss[5]*Xss[4]*( ( (th*((LSd/pd)^SI)) + (1-th)*((LSf/(tau*pfa))^SI) )^(1/(sig-1)) )
	  Xss[1] = ((LSf)^sig)*(1-th)*(((EP_*al*Xss[5]*Xss[4])/(tau*pfa))^sig)*(Xss[2]^(1-sig))
	  Xss[6] = ((LSd)^sig)*(th)*(((EP_*al*Xss[5]*Xss[4])/(pd))^sig)*(Xss[2]^(1-sig))
	  Xss[3] = ((EP_*(1-al)*Xss[5]*Xss[4])/(w))
    # vec = x1:xf, x2:x, x3:l, x4:y, x5:p, x6:xd
    return Xss # r.zero
end

function case_xfs_xdc(      nf0, # optimize over n
							nd0, # optimize over n
                            sf0::Float64, # state variable
							sd0::Float64, # state variable
                            lamf0::Float64, # state variable
							lamd0::Float64, # state variable
                            nu0::Float64, # state var -- demand shock
                            # other parameters
                            pa0::Float64,
                            ya0::Float64,
                            pd0::Float64,
                            pfa0::Float64,
                            tau0::Float64,
                            th0::Float64,
                            w0::Float64,
                            parameters0::Array{Float64},
							lamgrid0::Array{Float64}
                            )

	parameters = ones(11)
    parameters::Array{Float64} .= parameters0
    sig::Float64 = parameters[5]
    epsi::Float64 = parameters[2]
    al::Float64 = parameters[3]
	delta::Float64 = parameters[4]
	lamgridsize::Int64 = Int(parameters[7])
	lamgrid = zeros(lamgridsize, 3)
	lamgrid::Array{Float64} .= lamgrid0

    w::Float64 = w0
    pa::Float64 = pa0       # price of sector a
    ya::Float64 = ya0       # output of sector a
    pd::Float64 = pd0       # domestic input price
    pfa::Float64 = pfa0    # foreign input price from country i
    tau::Float64 = tau0     # tariff country i
    th::Float64 = th0       # sector specific
    sf::Float64 = sf0
    sd::Float64 = sd0
    nf::Float64 = nf0
    nd::Float64 = nd0
    lamf::Float64 = lamf0
	lamd::Float64 = lamd0
    nu::Float64 = nu0

	AL::Float64 = ((al)^al)*((1-al)^(1-al))
    EP::Float64 = epsi/(epsi-1)
    EP_::Float64 = (epsi-1)/epsi
    SI::Float64 = sig-1
    SIG::Float64 = (sig-1)/(sig)
    SIG_::Float64 = (sig)/(sig-1)
    sig_::Float64 = 1/sig
	LSf::Float64 = (1-lamf*delta)/(1-delta)
	LSd::Float64 = (1-lamd*delta)/(1-delta)

	Xguess = case_xfss_xdss(nf, nd, sf, sd, lamf, lamd, nu,
								pa, ya, pd, pfa, tau, th, w, parameters, lamgrid)

	error = Inf
	maxiter = 500
	iter = 0
	tol = 0.000001
	xdc = sd + lamd*nd
	x = 1.0
	xfnew = 1.0
	py = 1.0
	pynew = 1.0
	l = 1.0
	y = 1.0
	p = 1.0
	xfg = Xguess[1]

	while error > tol && iter < maxiter
	x = (( (th^sig_)*(xdc^SIG) + ((1-th)^sig_)*(xfg^SIG) )^(SIG_))
	py = ((xfg*(LSf^(-sig))*(x^(sig-1))*(pfa^sig)/(1-th))^(1/sig))*(EP/al)
	l = ((EP_*(1-al)*py)/(w))
	y = (x^al)*(l)^(1-al)
	p = pa*((nu*ya/y)^(1/epsi))

	pynew = p*y
	error = abs(py - pynew)
	xfnew = ((LSf)^sig)*(1-th)*(((EP_*al*pynew)/(pfa))^sig)*(x^(1-sig))
	#@show xfg, xfnew, py, pynew
	xfg = xfnew
	end
	# vec = x1:xf, x2:x, x3:l, x4:y, x5:p, x6:xd
	vec = [xfnew, x, l, y, p, xdc]
    return vec # r.zero
end

function case_xfc_xds(      nf0, # optimize over n
							nd0, # optimize over n
                            sf0::Float64, # state variable
							sd0::Float64, # state variable
                            lamf0::Float64, # state variable
							lamd0::Float64, # state variable
                            nu0::Float64, # state var -- demand shock
                            # other parameters
                            pa0::Float64,
                            ya0::Float64,
                            pd0::Float64,
                            pfa0::Float64,
                            tau0::Float64,
                            th0::Float64,
                            w0::Float64,
                            parameters0::Array{Float64},
							lamgrid0::Array{Float64}
                            )

	parameters = ones(11)
    parameters::Array{Float64} .= parameters0
    sig::Float64 = parameters[5]
    epsi::Float64 = parameters[2]
    al::Float64 = parameters[3]
	delta::Float64 = parameters[4]
	lamgridsize::Int64 = Int(parameters[7])
	lamgrid = zeros(lamgridsize, 3)
	lamgrid::Array{Float64} .= lamgrid0

    w::Float64 = w0
    pa::Float64 = pa0       # price of sector a
    ya::Float64 = ya0       # output of sector a
    pd::Float64 = pd0       # domestic input price
    pfa::Float64 = pfa0    # foreign input price from country i
    tau::Float64 = tau0     # tariff country i
    th::Float64 = th0       # sector specific
    sf::Float64 = sf0
    sd::Float64 = sd0
    nf::Float64 = nf0
    nd::Float64 = nd0
    lamf::Float64 = lamf0
	lamd::Float64 = lamd0
    nu::Float64 = nu0

	AL::Float64 = ((al)^al)*((1-al)^(1-al))
    EP::Float64 = epsi/(epsi-1)
    EP_::Float64 = (epsi-1)/epsi
    SI::Float64 = sig-1
    SIG::Float64 = (sig-1)/(sig)
    SIG_::Float64 = (sig)/(sig-1)
    sig_::Float64 = 1/sig
	LSf::Float64 = (1-lamf*delta)/(1-delta)
	LSd::Float64 = (1-lamd*delta)/(1-delta)

	Xguess = case_xfss_xdss(nf, nd, sf, sd, lamf, lamd, nu,
								pa, ya, pd, pfa, tau, th, w, parameters, lamgrid)

	error = Inf
	maxiter = 500
	iter = 0
	tol = 0.000001
	xfc = sf + lamf*nf
	x = 1.0
	xdnew = 1.0
	py = 1.0
	pynew = 1.0
	l = 1.0
	y = 1.0
	p = 1.0
	xdg = Xguess[6]

	while error > tol && iter < maxiter
	x = (( (th^sig_)*(xdg^SIG) + ((1-th)^sig_)*(xfc^SIG) )^(SIG_))
	py = ((xdg*(LSd^(-sig))*(x^(sig-1))*(pd^sig)/(th))^(1/sig))*(EP/al)
	l = ((EP_*(1-al)*py)/(w))
	y = (x^al)*(l)^(1-al)
	p = pa*((nu*ya/y)^(1/epsi))

	pynew = p*y
	error = abs(py - pynew)
	xdnew = ((LSd)^sig)*th*(((EP_*al*pynew)/(pd))^sig)*(x^(1-sig))
	#@show xdg, xdnew, py, pynew
	xdg = xdnew
	end
	# vec = x1:xf, x2:x, x3:l, x4:y, x5:p, x6:xd
	vec = [xfc, x, l, y, p, xdnew]
    return vec # r.zero
end

function case_xfc_xdc(      nf0, # optimize over n
							nd0, # optimize over n
                            sf0::Float64, # state variable
							sd0::Float64, # state variable
                            lamf0::Float64, # state variable
							lamd0::Float64, # state variable
                            nu0::Float64, # state var -- demand shock
                            # other parameters
                            pa0::Float64,
                            ya0::Float64,
                            pd0::Float64,
                            pfa0::Float64,
                            tau0::Float64,
                            th0::Float64,
                            w0::Float64,
                            parameters0::Array{Float64},
							lamgrid0::Array{Float64}
                            #Xguess0::Float64
                            )

	parameters = ones(11)
    parameters::Array{Float64} .= parameters0
    sig::Float64 = parameters[5]
    epsi::Float64 = parameters[2]
    al::Float64 = parameters[3]
	delta::Float64 = parameters[4]
	lamgridsize::Int64 = Int(parameters[7])
	lamgrid = zeros(lamgridsize, 3)
	lamgrid::Array{Float64} .= lamgrid0

    w::Float64 = w0
    pa::Float64 = pa0       # price of sector a
    ya::Float64 = ya0       # output of sector a
    pd::Float64 = pd0       # domestic input price
    pfa::Float64 = pfa0    # foreign input price from country i
    tau::Float64 = tau0     # tariff country i
    th::Float64 = th0       # sector specific
    sf::Float64 = sf0
    sd::Float64 = sd0
    nf::Float64 = nf0
    nd::Float64 = nd0
    lamf::Float64 = lamf0
	lamd::Float64 = lamd0
    nu::Float64 = nu0

	AL::Float64 = ((al)^al)*((1-al)^(1-al))
    EP::Float64 = epsi/(epsi-1)
    EP_::Float64 = (epsi-1)/epsi
    SI::Float64 = sig-1
    SIG::Float64 = (sig-1)/(sig)
    SIG_::Float64 = (sig)/(sig-1)
    sig_::Float64 = 1/sig

	Xguess = case_xfss_xdss(nf, nd, sf, sd, lamf, lamd, nu,
								pa, ya, pd, pfa, tau, th, w, parameters, lamgrid)

	error = Inf
	maxiter = 500
	iter = 0
	tol = 0.000001

	xdc = sd + lamd*nd
	xfc = sf + lamf*nf
	x = (( (th^sig_)*(xdc^SIG) + ((1-th)^sig_)*(xfc^SIG) )^(SIG_))
	yg = Xguess[4]
	#@show yg
	ynew = 1.0
	l = 1.0
	p = 1.0

	while error > tol && iter < maxiter
	p = pa*((nu*ya/yg)^(1/epsi))
	l = ((EP_*(1-al)*p*yg)/(w))
	ynew = (x^al)*(l)^(1-al)

	error = abs(yg - ynew)
	#@show yg, ynew, error
	yg = ynew
	end

	vec = [xfc, x, l, ynew, p, xdc]
    return vec # r.zero
end

# Step 2: Compute the value function as a fn of (n, s, lam) from m=0, and m=1
    # Creates V^i(s, n, lam, m): V(s,n,lam) for m=0,1 and country=C,NA
    # And it creates a function V(n)(s,lam) of n for specific values of s,lam
function value_function(    # state variables
                            nf,
							nd,
                            sf0::Float64,
							sd0::Float64,
                            lamf0::Float64,
							lamd0::Float64,
                            nu0::Float64,
                            lamgrid0::Array{Float64},
                            # other parameters
                            pa0::Float64,
                            ya0::Float64,
                            pd0::Float64,
                            pfa0::Float64,
                            tau0::Float64,
                            th0::Float64,
                            w0::Float64,
                            sfgrid0::Array{Float64},
							sdgrid0::Array{Float64},
                            vinit0::Array{Float64},
                            parameters0::Array{Float64},
							vitp
                            #Xguess0::Float64
                            )

    parameters = ones(11)
    parameters::Array{Float64} .= parameters0
    sig::Float64 = parameters[5]
    beta::Float64 = parameters[1]
    epsi::Float64 = parameters[2]
    al::Float64 = parameters[3]
    delta::Float64 = parameters[4]
    ome::Float64 = parameters[10]

    lamgridsize::Int64 = Int(parameters[7])
    lamgrid = zeros(lamgridsize, 3)
    lamgrid::Array{Float64} .= lamgrid0

    sgridsize::Int64 = Int(parameters[6])
    #sfgridmax::Float64 = parameters[8]   # this should be a function !
	#sdgridmax::Float64 = parameters[11]
    sfgrid::Array{Float64} = ones(sgridsize)
	sfgrid .= sfgrid0
	sdgrid::Array{Float64} = ones(sgridsize)
	sdgrid .= sdgrid0
    vinit = ones(sgridsize, sgridsize)
    vinit::Array{Float64} .= vinit0

    lamf::Float64 = lamf0
	lamd::Float64 = lamd0
    nu::Float64 = nu0
    sf::Float64 = sf0
	sd::Float64 = sd0

    w::Float64 = w0
    pa::Float64 = pa0       # price of sector a
    ya::Float64 = ya0       # output of sector a
    pd::Float64 = pd0       # domestic input price
    pfa::Float64 = pfa0    # foreign input price from country i
    tau::Float64 = tau0     # tariff country i
    th::Float64 = th0       # sector specific

	#XFss, XDss = unconstrained_xf(lamgrid, pa, ya, pd, pfa, tau, th, w, parameters)
    #X(nf, nd) = Array{Function}(undef, 6)
    #if sf > (2 - minimum(lamgrid[:,1])).*maximum(XFss) && sd > (2 - minimum(lamgrid[:,2])).*maximum(XDss)
    #X(nf, nd) = case_xfss_xdss(nf, nd, sf, sd, lamf, lamd, nu,
	#							pa, ya, pd, pfa, tai, th, w, parameters, lamgrid)
    #else
	X(nf, nd) = cases_fn(nf, nd, sf, sd, lamf, lamd, nu,
							pa, ya, pd, pfa, tau, th, w, parameters, lamgrid, 1)
    #end

	EP_::Float64 = (epsi-1)/epsi
    # vec = x1:xf, x2:x, x3:l, x4:y, x5:p, x6:xd
    xf(nf, nd) = X(nf, nd)[1]
    x(nf, nd) = X(nf, nd)[2]
    l(nf, nd) = X(nf, nd)[3]
    y(nf, nd) = X(nf, nd)[4]
    p(nf, nd) = X(nf, nd)[5]
    xd(nf, nd) = X(nf, nd)[6]

    py(nf, nd) = pa*((ya*nu)^(1/epsi))*( ((x(nf, nd)^al)*(l(nf, nd)^(1-al)) )^(EP_))
    sfp(nf, nd) = (1-delta)*(sf+lamf*nf-xf(nf, nd))+(1-lamf)*nf
	sdp(nf, nd) = (1-delta)*(sd+lamd*nd-xd(nf, nd))+(1-lamd)*nd

	if sfp(nf,nd) < 1e-10
		X, Y, XSS = cases_fn(nf, nd, sf, sd, lamf, lamd, nu,
								pa, ya, pd, pfa, tau, th, w, parameters, lamgrid, 2)
		@show X, Y, XSS
		@show sfp(nf,nd), sdp(nf,nd), xf(nf, nd), xd(nf, nd), p(nf, nd)
	end
	if sdp(nf,nd) < 1e-10
		X, Y, XSS = cases_fn(nf, nd, sf, sd, lamf, lamd, nu,
								pa, ya, pd, pfa, tau, th, w, parameters, lamgrid, 2)
		@show X, Y, XSS
		@show sfp(nf,nd), sdp(nf,nd), xf(nf, nd), xd(nf, nd), p(nf, nd), nf,  nd
	end
	if isnan(nf) | isnan(nd)
		@show "nf or nd is NAN"
	end

	vitp = interpolate( (sfgrid,sdgrid), vinit, Gridded(Linear()) )

    #VF(nf, nd) =  py(nf, nd) - w*l(nf, nd) - pd*nd - pfa*tau*nf + beta*vfuncmat(sfgrid, sdgrid, vinit)(sfp(nf, nd), sdp(nf, nd))
	VF(nf, nd) =   p(nf, nd)*y(nf, nd) - w*l(nf, nd) - pd*nd - pfa*tau*nf + beta*vitp(sfp(nf, nd), sdp(nf, nd))

return VF(nf, nd)
end

# 3 Obtain value function V(s,m): V(s)0, V(s)1
# which are equal to EV(s,n,lam,m) (one for each country) (step 1.2)
function compute_EVF(      # state variables
							n,
                            #nf::Float64,
							#nd::Float64,
                            sf0::Float64,
							sd0::Float64,
                            lamgrid0::Array{Float64}, # Matrix with all shocks (lamgridsizev, shocks)
                            # other parameters
                            pa0::Float64,
                            ya0::Float64,
                            pd0::Float64,
                            pfa0::Float64,
                            tau0::Float64,
                            th0::Float64,
                            w0::Float64,
                            sfgrid0::Array{Float64},
							sdgrid0::Array{Float64},
                            vinit0::Array{Float64},
                            parameters0::Array{Float64},
							vitp
                            #Xguess0::Float64
                            )

    # Useful Parameters
    parameters = ones(11)
    parameters::Array{Float64} .= parameters0
    sgridsize::Int64 = Int(parameters[6])
	lamgridsize::Int64 = Int(parameters[7])
	lamgrid = zeros(lamgridsize, 3)
	lamgrid::Array{Float64} .= lamgrid0

	#sgridmax::Float64 = parameters[8]   # this should be a function !
	sfgrid::Array{Float64} = ones(sgridsize)
	sfgrid .= sfgrid0
	sdgrid::Array{Float64} = ones(sgridsize)
	sdgrid .= sdgrid0
	vinit = ones(sgridsize, sgridsize)
	vinit::Array{Float64} .= vinit0

	sf::Float64 = sf0
	sd::Float64 = sd0
    w::Float64 = w0
    pa::Float64 = pa0       # price of sector a
    ya::Float64 = ya0       # output of sector a
    pd::Float64 = pd0
    pfa::Float64 = pfa0
    tau::Float64 = tau0
    th::Float64 = th0   # sector specific

    VVF = Array{Function}(undef, lamgridsize)
    # for each n, obtain the mean over lamgridsize, for each s value
    for i = 1:lamgridsize
        lamf = lamgrid[i, 1]
		lamd = lamgrid[i, 2]
        nu = lamgrid[i, 3]
        VVF[i] = (n) -> value_function(n[1], n[2], sf, sd, lamf, lamd, nu, lamgrid, pa, ya, pd, pfa, tau, th, w,
                        sfgrid, sdgrid, vinit, parameters, vitp)

    end

    EVF(n) = (sum(VVF[:])(n))/lamgridsize
    #EVF(n) = ((VVF[1]+VVF[2]+VVF[3]+VVF[4]+VVF[5])(n))/lamgridsize

return EVF(n)
end


# 4 create bounds for new orders (policy) to maximize over:
    # Compute the bounds for n: nmin, nmax
    # using the fact that sp remains in the grid:
    # sp >= 0 and sp <= sgridmax
function bounds(            sf0::Float64,
							sd0::Float64,
                            lamgrid0::Array{Float64}, # Matrix with all shocks (lamgridsizev, shocks)
                            pa0::Float64,
                            ya0::Float64,
                            pd0::Float64,
                            pfa0::Float64,
                            tau0::Float64,
                            th0::Float64,
                            w0::Float64,
                            sfgrid0::Array{Float64},
							sdgrid0::Array{Float64},
                            vinit0::Array{Float64},
                            parameters0::Array{Float64})

    parameters = ones(11)
    parameters::Array{Float64} .= parameters0
    sig::Float64 = parameters[5]
    beta::Float64 = parameters[1]
    epsi::Float64 = parameters[2]
    al::Float64 = parameters[3]
    delta::Float64 = parameters[4]
    #sectors::Int64 = Int(parameters[5])
    sgridsize::Int64 = Int(parameters[6])
    lamgridsize::Int64 = Int(parameters[7])
    sfgridmax::Float64 = parameters[8]   # this should be a function !
	sdgridmax::Float64 = parameters[11]
    lamgrid = ones(lamgridsize, 3)
    lamgrid::Array{Float64} .= lamgrid0

    sfgrid = sfgrid0
	sdgrid = sdgrid0
    vinit = ones(sgridsize, sgridsize)
    vinit::Array{Float64} .= vinit0

    w::Float64 = w0
    pa::Float64 = pa0       # price of sector a
    ya::Float64 = ya0       # output of sector a
    pd::Float64 = pd0
    pfa::Float64 = pfa0
    tau::Float64 = tau0
    th::Float64 = th0   # sector specific
    sf::Float64 = sf0
	sd::Float64 = sd0

    AL::Float64 = ((al)^al)*((1-al)^(1-al))
    EP::Float64 = epsi/(epsi-1)
    EP_::Float64 = (epsi-1)/epsi
    SI::Float64 = sig-1

    SIG::Float64 = (sig-1)/(sig)
    SIG_::Float64 = (sig)/(sig-1)
    sig_::Float64 = 1/sig

    lamf_min = minimum(lamgrid[:,1])
    lamf_max = maximum(lamgrid[:,1])
	lamd_min = minimum(lamgrid[:,2])
    lamd_max = maximum(lamgrid[:,2])
#=
    nu_min = minimum(lamgrid[:,3])
    nu_max = maximum(lamgrid[:,3])
    LSf_min::Float64 = (1-lamf_min*delta)/(1-delta)
    LSf_max::Float64 = (1-lamf_max*delta)/(1-delta)
	LSd_min::Float64 = (1-lamd_min*delta)/(1-delta)
	LSd_max::Float64 = (1-lamd_max*delta)/(1-delta)

    #Opcion A: Unsconstrained -- x < s+lamn
	p_min::Float64 = (EP)*(w^(1-al))*(( (th*((LSd_max/pd)^SI)) + (1-th)*((LSf_max/(tau*pfa))^SI) )^(al/(1-sig)))*(1/AL)
    y_min::Float64 = (pa^epsi)*(p_min^(-epsi))*(ya)*nu_min
	x_min::Float64 = (EP_)*al*p_min*y_min*( ( (th*((LSd_max/pd)^SI)) + (1-th)*((LSf_max/(tau*pfa))^SI) )^(1/(sig-1)) )
    xf_min::Float64 = ((LSf_max)^sig)*(1-th)*(((EP_*al*p_min*y_min)/(tau*pfa))^sig)*(x_min^(1-sig))
	xd_min::Float64 = ((LSd_max)^sig)*(th)*(((EP_*al*p_min*y_min)/(pd))^sig)*(x_min^(1-sig))
	#xf_min::Float64 = 1e-5
	#xd_min::Float64 = 1e-5

	p_max::Float64 = (EP)*(w^(1-al))*(( (th*((LSd_min/pd)^SI)) + (1-th)*((LSf_min/(tau*pfa))^SI) )^(al/(1-sig)))*(1/AL)
    y_max::Float64 = (pa^epsi)*(p_max^(-epsi))*(ya)*nu_max
	x_max::Float64 = (EP_)*al*p_max*y_max*( ( (th*((LSd_min/pd)^SI)) + (1-th)*((LSf_min/(tau*pfa))^SI) )^(1/(sig-1)) )
    xf_max::Float64 = ((LSf_min)^sig)*(1-th)*(((EP_*al*p_max*y_max)/(tau*pfa))^sig)*(x_max^(1-sig))
	xd_max::Float64 = ((LSd_min)^sig)*(th)*(((EP_*al*p_max*y_max)/(pd))^sig)*(x_max^(1-sig))
=#

    sfmin::Float64 = 1e-10
	sdmin::Float64 = 1e-10
	nfmin::Float64 = sfmin
	ndmin::Float64 = sdmin
	nfmax::Float64 = (sfgridmax/(1 - delta*lamf_min)) - (sf/lamf_min)
	ndmax::Float64 = (sdgridmax/(1 - delta*lamd_min)) - (sd/lamd_min)
	if nfmax < 1e-5
	nfmax = 0.0001
	end
	if ndmax < 1e-5
	ndmax = 0.0001
	end
	#nfmax::Float64 = (1.3*1.2 - sf)/lamf_min
	#ndmax::Float64 = (1.3*1.2 - sd)/lamd_min

#=
    #nfmin::Float64 = min( max(sfmin/(1-lamf_max), (sfmin-(1-delta)*(sf-xf_max))/(1-delta*lamf_max)), sfgridmax)
    nfmin::Float64 = min( (sfmin-(1-delta)*(sf-0.0))/(1-delta*lamf_max), sfgridmax - 0.01)

    nfmax::Float64 = max( min(sfgridmax/(1-lamf_min), (sfgridmax-(1-delta)*(sf-xf_min))/(1-delta*lamf_min)), sfmin)
    #nfmax::Float64 = max(  (sfgridmax-(1-delta)*(sf-xf_min))/(1-delta*lamf_min), sfmin)

	#ndmin::Float64 = min( max(sdmin/(1-lamd_max), (sdmin-(1-delta)*(sd-xd_max))/(1-delta*lamd_max)), sdgridmax)
	ndmin::Float64 = min( (sdmin-(1-delta)*(sd-0.0))/(1-delta*lamd_max), sdgridmax - 0.01)

    ndmax::Float64 = max( min(sdgridmax/(1-lamd_min), (sdgridmax-(1-delta)*(sd-xd_min))/(1-delta*lamd_min)), sdmin)
    #ndmax::Float64 = max( (sdgridmax-(1-delta)*(sd-xd_min))/(1-delta*lamd_min), sdmin)
=#
return nfmin, nfmax, ndmin, ndmax
end

# 5 Maximize EVF for n (given values of s, m)
# returns V(s,m) by country and policy function n by (s,m)
function maximize_EVF(      # state variables
                            sf0::Float64,
							sd0::Float64,
                            lamgrid0::Array{Float64}, # Matrix with all shocks (lamgridsizev, shocks)
                            # other parameters
                            pa0::Float64,
                            ya0::Float64,
                            pd0::Float64,
                            pfa0::Float64,
                            tau0::Float64,
                            th0::Float64,
                            w0::Float64,
                            sfgrid0::Array{Float64},
							sdgrid0::Array{Float64},
                            vinit0::Array{Float64},
                            parameters0::Array{Float64},
                            #Xguess0::Float64,
                            #i0::Int64,
                            Pnf0::Float64,
							Pnd0::Float64,
							Tv0::Float64,
                            iter0::Int64,
							vitp
                            )

    # Useful Parameters
    parameters = ones(11)
    parameters::Array{Float64} .= parameters0
    sgridsize::Int64 = Int(parameters[6])
    lamgridsize::Int64 = Int(parameters[7])
    lamgrid = ones(lamgridsize, 3)
    lamgrid::Array{Float64} .= lamgrid0
    #Xguess::Float64 = Xguess0
    iter::Int64 = iter0
	delta = parameters[4]
	beta = parameters[1]

	sfgrid::Array{Float64} = ones(sgridsize)
	sfgrid .= sfgrid0
	sdgrid::Array{Float64} = ones(sgridsize)
	sdgrid .= sdgrid0
    vinit = ones(sgridsize, sgridsize)
    vinit::Array{Float64} .= vinit0

    sf::Float64 = sf0
	sd::Float64 = sd0
    w::Float64 = w0
    pa::Float64 = pa0       # price of sector a
    ya::Float64 = ya0       # output of sector a
    pd::Float64 = pd0
    pfa::Float64 = pfa0
    tau::Float64 = tau0
    th::Float64 = th0   # sector specific
    #i::Int64 = i0
    nfbef::Float64 = Pnf0
	ndbef::Float64 = Pnd0
	nfmax::Float64 = Inf
	ndmax::Float64 = Inf
	Tv_::Float64 = Tv0
	nfini::Float64 = 0.0001
	ndini::Float64 = 0.0001

	nfmin, nfmax2, ndmin, ndmax2 = bounds(sf, sd, lamgrid, pa, ya, pd, pfa, tau,
						th, w, sfgrid, sdgrid, vinit, parameters)
	#@show nfmax2, ndmax2
# monotonocity in the bounds!
#=
        if iter > 100000
			if nfmax2 > nfbef
				nfmax = nfbef
			else
				nfmax = nfmax2
			end
			if ndmax2 > ndbef
				ndmax = ndbef
			else
				ndmax = ndmax2
			end
		else
			nfmax = nfmax2
			ndmax = ndmax2
		end

# how to deal with nini within bounds in nbef is 1e-5? redefine ndmax:
	if ndbef < 0.00009
		ndmax = 0.0001
		ndini = 0.00009
	else
		ndini = 0.0001
	end
	if nfbef < 0.00009
		nfmax = 0.0001
		nfini = 0.00009
	else
		nfini = 0.0001
	end
=#
# if not monotonicity in bounds, then choose the max from bounds function:
	nfmax = nfmax2
	ndmax = ndmax2
    nfmin = 0.00001
	ndmin = 0.00001
	nmin = [nfmin, ndmin]
	nmax = [nfmax, ndmax]
	Xguess = unconstrained_fn(lamgrid, pa, ya, pd, pfa, tau, th, w, parameters)
	niniss = [Xguess[1], Xguess[6]]
	nini = [Xguess[1], Xguess[6]]
	if niniss[1] < nfmax
		nini[1] = niniss[1]
	else
		nini[1] = (nfmin + nfmax)/2
	end
	if niniss[2] < ndmax
		nini[2] = niniss[2]
	else
		nini[2] = (ndmin + ndmax)/2
	end
	#@show nmin, nmax, sf, sd
	#size::Int64 = 3
	#NINI = ones(size,2)
	#TVFF = ones(size)
	#POL = ones(size,2)
	#NINI[:,1] = collect(range(nfmin*2; length = size, stop = nfmax*0.99))
	#NINI[:,2] = collect(range(ndmin*2; length = size, stop = ndmax*0.99))

	 EVF(n) = -compute_EVF(n, sf, sd, lamgrid, pa, ya, pd, pfa, tau,
				th, w, sfgrid, sdgrid, vinit, parameters, vitp)

	#for i = 1:size
	 results = optimize(EVF, nmin, nmax, nini, Fminbox(LBFGS()), Optim.Options(f_tol = 1e-8, x_tol = 1e-8))
	#results = optimize(EVF, nmin, nmax, NINI[i,:], Fminbox(LBFGS()), Optim.Options(f_tol = 1e-8, x_tol = 1e-8))
	TVF = Optim.minimum(results)
	policy = Optim.minimizer(results) # n choice
	#TVFF[i] = TVF
	#POL[i,:] = policy
	#end

	#value = minimum(TVFF)
	#ind = findmin(TVFF)[2]
	#policy = POL[ind,:]
	#@show policy
	#@show TVFF, POL, value, policy #, sf + mean(lamgrid[:,1])*policy[1], sd + mean(lamgrid[:,2])*policy[2]

#other otimization routine: do they give same results?
	#res = bboptimize(EVF; SearchRange = [(nfmin, nfmax), (ndmin, ndmax)], FitnessTolerance = 1e-6)
	#Tv = best_fitness(res)
	#pol = BlackBoxOptim.minimum(res)
	#@show TVF, Tv, policy[1], pol[1], policy[2], pol[2]

	return TVF, policy[1], policy[2]
	#return value, policy[1], policy[2]
end

# 6.  Apply the maximization process for all values of s in the grid:
function Tf(                lamgrid0::Array{Float64},
                            # other parameters
                            pa0::Float64,
                            ya0::Float64,
                            pd0::Float64,
                            pfa0::Float64,
                            tau0::Float64,
                            th0::Float64,
                            w0::Float64,
                            sfgrid0::Array{Float64},
							sdgrid0::Array{Float64},
                            vinit0::Array{Float64},
                            parameters0::Array{Float64},
                            #Xguess0::Float64,
                            pass0::Int64,
                            iter0::Int64)

    parameters = ones(11)
    parameters::Array{Float64} .= parameters0
    #sectors::Int64 = Int(parameters[5])
    sgridsize::Int64 = Int(parameters[6])
    lamgridsize::Int64 = Int(parameters[7])
    lamgrid = ones(lamgridsize, 3)
    lamgrid::Array{Float64} .= lamgrid0
    pass::Int64 = pass0
    #Xguess::Float64 = Xguess0
    iter::Int64 = iter0

    #sfgridmax::Float64 = parameters[8]   # this should be a function !
	#sdgridmax::Float64 = parameters[11]
    sig::Float64 = parameters[5]
    beta::Float64 = parameters[1]
    epsi::Float64 = parameters[2]
    al::Float64 = parameters[3]
    delta::Float64 = parameters[4]

	sfgrid::Array{Float64} = ones(sgridsize)
	sfgrid .= sfgrid0
	sdgrid::Array{Float64} = ones(sgridsize)
	sdgrid .= sdgrid0
    vinit = ones(sgridsize, sgridsize)
    vinit::Array{Float64} .= vinit0

    w::Float64 = w0
    pa::Float64 = pa0       # price of sector a
    ya::Float64 = ya0       # output of sector a
    pd::Float64 = pd0
    pfa::Float64 = pfa0
    tau::Float64 = tau0
    th::Float64 = th0       # sector specific

    # preallocate memory for vectors
    Tv::Array{Float64} = ones(sgridsize, sgridsize)
    Pnf::Array{Float64} = ones(sgridsize, sgridsize)
	Pnd::Array{Float64} = ones(sgridsize, sgridsize)
#@show sfgrid,sdgrid
    # Fill in the column for n, Tv
	vitp = interpolate( (sfgrid,sdgrid), vinit, Gridded(Linear()) )

	  Tv[1,1], Pnf[1,1], Pnd[1,1] = maximize_EVF(sfgrid[1], sdgrid[1], lamgrid, pa, ya, pd, pfa,
						tau, th, w, sfgrid, sdgrid, vinit, parameters, 1000.0, 1000.0, 1000.0, iter, vitp)
	 #@show Tv[1,1], Pnf[1,1], Pnd[1,1], sfgrid[1] + mean(lamgrid[:,1])*Pnf[1,1], sdgrid[1] + mean(lamgrid[:,2])Pnd[1,1]

for i = 2:sgridsize
         Tv[i,1], Pnf[i,1], Pnd[i,1] = maximize_EVF(sfgrid[i], sdgrid[1], lamgrid, pa, ya, pd, pfa,
                            tau, th, w, sfgrid, sdgrid, vinit, parameters, Pnf[i-1,1], Pnd[i-1,1], Tv[i-1,1], iter, vitp)
    #     Tv[i,1], Pnf[i,1], Pnd[i,1], sfgrid[i] + mean(lamgrid[:,1])*Pnf[i,1], sdgrid[1] + mean(lamgrid[:,2])*Pnd[i,1]
end

for j = 2:sgridsize
       Tv[1,j], Pnf[1,j], Pnd[1,j] = maximize_EVF(sfgrid[1], sdgrid[j], lamgrid, pa, ya, pd, pfa,
                            tau, th, w, sfgrid, sdgrid, vinit, parameters, Pnf[1,j-1], Pnd[1,j-1], Tv[1,j-1], iter, vitp)
    #     Tv[1,j], Pnf[1,j], Pnd[1,j], sfgrid[1] + mean(lamgrid[:,1])*Pnf[1,j], sdgrid[j] + mean(lamgrid[:,2])*Pnd[1,j]
end

for j = 2:sgridsize
    for i = 2:sgridsize
        #Tv[i,j], Pnf[i,j], Pnd[i,j] = maximize_EVF(sfgrid[i], sdgrid[j], lamgrid, pa, ya, pd, pfa,
         #                   tau, th, w, sfgrid, sdgrid, vinit, parameters, Pnf[i-1,j], Pnd[i-1,j], Tv[i-1,j], iter)
		 Tv[i,j], Pnf[i,j], Pnd[i,j] = maximize_EVF(sfgrid[i], sdgrid[j], lamgrid, pa, ya, pd, pfa,
                            tau, th, w, sfgrid, sdgrid, vinit, parameters,
							min(Pnf[i-1,j], Pnf[i,j-1]), min(Pnd[i-1,j], Pnd[i,j-1]), min(Tv[i-1,j], Tv[i,j-1]), iter, vitp)
       # @show Tv[i,j], Pnf[i,j], Pnd[i,j], sfgrid[i] + mean(lamgrid[:,1])*Pnf[i,j], sdgrid[j]+ mean(lamgrid[:,2])*Pnd[i,j]
    end
end

#@show Tv[1:4, 1:4]

if pass == 1
    return Pnf, Pnd # returns the entire column (sgrid)
elseif pass == 2
    return Tv, Pnf, Pnd
#elseif pass == 3
#    return PROF
end

end

# 7. Compute the value function iteration of final V(s,m) (vinit replacement)
    # Iterate the Bellman Operator Matrix: Find fixed point for a given sector a (pa, w)
function Vfn(         lamgrid0::Array{Float64}, # full matrix
                      sector0::Int64,
                      pa0::Array{Float64}, # all sectors
                      ya0::Array{Float64}, # all sectors
                      pd0::Float64,
                      pfa0::Array{Float64}, # all sectors
                      tau0::Float64, # both countries
                      th0::Array{Float64}, # all sectors
                      w0::Float64,
                      sfgrid0::Array{Float64},
					  sdgrid0::Array{Float64},
                      vinit0::Array{Float64},
                      parameters0::Array{Float64},
                      #Xguess0::Float64
                      )

    #both::Float64 = both0
    iter::Int64 = 0
    max_iter::Int64 = 500
    tol::Float64 = 1e-3
    error::Float64 = Inf
	errorF::Float64 = Inf
	errorD::Float64 = Inf
    error0::Float64 = 100

    parameters = ones(11)
    parameters::Array{Float64} .= parameters0
    sgridsize::Int64 = Int(parameters[6])
    lamgridsize::Int64 = Int(parameters[7])
    sectot::Int64 = Int(parameters[9])
    lamgrid = zeros(lamgridsize, 3)
    lamgrid::Array{Float64} .= lamgrid0
	sector::Int64 = sector0

    w::Float64 = w0
    tau::Float64 = tau0
    pa = ones(sectot)
    pa::Array{Float64} .= pa0
    ya = ones(sectot)
    ya::Array{Float64} .= ya0
    pas::Float64 = pa[sector]
    yas::Float64 = ya[sector]
    tha = ones(sectot)
    tha::Array{Float64} .= th0
    th::Float64 = tha[sector]
    pd::Float64 = pd0
    pfa = ones(sectot)
    pfa::Array{Float64} .= pfa0
    pfas::Float64 = pfa[sector]
    psi::Float64 = 0.5

	sfgrid::Array{Float64} = ones(sgridsize)
	sfgrid .= sfgrid0
	sdgrid::Array{Float64} = ones(sgridsize)
	sdgrid .= sdgrid0
    v0 = ones(sgridsize, sgridsize)
    v0::Array{Float64} .= vinit0

    Tv0::Array{Float64} = ones(sgridsize, sgridsize)
    Pnf0::Array{Float64} = ones(sgridsize, sgridsize)
    Pnf00::Array{Float64} = ones(sgridsize, sgridsize)
	Pnd0::Array{Float64} = ones(sgridsize, sgridsize)
	Pnd00::Array{Float64} = ones(sgridsize, sgridsize)

while iter <= max_iter && error > tol
      Tv0, Pnf0, Pnd0 = Tf(lamgrid, pas, yas, pd, pfas, tau, th, w, sfgrid, sdgrid, v0, parameters, 2, iter)
    #@show Tv0[1,1], Pnf0[1,1], Pnd0[1,1]
    #if iter > 0
        #resultsTv = Tv0, Pnf0, Pnd0
        #save("/home/carre114/C11_08_21/C_sd/Res/resTv$iter.jld2", "resultsTv", resultsTv)
    #end

    error2 = maximum(abs.(Tv0 .- v0))
    errorF = maximum(abs.(Pnf0 .- Pnf00))
	errorD = maximum(abs.(Pnd0 .- Pnd00))
	error = max(errorF, errorD)
	#@show iter, error, error2
    if error < 0.005 && abs(error - error0) < 1e-6
    error = 1e-8
    end
    if abs(error - error0) < 1e-6
    error = 1e-7
    end
    error0 = error
    #@show iter, error, error2

    v0 .= -Tv0
	#v0 .= 0.5.*Tv0 .+ 0.5.*v0
    Pnf00 .= Pnf0
	Pnd00 .= Pnd0
    iter += 1
end

return Tv0, Pnf0, Pnd0
end

##

function unconstrained_xf(
                            lamgrid0::Array{Float64},
                            # other parameters
                            pa0::Float64, # all sectors
                            ya0::Float64, # all sectors
                            pd0::Float64,
                            pfa0::Float64, # all sectors
                            tau0::Float64, # both countries
                            th0::Float64, # all sectors
                            w0::Float64,
                            #sgrid0::StepRangeLen{Float64},
                            parameters0::Array{Float64})

    parameters = ones(11)
    parameters::Array{Float64} .= parameters0

    sectot::Int64 = Int(parameters[9])
    sig::Float64 = parameters[5]
    beta::Float64 = parameters[1]
    epsi::Float64 = parameters[2]
    al::Float64 = parameters[3]
    delta::Float64 = parameters[4]
    ome::Float64 = parameters[3]

    lamgridsize::Int64 = Int(parameters[7])
    lamgrid = zeros(lamgridsize, 3)
    lamgrid::Array{Float64} .= lamgrid0
    #sector::Int64 = sector0
    w::Float64 = w0
    tau::Float64 = tau0    # China is in the first element
    pa::Float64 = pa0
    ya::Float64 = ya0
    th::Float64 = th0
    pd::Float64 = pd0
    pfa::Float64 = pfa0 # China is in the first column

    AL::Float64 = ((al)^al)*((1-al)^(1-al))
    EP::Float64 = epsi/(epsi-1)
    EP_::Float64 = (epsi-1)/epsi
    SI::Float64 = sig-1
    SIG::Float64 = (sig-1)/(sig)
    SIG_::Float64 = (sig)/(sig-1)
    sig_::Float64 = 1/sig

    p::Float64 = 0.0
    y::Float64 = 0.0
    x::Float64 = 0.0
    xf::Float64 = 0.0
    LS::Float64 = 0.0
    lam::Float64 = 0.0
    nu::Float64 = 0.0
    XF::Array{Float64} = zeros(lamgridsize)
	XD::Array{Float64} = zeros(lamgridsize)

	for i = 1:lamgridsize
    lamf = lamgrid[i, 1]
	lamd = lamgrid[i, 2]
    nu = lamgrid[i, 3]
    LSf = (1-lamf*delta)/(1-delta)
	LSd = (1-lamd*delta)/(1-delta)

    p = (EP)*(w^(1-al))*(( (th*((LSd/pd)^SI)) + (1-th)*((LSf/(tau*pfa))^SI) )^(al/(1-sig)))*(1/AL)
    y = (pa^epsi)*(p^(-epsi))*ya*nu
    x = (EP_)*al*p*y*( ( (th*((LSd/pd)^SI)) + (1-th)*((LSf/(tau*pfa))^SI) )^(1/(sig-1)) )
    xf = ((LSf)^sig)*(1-th)*(((EP_*al*p*y)/(tau*pfa))^sig)*(x^(1-sig))
	xd = ((LSd)^sig)*(th)*(((EP_*al*p*y)/(pd))^sig)*(x^(1-sig))
    l = ((EP_*(1-al)*p*y)/(w))

    XF[i] = xf
    XD[i] = xd
    end
    # vec = x1:xf, x2:x, x3:l, x4:y, x5:p, x6:xd

    return XF, XD
end

function sgrid_initial(beta0::Float64,
                      epsi0::Float64,
                      al0::Float64,
                      delta0::Float64,
                      sigma0::Float64,
                      sgridsize0::Int64,
                      lamgridsize0::Int64,
                      sectot0::Int64,
                      ome0::Float64,
                      Cg0::Float64,
                      Ng0::Float64,
                      pag0::Array{Float64},
                      wg0::Float64,
                      L0::Float64,
                      lamgrid0::Array{Float64},
                      Gamma0::Array{Float64},
                      tau0::Float64,
                      th0::Array{Float64}
                      )

  beta::Float64 = beta0
  epsi::Float64 = epsi0
  al::Float64 = al0
  delta::Float64 = delta0
  sigma::Float64 = sigma0
  sgridsize::Int64 = sgridsize0
  lamgridsize::Int64 = lamgridsize0
  sectot::Int64 = sectot0
  ome::Float64 = ome0
  Ng::Float64 = Ng0
  Cg::Float64 = Cg0
  pag::Array{Float64} = ones(sectot)
  pag .= pag0
  wg::Float64 = wg0
  L::Float64 = L0
  lamgrid = zeros(lamgridsize, sectot .+ 2)
  lamgrid::Array{Float64} .= lamgrid0
  Gamma::Array{Float64} = ones(sectot)
  Gamma .= Gamma0
  th = ones(sectot)
  th .= th0
  tau = tau0

  parameters2 = [beta, epsi, al, delta, sigma, sgridsize, lamgridsize, sectot, ome]
  yaini, pdini, Piniv = Aggregates0(Cg, Ng, pag, wg, Gamma, L, parameters2)
  parameters = [beta, epsi, al, delta, sigma, sgridsize, lamgridsize, 20.0, sectot, ome, 20.0]

  pfmin = ones(sectot).*pdini
  XFC = ones(lamgridsize, sectot)
  XDC = ones(lamgridsize, sectot)
  lamgridsec = ones(lamgridsize, 3)

  lamgridsec[:,1] .= lamgrid[:,1]
  lamgridsec[:,2] .= lamgrid[:,2]
  for sec = 1:sectot
  lamgridsec[:,3] = lamgrid[:, sec+2]
  XFC[:,sec], XDC[:,sec] = unconstrained_xf(lamgridsec, pag[sec], yaini[sec], pdini, pfmin[sec], tau, th[sec], wg, parameters)
  end

  #sfgridmax = ((1-delta*minimum(lamgrid[:,1]))*maximum(XFC)*1.35/minimum(lamgrid[:,1]))
  #sdgridmax = ((1-delta*minimum(lamgrid[:,2]))*maximum(XDC)*1.35/minimum(lamgrid[:,2]))
 	sfgridmax = maximum(XFC)*10
 	sdgridmax = maximum(XDC)*2
 	#sfgridmax = mean(XFC)
 	#sdgridmax = mean(XDC)

 #sfgrid = collect(range(1e-10; length = sgridsize, stop = sfgridmax))
 #sdgrid = collect(range(1e-10; length = sgridsize, stop = sdgridmax))

 sf1 = collect(range(1e-10; length = sgridsize, stop = sfgridmax))
 sd1 = collect(range(1e-10; length = sgridsize, stop = sdgridmax))
 sf_c = 1.0
 sd_c = 1.0
 sf_c = sf1[Int((1/3)*sgridsize)]
 sd_c = sd1[Int((1/3)*sgridsize)]

#1e-10
 sf2 = collect(range(1e-10; length = Int((2/3)*sgridsize), stop = sf_c))
 sd2 = collect(range(1e-10; length = Int((2/3)*sgridsize), stop = sd_c))

 sf3 = collect(range(sf_c; length = Int((1/3)*sgridsize+1), stop = sfgridmax))
 sd3 = collect(range(sd_c; length = Int((1/3)*sgridsize+1), stop = sdgridmax))

 sfgrid = ones(sgridsize)
 sdgrid = ones(sgridsize)
 sfgrid[1:Int((2/3)*sgridsize)] = sf2
 sfgrid[Int((2/3)*sgridsize +1):sgridsize] = sf3[2:Int((1/3)*sgridsize+1)]
 sdgrid[1:Int((2/3)*sgridsize)] = sd2
 sdgrid[Int((2/3)*sgridsize +1):sgridsize] = sd3[2:Int((1/3)*sgridsize+1)]

@show XFC, XDC, lamgrid,  mean(XFC), mean(XDC), sfgridmax, sdgridmax

  return sfgridmax, sdgridmax, sfgrid, sdgrid
end

function check1(sf, sd,
	lamgrid::Array{Float64}, # Matrix with all shocks (lamgridsizev, shocks)
	# other parameters
	pa::Array{Float64},
	ya::Array{Float64},
	pd::Float64,
	pfa::Array{Float64},
	tau::Float64,
	th::Array{Float64},
	w::Float64,
	sfgrid::Array{Float64},
	sdgrid::Array{Float64},
	vinit::Array{Float64},
	parameters::Array{Float64})

	nfmin, nfmax, ndmin, ndmax = bounds(sf, sd, lamgrid, pa[1], ya[1], pd, pfa[1], tau,
						th[1], w, sfgrid, sdgrid, vinit, parameters)
@show sf, sd, mean(lamgrid[:,1]), mean(lamgrid[:,2]), mean(lamgrid[:,3]),
							pa[1], ya[1], pd, pfa[1], tau, th[1], w, parameters, lamgrid
size::Int64 = 10
NINI = ones(size,2)
NINI[:,1] = collect(range(0.0001; length = size, stop = nfmax*0.99))
NINI[:,2] = collect(range(0.0001; length = size, stop = ndmax*0.99))

eval = ones(size)
for i = 1:size
	eval[i] = -compute_EVF(NINI[i,:], sf, sd, lamgrid, pa[1], ya[1], pd, pfa[1], tau, th[1], w, sfgrid, sdgrid, vinit, parameters)
X, Y, X_1 = cases_fn(NINI[i,1], NINI[i,2], sf, sd, mean(lamgrid[:,1]), mean(lamgrid[:,2]), mean(lamgrid[:,3]),
						pa[1], ya[1], pd, pfa[1], tau, th[1], w, parameters, lamgrid, 2)
# vec = x1:xf, x2:x, x3:l, x4:y, x5:p, x6:xd
xf = X[1]
x = X[2]
l = X[3]
y = X[4]
p = X[5]
xd = X[6]

Prof = p.*y .- w.*l .- pd.*NINI[i,2] .- pfa.*tau.*NINI[i,1]
@show Prof, p*y, p, y, xf, xd, l, Y, X_1
#@show Prof, p(nf, nd), y(nf, nd), p(nf, nd)*y(nf, nd)
end
@show eval
end
