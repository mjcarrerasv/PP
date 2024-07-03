# Value function algorithm
# May 2024

# Note: Timing is changed:
    # shocks are realized
    # then decisions (all, including dynamic) are made

##############################################################################################################################
#@everywhere using Optim
@everywhere using Interpolations, Optim
@everywhere using Statistics
#parametersv = [betav, epsiv, sigmav, deltav, thv, sgridsizev, shockgridsizev]

#@everywhere V_int(sgrid::Array{Float64}, col::Int64, vinit::Array{Float64}) = interpolate( (sgrid,), vinit[:,col], Gridded(Linear()) )
#@everywhere V_int(sdgrid::Array{Float64}, sfgrid::Array{Float64}, col::Int64, vinit::Array{Float64}) = interpolate( (sdgrid,sfgrid), vinit[:,:,col], Gridded(Linear()) )

@everywhere vfuncmat(sfgrid::Array{Float64}, sdgrid::Array{Float64}, vinit::Array{Float64}) =
interpolate( (sfgrid,sdgrid), vinit, Gridded(Linear()) )

#@everywhere vitp(sfgrid::Array{Float64}, sdgrid::Array{Float64}, vinit::Array{Float64}) = interpolate( (sfgrid,sdgrid), vinit, Gridded(Linear()) )
##############################################################################################################################
@everywhere function unconstrained_fn(
                            nu0::Float64,
                            pd0::Float64,
                            pf0::Float64,
                            parameters0::Array{Float64})

nu::Float64 = nu0

parameters = ones(7)
parameters::Array{Float64} .= parameters0
beta::Float64 = parameters[1]
epsi::Float64 = parameters[2]
sig::Float64 = parameters[3]
delta::Float64 = parameters[4]
th::Float64 = parameters[5]              

pd::Float64 = pd0                                        # domestic input price
pf::Float64 = pf0                                       # foreign input price from country i

EP::Float64 = epsi/(epsi-1)
EP_::Float64 = (epsi-1)/epsi
SI::Float64 = 1-sig
sig_::Float64 = 1/sig

p::Float64 = 0.0
y::Float64 = 0.0
x::Float64 = 0.0
xf::Float64 = 0.0
xd::Float64 = 0.0
X_UNC::Array{Float64} = zeros(5)

    p = (EP)*(( th*((pd)^SI) + (1-th)*((pf)^SI)  )^(1/SI))
    y = nu*(p^(-epsi))
    x = ( ( (p*y*EP_)^(sig-1) )*( th*((pd)^SI) + (1-th)*((pf)^SI) ) )^(sig_)
    xf = (1-th)*(((EP_*p*y)/(x*pf))^sig)
    xd = (th)*(((EP_*p*y)/(x*pd))^sig)

X_UNC[1] = p
X_UNC[2] = y
X_UNC[3] = x
X_UNC[4] = xf
X_UNC[5] = xd

return X_UNC
end

##############################################################################################################################
@everywhere function constrained_xf(
                            nf0::Float64,
                            sf0::Float64,
                            nu0::Float64,
                            lamf0::Float64,
                            pd0::Float64,
                            pf0::Float64,
                            parameters0::Array{Float64})

nu::Float64 = nu0
lam::Float64 = lamf0
s::Float64 = sf0
n::Float64 = nf0

parameters = ones(7)
parameters::Array{Float64} .= parameters0
epsi::Float64 = parameters[2]
sig::Float64 = parameters[3]
th::Float64 = parameters[5]              

pd::Float64 = pd0                                        # domestic input price
pf::Float64 = pf0                                       # foreign input price from country i

EP::Float64 = epsi/(epsi-1)
EP_::Float64 = (epsi-1)/epsi
SIG::Float64 = (sig-1)/sig
sig_::Float64 = 1/sig

p::Float64 = 0.0
y::Float64 = 0.0
x::Float64 = 0.0
xdg::Float64 = 0.0
xdnew::Float64 = 0.0

X_UNC::Array{Float64} = zeros(5)
X_UNC = unconstrained_fn(nu, pd, pf, parameters)

xf::Float64 = s + lam*n
#xdg = X_UNC[5]
xdg = 0.1

error = Inf
maxiter = 500
iter = 0
tol = 1e-7

    while error > tol && iter < maxiter

    x = ((th)^(sig_)*(xdg)^(SIG)) + ((1-th)^(sig_)*(xf)^(SIG))
    y = x^(1/SIG)
    p = (nu/y)^(1/epsi)

    xdnew = (th)*(((EP_*p*y)/(x*pd))^sig)

    error = abs(xdg - xdnew)
    xdg = xdnew
    iter += 1

    end
 
    X_CON::Array{Float64} = zeros(5)
    X_CON[1] = p
    X_CON[2] = y
    X_CON[3] = x
    X_CON[4] = xf
    X_CON[5] = xdnew
# x1 = p, x2 = y, x3 = x, x4 = xf, x5 = xd 

return X_CON
end

##############################################################################################################################
@everywhere function constrained_xd(
                                nd0::Float64,
                                sd0::Float64,
                                nu0::Float64,
                                lamd0::Float64,
                                pd0::Float64,
                                pf0::Float64,
                                parameters0::Array{Float64})

nu::Float64 = nu0
lam::Float64 = lamd0
s::Float64 = sd0
n::Float64 = nd0

parameters = ones(7)
parameters::Array{Float64} .= parameters0
epsi::Float64 = parameters[2]
sig::Float64 = parameters[3]
th::Float64 = parameters[5]              

pd::Float64 = pd0                                        # domestic input price
pf::Float64 = pf0                                       # foreign input price from country i

EP::Float64 = epsi/(epsi-1)
EP_::Float64 = (epsi-1)/epsi
SIG::Float64 = (sig-1)/sig
sig_::Float64 = 1/sig

p::Float64 = 0.0
y::Float64 = 0.0
x::Float64 = 0.0
xfg::Float64 = 0.0
xfnew::Float64 = 0.0

X_UNC::Array{Float64} = zeros(5)
X_UNC = unconstrained_fn(nu, pd, pf, parameters)

xd::Float64 = s + lam*n
#xdg = X_UNC[5]
xfg = 0.1

error = Inf
maxiter = 500
iter = 0
tol = 1e-7

while error > tol && iter < maxiter

x = ((th)^(sig_)*(xd)^(SIG)) + ((1-th)^(sig_)*(xfg)^(SIG))
y = x^(1/SIG)
p = (nu/y)^(1/epsi)

xfnew = (1-th)*(((EP_*p*y)/(x*pf))^sig)

error = abs(xfg - xfnew)
xfg = xfnew
iter += 1

end

X_CON::Array{Float64} = zeros(5)
X_CON[1] = p
X_CON[2] = y
X_CON[3] = x
X_CON[4] = xfnew
X_CON[5] = xd
# x1 = p, x2 = y, x3 = x, x4 = xf, x5 = xd 

return X_CON
end

##############################################################################################################################
@everywhere function constrained_both(
                                nd0::Float64,
                                nf0::Float64,
                                sd0::Float64,
                                sf0::Float64,
                                nu0::Float64,
                                lamd0::Float64,
                                lamf0::Float64,
                                parameters0::Array{Float64})

nu::Float64 = nu0
lamd::Float64 = lamd0
sd::Float64 = sd0
nd::Float64 = nd0
lamf::Float64 = lamf0
sf::Float64 = sf0
nf::Float64 = nf0

parameters = ones(7)
parameters::Array{Float64} .= parameters0
epsi::Float64 = parameters[2]
sig::Float64 = parameters[3]
th::Float64 = parameters[5]              

EP::Float64 = epsi/(epsi-1)
EP_::Float64 = (epsi-1)/epsi
SIG::Float64 = (sig-1)/sig
sig_::Float64 = 1/sig

p::Float64 = 0.0
y::Float64 = 0.0
x::Float64 = 0.0
xf::Float64 = 0.0
xd::Float64 = 0.0

xd = sd + lamd*nd
xf = sf + lamf*nf
x = ((th)^(sig_)*(xd)^(SIG)) + ((1-th)^(sig_)*(xf)^(SIG))
y = x^(1/SIG)
p = (nu/y)^(1/epsi)

X_CON::Array{Float64} = zeros(5)
X_CON[1] = p
X_CON[2] = y
X_CON[3] = x
X_CON[4] = xf
X_CON[5] = xd
# x1 = p, x2 = y, x3 = x, x4 = xf, x5 = xd 

return X_CON
end

##############################################################################################################################
@everywhere function cases_fn(
                    nd,
                    nf,
                    sd0::Float64,
                    sf0::Float64,
                    nu0::Float64,
                    lamd0::Float64,
                    lamf0::Float64,
                    pd0::Float64,
                    pf0::Float64,
                    parameters0::Array{Float64})

nu::Float64 = nu0
lamd::Float64 = lamd0
sd::Float64 = sd0
lamf::Float64 = lamf0
sf::Float64 = sf0

parameters = ones(7)
parameters::Array{Float64} .= parameters0
            
pd::Float64 = pd0                                        # domestic input price
pf::Float64 = pf0                                       # foreign input price from country i
xf_unc::Float64  = 0.0
xd_unc::Float64  = 0.0

X_UNC::Array{Float64} = zeros(5)
X::Array{Float64} = zeros(5)
case::Float64 = -10.0

X_UNC = unconstrained_fn(nu, pd, pf, parameters)
xf_unc = X_UNC[4]
xd_unc = X_UNC[5]

    if xd_unc <= sd + lamd*nd && xf_unc <= sf + lamf*nf
        X = X_UNC
        case = 0.0  #unconstrained

    elseif xd_unc > sd + lamd*nd && xf_unc > sf + lamf*nf
        X  = constrained_both(nd, nf, sd, sf, nu, lamd, lamf, parameters)
        case = 3.0  #both constrained

    elseif xd_unc > sd + lamd*nd && xf_unc <= sf + lamf*nf
        X  = constrained_xd(nd, sd, nu, lamd, pd, pf, parameters)
        case = 1.0      # xd constrained

    elseif xd_unc <= sd + lamd*nd && xf_unc > sf + lamf*nf
        X  = constrained_xf(nf, sf, nu, lamf, pd, pf, parameters)
        case = 2.0      # xf constraned

    else 
        @show "error", xd_unc,  sd + lamd*nd,  xf_unc, sf + lamf*nf
    end

    # x1 = p, x2 = y, x3 = x, x4 = xf, x5 = xd
return X, case

end

##############################################################################################################################
## Creates a function V(n)(s,lam) of n for specific values of s,lam
@everywhere function value_fn(                
                    n,            
                    #nd,
                    #nf,
                    sd0::Float64,
                    sf0::Float64,
                    nu0::Float64,
                    lamd0::Float64,
                    lamf0::Float64,
                    pd0::Float64,
                    pf0::Float64,
                    parameters0::Array{Float64},
                    sdgrid0::Array{Float64},
                    sfgrid0::Array{Float64},
                    v0::Array{Float64}
                    )

nu::Float64 = nu0
sd::Float64 = sd0
lamd::Float64 = lamd0
sf::Float64 = sf0
lamf::Float64 = lamf0

parameters = ones(7)
parameters::Array{Float64} .= parameters0
beta::Float64 = parameters[1]
delta::Float64 = parameters[4]  
sgridsize::Int64 = Int(parameters[6])
shockgridsize::Int64 = Int(parameters[7])

vinit::Array{Float64} = ones(sgridsize, sgridsize)
vinit .= v0
sdgrid::Array{Float64} = ones(sgridsize)
sdgrid .= sdgrid0
sfgrid::Array{Float64} = ones(sgridsize)
sfgrid .= sfgrid0
#Vgrid = Array{Function}(undef, shockgridsize)

pd::Float64 = pd0                                        # domestic input price
pf::Float64 = pf0       

# note n = [nd, nf]
X(n) = cases_fn(n[1], n[2], sd, sf, nu, lamd, lamf, pd, pf, parameters)
# x1 = p, x2 = y, x3 = x, x4 = xf, x5 = xd

p(n) = X(n)[1][1]
y(n) = X(n)[1][2]
x(n) = X(n)[1][3]
xf(n) = X(n)[1][4]
xd(n) = X(n)[1][5]
sfp(n) = (1-delta)*(sf + n[2] - xf(n))
sdp(n) = (1-delta)*(sd + n[1] - xd(n))

#=
if sfp(n) > sfgrid[sgridsize]
    @show "sfp larger than grid"
    sfp(n) = sfgrid[sgridsize]
end
if sdp(n) > sdgrid[sgridsize]
    @show "sdp larger than grid"
    sdp(n) = sdgrid[sgridsize]
end
=#

#@everywhere vfuncmat(sfgrid::Array{Float64}, sdgrid::Array{Float64}, vinit::Array{Float64}) =
#interpolate( (sfgrid,sdgrid), vinit, Gridded(Linear()) )

#VF(n) =  p(n)*y(n) - pd*n[1]- pf*n[2] + beta*(sum(Vgrid[:])(n))/shockgridsize
VF(n) =  p(n)*y(n) - pd*n[1] - pf*n[2] + beta*(vfuncmat(sdgrid, sfgrid, vinit)(sdp(n), sfp(n)))

#for i = 1:shockgridsize
#    Vgrid[i] = (n) -> V_int(sdgrid, sfgrid, i, vinit)(sdp(n), sfp(n))
#end

return VF(n)

end

##############################################################################################################################
@everywhere function compute_EVF(                
    n,            
    #nd,
    #nf,
    sd0::Float64,
    sf0::Float64,
    pd0::Float64,
    pf0::Float64,
    parameters0::Array{Float64},
    shockgrid0::Array{Float64},
    sdgrid0::Array{Float64},
    sfgrid0::Array{Float64},
    v0::Array{Float64}
    )


sd::Float64 = sd0
sf::Float64 = sf0

parameters = ones(7)
parameters::Array{Float64} .= parameters0 
sgridsize::Int64 = Int(parameters[6])
shockgridsize::Int64 = Int(parameters[7])
shockgrid = zeros(shockgridsize, 3) 
shockgrid::Array{Float64} .= shockgrid0  

vinit::Array{Float64} = ones(sgridsize, sgridsize)
vinit .= v0
sdgrid::Array{Float64} = ones(sgridsize)
sdgrid .= sdgrid0
sfgrid::Array{Float64} = ones(sgridsize)
sfgrid .= sfgrid0

pd::Float64 = pd0                                        # domestic input price
pf::Float64 = pf0       

VVF = Array{Function}(undef, shockgridsize)
#VVF = SharedArray{Function}(shockgridsize)

#@sync @distributed  
for i = 1:shockgridsize

    nu = shockgrid[i,1]
    lamd = shockgrid[i,2]
    lamf = shockgrid[i,3]

    VVF[i] = (n) -> value_fn(n, sd, sf, nu, lamd, lamf, pd, pf, parameters, sdgrid, sfgrid, vinit)                     

end

EVF(n) = (sum(VVF[:])(n))/shockgridsize

return EVF(n)
end


##############################################################################################################################
@everywhere function maximize_EVF(
                        sd0::Float64,
                        sf0::Float64,
                        pd0::Float64,
                        pf0::Float64,
                        parameters0::Array{Float64},
                        shockgrid0::Array{Float64},
                        sdgrid0::Array{Float64},
                        sfgrid0::Array{Float64},
                        v0::Array{Float64}
                        )
                        

sd::Float64 = sd0
sf::Float64 = sf0

parameters = ones(7)
parameters::Array{Float64} .= parameters0    
shockgridsize::Int64 = Int(parameters[7])
shockgrid = zeros(shockgridsize, 3) 
shockgrid::Array{Float64} .= shockgrid0  
sgridsize::Int64 = Int(parameters[6])
sdgrid::Array{Float64} = ones(sgridsize)
sdgrid .= sdgrid0
sfgrid::Array{Float64} = ones(sgridsize)
sfgrid .= sfgrid0
delta::Float64 = parameters[4]
vinit::Array{Float64} = ones(sgridsize, sgridsize)
vinit .= v0

pd::Float64 = pd0                                       
pf::Float64 = pf0   

#establish bounds of n where sp is within the sgridv
nu_max = maximum(shockgrid[:,1])
nu = mean(shockgrid[:,1])

X = unconstrained_fn(nu, pd, pf, parameters)
X_UNC_max = unconstrained_fn(nu_max, pd, pf, parameters)
xf_unc = X[4]
xd_unc = X[5]

sdpmin::Float64 = sdgrid[1] + 1e-5
sdpmax::Float64 = sdgrid[sgridsize] - 1e-5
sfpmin::Float64 = sfgrid[1] + 1e-5
sfpmax::Float64 = sfgrid[sgridsize] - 1e-5
xd_unc_max = X_UNC_max[5]
xf_unc_max = X_UNC_max[4]

if sf < xf_unc*(0.8)    
    #nfmax = xf_unc_max/(1-delta)
    #nfmax = xf_unc_max
    nfini = (xf_unc/(1-delta))*(0.8)
else 
    #nfmax = (xf_unc_max)/2
    nfini = 0.01
end

if sd < xd_unc*(0.8) 
    #ndmax = xd_unc_max/(1-delta)
    #ndmax = xd_unc_max
    ndini = (xd_unc/(1-delta))*(0.8)
else 
    #ndmax = (xd_unc_max)/2
    ndini = 0.01
end

#nfmax::Float64 = (sfgrid[sgridsize]/(1 - delta)) - sf + xf_unc*0.5
#ndmax::Float64 = (sdgrid[sgridsize]/(1 - delta)) - sd + xd_unc*0.5

nfmax::Float64 = (sfgrid[sgridsize]/(1 - delta)) - sf
ndmax::Float64 = (sdgrid[sgridsize]/(1 - delta)) - sd

if nfmax < 1e-5
	nfmax = 0.0001
end
if ndmax < 1e-5
	ndmax = 0.0001
end

ndmin::Float64 = sdpmin
nfmin::Float64 = sfpmin
nmin = [ndmin, nfmin]
nmax = [ndmax, nfmax]
nini = [0.01, 0.01]

EVF(n) = -compute_EVF(n, sd, sf, pd, pf, parameters, shockgrid, sdgrid, sfgrid, vinit)                     

results = optimize(EVF, nmin, nmax, nini, Fminbox(LBFGS()), Optim.Options(f_tol = 1e-6, x_tol = 1e-6))
TVF = Optim.minimum(results)
policy = Optim.minimizer(results) 

#@show policy

return TVF, policy

end


##############################################################################################################################
# 6.  Apply the maximization process for all values of s in the grid:
@everywhere function Tf_d(  
                pd0::Float64,
                pf0::Float64,
                #lamd0::Float64,
                parameters0::Array{Float64},
                shockgrid0::Array{Float64},
                sdgrid0::Array{Float64},
                sfgrid0::Array{Float64},
                v0::Array{Float64}
                )
                
#lamd::Float64 = lamd0
parameters = ones(7)
parameters::Array{Float64} .= parameters0    
shockgridsize::Int64 = Int(parameters[7])
shockgrid = zeros(shockgridsize, 3) 
shockgrid::Array{Float64} .= shockgrid0  
sgridsize::Int64 = Int(parameters[6])
sdgrid::Array{Float64} = ones(sgridsize)
sdgrid .= sdgrid0
sfgrid::Array{Float64} = ones(sgridsize)
sfgrid .= sfgrid0
vinit::Array{Float64} = ones(sgridsize, sgridsize)
vinit .= v0

pd::Float64 = pd0                                       
pf::Float64 = pf0   

Tv = SharedArray{Float64}(sgridsize, sgridsize)
Pnd = SharedArray{Float64}(sgridsize, sgridsize)
Pnf = SharedArray{Float64}(sgridsize, sgridsize)

#vitp = interpolate( sgrid, vinit, Gridded(Linear()) )

 for i = 1:sgridsize                 #sd rows
    @sync @distributed     for k = 1:sgridsize             #sf column
        Tv[i,k], p = maximize_EVF(sdgrid[i], sfgrid[k], pd, pf, parameters, shockgrid, sdgrid, sfgrid, vinit)
        Pnd[i,k] = p[1]
        Pnf[i,k] = p[2]

    end
end

                        

    return Tv, Pnd, Pnf
end

   
##############################################################################################################################
@everywhere function Vfn_d( 
                pd0::Float64,
                pf0::Float64,
                #lamd0::Float64,
                parameters0::Array{Float64},
                shockgrid0::Array{Float64},
                sdgrid0::Array{Float64},
                sfgrid0::Array{Float64}
                )

#lamd::Float64 = lamd0
parameters = ones(7)
parameters::Array{Float64} .= parameters0    
shockgridsize::Int64 = Int(parameters[7])
shockgrid = zeros(shockgridsize, 3) 
shockgrid::Array{Float64} .= shockgrid0  
sgridsize::Int64 = Int(parameters[6])
sdgrid::Array{Float64} = ones(sgridsize)
sdgrid .= sdgrid0
sfgrid::Array{Float64} = ones(sgridsize)
sfgrid .= sfgrid0
delta::Float64 = parameters[4]

pd::Float64 = pd0                                       
pf::Float64 = pf0   

nu_mean = mean(shockgrid[:,1])
X_UNC::Array{Float64} = zeros(5)
X_UNC = unconstrained_fn(nu_mean, pd, pf, parameters)
xf_unc = X_UNC[4]
Tv::Array{Float64} = ones(sgridsize, sgridsize)*(xf_unc/(1-delta))
Tv0::Array{Float64} = ones(sgridsize, sgridsize)
Pnd::Array{Float64} = ones(sgridsize, sgridsize)
Pnd0::Array{Float64} = ones(sgridsize, sgridsize)
Pnf::Array{Float64} = ones(sgridsize, sgridsize)
Pnf0::Array{Float64} = ones(sgridsize, sgridsize)

iter::Int64 = 0
max_iter::Int64 = 70
tol1::Float64 = 1e-2 #1e-3 for Tv
tol2::Float64 = 1e-4 #1e-3 for policies
tol::Float64 = 1e-3
error::Float64 = Inf
error1::Float64 = Inf
error2::Float64 = Inf
errord::Float64 = Inf
errorf::Float64 = Inf
errord0::Float64 = Inf
errorf0::Float64 = Inf


#while iter <= max_iter && error > tol
while iter <= max_iter && error > 0.0009

@time Tv0, Pnd0, Pnf0 = Tf_d(pd, pf, parameters, shockgrid, sdgrid, sfgrid, Tv)
@show Pnd0[1,1,1], Pnf0[1,1,1]
error1 = maximum(abs.(Tv0 .+ Tv))
errord = maximum(abs.(Pnd0 .- Pnd))
errorf = maximum(abs.(Pnf0 .- Pnf))
error2 = max(errord, errorf)
error = min(error1, error2)
xx = argmin([error1, error2])
   if xx == 1
        tol = tol1
   else 
        tol = tol2
   end
   @show iter, error1, errord, errorf

# errors bumping around
if errord < 0.0009 && errorf < 0.0009 && abs(errord - errord0) < 0.001 && abs(errorf - errorf0) < 0.001 
error = 1e-10
end

Tv .= -Tv0
Pnf .= Pnf0
Pnd .= Pnd0
iter += 1
errord0 = errord
errorf0 = errorf

end

return Tv, Pnd, Pnf

end


##############################################################################################################################
@everywhere function policies(
    Pnd0::Array{Float64},
    Pnf0::Array{Float64},
    pd0::Float64,
    pf0::Float64,
    #lamd0::Float64,
    parameters0::Array{Float64},
    shockgrid0::Array{Float64},
    sdgrid0::Array{Float64},
    sfgrid0::Array{Float64}
    )

parameters = ones(7)
parameters::Array{Float64} .= parameters0    
shockgridsize::Int64 = Int(parameters[7])
shockgrid = zeros(shockgridsize, 3) 
shockgrid::Array{Float64} .= shockgrid0  
sgridsize::Int64 = Int(parameters[6])
sdgrid::Array{Float64} = ones(sgridsize)
sdgrid .= sdgrid0
sfgrid::Array{Float64} = ones(sgridsize)
sfgrid .= sfgrid0
delta::Float64 = parameters[4]

pd::Float64 = pd0                                       
pf::Float64 = pf0  
#lamd::Float64 = lamd0
n::Float64 = 0.0 
nu::Float64 = 0.0
lam::Float64 = 0.0 
s::Float64 = 0.0

X::Array{Float64} = zeros(5)
Pnd::Array{Float64} = ones(sgridsize, sgridsize)
Pnd .= Pnd0
Pnf::Array{Float64} = ones(sgridsize, sgridsize)
Pnf .= Pnf0

Px = SharedArray{Float64}(sgridsize, sgridsize, shockgridsize)
Pxd = SharedArray{Float64}(sgridsize, sgridsize, shockgridsize)
Pxf = SharedArray{Float64}(sgridsize, sgridsize, shockgridsize)
Py = SharedArray{Float64}(sgridsize, sgridsize, shockgridsize)
Pp = SharedArray{Float64}(sgridsize, sgridsize, shockgridsize)
Psdp = SharedArray{Float64}(sgridsize, sgridsize, shockgridsize)
Psfp = SharedArray{Float64}(sgridsize, sgridsize, shockgridsize)
Pcase = SharedArray{Float64}(sgridsize, sgridsize, shockgridsize)

for i = 1:sgridsize                 #sd rows
    for k = 1:sgridsize             #sf column
        @sync @distributed  for j = 1:shockgridsize      # different matrix for shocks

        nf = Pnf[i,k]
        nd = Pnd[i,k]
        sd = sdgrid[i]
        sf = sfgrid[k]
        nu = shockgrid[j,1]
        lamd = shockgrid[j,2]
        lamf = shockgrid[j,3]
            
        X, case = cases_fn(nd, nf, sd, sf, nu, lamd, lamf, pd, pf, parameters)
         # x1 = p, x2 = y, x3 = x, x4 = xf, x5 = xd

        Pp[i,k,j] = X[1]
        Py[i,k,j] = X[2]
        Px[i,k,j] = X[3]
        Pxf[i,k,j] = X[4]
        Pxd[i,k,j] = X[5]   
        Psdp[i,k,j] = (1-delta)*(sd + nd - X[5])
        Psfp[i,k,j] = (1-delta)*(sf + nf - X[4])
        Pcase[i,k,j] = case

        end
    end
end

return Pp, Py, Px, Pxf, Pxd, Psdp, Psfp, Pcase
end
































##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
function compute_EVF(                     
    sp,
    s0::Float64,
    pd0::Float64,
    pf0::Float64,
    parameters0::Array{Float64},
    shockgrid0::Array{Float64},      
    sgrid0::Array{Float64},
    v0::Array{Float64}
    )

s::Float64 = s0
#sp::Float64 = sp0

parameters = ones(7)
parameters::Array{Float64} .= parameters0     
shockgridsize::Int64 = Int(parameters[7])
shockgrid = zeros(shockgridsize, 2) 
shockgrid::Array{Float64} .= shockgrid0  
sgridsize::Int64 = Int(parameters[6])
vinit::Array{Float64} = ones(sgridsize)
vinit .= v0
sgrid::Array{Float64} = ones(sgridsize)
sgrid .= sgrid0

pd::Float64 = pd0                                       
pf::Float64 = pf0   

VVF = Array{Function}(undef, shockgridsize)

for i = 1:shockgridsize

lam = shockgrid[i, 1]
nu = shockgrid[i, 2]
VVF[i] = (sp) -> value_fn(sp, s, nu, lam, pd, pf, parameters, sgrid, vinit)                            

end

# iid shocks
EVF(sp) = (sum(VVF[:])(sp))/shockgridsize

return EVF(sp)
end

##############################################################################################################################
function maximize_EVF_grid(
    s0::Float64,
    eta0::Int64,
    pd0::Float64,
    pf0::Float64,
    parameters0::Array{Float64},
    shockgrid0::Array{Float64},
    sgrid0::Array{Float64},
    v0::Array{Float64}
    )

s::Float64 = s0
eta::Int64 = eta0

parameters = ones(7)
parameters::Array{Float64} .= parameters0    
shockgridsize::Int64 = Int(parameters[7])
shockgrid = zeros(shockgridsize, 2) 
shockgrid::Array{Float64} .= shockgrid0  
sgridsize::Int64 = Int(parameters[6])
sgrid::Array{Float64} = ones(sgridsize)
sgrid .= sgrid0

vinit::Array{Float64} = ones(sgridsize, shockgridsize)
vinit .= v0
VFF::Array{Float64} = ones(sgridsize)
Tv::Float64 = 1.0
policy::Int64 = 1.0

pd::Float64 = pd0                                       
pf::Float64 = pf0 

for i = 1:sgridsize
VFF[i] = value_fn(sgrid[i], s, shockgrid[eta,2], shockgrid[eta,1], pd, pf, parameters, sgrid, vinit)
end

vitp = interpolate( (sgrid,),  VFF , Gridded(Linear()) )
opt = optimize(sp -> vitp(sp), sgrid[1], sgrid[sgridsize])
Pol = Optim.minimizer(opt)
TVF = Optim.minimum(opt)

return TVF, Pol

end


