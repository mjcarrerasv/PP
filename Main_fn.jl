# PP 2024
# C2_05_2024
# Main file May 2025
# Two inputs, one input with DT and inventories
# Timing: Shocks (demand and DT), then decisions
# Code written as a funciton of n 

# Algorith where the dynamic choice is solved via function (see Main_mat) for grid 

using Interpolations, Optim, Random
using Statistics
using JLD2
using Distributions
using NLsolve
using FileIO
using StableRNGs

import Base.+
+(f::Function, g::Function) = (x...) -> f(x...) + g(x...)


##############################################################################################################################
# 0. Paremeters 

rng = StableRNG(2904)
sgridsizev =  21
sgridmaxv = 1.0
sgridv = collect(range(1e-10; length = sgridsizev, stop = sgridmaxv))

pdv = 1.0
pfv = 1.0

# shocks
include("lam.jl")
shocks = 2                          #  demand and dt shocks
shockgridsizev = 200                    # size of all shocks (demand and delivery): High, medium, low
shockgridv = ones(shockgridsizev, 2)

TTv = 30*3.0                        # a period is a quarter
day_meanv = 35.0 
day_delayv = 130.0
#delaypctv = 0.5 #0.5 #0.9 
#lamv, day = lam_grid(shockgridsizev, day_meanv, delaypctv, TTv)
lamv, day = lam_grid2(shockgridsizev, day_meanv, day_delayv, TTv)
shockgridv[:,1] .= lamv    # First column is for delivery time shocks
mean_lam = mean(lamv)
std_lam = std(lamv)
@show mean_lam, std_lam

#=
lamv2, day2 = lam_grid(shockgridsizev, day_meanv, 0.5, TTv)
mean_lam2 = mean(lamv2)
std_lam2 = std(lamv2)
@show mean_lam2, std_lam2

lamv3, day3 = lam_grid(shockgridsizev, day_meanv, 0.95, TTv)
mean_lam3 = mean(lamv3)
std_lam3 = std(lamv3)
@show mean_lam3, std_lam3

b_range = range(0.0; length=80, stop = 1)
histogram(lamv,  bins=b_range)
histogram!(lamv2,  bins=b_range)
histogram(lamv3,  bins=b_range)
=#

mean_dem = 1.0
sig_dem = 0.38 #0.38
nu = nu_grid(shockgridsizev, mean_dem, sig_dem)

shockgridv[:,2] .= nu                    # Second column is for demand shocks, First column is for delivery time shocks

Tv0 = ones(sgridsizev, shockgridsizev)
vinitv = zeros(sgridsizev, shockgridsizev)

# other parameters
betav = (1-0.04)^(1/4)  
epsiv = 4.0 #4.0 #1.5                               # demand elasticity
sigmav = 4.0 #0.8                              # input elasticity
thv = 0.5                       # domestic input weight 
deltav = 0.30/4    #0.01                        # depreciation rate

parametersv = [betav, epsiv, sigmav, deltav, thv, sgridsizev, shockgridsizev]

# simulation parameters -- Monte Carlo
Nfirmsv = 50000                          # number of firms
Tfirmsv = 200                           # number of periods

include("1Value_fn_v1.jl")
include("2Simulate.jl")
##############################################################################################################################
# 1. Set up initial matrices

nu_max = maximum(shockgridv[:,2])
X = unconstrained_fn( nu_max, pdv, pfv, parametersv)
sgridmaxv = X[4]*(1.1)
sgridv = collect(range(1e-10; length = sgridsizev, stop = sgridmaxv))
parametersv = [betav, epsiv, sigmav, deltav, thv, sgridsizev, shockgridsizev]

@time Tv,Pn = Vfn(pdv, pfv, parametersv, shockgridv, sgridv)

@time Pp, Py, Px, Pxf, Pxd, Psp, Pcase = policies(Pn, pdv, pfv, parametersv, shockgridv, sgridv )

@time shockMC, PsMC = sim_initial(Nfirmsv, Tfirmsv, parametersv, sgridv)

@time PsMC, PpMC, PyMC, PxMC, PxfMC, PxdMC, PnMC, PcaseMC = simulation(Pn, pdv, pfv, Nfirmsv, Tfirmsv, parametersv, shockgridv, sgridv, shockMC, PsMC )


@time PsMC2, PpMC2, PyMC2, PxMC2, PxfMC2, PxdMC2, PnMC2, PcaseMC2 = simulation_int(Nfirmsv, Tfirmsv, parametersv, shockgridv, sgridv,
Pn, Pp, Py, Px, Pxf, Pxd, Psp, Pcase, shockMC, PsMC )



##############################################################################################################################
#
using Plots
histogram(PsMC[:, Tfirmsv])
minimum(PsMC[:, Tfirmsv])

a1 = mean(PcaseMC[:, Tfirmsv])
a2 = mean(PyMC[:, Tfirmsv])
nu_mean = mean(shockgridv[:,2])
a3 = X_mean = unconstrained_fn( nu_mean, pdv, pfv, parametersv)

inv_sale = ones(Nfirmsv)*(-10.0)
inv_prod = ones(Nfirmsv)*(-10.0)
inv_sale .= pfv.*PsMC[:, Tfirmsv+1]./(PyMC[:, Tfirmsv].*PpMC[:, Tfirmsv])
inv_prod .= PsMC[:, Tfirmsv+1]./(PyMC[:, Tfirmsv])
a4 = mean(inv_sale[:])
a5 = mean(inv_prod[:])

xf_sale = ones(Nfirmsv)*(-10.0)
xf_y = ones(Nfirmsv)*(-10.0)
xf_inp = ones(Nfirmsv)*(-10.0)
xf_sale .= PxfMC[:, Tfirmsv]./(PyMC[:, Tfirmsv].*PpMC[:, Tfirmsv])
xf_y .= PxfMC[:, Tfirmsv]./(PyMC[:, Tfirmsv])
xf_inp .= PxfMC[:, Tfirmsv]./(PxdMC[:, Tfirmsv] .+ PxfMC[:, Tfirmsv])

a6 = mean(xf_sale[:])
a7 = mean(xf_y[:])
a8 =  mean(xf_inp[:])
a9 = mean(PpMC[:, Tfirmsv])

@show a1, a2, a3, a4, a5, a6, a7, a8, a9


lam_mean = mean(shockgridv[:,1])
lam_var = var(shockgridv[:,1])
lam_std = std(shockgridv[:,1])
histogram(shockgridv[:,1])
histogram(shockgridv[:,2])
maximum(Pn[sgridsizev,:])




##############################################################################################################################
stock = ones(Nfirmsv)*(-10.0)
stock[:] .= PsMC[:, Tfirmsv] .+ shockgridv[ shockMC[:, Tfirmsv] , 1].*PnMC[:, Tfirmsv] - PxfMC[:, Tfirmsv]
histogram(stock[:])

AA = ones(sgridsizev)
AA .=(1 .- deltav).*(sgridv[:] .+ Pn[:,4] .- Pxf[:,4])

##############################################################################################################################

#
EP = epsiv/(epsiv-1)
EP_ = (epsiv-1)/epsiv
SIG = 1-sigmav
SIG2 = (sigmav-1)/sigmav
sig_ = 1/sigmav

function f(x)
[ 0.1 +  0.59124*0.1 - x[1],
(thv)*(((EP_*x[5]*x[4])/(x[3]*pdv))^sigmav) - x[2],
((thv)^(sig_)*(x[2])^(SIG2)) + ((1-thv)^(sig_)*(x[1])^(SIG2)) - x[3],
x[3]^(sigmav/(sigmav-1)) - x[4],
(3.96482/x[4])^(1/epsiv) - x[5] ]
end
#xf = 1, xd = 2, x = 3 y = 4, p = 5

@time sol = nlsolve(f, [ 0.15912,  0.3815149,  1.15777,  0.4811, 4.0796])
sol.zero

a = f([ 0.15912400000000002,  0.38151498238140197,  1.2006764187074053 ,  0.48116726696259804,  4.079606421847822 ])

@time X_con = constrained_fn(0.1, 0.1, 3.96482, 0.59124, pdv, pfv, parametersv)



####
nu_mean = mean(shockgridv[:,2])
nu_max = maximum(shockgridv[:,2])
nu_min = minimum(shockgridv[:,2])
lam_max = maximum(shockgridv[:,1])
lam_min = minimum(shockgridv[:,1])


X_UNC = unconstrained_fn(nu_mean, pdv, pfv, parametersv)
X_UNC_max = unconstrained_fn(nu_max, pdv, pfv, parametersv)
X_UNC_min = unconstrained_fn(nu_min, pdv, pfv, parametersv)

xf_unc_mean = X_UNC[4]
nini = xf_unc_mean/(1-deltav)
xf_unc_max = X_UNC_max[4]
xf_unc_min = X_UNC_min[4]

spmin = sgridv[1] + 1e-5
spmax = sgridv[sgridsizev] - 1e-5



nmax1  = spmax*deltav/(1-deltav) + xf_unc_min
nmax2 = spmax/((1-deltav)*(1-lam_min))
nmax3 = xf_unc_max/(1-deltav)

nmax = min(nmax1, nmax2)

nmin1  = spmin*deltav/(1-deltav) + xf_unc_max
nmin2 = spmin/((1-deltav)*(1-lam_max))
nmin = max(nmin1, nmin2)
nmin = spmin


nmin::Float64  = spmin

spmin = sgridv[1] + 1e-5
spmax = sgridv[sgridsizev] - 1e-5

####
lamv2, day2 = lam_grid2(shockgridsizev, day_meanv, 12000.0, TTv)
lamv3 = nu_grid(shockgridsizev, 1- (35/TTv), 0.3)

mean_lam = mean(lamv)
std_lam = std(lamv)
mean_lam2 = mean(lamv2)
std_lam2 = std(lamv2)
mean_lam3 = mean(lamv3)
std_lam3 = std(lamv3)
@show mean_lam, std_lam, mean_lam2, std_lam2
b_range = range(0.0; length=80, stop = 1)
histogram(lamv,  bins=b_range)
histogram!(lamv2,  bins=b_range)

histogram(lamv3)

histogram(lamv2,  bins=b_range)

nu[:] = exp.(mean_dem .+ sig_dem .* randn(rng, lamgridsizev)) 
######