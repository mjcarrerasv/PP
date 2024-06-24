# PP 2024
# C4_05_2024
# Main file May 2025
# Two inputs: both have inventories 
# foreign input has stoch dt, and doomestic input has deterministic dt
# Timing: Shocks (demand and DT), then decisions
# Code written as a funciton of n 

# Algorith where the dynamic choice is solved via function (see Main_mat) for grid 
# using Pkg; Pkg.add(")

using Distributed
@everywhere using Interpolations, Optim, Random
@everywhere using Statistics
using JLD2
@everywhere using Distributions
@everywhere using NLsolve
using FileIO
using StableRNGs
using BenchmarkTools
#using Base.Threads
using SharedArrays
addprocs(6) 
workers()
Threads.nthreads()
 

@everywhere import Base.+
@everywhere +(f::Function, g::Function) = (x...) -> f(x...) + g(x...)

##############################################################################################################################
# 0. Parameters 

rng = StableRNG(2904)
@everywhere sgridsizev =  20
@everywhere sdgridmaxv = 1.0
@everywhere sdgridv = collect(range(1e-10; length = sgridsizev, stop = sdgridmaxv))
@everywhere sfgridmaxv = 1.0
@everywhere sfgridv = collect(range(1e-10; length = sgridsizev, stop = sfgridmaxv))

@everywhere pdv = 1.0
@everywhere pfv = 1.0/1.2 #0.1

# shocks
include("lam.jl")
include("halton.jl")
shocks = 3                          #  demand and dt shocks
@everywhere shockgridsizev = 100               # size of all shocks (demand and delivery): High, medium, low
@everywhere shockgridv = ones(shockgridsizev, 3)

@everywhere TTv = 30*3.0                        # a period is a quarter 
dayd_meanv = 10.0 + 14.0
dayd_delayv = 40.0 #30.0 # 20, 60, 100                  # domestic delivery times is second column
dayf_meanv = 35.0 + 14.0
dayf_delayv = 40.0 #30.0 # 20, 60, 100
lamd2v = (TTv - dayd_meanv)/TTv
lamf2v = (TTv - dayf_meanv)/TTv
lamdv, dayd = lam_grid3(shockgridsizev, dayd_meanv, dayd_delayv, TTv, lamd2v)
lamfv, dayf = lam_grid3(shockgridsizev, dayf_meanv, dayf_delayv, TTv, lamf2v)
#shockgridv[:,2] .= lamdv[:];                    # Second column: domestic delivery times 
#shockgridv[:,3] .= lamfv[:];                  # Third column is for foreign delivery time shocks

shockgridv[:,2] .= lamd2v;                     # Second column: domestic delivery times 
shockgridv[:,3] .= lamf2v;                  # Third column is for foreign delivery time shocks
#=
mean_lamd = mean(lamdv)
std_lamd = std(lamdv)
mean_lamf = mean(lamfv)
std_lamf = std(lamfv)
@show lamd2v, mean_lamd, std_lamd, lamf2v, mean_lamf, std_lamf
using Plots
histogram!(lamfv)
=#

mean_dem = 1.0
sig_dem = 1.2 #0.50 #0.38
nu = nu_grid(shockgridsizev, mean_dem, sig_dem)
shockgridv[:,1] .= nu[:]                    # Second column is for demand shocks, First column is for delivery time shocks

@everywhere Tv0 = ones(sgridsizev, sgridsizev, shockgridsizev)
@everywhere vinitv = zeros(sgridsizev, sgridsizev, shockgridsizev)

# other parameters
@everywhere betav = (1-0.04)^(1/4)  
@everywhere epsiv = 4.0 #4.0 #1.5                               # demand elasticity
@everywhere sigmav = 2.5 #2.5 #1.5  #4.0 #0.8                              # input elasticity
@everywhere thv = 0.75 #0.9110 #0.5                       # domestic input weight 
@everywhere deltav = 0.30/4    #0.01                        # depreciation rate

@everywhere parametersv = [betav, epsiv, sigmav, deltav, thv, sgridsizev, shockgridsizev]

# simulation parameters -- Monte Carlo
Nfirmsv = 50000                          # number of firms
Tfirmsv = 200                         # number of periods

include("1Value_fn_v2.jl")
include("2Simulate_v2.jl")

##############################################################################################################################
# 1. initial point
nu_max = maximum(shockgridv[:,1])
X = unconstrained_fn( nu_max, pdv, pfv, parametersv)
sdgridmaxv = X[5]*(1.4)
sdgridv = collect(range(1e-10; length = sgridsizev, stop = sdgridmaxv))
sfgridmaxv = X[4]*(1.4)
sfgridv = collect(range(1e-10; length = sgridsizev, stop = sfgridmaxv))
parametersv = [betav, epsiv, sigmav, deltav, thv, sgridsizev, shockgridsizev];

@show "initial", lamd2v, lamf2v, pfv

@time Tv_1d, Pnd_1d, Pnf_1d = Vfn_d(pdv, pfv, parametersv, shockgridv, sdgridv, sfgridv);
@time Pp_1d, Py_1d, Px_1d, Pxf_1d, Pxd_1d, Psdp_1d, Psfp_1d, Pcase_1d = policies(Pnd_1d, Pnf_1d, pdv, pfv,parametersv, shockgridv, sdgridv, sfgridv);

@time shockMC_1d, PsdMC_1d, PsfMC_1d = sim_initial(Nfirmsv, Tfirmsv, parametersv, sdgridv, sfgridv);
@time PsdMC_1d, PsfMC_1d, PpMC_1d, PyMC_1d, PxMC_1d, PxfMC_1d, PxdMC_1d, PndMC_1d, PnfMC_1d, PcaseMC_1d = 
simulation(Pnd_1d, Pnf_1d, pdv, pfv, Nfirmsv, Tfirmsv, parametersv, shockgridv, sdgridv, sfgridv, shockMC_1d, PsdMC_1d, PsfMC_1d);

##############################################################################################################################
# 2. pf change only 
@everywhere pfv = 1.0 #1.0/1.2 #0.1

nu_max = maximum(shockgridv[:,1])
X = unconstrained_fn( nu_max, pdv, pfv, parametersv)
sdgridmaxv = X[5]*(1.4)
sdgridv = collect(range(1e-10; length = sgridsizev, stop = sdgridmaxv))
sfgridmaxv = X[4]*(1.4)
sfgridv = collect(range(1e-10; length = sgridsizev, stop = sfgridmaxv))
parametersv = [betav, epsiv, sigmav, deltav, thv, sgridsizev, shockgridsizev];

@show "price change", lamd2v, lamf2v, pfv

@time Tv_2c, Pnd_2c, Pnf_2c = Vfn_d(pdv, pfv, parametersv, shockgridv, sdgridv, sfgridv);
@time Pp_2c, Py_2c, Px_2c, Pxf_2c, Pxd_2c, Psdp_2c, Psfp_2c, Pcase_2c = policies(Pnd_2c, Pnf_2c, pdv, pfv,parametersv, shockgridv, sdgridv, sfgridv);

@time shockMC_2c, PsdMC_2c, PsfMC_2c = sim_initial(Nfirmsv, Tfirmsv, parametersv, sdgridv, sfgridv);
@time PsdMC_2c, PsfMC_2c, PpMC_2c, PyMC_2c, PxMC_2c, PxfMC_2c, PxdMC_2c, PndMC_2c, PnfMC_2c, PcaseMC_2c = 
simulation(Pnd_2c, Pnf_2c, pdv, pfv, Nfirmsv, Tfirmsv, parametersv, shockgridv, sdgridv, sfgridv, shockMC_2c, PsdMC_2c, PsfMC_2c);


##############################################################################################################################
# 3. increase in uncertainty 
@everywhere pfv = 1.0 #1.0/1.2 #0.1

@everywhere sigmav = 2.5 #2.5 #1.5  #4.0 #0.8  

dayd_meanv = 10.0 + 14.0
dayd_delayv = 1.0 #30.0 # 20, 60, 100                  # domestic delivery times is second column
dayf_meanv = 35.0 + 14.0
dayf_delayv = 20.0 #30.0 # 20, 60, 100
lamdv, dayd = lam_grid3(shockgridsizev, dayd_meanv, dayd_delayv, TTv, lamd2v)
lamfv, dayf = lam_grid3(shockgridsizev, dayf_meanv, dayf_delayv, TTv, lamf2v)
shockgridv[:,2] .= lamdv[:];                    # Second column: domestic delivery times 
shockgridv[:,3] .= lamfv[:];                  # Third column is for foreign delivery time shocks

#
mean_lamd = mean(lamdv)
std_lamd = std(lamdv)
mean_lamf = mean(lamfv)
std_lamf = std(lamfv)
@show "uncertainty", pfv, lamd2v, mean_lamd, std_lamd, lamf2v, mean_lamf, std_lamf
#histogram!(lamv)
#

nu_max = maximum(shockgridv[:,1])
X = unconstrained_fn( nu_max, pdv, pfv, parametersv)
sdgridmaxv = X[5]*(1.1)
sdgridv = collect(range(1e-10; length = sgridsizev, stop = sdgridmaxv))
sfgridmaxv = X[4]*(1.1)
sfgridv = collect(range(1e-10; length = sgridsizev, stop = sfgridmaxv))
parametersv = [betav, epsiv, sigmav, deltav, thv, sgridsizev, shockgridsizev];


@time Tv_3, Pnd_3, Pnf_3 = Vfn_d(pdv, pfv, parametersv, shockgridv, sdgridv, sfgridv);
@time Pp_3, Py_3, Px_3, Pxf_3, Pxd_3, Psdp_3, Psfp_3, Pcase_3 = policies(Pnd_3, Pnf_3, pdv, pfv, parametersv, shockgridv, sdgridv, sfgridv);

@time shockMC_3, PsdMC_3, PsfMC_3 = sim_initial(Nfirmsv, Tfirmsv, parametersv, sdgridv, sfgridv);
@time PsdMC_3, PsfMC_3, PpMC_3, PyMC_3, PxMC_3, PxfMC_3, PxdMC_3, PndMC_3, PnfMC_3, PcaseMC_3 = 
simulation(Pnd_3, Pnf_3, pdv, pfv, Nfirmsv, Tfirmsv, parametersv, shockgridv, sdgridv, sfgridv, shockMC_3, PsdMC_3, PsfMC_3);


##############################################################################################################################
##############################################################################################################################
###############################################################################################################################
# 1. initial point
lamd2v = (TTv - dayd_meanv)/TTv
lamf2v = (TTv - dayf_meanv)/TTv
shockgridv[:,2] .= lamd2v;                     # Second column: domestic delivery times 
shockgridv[:,3] .= lamf2v;   

@everywhere pfv = 1.0/1.2 #0.1

a1 = count(<=(0.0), PcaseMC_1[:, Tfirmsv])/Nfirmsv
nu_mean = mean(shockgridv[:,2])
a2 = X_mean = unconstrained_fn( nu_mean, pdv, pfv, parametersv)
a3 = mean(PyMC_1[:, Tfirmsv])
a4 = mean(PpMC_1[:, Tfirmsv])

a5 = mean(PndMC_1[:, Tfirmsv])
a6 = mean(PnfMC_1[:, Tfirmsv])
a7 = pfv.*mean(PnfMC_1[:, Tfirmsv])

xf_sale = ones(Nfirmsv)*(-10.0)
xf_y = ones(Nfirmsv)*(-10.0)
xf_inp = ones(Nfirmsv)*(-10.0)
xd_sale = ones(Nfirmsv)*(-10.0)
xd_y = ones(Nfirmsv)*(-10.0)
xd_inp = ones(Nfirmsv)*(-10.0)

xf_sale .= pfv.*PxfMC_1[:, Tfirmsv]./(PyMC_1[:, Tfirmsv].*PpMC_1[:, Tfirmsv])
xf_y .= PxfMC_1[:, Tfirmsv]./(PyMC_1[:, Tfirmsv])
xf_inp .= pfv.*PxfMC_1[:, Tfirmsv]./(pdv.*PxdMC_1[:, Tfirmsv] .+ pfv.*PxfMC_1[:, Tfirmsv])

xd_sale .= pdv.*PxdMC_1[:, Tfirmsv]./(PyMC_1[:, Tfirmsv].*PpMC_1[:, Tfirmsv])
xd_y .= PxdMC_1[:, Tfirmsv]./(PyMC_1[:, Tfirmsv])
xd_inp .= pdv.*PxdMC_1[:, Tfirmsv]./(pdv.*PxdMC_1[:, Tfirmsv] .+ pfv.*PxfMC_1[:, Tfirmsv])

a8 = mean(PxfMC_1[:, Tfirmsv])
a9 = pfv.*mean(PxfMC_1[:, Tfirmsv])
a10 = mean(xf_sale[:])
a11 = mean(xf_y[:])
a12 =  mean(xf_inp[:])

a13 = mean(PxdMC_1[:, Tfirmsv])
a14 = mean(xd_sale[:])
a15 = mean(xd_y[:])
a16 =  mean(xd_inp[:])

invf_prod = ones(Nfirmsv)*(-10.0)
invd_prod = ones(Nfirmsv)*(-10.0)
inv_prod = ones(Nfirmsv)*(-10.0)
invf_sale = ones(Nfirmsv)*(-10.0)
invd_sale = ones(Nfirmsv)*(-10.0)
inv_sale = ones(Nfirmsv)*(-10.0)

invf_sale .= pfv.*PsfMC_1[:, Tfirmsv+1]./(PyMC_1[:, Tfirmsv].*PpMC_1[:, Tfirmsv])
invd_sale .= pdv.*PsdMC_1[:, Tfirmsv+1]./(PyMC_1[:, Tfirmsv].*PpMC_1[:, Tfirmsv])
inv_sale .= (pfv.*PsfMC_1[:, Tfirmsv+1].+ pdv.*PsdMC_1[:, Tfirmsv+1])./(PyMC_1[:, Tfirmsv].*PpMC_1[:, Tfirmsv])

invf_prod .= PsfMC_1[:, Tfirmsv+1]./(PyMC_1[:, Tfirmsv])
invd_prod .= PsdMC_1[:, Tfirmsv+1]./(PyMC_1[:, Tfirmsv])
inv_prod .= (PsfMC_1[:, Tfirmsv+1] .+ PsdMC_1[:, Tfirmsv+1])./(PyMC_1[:, Tfirmsv])

invf = ones(Nfirmsv)*(-10.0)
invf_val = ones(Nfirmsv)*(-10.0)
invd = ones(Nfirmsv)*(-10.0)
inv = ones(Nfirmsv)*(-10.0)
inv_val = ones(Nfirmsv)*(-10.0)

invf.= PsfMC_1[:, Tfirmsv+1]
invf_val.= pfv.*PsfMC_1[:, Tfirmsv+1]
invd .= PsdMC_1[:, Tfirmsv+1]
inv .= (PsfMC_1[:, Tfirmsv+1].+ PsdMC_1[:, Tfirmsv+1])
inv_val .= (pfv.*PsfMC_1[:, Tfirmsv+1].+ pdv.*PsdMC_1[:, Tfirmsv+1])

a17 = mean(invf[:])
a18 = mean(invf_val[:])
a19 = mean(invf_sale[:])
a20 = mean(invf_prod[:])

a21 = mean(invd[:])
a22 = mean(invd_sale[:])
a23 = mean(invd_prod[:])

a24 = mean(inv[:])
a25 = mean(inv_val[:])
a26 = mean(inv_sale[:])
a27 = mean(inv_prod[:])

@show a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27


##############################################################################################################################
##############################################################################################################################
###############################################################################################################################
# 2. pf change only 
lamd2v = (TTv - dayd_meanv)/TTv
lamf2v = (TTv - dayf_meanv)/TTv
shockgridv[:,2] .= lamd2v;                     # Second column: domestic delivery times 
shockgridv[:,3] .= lamf2v;   

@everywhere pfv = 1.0 #1.0/1.2 #0.1

a1 = count(<=(0.0), PcaseMC_2[:, Tfirmsv])/Nfirmsv
nu_mean = mean(shockgridv[:,2])
a2 = X_mean = unconstrained_fn( nu_mean, pdv, pfv, parametersv)
a3 = mean(PyMC_2[:, Tfirmsv])
a4 = mean(PpMC_2[:, Tfirmsv])

a5 = mean(PndMC_2[:, Tfirmsv])
a6 = mean(PnfMC_2[:, Tfirmsv])
a7 = pfv.*mean(PnfMC_2[:, Tfirmsv])

xf_sale = ones(Nfirmsv)*(-10.0)
xf_y = ones(Nfirmsv)*(-10.0)
xf_inp = ones(Nfirmsv)*(-10.0)
xd_sale = ones(Nfirmsv)*(-10.0)
xd_y = ones(Nfirmsv)*(-10.0)
xd_inp = ones(Nfirmsv)*(-10.0)

xf_sale .= pfv.*PxfMC_2[:, Tfirmsv]./(PyMC_2[:, Tfirmsv].*PpMC_2[:, Tfirmsv])
xf_y .= PxfMC_2[:, Tfirmsv]./(PyMC_2[:, Tfirmsv])
xf_inp .= pfv.*PxfMC_2[:, Tfirmsv]./(pdv.*PxdMC_2[:, Tfirmsv] .+ pfv.*PxfMC_2[:, Tfirmsv])

xd_sale .= pdv.*PxdMC_2[:, Tfirmsv]./(PyMC_2[:, Tfirmsv].*PpMC_2[:, Tfirmsv])
xd_y .= PxdMC_2[:, Tfirmsv]./(PyMC_2[:, Tfirmsv])
xd_inp .= pdv.*PxdMC_2[:, Tfirmsv]./(pdv.*PxdMC_2[:, Tfirmsv] .+ pfv.*PxfMC_2[:, Tfirmsv])

a8 = mean(PxfMC_2[:, Tfirmsv])
a9 = pfv.*mean(PxfMC_2[:, Tfirmsv])
a10 = mean(xf_sale[:])
a11 = mean(xf_y[:])
a12 =  mean(xf_inp[:])

a13 = mean(PxdMC_2[:, Tfirmsv])
a14 = mean(xd_sale[:])
a15 = mean(xd_y[:])
a16 =  mean(xd_inp[:])

invf_prod = ones(Nfirmsv)*(-10.0)
invd_prod = ones(Nfirmsv)*(-10.0)
inv_prod = ones(Nfirmsv)*(-10.0)
invf_sale = ones(Nfirmsv)*(-10.0)
invd_sale = ones(Nfirmsv)*(-10.0)
inv_sale = ones(Nfirmsv)*(-10.0)

invf_sale .= pfv.*PsfMC_2[:, Tfirmsv+1]./(PyMC_2[:, Tfirmsv].*PpMC_2[:, Tfirmsv])
invd_sale .= pdv.*PsdMC_2[:, Tfirmsv+1]./(PyMC_2[:, Tfirmsv].*PpMC_2[:, Tfirmsv])
inv_sale .= (pfv.*PsfMC_2[:, Tfirmsv+1].+ pdv.*PsdMC_2[:, Tfirmsv+1])./(PyMC_2[:, Tfirmsv].*PpMC_2[:, Tfirmsv])

invf_prod .= PsfMC_2[:, Tfirmsv+1]./(PyMC_2[:, Tfirmsv])
invd_prod .= PsdMC_2[:, Tfirmsv+1]./(PyMC_2[:, Tfirmsv])
inv_prod .= (PsfMC_2[:, Tfirmsv+1] .+ PsdMC_2[:, Tfirmsv+1])./(PyMC_2[:, Tfirmsv])

invf = ones(Nfirmsv)*(-10.0)
invf_val = ones(Nfirmsv)*(-10.0)
invd = ones(Nfirmsv)*(-10.0)
inv = ones(Nfirmsv)*(-10.0)
inv_val = ones(Nfirmsv)*(-10.0)

invf.= PsfMC_2[:, Tfirmsv+1]
invf_val.= pfv.*PsfMC_2[:, Tfirmsv+1]
invd .= PsdMC_2[:, Tfirmsv+1]
inv .= (PsfMC_2[:, Tfirmsv+1].+ PsdMC_2[:, Tfirmsv+1])
inv_val .= (pfv.*PsfMC_2[:, Tfirmsv+1].+ pdv.*PsdMC_2[:, Tfirmsv+1])

a17 = mean(invf[:])
a18 = mean(invf_val[:])
a19 = mean(invf_sale[:])
a20 = mean(invf_prod[:])


a21 = mean(invd[:])
a22 = mean(invd_sale[:])
a23 = mean(invd_prod[:])

a24 = mean(inv[:])
a25 = mean(inv_val[:])
a26 = mean(inv_sale[:])
a27 = mean(inv_prod[:])

@show a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27


##############################################################################################################################
##############################################################################################################################
###############################################################################################################################
# 3. increase in uncertainty 
@everywhere pfv = 1.0 #1.0/1.2 #0.1

dayd_meanv = 10.0 + 14.0
dayd_delayv = 40.0 #30.0 # 20, 60, 100                  # domestic delivery times is second column
dayf_meanv = 35.0 + 14.0
dayf_delayv = 40.0 #30.0 # 20, 60, 100
lamdv, dayd = lam_grid3(shockgridsizev, dayd_meanv, dayd_delayv, TTv, lamd2v)
lamfv, dayf = lam_grid3(shockgridsizev, dayf_meanv, dayf_delayv, TTv, lamf2v)
shockgridv[:,2] .= lamdv[:];                    # Second column: domestic delivery times 
shockgridv[:,3] .= lamfv[:];  

a1 = count(<=(0.0), PcaseMC_3[:, Tfirmsv])/Nfirmsv
nu_mean = mean(shockgridv[:,2])
a2 = X_mean = unconstrained_fn( nu_mean, pdv, pfv, parametersv)
a3 = mean(PyMC_3[:, Tfirmsv])
a4 = mean(PpMC_3[:, Tfirmsv])

a5 = mean(PndMC_3[:, Tfirmsv])
a6 = mean(PnfMC_3[:, Tfirmsv])
a7 = pfv.*mean(PnfMC_3[:, Tfirmsv])

xf_sale = ones(Nfirmsv)*(-10.0)
xf_y = ones(Nfirmsv)*(-10.0)
xf_inp = ones(Nfirmsv)*(-10.0)
xd_sale = ones(Nfirmsv)*(-10.0)
xd_y = ones(Nfirmsv)*(-10.0)
xd_inp = ones(Nfirmsv)*(-10.0)

xf_sale .= pfv.*PxfMC_3[:, Tfirmsv]./(PyMC_3[:, Tfirmsv].*PpMC_3[:, Tfirmsv])
xf_y .= PxfMC_3[:, Tfirmsv]./(PyMC_3[:, Tfirmsv])
xf_inp .= pfv.*PxfMC_3[:, Tfirmsv]./(pdv.*PxdMC_3[:, Tfirmsv] .+ pfv.*PxfMC_3[:, Tfirmsv])

xd_sale .= pdv.*PxdMC_3[:, Tfirmsv]./(PyMC_3[:, Tfirmsv].*PpMC_3[:, Tfirmsv])
xd_y .= PxdMC_3[:, Tfirmsv]./(PyMC_3[:, Tfirmsv])
xd_inp .= pdv.*PxdMC_3[:, Tfirmsv]./(pdv.*PxdMC_3[:, Tfirmsv] .+ pfv.*PxfMC_3[:, Tfirmsv])

a8 = mean(PxfMC_3[:, Tfirmsv])
a9 = pfv.*mean(PxfMC_3[:, Tfirmsv])
a10 = mean(xf_sale[:])
a11 = mean(xf_y[:])
a12 =  mean(xf_inp[:])

a13 = mean(PxdMC_3[:, Tfirmsv])
a14 = mean(xd_sale[:])
a15 = mean(xd_y[:])
a16 =  mean(xd_inp[:])

invf_prod = ones(Nfirmsv)*(-10.0)
invd_prod = ones(Nfirmsv)*(-10.0)
inv_prod = ones(Nfirmsv)*(-10.0)
invf_sale = ones(Nfirmsv)*(-10.0)
invd_sale = ones(Nfirmsv)*(-10.0)
inv_sale = ones(Nfirmsv)*(-10.0)

invf_sale .= pfv.*PsfMC_3[:, Tfirmsv+1]./(PyMC_3[:, Tfirmsv].*PpMC_3[:, Tfirmsv])
invd_sale .= pdv.*PsdMC_3[:, Tfirmsv+1]./(PyMC_3[:, Tfirmsv].*PpMC_3[:, Tfirmsv])
inv_sale .= (pfv.*PsfMC_3[:, Tfirmsv+1].+ pdv.*PsdMC_3[:, Tfirmsv+1])./(PyMC_3[:, Tfirmsv].*PpMC_3[:, Tfirmsv])

invf_prod .= PsfMC_3[:, Tfirmsv+1]./(PyMC_3[:, Tfirmsv])
invd_prod .= PsdMC_3[:, Tfirmsv+1]./(PyMC_3[:, Tfirmsv])
inv_prod .= (PsfMC_3[:, Tfirmsv+1] .+ PsdMC_3[:, Tfirmsv+1])./(PyMC_3[:, Tfirmsv])

invf = ones(Nfirmsv)*(-10.0)
invf_val = ones(Nfirmsv)*(-10.0)
invd = ones(Nfirmsv)*(-10.0)
inv = ones(Nfirmsv)*(-10.0)
inv_val = ones(Nfirmsv)*(-10.0)

invf.= PsfMC_3[:, Tfirmsv+1]
invf_val.= pfv.*PsfMC_3[:, Tfirmsv+1]
invd .= PsdMC_3[:, Tfirmsv+1]
inv .= (PsfMC_3[:, Tfirmsv+1].+ PsdMC_3[:, Tfirmsv+1])
inv_val .= (pfv.*PsfMC_3[:, Tfirmsv+1].+ pdv.*PsdMC_3[:, Tfirmsv+1])

a17 = mean(invf[:])
a18 = mean(invf_val[:])
a19 = mean(invf_sale[:])
a20 = mean(invf_prod[:])


a21 = mean(invd[:])
a22 = mean(invd_sale[:])
a23 = mean(invd_prod[:])

a24 = mean(inv[:])
a25 = mean(inv_val[:])
a26 = mean(inv_sale[:])
a27 = mean(inv_prod[:])

@show a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27



##############################################################################################################################
##############################################################################################################################
###############################################################################################################################
# 1. initial point
lamd2v = (TTv - dayd_meanv)/TTv
lamf2v = (TTv - dayf_meanv)/TTv
shockgridv[:,2] .= lamd2v;                     # Second column: domestic delivery times 
shockgridv[:,3] .= lamf2v;   

@everywhere pfv = 1.0/1.2 #0.1

a1 = count(<=(0.0), PcaseMC_1b[:, Tfirmsv])/Nfirmsv
nu_mean = mean(shockgridv[:,2])
a2 = X_mean = unconstrained_fn( nu_mean, pdv, pfv, parametersv)
a3 = mean(PyMC_1b[:, Tfirmsv])
a4 = mean(PpMC_1b[:, Tfirmsv])

a5 = mean(PndMC_1b[:, Tfirmsv])
a6 = mean(PnfMC_1b[:, Tfirmsv])
a7 = pfv.*mean(PnfMC_1b[:, Tfirmsv])

xf_sale = ones(Nfirmsv)*(-10.0)
xf_y = ones(Nfirmsv)*(-10.0)
xf_inp = ones(Nfirmsv)*(-10.0)
xd_sale = ones(Nfirmsv)*(-10.0)
xd_y = ones(Nfirmsv)*(-10.0)
xd_inp = ones(Nfirmsv)*(-10.0)

xf_sale .= pfv.*PxfMC_1b[:, Tfirmsv]./(PyMC_1b[:, Tfirmsv].*PpMC_1b[:, Tfirmsv])
xf_y .= PxfMC_1b[:, Tfirmsv]./(PyMC_1b[:, Tfirmsv])
xf_inp .= pfv.*PxfMC_1b[:, Tfirmsv]./(pdv.*PxdMC_1b[:, Tfirmsv] .+ pfv.*PxfMC_1b[:, Tfirmsv])

xd_sale .= pdv.*PxdMC_1b[:, Tfirmsv]./(PyMC_1b[:, Tfirmsv].*PpMC_1[:, Tfirmsv])
xd_y .= PxdMC_1b[:, Tfirmsv]./(PyMC_1b[:, Tfirmsv])
xd_inp .= pdv.*PxdMC_1b[:, Tfirmsv]./(pdv.*PxdMC_1b[:, Tfirmsv] .+ pfv.*PxfMC_1b[:, Tfirmsv])

a8 = mean(PxfMC_1b[:, Tfirmsv])
a9 = pfv.*mean(PxfMC_1b[:, Tfirmsv])
a10 = mean(xf_sale[:])
a11 = mean(xf_y[:])
a12 =  mean(xf_inp[:])

a13 = mean(PxdMC_1b[:, Tfirmsv])
a14 = mean(xd_sale[:])
a15 = mean(xd_y[:])
a16 =  mean(xd_inp[:])

invf_prod = ones(Nfirmsv)*(-10.0)
invd_prod = ones(Nfirmsv)*(-10.0)
inv_prod = ones(Nfirmsv)*(-10.0)
invf_sale = ones(Nfirmsv)*(-10.0)
invd_sale = ones(Nfirmsv)*(-10.0)
inv_sale = ones(Nfirmsv)*(-10.0)

invf_sale .= pfv.*PsfMC_1b[:, Tfirmsv+1]./(PyMC_1b[:, Tfirmsv].*PpMC_1b[:, Tfirmsv])
invd_sale .= pdv.*PsdMC_1b[:, Tfirmsv+1]./(PyMC_1b[:, Tfirmsv].*PpMC_1b[:, Tfirmsv])
inv_sale .= (pfv.*PsfMC_1b[:, Tfirmsv+1].+ pdv.*PsdMC_1b[:, Tfirmsv+1])./(PyMC_1b[:, Tfirmsv].*PpMC_1b[:, Tfirmsv])

invf_prod .= PsfMC_1b[:, Tfirmsv+1]./(PyMC_1b[:, Tfirmsv])
invd_prod .= PsdMC_1b[:, Tfirmsv+1]./(PyMC_1b[:, Tfirmsv])
inv_prod .= (PsfMC_1b[:, Tfirmsv+1] .+ PsdMC_1b[:, Tfirmsv+1])./(PyMC_1b[:, Tfirmsv])

invf = ones(Nfirmsv)*(-10.0)
invf_val = ones(Nfirmsv)*(-10.0)
invd = ones(Nfirmsv)*(-10.0)
inv = ones(Nfirmsv)*(-10.0)
inv_val = ones(Nfirmsv)*(-10.0)

invf.= PsfMC_1[:, Tfirmsv+1]
invf_val.= pfv.*PsfMC_1[:, Tfirmsv+1]
invd .= PsdMC_1[:, Tfirmsv+1]
inv .= (PsfMC_1[:, Tfirmsv+1].+ PsdMC_1[:, Tfirmsv+1])
inv_val .= (pfv.*PsfMC_1[:, Tfirmsv+1].+ pdv.*PsdMC_1[:, Tfirmsv+1])

a17 = mean(invf[:])
a18 = mean(invf_val[:])
a19 = mean(invf_sale[:])
a20 = mean(invf_prod[:])

a21 = mean(invd[:])
a22 = mean(invd_sale[:])
a23 = mean(invd_prod[:])

a24 = mean(inv[:])
a25 = mean(inv_val[:])
a26 = mean(inv_sale[:])
a27 = mean(inv_prod[:])

@show a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27
