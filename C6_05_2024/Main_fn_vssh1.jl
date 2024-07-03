# PP 2024
# C4_05_2024
# Main file May 2025
# Two inputs: both have inventories 
# foreign input has stoch dt, and doomestic input has deterministic dt
# Timing: Shocks (demand and DT), then decisions
# Code written as a funciton of n 

# Algorith where the dynamic choice is solved via function (see Main_mat) for grid 
#using Pkg
#Pkg.add("Interpolations")

using Distributed
@everywhere using Interpolations, Optim, Random
@everywhere using Statistics
using JLD2
@everywhere using Distributions
@everywhere using NLsolve
using FileIO
using StableRNGs
#using BenchmarkTools
#using Base.Threads
@everywhere using SharedArrays
addprocs(6) 
workers()
Threads.nthreads()
 

@everywhere import Base.+
@everywhere +(f::Function, g::Function) = (x...) -> f(x...) + g(x...)

##############################################################################################################################
# 0. Parameters 

#rng = StableRNG(2904)
#Random.seed!(rng, 2905)
Random.seed!(2905)

@everywhere sgridsizev =  15
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
sig_dem = 0.38 #1.2 #0.50 #0.38
nu = nu_grid(shockgridsizev, mean_dem, sig_dem)
shockgridv[:,1] .= nu[:]                    # Second column is for demand shocks, First column is for delivery time shocks

@everywhere Tv0 = ones(sgridsizev, sgridsizev)
@everywhere vinitv = zeros(sgridsizev, sgridsizev)

# other parameters
@everywhere betav = (1-0.04)^(1/4)  
@everywhere epsiv = 4.0 #4.0 #1.5                               # demand elasticity
@everywhere sigmav = 2.5 #2.5 #2.5 #1.5  #4.0 #0.8                              # input elasticity
@everywhere thv = 0.75 #0.9110 #0.5                       # domestic input weight 
@everywhere deltav = 0.30/4    #0.01                        # depreciation rate

@everywhere parametersv = [betav, epsiv, sigmav, deltav, thv, sgridsizev, shockgridsizev]

# simulation parameters -- Monte Carlo
Nfirmsv = 50000                          # number of firms
Tfirmsv = 200                         # number of periods

include("1Value_fn_v1.jl")
include("2Simulate_v1.jl")

##############################################################################################################################
# 1. initial point
nu_max = maximum(shockgridv[:,1])
X = unconstrained_fn( nu_max, pdv, pfv, parametersv)
sdgridmaxv = X[5]*(2.0)
sdgridv = collect(range(1e-10; length = sgridsizev, stop = sdgridmaxv))
sfgridmaxv = X[4]*(2.0)
sfgridv = collect(range(1e-10; length = sgridsizev, stop = sfgridmaxv))
parametersv = [betav, epsiv, sigmav, deltav, thv, sgridsizev, shockgridsizev];

@show "initial", lamd2v, lamf2v, pfv, sdgridmaxv, sfgridmaxv, nu_max
@show shockgridv


include("1Value_fn_v1.jl")
@time Tv_1, Pnd_1, Pnf_1 = Vfn_d(pdv, pfv, parametersv, shockgridv, sdgridv, sfgridv);

@time Pp_1, Py_1, Px_1, Pxf_1, Pxd_1, Psdp_1, Psfp_1, Pcase_1 = policies(Pnd_1, Pnf_1, pdv, pfv,parametersv, shockgridv, sdgridv, sfgridv);

@time shockMC_1, PsdMC_1, PsfMC_1 = sim_initial(Nfirmsv, Tfirmsv, parametersv, sdgridv, sfgridv);
@time PsdMC_1, PsfMC_1, PpMC_1, PyMC_1, PxMC_1, PxfMC_1, PxdMC_1, PndMC_1, PnfMC_1, PcaseMC_1 = 
simulation(Pnd_1, Pnf_1, pdv, pfv, Nfirmsv, Tfirmsv, parametersv, shockgridv, sdgridv, sfgridv, shockMC_1, PsdMC_1, PsfMC_1);

A_1 = stat_dist( pdv, pfv, Nfirmsv, Tfirmsv, parametersv, shockgridv, PsdMC_1, PsfMC_1, PpMC_1, PyMC_1, PxMC_1, PxfMC_1, PxdMC_1, PndMC_1, PnfMC_1, PcaseMC_1)

#v3_1 = Tv_1, Pnd_1, Pnf_1 
#save("V_v3_1.jld2","v3_1",v3_1)
#utput_1 = load("V_v3_1.jld2","v3_1")


##############################################################################################################################
# 2. pf change only 
@everywhere pfv = 1.0 #1.0/1.2 #0.1

#=
nu_max = maximum(shockgridv[:,1])
X = unconstrained_fn( nu_max, pdv, pfv, parametersv)
sdgridmaxv = X[5]*(1.1)
sdgridv = collect(range(1e-10; length = sgridsizev, stop = sdgridmaxv))
sfgridmaxv = X[4]*(1.1)
sfgridv = collect(range(1e-10; length = sgridsizev, stop = sfgridmaxv))
parametersv = [betav, epsiv, sigmav, deltav, thv, sgridsizev, shockgridsizev];
=#

@show "price change", lamd2v, lamf2v, pfv, sdgridmaxv, sfgridmaxv, nu_max
@show shockgridv

include("1Value_fn_v1.jl")
@time Tv_2, Pnd_2, Pnf_2 = Vfn_d(pdv, pfv, parametersv, shockgridv, sdgridv, sfgridv);

@time Pp_2, Py_2, Px_2, Pxf_2, Pxd_2, Psdp_2, Psfp_2, Pcase_2 = policies(Pnd_2, Pnf_2, pdv, pfv,parametersv, shockgridv, sdgridv, sfgridv);

@time shockMC_2, PsdMC_2, PsfMC_2 = sim_initial(Nfirmsv, Tfirmsv, parametersv, sdgridv, sfgridv);
@time PsdMC_2, PsfMC_2, PpMC_2, PyMC_2, PxMC_2, PxfMC_2, PxdMC_2, PndMC_2, PnfMC_2, PcaseMC_2 = 
simulation(Pnd_2, Pnf_2, pdv, pfv, Nfirmsv, Tfirmsv, parametersv, shockgridv, sdgridv, sfgridv, shockMC_2, PsdMC_2, PsfMC_2);

A_2 = stat_dist( pdv, pfv, Nfirmsv, Tfirmsv, parametersv, shockgridv,PsdMC_2, PsfMC_2, PpMC_2, PyMC_2, PxMC_2, PxfMC_2, PxdMC_2, PndMC_2, PnfMC_2, PcaseMC_2 )

#v3_2 = Tv_2, Pnd_2, Pnf_2 
#save("V_v3_2.jld2","v3_2",v3_2)
#output_2 = load("V_v3_2.jld2","v3_2")


#=
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

=#