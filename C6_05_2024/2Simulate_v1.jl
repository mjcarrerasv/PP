# May 2024
# simulate the economy

include("1Value_fn_v1.jl")
#@everywhere PolFunc(i::Int64, PolicyMat::Array{Float64}) = interpolate( (sdgrid,sfgrid), PolicyMat[:,:,i], Gridded(Linear()) )
@everywhere PolFunc(PolicyMat::Array{Float64}, sdgrid::Array{Float64}, sfgrid::Array{Float64}) = interpolate( (sdgrid,sfgrid), PolicyMat[:,:], Gridded(Linear()) )


@everywhere function sim_initial(
                Nfirms0::Int64,
                Tperiod0::Int64,
                parameters0::Array{Float64},
                sdgrid0::Array{Float64},
                sfgrid0::Array{Float64}
                )

    Nf::Int64 = Nfirms0
    Tp::Int64 = Tperiod0

    parameters = ones(7)
    parameters::Array{Float64} .= parameters0    
    shockgridsize::Int64 = Int(parameters[7])
    sgridsize::Int64 = Int(parameters[6])
    sdgrid::Array{Float64} = ones(sgridsize)
    sdgrid .= sdgrid0
    sfgrid::Array{Float64} = ones(sgridsize)
    sfgrid .= sfgrid0
    
    shockMC::Array{Int64} = zeros(Int, Nf, Tp)
    rng = StableRNG(2904)
    Random.seed!(rng, 3012)
    shockMC = rand(rng, 1:shockgridsize, Nf, Tp)

    PsdMC::Array{Float64} = zeros(Nf, Tp+1)
    PsfMC::Array{Float64} = zeros(Nf, Tp+1)

    for n = 1:Nf
        PsdMC[n,1] = rand(rng, sdgrid)
        PsfMC[n,1] = rand(rng, sfgrid)
    end

    return shockMC, PsdMC, PsfMC
end



##############################################################################################################################
@everywhere function simulation(    
                    Pnd0::Array{Float64},
                    Pnf0::Array{Float64},
                    pd0::Float64,
                    pf0::Float64,
                    #lamd0::Float64,
                    Nfirms0::Int64,
                    Tperiod0::Int64,
                    parameters0::Array{Float64},
                    shockgrid0::Array{Float64},
                    sdgrid0::Array{Float64},
                    sfgrid0::Array{Float64},
                    shockMC0::Array{Int64},
                    PsdMC0::Array{Float64},
                    PsfMC0::Array{Float64}
                    )

    pd::Float64 = pd0                                       
    pf::Float64 = pf0  
    #lamd::Float64 = lamd0

    Nf::Int64 = Nfirms0
    Tp::Int64 = Tperiod0

    parameters = ones(7)
    parameters::Array{Float64} .= parameters0  
    delta::Float64 = parameters[4]  
    shockgridsize::Int64 = Int(parameters[7])
    shockgrid = zeros(shockgridsize, 3) 
    shockgrid::Array{Float64} .= shockgrid0  
    sgridsize::Int64 = Int(parameters[6])
    sdgrid::Array{Float64} = ones(sgridsize)
    sdgrid .= sdgrid0
    sfgrid::Array{Float64} = ones(sgridsize)
    sfgrid .= sfgrid0

    shockMC::Array{Int64} = zeros(Int, Nf, Tp)
    shockMC .= shockMC0
    PsdMC::Array{Float64} = SharedArray{Float64}(Nf, Tp+1)
    PsdMC .= PsdMC0
    PsfMC::Array{Float64} = SharedArray{Float64}(Nf, Tp+1)
    PsfMC .= PsfMC0

    Pnd::Array{Float64} = ones(sgridsize, sgridsize)
    Pnd .= Pnd0
    Pnf::Array{Float64} = ones(sgridsize, sgridsize)
    Pnf .= Pnf0

    PndMC = SharedArray{Float64}(Nf, Tp)
    PnfMC = SharedArray{Float64}(Nf, Tp)
    PxMC = SharedArray{Float64}(Nf, Tp)
    PxdMC = SharedArray{Float64}(Nf, Tp)
    PxfMC = SharedArray{Float64}(Nf, Tp)
    PpMC = SharedArray{Float64}(Nf, Tp)
    PyMC = SharedArray{Float64}(Nf, Tp)
    PcaseMC = SharedArray{Float64}(Nf, Tp)



    for t = 1:Tp 
       # @sync @distributed  for n = 1:Nf
         for n = 1:Nf   
                
            PndMC[n,t] = PolFunc(Pnd, sdgrid, sfgrid)(PsdMC[n,t], PsfMC[n,t])
            PnfMC[n,t] = PolFunc(Pnf, sdgrid, sfgrid)(PsdMC[n,t], PsfMC[n,t])
            
            nu = shockgrid[shockMC[n,t], 1]
            lamd = shockgrid[shockMC[n,t], 2]
            lamf = shockgrid[shockMC[n,t], 3]

            X, case = cases_fn(PndMC[n,t], PnfMC[n,t], PsdMC[n,t], PsfMC[n,t], nu, lamd, lamf, pd, pf, parameters)
            # x1 = p, x2 = y, x3 = x, x4 = xf, x5 = xd
            PpMC[n,t] = X[1]
            PyMC[n,t] = X[2]
            PxMC[n,t] = X[3]
            PxfMC[n,t] = X[4]
            PxdMC[n,t] = X[5]   
            PsdMC[n,t+1] = (1-delta)*(PsdMC[n,t] + PndMC[n,t] - X[5])
            PsfMC[n,t+1] = (1-delta)*(PsfMC[n,t] + PnfMC[n,t] - X[4])
            PcaseMC[n,t] = case
            #@show n,t, PsdMC[n,t+1], PsdMC[n,t], PndMC[n,t], X[5]
            #@show nu,lamd, lamf, PsdMC[n,t], PsdMC[n,t+1], PndMC[n,t], X[5], PsfMC[n,t], PsfMC[n,t+1], PnfMC[n,t], X[4]

        end
    end


return PsdMC, PsfMC, PpMC, PyMC, PxMC, PxfMC, PxdMC, PndMC, PnfMC, PcaseMC
end


##############################################################################################################################
@everywhere function stat_dist(    

    pd0::Float64,
    pf0::Float64,
    #lamd0::Float64,
    Nfirms0::Int64,
    Tperiod0::Int64,
    parameters0::Array{Float64},
    shockgrid0::Array{Float64},
    #sdgrid0::Array{Float64},
    #sfgrid0::Array{Float64},
    #shockMC0::Array{Int64},
    PsdMC0,
    PsfMC0,
    PpMC0,
     PyMC0, 
     PxMC0, 
     PxfMC0, 
     PxdMC0,
     PndMC0,
     PnfMC0, 
     PcaseMC0
    )

    pdv::Float64 = pd0                                       
    pfv::Float64 = pf0  
    #lamd::Float64 = lamd0

    Nfirmsv::Int64 = Nfirms0
    Tfirmsv::Int64 = Tperiod0

    parametersv = ones(7)
    parametersv::Array{Float64} .= parameters0   
    shockgridsize::Int64 = Int(parametersv[7])
    shockgridv = zeros(shockgridsize, 3) 
    shockgridv::Array{Float64} .= shockgrid0  

 
    PsdMC::Array{Float64} = ones(Nfirmsv, Tfirmsv+1)
    PsdMC .= PsdMC0
    PsfMC::Array{Float64} = ones(Nfirmsv, Tfirmsv+1)
    PsfMC .= PsfMC0
    PndMC::Array{Float64} = ones(Nfirmsv, Tfirmsv)
    PndMC .= PndMC0
    PnfMC::Array{Float64} =ones(Nfirmsv, Tfirmsv)
    PnfMC .= PnfMC0
    PxMC::Array{Float64} = ones(Nfirmsv, Tfirmsv)
    PxMC .= PxMC0
    PxdMC::Array{Float64} = ones(Nfirmsv, Tfirmsv)
    PxdMC .= PxdMC0
    PxfMC::Array{Float64} = ones(Nfirmsv, Tfirmsv)
    PxfMC .= PxfMC0
    PpMC::Array{Float64} = ones(Nfirmsv, Tfirmsv)
    PpMC .= PpMC0
    PyMC::Array{Float64} = ones(Nfirmsv, Tfirmsv)
    PyMC .= PyMC0
    PcaseMC::Array{Float64} = ones(Nfirmsv, Tfirmsv)
    PcaseMC .= PcaseMC0

a1 = count(<=(0.0), PcaseMC[:, Tfirmsv])/Nfirmsv
nu_mean = mean(shockgridv[:,1])
a2 = unconstrained_fn( nu_mean, pdv, pfv, parametersv)
a3 = mean(PyMC[:, Tfirmsv])
a4 = mean(PpMC[:, Tfirmsv])

a5 = mean(PndMC[:, Tfirmsv])
a6 = mean(PnfMC[:, Tfirmsv])
a7 = pfv.*mean(PnfMC[:, Tfirmsv])

xf_sale = ones(Nfirmsv)*(-10.0)
xf_y = ones(Nfirmsv)*(-10.0)
xf_inp = ones(Nfirmsv)*(-10.0)
xd_sale = ones(Nfirmsv)*(-10.0)
xd_y = ones(Nfirmsv)*(-10.0)
xd_inp = ones(Nfirmsv)*(-10.0)

xf_sale .= pfv.*PxfMC[:, Tfirmsv]./(PyMC[:, Tfirmsv].*PpMC[:, Tfirmsv])
xf_y .= PxfMC[:, Tfirmsv]./(PyMC[:, Tfirmsv])
xf_inp .= pfv.*PxfMC[:, Tfirmsv]./(pdv.*PxdMC[:, Tfirmsv] .+ pfv.*PxfMC[:, Tfirmsv])

xd_sale .= pdv.*PxdMC[:, Tfirmsv]./(PyMC[:, Tfirmsv].*PpMC[:, Tfirmsv])
xd_y .= PxdMC[:, Tfirmsv]./(PyMC[:, Tfirmsv])
xd_inp .= pdv.*PxdMC[:, Tfirmsv]./(pdv.*PxdMC[:, Tfirmsv] .+ pfv.*PxfMC[:, Tfirmsv])

a8 = mean(PxfMC[:, Tfirmsv])
a9 = pfv.*mean(PxfMC[:, Tfirmsv])
a10 = mean(xf_sale[:])
a11 = mean(xf_y[:])
a12 =  mean(xf_inp[:])

a13 = mean(PxdMC[:, Tfirmsv])
a14 = mean(xd_sale[:])
a15 = mean(xd_y[:])
a16 =  mean(xd_inp[:])

invf_prod = ones(Nfirmsv)*(-10.0)
invd_prod = ones(Nfirmsv)*(-10.0)
inv_prod = ones(Nfirmsv)*(-10.0)
invf_sale = ones(Nfirmsv)*(-10.0)
invd_sale = ones(Nfirmsv)*(-10.0)
inv_sale = ones(Nfirmsv)*(-10.0)
invf_xf = ones(Nfirmsv)*(-10.0)
invd_xd = ones(Nfirmsv)*(-10.0)

invf_sale .= pfv.*PsfMC[:, Tfirmsv+1]./(PyMC[:, Tfirmsv].*PpMC[:, Tfirmsv])
invd_sale .= pdv.*PsdMC[:, Tfirmsv+1]./(PyMC[:, Tfirmsv].*PpMC[:, Tfirmsv])
inv_sale .= (pfv.*PsfMC[:, Tfirmsv+1].+ pdv.*PsdMC[:, Tfirmsv+1])./(PyMC[:, Tfirmsv].*PpMC[:, Tfirmsv])

invf_prod .= PsfMC[:, Tfirmsv+1]./(PyMC[:, Tfirmsv])
invd_prod .= PsdMC[:, Tfirmsv+1]./(PyMC[:, Tfirmsv])
inv_prod .= (PsfMC[:, Tfirmsv+1] .+ PsdMC[:, Tfirmsv+1])./(PyMC[:, Tfirmsv])

invf_xf .= PsfMC[:, Tfirmsv+1]./(PxfMC[:, Tfirmsv])
invd_xd .= PsdMC[:, Tfirmsv+1]./(PxdMC[:, Tfirmsv])

invf = ones(Nfirmsv)*(-10.0)
invf_val = ones(Nfirmsv)*(-10.0)
invd = ones(Nfirmsv)*(-10.0)
inv = ones(Nfirmsv)*(-10.0)
inv_val = ones(Nfirmsv)*(-10.0)

invf.= PsfMC[:, Tfirmsv+1]
invf_val.= pfv.*PsfMC[:, Tfirmsv+1]
invd .= PsdMC[:, Tfirmsv+1]
inv .= (PsfMC[:, Tfirmsv+1].+ PsdMC[:, Tfirmsv+1])
inv_val .= (pfv.*PsfMC[:, Tfirmsv+1].+ pdv.*PsdMC[:, Tfirmsv+1])

a17 = mean(invf[:])
a18 = mean(invf_val[:])
a19 = mean(invf_sale[:])
a20 = mean(invf_prod[:])
a21 = mean(invf_xf[:])

a22 = mean(invd[:])
a23 = mean(invd_sale[:])
a24 = mean(invd_prod[:])
a25 = mean(invd_xd[:])

a26 = mean(inv[:])
a27 = mean(inv_val[:])
a28 = mean(inv_sale[:])
a29 = mean(inv_prod[:])

@show a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29

return  a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29
end

##############################################################################################################################
@everywhere function moments_initial(
                                    pd0::Float64,
                                    pf0::Float64,
                                    parameters0::Array{Float64},
                                    shockgrid0::Array{Float64},
                                    sdgrid0::Array{Float64},
                                    sfgrid0::Array{Float64},
                                    sigma_nu0::Float64,
                                    Nfirms0::Int64,
                                    Tperiod0::Int64,
                                    #data moments
                                    invdata0::Float64,
                                    impdata0::Float64
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

pd::Float64 = pd0                                       
pf::Float64 = pf0   
Nfirms::Int64 = Nfirms0
Tperiod::Int64 = Tperiod0

invdata::Float64 =invdata0
impdata::Float64 = impdata0

Tv::Array{Float64} = ones(sgridsize, sgridsize)
Pnd::Array{Float64} = ones(sgridsize, sgridsize)
Pnf::Array{Float64} = ones(sgridsize, sgridsize)

iter::Int64 = 0
max_iter::Int64 = 70
tol::Float64 = 1e-3
nu_tol::Float64 = 1e-3
th_tol::Float64 = 1e-3
error::Float64 = Inf
error_inv::Float64 = Inf
error_imp::Float64 = Inf

th_min::Float64 = 0.0001
th_max::Float64 = 0.999
th_new::Float64 = 0.5

nu_min::Float64 = 0.0001
nu_max::Float64 = 2.0
nu_new::Float64 = 0.5

mean_dem::Float64 = 1.0
nu::Float64 = sigma_nu0
th::Float64 = parameters[5]
#parametersv = [betav, epsiv, sigmav, deltav, thv, sgridsizev, shockgridsizev];

while iter <= max_iter && error > tol

        ### OBTAIN EQUILIBRIUM 
        parameters = [beta, epsi, sigma, delta, th, sgridsize, shockgridsize]
        nu_vec = nu_grid(shockgridsize, mean_dem, nu)
        shockgrid[:,1] .= nu_vec[:]                    # Second column is for demand shocks, First column is for delivery time shocks
        @show shockgrid[1:10,1], nu, th

        nu_max = maximum(shockgrid[:,1])
        X = unconstrained_fn( nu_max, pd, pf, parameters)
        sdgridmax = X[5]*(1.1)
        sdgrid = collect(range(1e-10; length = sgridsize, stop = sdgridmax))
        sfgridmax = X[4]*(1.1)
        sfgrid = collect(range(1e-10; length = sgridsize, stop = sfgridmax))

        #1 Obtain VF and policy functions for orders
        @time Tv, Pnd, Pnf = Vfn_d(pd, pf, parameters, shockgrid, sdgrid, sfgrid);

        #2 Obtain other policy functions: function of stochastic shocks
        #Pp, Py, Px, Pxf, Pxd, Psdp, Psfp, Pcase = policies(Pnd, Pnf, pd, pf, parameters, shockgrid, sdgrid, sfgrid);

        #3 Obtain stationary distribution
        shockMC, PsdMC, PsfMC = sim_initial(Nfirms, Tfirms, parameters, sdgrid, sfgrid);
         PsdMC, PsfMC, PpMC, PyMC, PxMC, PxfMC, PxdMC, PndMC, PnfMC, PcaseMC = 
        simulation(Pnd, Pnf, pd, pf, Nfirms, Tfirms, parameters, shockgrid, sdgrid, sfgrid, shockMC, PsdMC, PsfMC);

        #4 Obtain moments of the stationary distribution
        Moments = stat_dist(pdv, pfv, Nfirmsv, Tfirmsv, parametersv, shockgridv, PsdMC, PsfMC, PpMC, PyMC, PxMC, PxfMC, PxdMC, PndMC, PnfMC, PcaseMC)

        ### CHECK WITH DATA
        inv_sale = Moments[28] 
        inv_prod = Moments[29] 
        xf_sale = Moments[10] 
        xd_sale = Moments[14] 
        xf_inp = Moments[12] 

        ### UPDATE Parameters: theta and sigma_nu
        error_inv = inv_sale - invdata
        error_imp = xf_inp - impdata
        error = max(abs(error_inv), abs(error_imp))

        if error_inv > nu_tol
            nu_max = nu
            if nu_min > 0.0001
              nu_new = (nu_min + nu_max)/2
            else
              nu_new = nu - 0.01
            end

        elseif error_inv < -1*nu_tol
            nu_min = th
             if nu_max < 2.0
               nu_new = (nu_min + nu_max)/2
             else
               nu_new = nu + 0.01
             end

        elseif error_inv < nu_tol && error_inv < -1*nu_tol
             error_inv = 0.0
             nu_new = nu
        end

         @show inv_sale, inv_prod, invdata, error_inv, nu_min, nu_max, nu, nu_new

        if error_imp < -1*th_tol
            th_max = th
            if th_min > 0.0001
              th_new = (th_min + th_max)/2
            else
              th_new = th - 0.05
            end

        elseif error_imp > th_tol
            th_min = th
             if th_max < 0.999
               th_new = (th_min + th_max)/2
             else
               th_new = th + 0.05
             end

        elseif error_imp < th_tol && error_imp < -1*th_tol
             error_imp = 0.0
             th_new = th
        end
        
        @show  xf_sale, xd_sale, xf_inp, impdata, error_imp, th_min, th_max, th_new

        @show iter, error
        th = th_new
        nu = nu_new
        iter += 1

end

return Tv, Pnd, Pnf 

end


