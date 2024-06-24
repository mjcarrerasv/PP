# May 2024
# simulate the economy

include("1Value_fn_v2.jl")
#@everywhere PolFunc(i::Int64, PolicyMat::Array{Float64}) = interpolate( (sdgrid,sfgrid), PolicyMat[:,:,i], Gridded(Linear()) )
@everywhere PolFunc(i::Int64, PolicyMat::Array{Float64}, sdgrid::Array{Float64}, sfgrid::Array{Float64}) = interpolate( (sdgrid,sfgrid), PolicyMat[:,:,i], Gridded(Linear()) )


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

    Pnd::Array{Float64} = ones(sgridsize, sgridsize, shockgridsize)
    Pnd .= Pnd0
    Pnf::Array{Float64} = ones(sgridsize, sgridsize, shockgridsize)
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
                
            PndMC[n,t] = PolFunc(shockMC[n,t], Pnd, sdgrid, sfgrid)(PsdMC[n,t], PsfMC[n,t])
            PnfMC[n,t] = PolFunc(shockMC[n,t], Pnf, sdgrid, sfgrid)(PsdMC[n,t], PsfMC[n,t])
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
            #@show PsdMC[n,t+1], t
            PsfMC[n,t+1] = (1-delta)*(PsfMC[n,t] + PnfMC[n,t] - X[4])
            PcaseMC[n,t] = case
            #@show nu,lamd, lamf, PsdMC[n,t], PsdMC[n,t+1], PndMC[n,t], X[5], PsfMC[n,t], PsfMC[n,t+1], PnfMC[n,t], X[4]

        end
    end

return PsdMC, PsfMC, PpMC, PyMC, PxMC, PxfMC, PxdMC, PndMC, PnfMC, PcaseMC
end


##############################################################################################################################
function simulation_int(
    Nfirms0::Int64,
    Tperiod0::Int64,
    parameters0::Array{Float64},
    shockgrid0::Array{Float64},
    sgrid0::Array{Float64},

    Pn0::Array{Float64},
    Pp0::Array{Float64},
    Py0::Array{Float64}, 
    Px0::Array{Float64}, 
    Pxf0::Array{Float64}, 
    Pxd0::Array{Float64}, 
    Psp0::Array{Float64}, 
    Pcase0::Array{Float64},

    shockMC0::Array{Int64},
    PsMC0::Array{Float64}
    )


    Nf::Int64 = Nfirms0
    Tp::Int64 = Tperiod0

    parameters = ones(7)
    parameters::Array{Float64} .= parameters0    
    shockgridsize::Int64 = Int(parameters[7])
    shockgrid = zeros(shockgridsize, 2) 
    shockgrid::Array{Float64} .= shockgrid0  
    sgridsize::Int64 = Int(parameters[6])
    sgrid::Array{Float64} = ones(sgridsize)
    sgrid .= sgrid0

    Px::Array{Float64} = ones(sgridsize, shockgridsize)
    Px .= Px0
    Pxd::Array{Float64} = ones(sgridsize, shockgridsize)
    Pxd .= Pxd0
    Pxf::Array{Float64} = ones(sgridsize, shockgridsize)
    Pxf .= Pxf0
    Py::Array{Float64} = ones(sgridsize, shockgridsize)
    Py .= Py0
    Pp::Array{Float64} = ones(sgridsize, shockgridsize)
    Pp .= Pp0
    Pn::Array{Float64} = ones(sgridsize, shockgridsize)
    Pn .= Pn0
    Psp::Array{Float64} = ones(sgridsize, shockgridsize)
    Psp .= Psp0
    Pcase::Array{Float64} = ones(sgridsize, shockgridsize)
    Pcase .= Pcase0

    shockMC::Array{Int64} = zeros(Int, Nf, Tp)
    shockMC .= shockMC0
    PsMC::Array{Float64} = zeros(Nf, Tp+1)
    PsMC .= PsMC0

    PnMC::Array{Float64} = zeros(Nf, Tp)
    PxMC::Array{Float64} = zeros(Nf, Tp)
    PxdMC::Array{Float64} = zeros(Nf, Tp)
    PxfMC::Array{Float64} = zeros(Nf, Tp)
    PpMC::Array{Float64} = zeros(Nf, Tp)
    PyMC::Array{Float64} = zeros(Nf, Tp)
    PcaseMC::Array{Float64} = zeros(Nf, Tp)

    PolFunc(i::Int64, PolicyMat::Array{Float64}) =
    interpolate( (sgrid,), PolicyMat[:,i], Gridded(Linear()) )

    for t = 1:Tp       
        for n = 1:Nf
            PsMC[n,t+1] = PolFunc(shockMC[n,t], Psp0)(PsMC[n,t])

            PxMC[n,t] = PolFunc(shockMC[n,t], Px)(PsMC[n,t])
            PxdMC[n,t] = PolFunc(shockMC[n,t], Pxd)(PsMC[n,t])
            PxfMC[n,t] = PolFunc(shockMC[n,t], Pxf)(PsMC[n,t])
            PyMC[n,t] = PolFunc(shockMC[n,t], Py)(PsMC[n,t])
            PpMC[n,t] = PolFunc(shockMC[n,t], Pp)(PsMC[n,t])
            PnMC[n,t] = PolFunc(shockMC[n,t], Pn)(PsMC[n,t])
            PcaseMC[n,t] = PolFunc(shockMC[n,t], Pcase)(PsMC[n,t])

        end
    end

return PsMC, PpMC, PyMC, PxMC, PxfMC, PxdMC, PnMC, PcaseMC
end
