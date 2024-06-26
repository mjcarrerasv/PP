
##############################################################################################################################
##############################################################################################################################
###############################################################################################################################
# 1. initial point
lamd2v = (TTv - dayd_meanv)/TTv
lamf2v = (TTv - dayf_meanv)/TTv
shockgridv[:,2] .= lamd2v;                     # Second column: domestic delivery times 
shockgridv[:,3] .= lamf2v;   

@everywhere pfv = 1.0 #0.1
@everywhere pfv = 1.0/1.2 #0.1

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
invf_xf = ones(Nfirmsv)*(-10.0)
invd_xd = ones(Nfirmsv)*(-10.0)

invf_sale .= pfv.*PsfMC_3[:, Tfirmsv+1]./(PyMC_3[:, Tfirmsv].*PpMC_3[:, Tfirmsv])
invd_sale .= pdv.*PsdMC_3[:, Tfirmsv+1]./(PyMC_3[:, Tfirmsv].*PpMC_3[:, Tfirmsv])
inv_sale .= (pfv.*PsfMC_3[:, Tfirmsv+1].+ pdv.*PsdMC_3[:, Tfirmsv+1])./(PyMC_3[:, Tfirmsv].*PpMC_3[:, Tfirmsv])

invf_prod .= PsfMC_3[:, Tfirmsv+1]./(PyMC_3[:, Tfirmsv])
invd_prod .= PsdMC_3[:, Tfirmsv+1]./(PyMC_3[:, Tfirmsv])
inv_prod .= (PsfMC_3[:, Tfirmsv+1] .+ PsdMC_3[:, Tfirmsv+1])./(PyMC_3[:, Tfirmsv])

invf_xf .= PsfMC_3[:, Tfirmsv+1]./(PxfMC_3[:, Tfirmsv])
invd_xd .= PsdMC_3[:, Tfirmsv+1]./(PxdMC_3[:, Tfirmsv])

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




#######
shockMC_1

shockMC_v2



mean(Py_1.-Py_v2)

mean(Pnd_1.-Pnd_v2)

mean(Pnf_1.-Pnf_v2)
mean(abs.(Psdp_1.-Psdp_v2))
mean(Psfp_1.-Psfp_v2)
aa = abs.(Psdp_1[1,1,:].-Psdp_v2[1,1,:])
aa2 = abs.(Psdp_1[2,1,:].-Psdp_v2[2,1,:])
aa3 = abs.(Psdp_1[3,1,:].-Psdp_v2[3,1,:])
aa4 = abs.(Psdp_1[4,1,:].-Psdp_v2[4,1,:])

using Plots
histogram(aa)
histogram!(aa2)
histogram!(aa3)
histogram!(aa4)

mean(Py_1.-Py_v2)

#####
shockMC_1 .- shockMC_v2
shockMC_1v .- shockMC_v2
shockMC_v2v .- shockMC_v2


PnfMC_1 .- PnfMC_v2
mean(PnfMC_1[:,1] .- PnfMC_v2[:,1])
mean(PnfMC_1[:,Tfirmsv] .- PnfMC_v2[:,Tfirmsv])

PndMC_1 .- PndMC_v2

mean(PndMC_1[:,1] .- PndMC_v2[:,1])
mean(PndMC_1[:,Tfirmsv] .- PndMC_v2[:,Tfirmsv])

PsdMC_1 .- PsdMC_v2
PsdMC_1v .- PsdMC_v2
PsdMC_1v .- PsdMC_v2v


mean(PndMC_1[:,1] .- PndMC_v2[:,1])
mean(PndMC_1[:,Tfirmsv] .- PndMC_v2[:,Tfirmsv])


@time a, b, c = sim_initial(Nfirmsv, Tfirmsv, parametersv, sdgridv, sfgridv);
@time a1, b1, c1 = sim_initial(Nfirmsv, Tfirmsv, parametersv, sdgridv, sfgridv);


#v2v vs v2
PsdMC_v2 .- PsdMC_v2v
PndMC_v2 .- PndMC_v2v
PsdMC_1 .- PsdMC_1v 
PsdMC_v2v .- PsdMC_1v 

a = ones(10)*5.0
b = ones(5)*2
c = a, b
save("XX.jld2","c",c)
output = load("XX.jld2","c")
