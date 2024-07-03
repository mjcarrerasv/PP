# matrix of shocks for delivery times


function lam_grid( lamgridsize0::Int64,
                    mean0::Float64,
                    delaypct0::Float64,
                    t0::Float64
                    )

t::Float64 = t0      # a period in the model equals to a month
lamgridsizevv = lamgridsize0

#1.  data of delivery times and delays
mean = mean0
delaypct = delaypct0
dif = mean*delaypct

#2. obtain the mean and sd for the lognormal dist
    # that gives data above
mu = log(mean)

# obtain the sd using scatter intervals 95%
    # [mu/var, mu*var]
low = mean - dif
hi = mean + dif
sd_low = (mean/low)^(0.5)
sd_hi = (hi/mean)^(0.5)
sd_mean = (sd_low + sd_hi)/2
sd = log(sd_mean)

#3. include linear transformation to go from data to lam:
    # lam = 1 - x/30
    # so first convert log normal to y = ax, a = 1/t
mu_tr = mu + log(1/t)
sd_tr = sd


#4. draw from lognormal, and apply final part of
    # linear transformation
#rng = StableRNG(2904)
#Random.seed!(rng, 2905)

# halton draw
U = MVHalton(lamgridsizevv, 1)
#0,1 normal cdf
Z = norminvcdf.(U)

dves = exp.(mu_tr .+ sd_tr.*Z)

lam = max.(0.01, 1 .- dves)

return lam, dves
end

#####################################################################
function nu_grid(lamgridsize0::Int64,
    mean_dem0::Float64,
    sig_dem0::Float64
    )

    mean_dem = mean_dem0
    sig_dem = sig_dem0
    lamgridsizev = lamgridsize0
    nu = ones(lamgridsizev)

    #rng = StableRNG(2904)
    #Random.seed!(rng, 2905)

    # halton draw
    U = MVHalton(lamgridsizev, 1)
    #0,1 normal cdf
    Z = norminvcdf.(U)
    nu[:] = exp.(mean_dem .+ sig_dem .* Z) # log normal

return nu
end


#####################################################################
function lam_grid2( 
    
    lamgridsize0::Int64,
    mean0::Float64,
    delay_days0::Float64,
    t0::Float64
    )

t::Float64 = t0      # a period in the model equals to a month
lamgridsizevv = lamgridsize0

#1.  data of delivery times and delays
mean = mean0
dif = delay_days0

#2. obtain the mean and sd for the lognormal dist
# that gives data above
mu = log(mean)

# obtain the sd using scatter intervals 95%
# [mu/var, mu*var]
low = mean 
hi = mean + dif
sd_low = (mean/low)^(0.5)
sd_hi = (hi/mean)^(0.5)
sd_mean = (sd_low + sd_hi)/2
sd = log(sd_mean)

#3. include linear transformation to go from data to lam:
# lam = 1 - x/30
# so first convert log normal to y = ax, a = 1/t
mu_tr = mu + log(1/t)
sd_tr = sd


#4. draw from lognormal, and apply final part of
# linear transfo2rmation
#rng = StableRNG(2904)
#Random.seed!(rng, 2905)

    # halton draw
    U = MVHalton(lamgridsizevv, 1)
    #0,1 normal cdf
    Z = norminvcdf.(U)

dves = exp.(mu_tr .+ sd_tr.*Z)

lam = max.(0.01, 1 .- dves)

return lam, dves
end

#####################################################################
function lam_grid3( 
    
    lamgridsize0::Int64,
    mean0::Float64,
    delay_days0::Float64,
    t0::Float64,
    lamdeter0::Float64
    )

t::Float64 = t0      # a period in the model equals to a month
lamgridsizevv = lamgridsize0
lamdeter = lamdeter0
#1.  data of delivery times and delays
mean = mean0
dif = delay_days0

#2. obtain the mean and sd for the lognormal dist
# that gives data above
mu = log(mean)

# obtain the sd using scatter intervals 95%
# [mu/var, mu*var]
hi = mean + dif
sd_hi = (hi/mean)^(0.5)
sd = log(sd_hi)

#3. include linear transformation to go from data to lam:
# lam = 1 - x/30
# so first convert log normal to y = ax, a = 1/t
mu_tr = mu + log(1/t)
sd_tr = sd

#4. draw from lognormal, and apply final part of
# linear transfo2rmation
#rng = StableRNG(2904)
#Random.seed!(rng, 2905)

    # halton draw
    U = MVHalton(lamgridsizevv, 1)
    #0,1 normal cdf
    Z = norminvcdf.(U)

dves = exp.(mu_tr .+ sd_tr.*Z)

lam = max.(0.01, 1 .- dves)
lam2 = min.(lamdeter, lam)

return lam2, dves
end
