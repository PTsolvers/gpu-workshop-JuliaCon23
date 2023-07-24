using Printf, LinearAlgebra
using CairoMakie
using Enzyme
using Optim
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using ImplicitGlobalGrid
using MPI

# @init_parallel_stencil(Threads, Float64, 2)
@init_parallel_stencil(CUDA, Float64, 2)

@views inn(A) = A[2:end-1, 2:end-1]
@views avx(A) = 0.5 .* (A[1:end-1, :] .+ A[2:end, :])
@views avy(A) = 0.5 .* (A[:, 1:end-1] .+ A[:, 2:end])
@views maxloc(A) = max.(A[2:end-1, 2:end-1], max.(max.(A[1:end-2, 2:end-1], A[3:end, 2:end-1]),
                                                  max.(A[2:end-1, 1:end-2], A[2:end-1, 3:end])))

sum_g(A) = (sum_l = sum(A); MPI.Allreduce(sum_l, MPI.SUM, MPI.COMM_WORLD))
max_g(A) = (max_l = maximum(A); MPI.Allreduce(max_l, MPI.MAX, MPI.COMM_WORLD))

@parallel function smooth_d!(A2, A)
        @inn(A2) = @inn(A) + 0.2 * (@d2_xi(A) + @d2_yi(A))
    return
end

function smooth!(A2, A; nsm=1)
    for _ ∈ 1:nsm
        @hide_communication (16, 8) begin
            @parallel smooth_d!(A2, A)
            A, A2 = A2, A
            update_halo!(A)
        end
    end
    return
end

@parallel_indices (ix) function bc_x!(A, val)
    A[ix,   1] = val
    A[ix, end] = val
    return
end

@parallel_indices (iy) function bc_y!(A, val)
    A[  1, iy] = val
    A[end, iy] = val
    return
end

@parallel function residual_fluxes!(Rqx, Rqy, qx, qy, Pf, K, dx, dy)
    @inn_x(Rqx) = @inn_x(qx) + @av_xa(K) * @d_xa(Pf) / dx
    @inn_y(Rqy) = @inn_y(qy) + @av_ya(K) * @d_ya(Pf) / dy
return
end

@parallel function residual_pressure!(RPf, qx, qy, Qf, dx, dy)
    @all(RPf) = @d_xa(qx) / dx + @d_ya(qy) / dy - @all(Qf)
    return
end

@parallel function update_fluxes!(qx, qy, Rqx, Rqy, cfl, nx, ny, re)
    @inn_x(qx) = @inn_x(qx) - @inn_x(Rqx) / (1.0 + 2cfl * nx / re)
    @inn_y(qy) = @inn_y(qy) - @inn_y(Rqy) / (1.0 + 2cfl * ny / re)
    return
end

@parallel function update_pressure!(Pf, RPf, K_max, vdτ, ly, re)
    @all(Pf) = @all(Pf) - @all(RPf) * (vdτ * ly / re) / @all(K_max)
    return
end

@views function forward_solve!(me, logK, fields, scalars, iter_params; visu=nothing)
    (;Pf, qx, qy, Qf, RPf, Rqx, Rqy, K)               = fields
    (;nxg, nyg, dx, dy)                               = scalars
    (;cfl, re, vdτ, ly, ϵtol, maxiter, ncheck, K_max) = iter_params
    isnothing(visu) || ((;qx_c, qy_c, qM, Pf_v, K_v, qM_v, qx_c_v, qy_c_v, Pf_inn, K_inn, fig, plt, st) = visu)
    K .= exp.(logK)
    # approximate diagonal (Jacobi) preconditioner
    K_max .= K; K_max[2:end-1, 2:end-1] .= maxloc(K); K_max[:, [1, end]] .= K_max[:, [2, end-1]]
    update_halo!(K_max)
    # iterative loop
    iters_evo = Float64[]; errs_evo = Float64[]
    err = 2ϵtol; iter = 1
    while err >= ϵtol && iter <= maxiter
        @parallel residual_fluxes!(Rqx, Rqy, qx, qy, Pf, K, dx, dy)
        @hide_communication (16, 8) begin
            @parallel update_fluxes!(qx, qy, Rqx, Rqy, cfl, nxg, nyg, re)
            update_halo!(qx, qy)
        end
        @parallel residual_pressure!(RPf, qx, qy, Qf, dx, dy)
        @parallel update_pressure!(Pf, RPf, K_max, vdτ, ly, re)
        if iter % ncheck == 0
            err = max_g(abs.(RPf))
            push!(iters_evo, iter/nxg); push!(errs_evo, err)
            (me==0) && @printf("  #iter/nx=%.1f, max(err)=%1.3e\n", iter/nxg, err)
        end
        iter += 1
    end
    if !isnothing(visu)
        qx_c .= Array(avx(qx))[2:end-1,2:end-1]; qy_c .= Array(avy(qy))[2:end-1,2:end-1]; qM .= sqrt.(qx_c.^2 .+ qy_c.^2)
        qx_c ./= qM; qy_c ./= qM
        Pf_inn .= Array(inn(Pf)); gather!(Pf_inn, Pf_v)
        K_inn  .= Array(inn(K));  gather!(K_inn, K_v)
        gather!(qM, qM_v)
        gather!(qx_c, qx_c_v)
        gather!(qy_c, qy_c_v)
        if me==0
            plt.fld.Pf[3] = Pf_v
            plt.fld.K[3]  = log10.(K_v)
            plt.fld.qM[3] = qM_v
            plt.fld.ar[3] = qx_c_v[1:st:end, 1:st:end]
            plt.fld.ar[4] = qy_c_v[1:st:end, 1:st:end]
            plt.err[1] = Point2.(iters_evo, errs_evo)
            # display(fig)
            CairoMakie.save("out_fwd.png", fig)
        end
    end
    return
end

@views function adjoint_solve!(me, logK, fwd_params, adj_params, loss_params)
    # unpack forward
    (;Pf, qx, qy, Qf, RPf, Rqx, Rqy, K) = fwd_params.fields
    (;nxg, nyg, dx, dy)                 = fwd_params.scalars
    # unpack adjoint
    (;P̄f, q̄x, q̄y, R̄Pf, R̄qx, R̄qy, Ψ_qx, Ψ_qy, Ψ_Pf)      = adj_params.fields
    (;∂J_∂Pf)                                           = loss_params.fields
    (;cfl, re_a, vdτ, ly, ϵtol, maxiter, ncheck, K_max) = adj_params.iter_params
    # iterative loop
    iters_evo = Float64[]; errs_evo = Float64[]
    err = 2ϵtol; iter = 1
    while err >= ϵtol && iter <= maxiter
        R̄qx .= Ψ_qx
        R̄qy .= Ψ_qy
        P̄f  .= .-∂J_∂Pf
        q̄x  .= 0.0
        q̄y  .= 0.0
        @hide_communication (16, 8) begin
            @parallel ∇=(Rqx->R̄qx, Rqy->R̄qy, qx->q̄x, qy->q̄y, Pf->P̄f) residual_fluxes!(Rqx, Rqy, qx, qy, Pf, K, dx, dy)
            @parallel (1:size(P̄f,1)) bc_x!(P̄f, 0.0)
            @parallel (1:size(P̄f,2)) bc_y!(P̄f, 0.0)
            update_halo!(P̄f)
        end
        @parallel update_pressure!(Ψ_Pf, P̄f, K_max, vdτ, ly, re_a)
        R̄Pf .= Ψ_Pf
        @parallel ∇=(RPf->R̄Pf, qx->q̄x, qy->q̄y) residual_pressure!(RPf, qx, qy, Qf, dx, dy)
        @hide_communication (16, 8) begin
            @parallel update_fluxes!(Ψ_qx, Ψ_qy, q̄x, q̄y, cfl, nxg, nyg, re_a)
            update_halo!(Ψ_qx, Ψ_qy)
        end
        if iter % ncheck == 0
            err = max_g(abs.(P̄f))
            push!(iters_evo, iter/nxg); push!(errs_evo, err)
            (me==0) && @printf("  #iter/nx=%.1f, max(err)=%1.6e\n", iter/nxg, err)
        end
        iter += 1
    end
    return
end

@views function loss(me, logK, fwd_params, loss_params; kwargs...)
    (;Pf_obs)       = loss_params.fields
    (;ixobs, iyobs) = loss_params.scalars
    (me==0) && @info "Forward solve"
    forward_solve!(me, logK, fwd_params...; kwargs...)
    Pf = fwd_params.fields.Pf
    return 0.5 * sum_g((Pf[ixobs, iyobs] .- Pf_obs).^2)
end

function ∇loss!(me, logK̄, logK, fwd_params, adj_params, loss_params; reg=nothing, kwargs...)
    # unpack
    (;R̄qx, R̄qy, Ψ_qx, Ψ_qy)    = adj_params.fields
    (;Pf, qx, qy, Rqx, Rqy, K) = fwd_params.fields
    (;dx, dy)                  = fwd_params.scalars
    (;Pf_obs, ∂J_∂Pf)          = loss_params.fields
    (;ixobs, iyobs)            = loss_params.scalars
    (me==0) && @info "Forward solve"
    forward_solve!(me, logK, fwd_params...; kwargs...)
    # set tangent
    ∂J_∂Pf[ixobs, iyobs] .= Pf[ixobs, iyobs] .- Pf_obs
    (me==0) && @info "Adjoint solve"
    adjoint_solve!(me, logK, fwd_params, adj_params, loss_params)
    # evaluate gradient dJ_dK
    R̄qx .= .-Ψ_qx
    R̄qy .= .-Ψ_qy
    logK̄ .= 0.0
    @parallel ∇=(Rqx->R̄qx, Rqy->R̄qy, logK->logK̄) residual_fluxes!(Rqx, Rqy, qx, qy, Pf, logK, dx, dy)
    # Tikhonov regularisation (smoothing)
    if !isnothing(reg)
        (;nsm, Tmp) = reg
        Tmp .= logK̄; smooth!(logK̄, Tmp; nsm)
    end
    logK̄ .*= K # convert to dJ_dlogK by chain rule
    return
end

@views function main()
    # physics
    lx, ly   = 2.0, 1.0 # domain extend
    k0_μ     = 1.0      # background permeability / fluid viscosity
    kb_μ     = 1e-6     # barrier permeability / fluid viscosity
    Q_in     = 1.0      # injection flux
    b_w      = 0.02lx   # barrier width
    b_b      = 0.3ly    # barrier bottom location
    b_t      = 0.8ly    # barrier top location
    # observations
    xobs_rng = LinRange(-lx / 6, lx / 6, 8)
    yobs_rng = LinRange(0.25ly, 0.85ly , 8)
    # numerics
    ny       = 127 #255
    nx       = ceil(Int, (ny + 1) * lx / ly) - 1
    me,      = init_global_grid(nx, ny, 1; dimx=0, dimy=0, dimz=1)
    nxg, nyg = nx_g(), ny_g()
    cfl      = 1 / 2.1
    ϵtol     = 1e-6
    maxiter  = 30nxg
    ncheck   = 2nxg
    re       = 0.8π # fwd re
    st       = ceil(Int, nxg / 30)
    # preprocessing
    re_a     = 2re  # adjoint re
    dx, dy   = lx / nxg, ly / nyg
    xc, yc   = [x_g(ix, dx, zeros(nx, ny)) - lx / 2 + dx / 2 for ix=1:nx], [y_g(iy, dy, zeros(nx, ny)) + dy / 2 for iy=1:ny]
    vdτ      = cfl * min(dx, dy)
    ixobs    = filter(ix -> 1 <= ix <= nx, floor.(Int, (xobs_rng .- xc[1]) ./ dx) .+ 1)
    iyobs    = filter(iy -> 1 <= iy <= ny, floor.(Int, (yobs_rng .- yc[1]) ./ dy) .+ 1)
    # init
    Pf       = @zeros(nx, ny)
    RPf      = @zeros(nx, ny)
    qx       = @zeros(nx + 1, ny)
    Rqx      = @zeros(nx + 1, ny)
    qy       = @zeros(nx, ny + 1)
    Rqy      = @zeros(nx, ny + 1)
    Qf       = @zeros(nx, ny)
    K        = k0_μ .* @ones(nx, ny)
    logK     = @zeros(nx, ny)
    Tmp      = @zeros(nx, ny)
    # init adjoint storage
    Ψ_qx     = @zeros(nx + 1, ny)
    q̄x       = @zeros(nx + 1, ny)
    R̄qx      = @zeros(nx + 1, ny)
    Ψ_qy     = @zeros(nx, ny + 1)
    q̄y       = @zeros(nx, ny + 1)
    R̄qy      = @zeros(nx, ny + 1)
    Ψ_Pf     = @zeros(nx, ny)
    P̄f       = @zeros(nx, ny)
    R̄Pf      = @zeros(nx, ny)
    ∂J_∂Pf   = @zeros(nx, ny)
    # set low permeability barrier location
    K[-b_w .< xc .<= b_w, b_b .< yc .<= b_t] .= kb_μ
    logK .= log.(K)
    K_max = copy(K)
    # set wells location
    x_iw, x_ew, y_w = -lx / 2 + lx / 5, -lx / 2 + 4lx / 5, 0.45ly
    Qf[x_iw .< xc .< (x_iw + dx), y_w .< yc .< (y_w + dy)] .=  Q_in / dx / dy # injection
    Qf[x_ew .< xc .< (x_ew + dx), y_w .< yc .< (y_w + dy)] .= -Q_in / dx / dy # extraction    
    # init visu
    iters_evo = Float64[]; errs_evo = Float64[]; fig=[]; plt=[]
    qM, qx_c, qy_c, Pf_inn, K_inn = zeros(nx-2, ny-2), zeros(nx-2, ny-2), zeros(nx-2, ny-2), zeros(nx-2, ny-2), zeros(nx-2, ny-2)
    Pf_v, K_v, qM_v, qx_c_v, qy_c_v = zeros(nxg-2, nyg-2), ones(nxg-2, nyg-2), zeros(nxg-2, nyg-2), zeros(nxg-2, nyg-2), zeros(nxg-2, nyg-2)
    if me==0
        xci_v, yci_v = LinRange(-lx / 2 + 3dx / 2, lx / 2 - 3dx / 2, nxg - 2), LinRange(3dy / 2, ly - 3dy / 2, nyg - 2)
        fig = Figure(resolution=(2500, 1200), fontsize=32)
        ax = ( Pf  = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="Pf"),
               K   = Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="log10(K)"),
               qM  = Axis(fig[2, 1][1, 1]; aspect=DataAspect(), title="|q|"),
               err = Axis(fig[2, 2]; yscale=log10, title="Convergence", xlabel="# iter/nx", ylabel="error"), )
        plt = (fld = ( Pf   = heatmap!(ax.Pf, xci_v, yci_v, Pf_v; colormap=:turbo, colorrange=(-1,1)),
                       K    = heatmap!(ax.K , xci_v, yci_v, log10.(K_v); colormap=:turbo, colorrange=(-6,0)),
                       xobs = scatter!(ax.K , vec(Point2.(xobs_rng, yobs_rng')); color=:white),
                       qM   = heatmap!(ax.qM, xci_v, yci_v, qM_v; colormap=:turbo, colorrange=(0,30)),
                       ar   = arrows!(ax.Pf, xci_v[1:st:end], yci_v[1:st:end], qx_c_v[1:st:end, 1:st:end], qy_c_v[1:st:end, 1:st:end]; lengthscale=0.05, arrowsize=15), ),
               err = scatterlines!(ax.err, Point2.(iters_evo, errs_evo), linewidth=4), )
        Colorbar(fig[1, 1][1, 2], plt.fld.Pf)
        Colorbar(fig[1, 2][1, 2], plt.fld.K)
        Colorbar(fig[2, 1][1, 2], plt.fld.qM)
    end
    # action
    fwd_params = (
        fields      = (;Pf, qx, qy, Qf, RPf, Rqx, Rqy, K),
        scalars     = (;nxg, nyg, dx, dy),
        iter_params = (;cfl, re, vdτ, ly, ϵtol, maxiter, ncheck, K_max),
    )
    fwd_visu = (;qx_c, qy_c, qM, Pf_v, K_v, qM_v, qx_c_v, qy_c_v, Pf_inn, K_inn, fig, plt, st)
    (me==0) && @info "Synthetic solve"
    forward_solve!(me, logK, fwd_params...; visu=fwd_visu)
    # store true data
    Pf_obs = copy(Pf[ixobs, iyobs])
    adj_params = (
        fields  = (;P̄f, q̄x, q̄y, R̄Pf, R̄qx, R̄qy, Ψ_qx, Ψ_qy, Ψ_Pf),
        iter_params = (;cfl, re_a, vdτ, ly, ϵtol, maxiter, ncheck, K_max),
    )
    loss_params = (
        fields  = (;Pf_obs, ∂J_∂Pf),
        scalars = (;ixobs, iyobs),
    )
    reg = (;nsm=50, Tmp)
    # loss functions
    J(_logK) = loss(me, _logK, fwd_params, loss_params)
    ∇J!(_logK̄, _logK) = ∇loss!(me, _logK̄, _logK, fwd_params, adj_params, loss_params; reg)
    (me==0) && @info "Inversion for K"
    # initial guess
    K    .= k0_μ
    logK .= log.(K)
    dJ_dlogK = @zeros(nx, ny)
    # GD params
    ngd = 50
    Δγ  = 0.2
    cost_evo = Float64[]
    for igd in 1:ngd
        (me==0) && printstyled("> GD iter $igd \n"; bold=true, color=:green)
        # evaluate gradient of the cost function
        ∇J!(dJ_dlogK, logK)
        # update logK
        γ = Δγ / max_g(abs.(dJ_dlogK))
        @. logK -= γ * dJ_dlogK
        (me==0) && @printf "  min(K) = %1.2e \n" minimum(K)
        # loss
        push!(cost_evo, J(logK))
        (me==0) && @printf "  --> Loss J = %1.2e (γ = %1.2e)\n" last(cost_evo)/first(cost_evo) γ
        # visu
        qx_c .= Array(avx(qx))[2:end-1,2:end-1]; qy_c .= Array(avy(qy))[2:end-1,2:end-1]; qM .= sqrt.(qx_c.^2 .+ qy_c.^2)
        qx_c ./= qM; qy_c ./= qM
        Pf_inn .= Array(inn(Pf)); gather!(Pf_inn, Pf_v)
        K_inn  .= Array(inn(K));  gather!(K_inn, K_v)
        gather!(qM, qM_v)
        gather!(qx_c, qx_c_v)
        gather!(qy_c, qy_c_v)
        if me==0
            plt.fld.Pf[3] = Pf_v
            plt.fld.K[3]  = log10.(K_v)
            plt.fld.qM[3] = qM_v
            plt.fld.ar[3] = qx_c_v[1:st:end, 1:st:end]
            plt.fld.ar[4] = qy_c_v[1:st:end, 1:st:end]
            plt.err[1] = Point2.(1:igd, cost_evo ./ 0.999cost_evo[1])
            # display(fig)
            (igd % 5 == 0) && CairoMakie.save("out_inv_$igd.png", fig)
        end
    end

    finalize_global_grid(finalize_MPI=true)
    return
end

main()