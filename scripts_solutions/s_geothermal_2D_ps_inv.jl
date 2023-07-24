using Printf, LinearAlgebra
using CairoMakie
using Enzyme
using Optim
using ParallelStencil
using ParallelStencil.FiniteDifferences2D

@init_parallel_stencil(Threads, Float64, 2)
# @init_parallel_stencil(CUDA, Float64, 2)

@views avx(A) = 0.5 .* (A[1:end-1, :] .+ A[2:end, :])
@views avy(A) = 0.5 .* (A[:, 1:end-1] .+ A[:, 2:end])
@views maxloc(A) = max.(A[2:end-1, 2:end-1], max.(max.(A[1:end-2, 2:end-1], A[3:end, 2:end-1]),
                                                  max.(A[2:end-1, 1:end-2], A[2:end-1, 3:end])))

@parallel function smooth_d!(A2, A)
        @inn(A2) = @inn(A) + 0.2 * (@d2_xi(A) + @d2_yi(A))
    return
end

function smooth!(A2, A; nsm=1)
    for _ ∈ 1:nsm
        @parallel smooth_d!(A2, A)
        A, A2 = A2, A
    end
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

@views function forward_solve!(logK, fields, scalars, iter_params; visu=nothing)
    (;Pf, qx, qy, Qf, RPf, Rqx, Rqy, K)               = fields
    (;nx, ny, dx, dy)                                 = scalars
    (;cfl, re, vdτ, ly, ϵtol, maxiter, ncheck, K_max) = iter_params
    isnothing(visu) || ((;qx_c, qy_c, qM, fig, plt, st) = visu)
    K .= exp.(logK)
    # approximate diagonal (Jacobi) preconditioner
    K_max .= K; K_max[2:end-1, 2:end-1] .= maxloc(K); K_max[:, [1, end]] .= K_max[:, [2, end-1]]
    # iterative loop
    iters_evo = Float64[]; errs_evo = Float64[]
    err = 2ϵtol; iter = 1
    while err >= ϵtol && iter <= maxiter
        @parallel residual_fluxes!(Rqx, Rqy, qx, qy, Pf, K, dx, dy)
        @parallel update_fluxes!(qx, qy, Rqx, Rqy, cfl, nx, ny, re)
        @parallel residual_pressure!(RPf, qx, qy, Qf, dx, dy)
        @parallel update_pressure!(Pf, RPf, K_max, vdτ, ly, re)
        if iter % ncheck == 0
            err = maximum(abs.(RPf))
            push!(iters_evo, iter/nx); push!(errs_evo, err)
            @printf("  #iter/nx=%.1f, max(err)=%1.3e\n", iter/nx, err)
            if !isnothing(visu)
                qx_c .= Array(avx(qx)); qy_c .= Array(avy(qy)); qM .= sqrt.(qx_c.^2 .+ qy_c.^2)
                qx_c ./= qM; qy_c ./= qM
                plt.fld.Pf[3] = Array(Pf)
                plt.fld.K[3]  = Array(log10.(K))
                plt.fld.qM[3] = qM
                plt.fld.ar[3] = qx_c[1:st:end, 1:st:end]
                plt.fld.ar[4] = qy_c[1:st:end, 1:st:end]
                plt.err[1] = Point2.(iters_evo, errs_evo)
                display(fig)
            end
        end
        iter += 1
    end
    return
end

@views function adjoint_solve!(logK, fwd_params, adj_params, loss_params)
    # unpack forward
    (;Pf, qx, qy, Qf, RPf, Rqx, Rqy, K) = fwd_params.fields
    (;nx, ny, dx, dy)  = fwd_params.scalars
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
        @parallel ∇=(Rqx->R̄qx, Rqy->R̄qy, qx->q̄x, qy->q̄y, Pf->P̄f) residual_fluxes!(Rqx, Rqy, qx, qy, Pf, K, dx, dy)
        P̄f[[1, end], :] .= 0.0; P̄f[:, [1, end]] .= 0.0
        @parallel update_pressure!(Ψ_Pf, P̄f, K_max, vdτ, ly, re_a)
        R̄Pf .= Ψ_Pf
        @parallel ∇=(RPf->R̄Pf, qx->q̄x, qy->q̄y) residual_pressure!(RPf, qx, qy, Qf, dx, dy)
        @parallel update_fluxes!(Ψ_qx, Ψ_qy, q̄x, q̄y, cfl, nx, ny, re_a)
        if iter % ncheck == 0
            err = maximum(abs.(P̄f))
            push!(iters_evo, iter/nx); push!(errs_evo, err)
            @printf("  #iter/nx=%.1f, max(err)=%1.6e\n", iter/nx, err)
        end
        iter += 1
    end
    return
end

@views function loss(logK, fwd_params, loss_params; kwargs...)
    (;Pf_obs)       = loss_params.fields
    (;ixobs, iyobs) = loss_params.scalars
    @info "Forward solve"
    forward_solve!(logK, fwd_params...; kwargs...)
    Pf = fwd_params.fields.Pf
    return 0.5*sum((Pf[ixobs, iyobs] .- Pf_obs).^2)
end

function ∇loss!(logK̄, logK, fwd_params, adj_params, loss_params; reg=nothing, kwargs...)
    # unpack
    (;R̄qx, R̄qy, Ψ_qx, Ψ_qy)    = adj_params.fields
    (;Pf, qx, qy, Rqx, Rqy, K) = fwd_params.fields
    (;dx, dy)                  = fwd_params.scalars
    (;Pf_obs, ∂J_∂Pf)          = loss_params.fields
    (;ixobs, iyobs)            = loss_params.scalars
    @info "Forward solve"
    forward_solve!(logK, fwd_params...; kwargs...)
    # set tangent
    ∂J_∂Pf[ixobs, iyobs] .= Pf[ixobs, iyobs] .- Pf_obs
    @info "Adjoint solve"
    adjoint_solve!(logK, fwd_params, adj_params, loss_params)
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
    ny       = 127
    nx       = ceil(Int, (ny + 1) * lx / ly) - 1
    cfl      = 1 / 2.1
    ϵtol     = 1e-6
    maxiter  = 30nx
    ncheck   = 2nx
    re       = 0.8π # fwd re
    st       = ceil(Int, nx / 30)
    # preprocessing
    re_a     = 2re  # adjoint re
    dx, dy   = lx / nx, ly / ny
    xc, yc   = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx), LinRange(dy / 2, ly - dy / 2, ny)
    vdτ      = cfl * min(dx, dy)
    ixobs    = floor.(Int, (xobs_rng .- xc[1]) ./ dx) .+ 1
    iyobs    = floor.(Int, (yobs_rng .- yc[1]) ./ dy) .+ 1
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
    K[ceil(Int, (lx/2-b_w)/dx):ceil(Int, (lx/2+b_w)/dx), ceil(Int, b_b/dy):ceil(Int, b_t/dy)] .= kb_μ
    logK .= log.(K)
    K_max = copy(K)
    # set wells location
    x_iw, x_ew, z_w = ceil.(Int, (lx / 5 / dx, 4lx / 5 / dx, 0.45ly / dy))
    Qf[x_iw:x_iw, z_w:z_w] .=  Q_in / dx / dy # injection
    Qf[x_ew:x_ew, z_w:z_w] .= -Q_in / dx / dy # extraction
    # init visu
    iters_evo = Float64[]; errs_evo = Float64[]
    qM, qx_c, qy_c = zeros(nx, ny), zeros(nx, ny), zeros(nx, ny)
    fig = Figure(resolution=(2500, 1200), fontsize=32)
    ax = ( Pf  = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="Pf"),
           K   = Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="log10(K)"),
           qM  = Axis(fig[2, 1][1, 1]; aspect=DataAspect(), title="|q|"),
           err = Axis(fig[2, 2]; yscale=log10, title="Convergence", xlabel="# iter/nx", ylabel="error"), )
    plt = (fld = ( Pf   = heatmap!(ax.Pf, xc, yc, Array(Pf); colormap=:turbo, colorrange=(-1,1)),
                   K    = heatmap!(ax.K , xc, yc, Array(log10.(K)); colormap=:turbo, colorrange=(-6,0)),
                   xobs = scatter!(ax.K , vec(Point2.(xobs_rng, yobs_rng')); color=:white),
                   qM   = heatmap!(ax.qM, xc, yc, qM; colormap=:turbo, colorrange=(0,30)),
                   ar   = arrows!(ax.Pf, xc[1:st:end], yc[1:st:end], qx_c[1:st:end, 1:st:end], qy_c[1:st:end, 1:st:end]; lengthscale=0.05, arrowsize=15), ),
           err = scatterlines!(ax.err, Point2.(iters_evo, errs_evo), linewidth=4), )
    Colorbar(fig[1, 1][1, 2], plt.fld.Pf)
    Colorbar(fig[1, 2][1, 2], plt.fld.K)
    Colorbar(fig[2, 1][1, 2], plt.fld.qM)
    # action
    fwd_params = (
        fields      = (;Pf, qx, qy, Qf, RPf, Rqx, Rqy, K),
        scalars     = (;nx, ny, dx, dy),
        iter_params = (;cfl, re, vdτ, ly, ϵtol, maxiter, ncheck, K_max),
    )
    fwd_visu = (;qx_c, qy_c, qM, fig, plt, st)
    @info "Synthetic solve"
    forward_solve!(logK, fwd_params...; visu=fwd_visu)
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
    J(_logK) = loss(_logK, fwd_params, loss_params)
    ∇J!(_logK̄, _logK) = ∇loss!(_logK̄, _logK, fwd_params, adj_params, loss_params; reg)
    @info "Inversion for K"
    # initial guess
    K    .= k0_μ
    logK .= log.(K)
    # Optim
    opt = Optim.Options(
        f_tol      = 1e-2,
        g_tol      = 1e-6,
        iterations = 20,
        store_trace=true, show_trace=true,
    )
    result = optimize(J, ∇J!, logK, LBFGS(), opt)
    K .= exp.(Optim.minimizer(result))
    @show result
    # visu
    errs_evo = Optim.f_trace(result)
    errs_evo ./= errs_evo[1]
    iters_evo = 1:length(errs_evo)
    qx_c .= Array(avx(qx)); qy_c .= Array(avy(qy)); qM .= sqrt.(qx_c.^2 .+ qy_c.^2)
    qx_c ./= qM; qy_c ./= qM
    plt.fld.Pf[3] = Array(Pf)
    plt.fld.K[3]  = Array(log10.(K))
    plt.fld.qM[3] = qM
    plt.fld.ar[3] = qx_c[1:st:end, 1:st:end]
    plt.fld.ar[4] = qy_c[1:st:end, 1:st:end]
    plt.err[1] = Point2.(iters_evo, errs_evo)
    display(fig)
    return
end

main()