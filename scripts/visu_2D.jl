using Printf
using CairoMakie

@views avx(A) = 0.5 .* (A[1:end-1, :] .+ A[2:end, :])
@views avy(A) = 0.5 .* (A[:, 1:end-1] .+ A[:, 2:end])
@views maxloc(A) = max.(A[2:end-1, 2:end-1], max.(max.(A[1:end-2, 2:end-1], A[3:end, 2:end-1]),
                                                  max.(A[2:end-1, 1:end-2], A[2:end-1, 3:end])))

@views function main()
    # physics
    lx, ly  = 2.0, 1.0 # domain extend
    k0_μ    = 1.0      # background permeability / fluid viscosity
    kb_μ    = 1e-6     # barrier permeability / fluid viscosity
    Q_in    = 1.0      # injection flux
    b_w     = 0.02lx   # barrier width
    b_b     = 0.3ly    # barrier bottom location
    b_t     = 0.8ly    # barrier top location
    # numerics
    ny      = 63
    nx      = ceil(Int, (ny + 1) * lx / ly) - 1
    st      = ceil(Int, nx / 30)
    # preprocessing
    dx, dy  = lx / nx, ly / ny
    xc, yc  = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx), LinRange(dy / 2, ly - dy / 2, ny)
    # init
    Pf      = zeros(nx, ny)
    qx      = zeros(nx + 1, ny)
    qy      = zeros(nx, ny + 1)
    Qf      = zeros(nx, ny)
    K       = k0_μ .* ones(nx, ny)
    # set low permeability barrier location
    K[ceil(Int, (lx/2-b_w)/dx):ceil(Int, (lx/2+b_w)/dx), ceil(Int, b_b/dy):ceil(Int, b_t/dy)] .= kb_μ
    # approximate diagonal (Jacobi) preconditioner
    K_max = copy(K); K_max[2:end-1, 2:end-1] .= maxloc(K); K_max[:, [1, end]] .= K_max[:, [2, end-1]]
    # set wells location
    x_iw, x_ew, z_w = ceil.(Int, (lx / 5 / dx, 4lx / 5 / dx, 0.45ly / dy))
    Qf[x_iw, z_w] =  Q_in / dx / dy # injection
    Qf[x_ew, z_w] = -Q_in / dx / dy # extraction
    # init visu
    iters_evo = [1,2,3]; errs_evo = [3,2,1]
    qM, qx_c, qy_c = zeros(nx, ny), zeros(nx, ny), zeros(nx, ny)
    fig = Figure(resolution=(2500, 1200), fontsize=32)
    ax = ( Pf  = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="Pf"),
           K   = Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="log10(K)"),
           qM  = Axis(fig[2, 1][1, 1]; aspect=DataAspect(), title="|q|"),
           err = Axis(fig[2, 2]; yscale=log10, title="Convergence", xlabel="# iter/nx", ylabel="error"), )
    plt = (fld = ( Pf = heatmap!(ax.Pf, xc, yc, Pf; colormap=:turbo, colorrange=(-1,1)),
                   K  = heatmap!(ax.K , xc, yc, log10.(K); colormap=:turbo, colorrange=(-6,0)),
                   qM = heatmap!(ax.qM, xc, yc, qM; colormap=:turbo, colorrange=(0,30)),
                   ar = arrows!(ax.Pf, xc[1:st:end], yc[1:st:end], qx_c[1:st:end, 1:st:end], qy_c[1:st:end, 1:st:end]; lengthscale=0.05, arrowsize=15), ),
           err = scatterlines!(ax.err, Point2.(iters_evo, errs_evo), linewidth=4), )
    Colorbar(fig[1, 1][1, 2], plt.fld.Pf)
    Colorbar(fig[1, 2][1, 2], plt.fld.K)
    Colorbar(fig[2, 1][1, 2], plt.fld.qM)
    Pf .+= Qf
    # visu
    qx_c .= avx(qx); qy_c .= avy(qy); qM .= sqrt.(qx_c.^2 .+ qy_c.^2)
    qx_c ./= qM; qy_c ./= qM
    plt.fld.Pf[3] = Pf
    plt.fld.K[3]  = log10.(K)
    plt.fld.qM[3] = qM
    plt.fld.ar[3] = qx_c[1:st:end, 1:st:end]
    plt.fld.ar[4] = qy_c[1:st:end, 1:st:end]
    plt.err[1] = Point2.(iters_evo, errs_evo)
    display(fig)
    return
end

main()