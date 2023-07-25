using BenchmarkTools
using ParallelStencil
using ParallelStencil.FiniteDifferences2D

# @init_parallel_stencil(Threads, Float64, 2)
@init_parallel_stencil(CUDA, Float64, 2)

@parallel function diffusion_step!(C2, C, D, dt, _dx, _dy)
        @inn(C2) = @inn(C) + dt * @inn(D) * (@d2_xi(C) * _dx * _dx + @d2_yi(C) * _dy * _dy)
    return
end

function perftest()
    nx = ny = 512 * 64
    C  = @rand(nx, ny)
    D  = @rand(nx, ny)
    _dx = _dy = dt = rand()
    C2 = copy(C)
    t_it = @belapsed begin
        @parallel diffusion_step!($C2, $C, $D, $dt, $_dx, $_dy)
    end
    T_eff = (2 * 1 + 1) / 1e9 * nx * ny * sizeof(Float64) / t_it
    println("T_eff = $(T_eff) GiB/s using CUDA.jl on a Nvidia Tesla A100 GPU")
    println("So that's cool. We are getting close to hardware limit, running at $(round(T_eff/1355*100, sigdigits=2)) % of memory copy! 🚀")
    return
end

perftest()