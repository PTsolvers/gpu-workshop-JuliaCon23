using CUDA
using BenchmarkTools

macro d2_xi(A) esc(:(($A[ix+2, iy+1] - $A[ix+1, iy+1]) - ($A[ix+1, iy+1] - $A[ix, iy+1]))) end
macro d2_yi(A) esc(:(($A[ix+1, iy+2] - $A[ix+1, iy+1]) - ($A[ix+1, iy+1] - $A[ix+1, iy]))) end
macro inn(A)  esc(:($A[ix+1, iy+1])) end

function diffusion_step!(C2, C, D, dt, _dx, _dy)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if (ix <= size(C, 1) - 2 && iy <= size(C, 2) - 2)
        @inbounds @inn(C2) = @inn(C) + dt * @inn(D) * (@d2_xi(C) * _dx * _dx + @d2_yi(C) * _dy * _dy)
    end
    return
end

function perftest()
    nx = ny = 512 * 64
    C  = CUDA.rand(Float64, nx, ny)
    D  = CUDA.rand(Float64, nx, ny)
    _dx = _dy = dt = rand()
    C2 = copy(C)
    nthreads = (16, 16)
    nblocks  = cld.((nx, ny), nthreads)
    t_it = @belapsed begin
        CUDA.@sync @cuda threads=$nthreads blocks=$nblocks diffusion_step!($C2, $C, $D, $dt, $_dx, $_dy)
    end
    T_eff = (2 * 1 + 1) / 1e9 * nx * ny * sizeof(Float64) / t_it
    println("T_eff = $(T_eff) GiB/s using CUDA.jl on a Nvidia Tesla A100 GPU")
    println("So that's cool. We are getting close to hardware limit, running at $(T_eff/1355*100), sigdigits=2) % of memory copy! 🚀")
    return
end

perftest()