using laGP
using Test

@testset "laGP.jl" begin
    include("test_gp.jl")
    include("test_gp_sep.jl")
    include("test_gpsep_reference.jl")
    include("test_wingwt_reference.jl")
    include("test_mle.jl")
    include("test_acquisition.jl")
    include("test_local_gp.jl")
end
