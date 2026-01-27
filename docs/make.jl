using Documenter
using laGP

makedocs(
    sitename = "laGP.jl",
    modules = [laGP],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://joshualeond.github.io/laGP.jl/stable/",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Theory" => "theory.md",
        "Examples" => [
            "Local Approximate GP Demo" => "examples/demo.md",
            "Motorcycle Crash Test" => "examples/motorcycle.md",
            "Posterior Sampling" => "examples/sinusoidal.md",
            "Wing Weight Surrogate" => "examples/surrogates.md",
            "Satellite Drag Modeling" => "examples/satellite.md",
        ],
        "API Reference" => "api.md",
    ],
    checkdocs = :exports,
    warnonly = [:missing_docs],
)

deploydocs(
    repo = "github.com/joshualeond/laGP.jl.git",
    devbranch = "main",
    push_preview = true,
)
