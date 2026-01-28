# Chapter 2: Satellite Drag Coefficient Modeling
# Port of "Surrogates" Chapter 2.3 - GRACE Satellite Example
# Original R code by Robert Gramacy
# Julia port using laGP.jl
#
# This example demonstrates:
# 1. Downloading real GRACE satellite drag data from the TPM repository
# 2. Separable GP fitting with per-dimension lengthscales
# 3. Prediction and RMSPE calculation
# 4. Atmospheric mixture combination of species predictions
# 5. Main effects / sensitivity analysis

using laGP
using Downloads
using DelimitedFiles
using Random
using CairoMakie
using Statistics: mean

# Set random seed for reproducibility
Random.seed!(42)

# Output directory (same as this script)
const OUTPUT_DIR = @__DIR__

# ============================================================================
# PART 1: Data Download and Loading
# ============================================================================

println("="^70)
println("PART 1: Downloading GRACE Satellite Drag Data")
println("="^70)

const BASE_URL = "https://bitbucket.org/gramacylab/tpm/raw/master/data/GRACE"
const SPECIES = [:He, :O, :O2, :N, :N2, :H]

# Molecular masses for atmospheric mixture calculation (g/mol)
const MOLECULAR_MASS = Dict(
    :He => 4.003,
    :O  => 15.999,
    :O2 => 31.998,
    :N  => 14.007,
    :N2 => 28.014,
    :H  => 1.008
)

# Variable names for the 7 input dimensions
const VAR_NAMES = ["Umag", "Ts", "Ta", "theta", "phi", "alphan", "sigmat"]
const VAR_DESCRIPTIONS = [
    "Velocity magnitude (m/s)",
    "Surface temperature (K)",
    "Atmospheric temperature (K)",
    "Yaw angle (rad)",
    "Pitch angle (rad)",
    "Normal accommodation coeff",
    "Tangential accommodation coeff"
]

"""
    load_grace_data(species::Symbol, n::Int)

Download and parse GRACE satellite drag coefficient data for a given species.

# Arguments
- `species::Symbol`: One of :He, :O, :O2, :N, :N2, :H
- `n::Int`: Number of samples (1000 for training, 100 for test)

# Returns
- `Matrix{Float64}`: n x 9 matrix (7 inputs + index + Cd response)
"""
function load_grace_data(species::Symbol, n::Int)
    url = "$BASE_URL/CD_GRACE_$(n)_$(species).csv"
    println("  Downloading: $url")

    # Download to buffer and parse
    io = IOBuffer()
    Downloads.download(url, io)
    seekstart(io)
    content = String(take!(io))

    # Parse CSV with header
    lines = split(content, '\n')
    header = split(lines[1], ',')

    # Parse data rows
    data_rows = Float64[]
    for line in lines[2:end]
        if !isempty(strip(line))
            values = parse.(Float64, split(line, ','))
            append!(data_rows, values)
        end
    end

    # Reshape to matrix
    n_cols = length(header)
    n_rows = length(data_rows) ÷ n_cols
    data = reshape(data_rows, n_cols, n_rows)'

    return Matrix{Float64}(data)
end

# Download data for all species
println("\nDownloading training data (n=1000)...")
train_data = Dict{Symbol,Matrix{Float64}}()
for species in SPECIES
    train_data[species] = load_grace_data(species, 1000)
end

println("\nDownloading test data (n=100)...")
test_data = Dict{Symbol,Matrix{Float64}}()
for species in SPECIES
    test_data[species] = load_grace_data(species, 100)
end

# Check data structure
println("\nData structure for He:")
println("  Training: $(size(train_data[:He])) - columns are: $(join(VAR_NAMES, ", ")), Cd, Cd_old")
println("  Test: $(size(test_data[:He]))")

# The CSV columns are: Umag, Ts, Ta, theta, phi, alphan, sigmat, Cd, Cd_old
# We want columns 1:7 as inputs (X) and column 8 as response (Y)

# ============================================================================
# PART 2: Data Normalization
# ============================================================================

println("\n" * "="^70)
println("PART 2: Data Normalization to [0, 1]")
println("="^70)

"""
    normalize_data(train::Matrix, test::Matrix)

Normalize training and test data to [0, 1] range based on combined min/max.

# Arguments
- `train::Matrix`: Training data (n_train x 9, with index in column 1)
- `test::Matrix`: Test data (n_test x 9, with index in column 1)

# Returns
- `X_train, X_test, Y_train, Y_test, ranges`: Normalized inputs/outputs and ranges
"""
function normalize_data(train::Matrix{Float64}, test::Matrix{Float64})
    # Extract inputs (columns 1:7) and response (column 8)
    # CSV format: Umag, Ts, Ta, theta, phi, alphan, sigmat, Cd, Cd_old
    X_train_raw = train[:, 1:7]
    X_test_raw = test[:, 1:7]
    Y_train = train[:, 8]
    Y_test = test[:, 8]

    # Compute global ranges from combined train + test
    n_inputs = 7
    ranges = Vector{Tuple{Float64,Float64}}(undef, n_inputs)

    for j in 1:n_inputs
        combined = vcat(X_train_raw[:, j], X_test_raw[:, j])
        ranges[j] = (minimum(combined), maximum(combined))
    end

    # Normalize to [0, 1]
    X_train = similar(X_train_raw)
    X_test = similar(X_test_raw)

    for j in 1:n_inputs
        lo, hi = ranges[j]
        X_train[:, j] = (X_train_raw[:, j] .- lo) ./ (hi - lo)
        X_test[:, j] = (X_test_raw[:, j] .- lo) ./ (hi - lo)
    end

    return X_train, X_test, Y_train, Y_test, ranges
end

# Normalize data for each species
normalized_data = Dict{Symbol,NamedTuple}()
for species in SPECIES
    X_train, X_test, Y_train, Y_test, ranges = normalize_data(
        train_data[species], test_data[species]
    )
    normalized_data[species] = (
        X_train=X_train, X_test=X_test,
        Y_train=Y_train, Y_test=Y_test,
        ranges=ranges
    )
end

# Print normalization ranges for first species
println("\nNormalization ranges for He:")
for (j, name) in enumerate(VAR_NAMES)
    lo, hi = normalized_data[:He].ranges[j]
    println("  $name: [$lo, $hi]")
end

# ============================================================================
# PART 3: Fit Separable GP with MLE Optimization
# ============================================================================

println("\n" * "="^70)
println("PART 3: Fitting Separable GPs for Each Species")
println("="^70)

# Storage for GP models and results
gp_models = Dict{Symbol,GPsep{Float64}}()
mle_results = Dict{Symbol,NamedTuple}()

for species in SPECIES
    println("\n--- Fitting GP for $species ---")

    X = normalized_data[species].X_train
    Y = normalized_data[species].Y_train

    # Get hyperparameter ranges
    d_range_sep = darg_sep(X)
    g_range = garg(Y)

    println("  Nugget range: min=$(round(g_range.min, sigdigits=3)), max=$(round(g_range.max, sigdigits=3))")

    # Initial lengthscales from darg_sep
    d_start = [r.start for r in d_range_sep.ranges]
    d_ranges = [(r.min, r.max) for r in d_range_sep.ranges]

    # Create and fit GP
    gp = new_gp_sep(X, Y, d_start, g_range.start)
    println("  Initial log-likelihood: $(round(llik_gp_sep(gp), digits=2))")

    # MLE optimization
    result = jmle_gp_sep!(gp; drange=d_ranges, grange=(g_range.min, g_range.max))

    println("  After MLE: $(result.msg)")
    println("  Nugget g: $(round(gp.g, sigdigits=4))")
    println("  Final log-likelihood: $(round(llik_gp_sep(gp), digits=2))")

    # Store results
    gp_models[species] = gp
    mle_results[species] = result
end

# ============================================================================
# PART 4: Predictions and RMSPE Calculation
# ============================================================================

println("\n" * "="^70)
println("PART 4: Predictions and RMSPE Calculation")
println("="^70)

# Storage for predictions and RMSPE
predictions = Dict{Symbol,NamedTuple}()
rmspe_values = Dict{Symbol,Float64}()

for species in SPECIES
    X_test = normalized_data[species].X_test
    Y_test = normalized_data[species].Y_test
    gp = gp_models[species]

    # Make predictions
    pred = pred_gp_sep(gp, X_test; lite=true)

    # Calculate RMSPE (Root Mean Square Percentage Error)
    pct_errors = ((pred.mean .- Y_test) ./ Y_test) .* 100
    rmspe = sqrt(mean(pct_errors.^2))

    predictions[species] = (mean=pred.mean, s2=pred.s2, Y_test=Y_test)
    rmspe_values[species] = rmspe

    println("  $species RMSPE: $(round(rmspe, digits=3))%")
end

# ============================================================================
# PART 5: Atmospheric Mixture Combination
# ============================================================================

println("\n" * "="^70)
println("PART 5: Atmospheric Mixture Combination")
println("="^70)

# For the mixture calculation, we need mole fractions (chi) for each species
# These would typically come from atmospheric composition data
# For this demonstration, we use example mole fractions from thermospheric conditions

# Example mole fractions at ~400 km altitude (typical for GRACE)
# These are approximate values that vary with solar activity and location
const EXAMPLE_MOLE_FRACTIONS = Dict(
    :O  => 0.70,   # Atomic oxygen dominates at this altitude
    :N2 => 0.15,   # Molecular nitrogen
    :He => 0.08,   # Helium
    :O2 => 0.04,   # Molecular oxygen
    :N  => 0.02,   # Atomic nitrogen
    :H  => 0.01    # Hydrogen
)

"""
    compute_mixture_cd(predictions::Dict, mole_fractions::Dict, molecular_mass::Dict)

Compute mixture drag coefficient using mass-weighted average:
    Cd_mix = sum(Cd_j * chi_j * m_j) / sum(chi_j * m_j)

Where chi_j = mole fraction, m_j = molecular mass for species j.
"""
function compute_mixture_cd(
    predictions::Dict{Symbol,<:NamedTuple},
    mole_fractions::Dict{Symbol,Float64},
    molecular_mass::Dict{Symbol,Float64}
)
    # Get test set size from first species
    n_test = length(first(values(predictions)).mean)

    # Initialize mixture predictions
    numerator = zeros(Float64, n_test)
    denominator = zeros(Float64, n_test)
    true_numerator = zeros(Float64, n_test)

    for species in SPECIES
        chi = mole_fractions[species]
        m = molecular_mass[species]
        weight = chi * m

        numerator .+= predictions[species].mean .* weight
        denominator .+= weight
        true_numerator .+= predictions[species].Y_test .* weight
    end

    Cd_mix_pred = numerator ./ denominator
    Cd_mix_true = true_numerator ./ denominator

    return Cd_mix_pred, Cd_mix_true
end

# Compute mixture predictions
Cd_mix_pred, Cd_mix_true = compute_mixture_cd(
    predictions, EXAMPLE_MOLE_FRACTIONS, MOLECULAR_MASS
)

# Calculate mixture RMSPE
pct_errors_mix = ((Cd_mix_pred .- Cd_mix_true) ./ Cd_mix_true) .* 100
rmspe_mixture = sqrt(mean(pct_errors_mix.^2))

println("\nMixture RMSPE: $(round(rmspe_mixture, digits=3))%")
println("(Using example mole fractions for ~400 km altitude)")

# ============================================================================
# PART 6: Visualizations
# ============================================================================

println("\n" * "="^70)
println("PART 6: Creating Visualizations")
println("="^70)

# --- Figure 1: Lengthscales by Species ---
println("\nCreating lengthscale comparison plot...")

fig_ls = Figure(size=(900, 500))
ax_ls = Axis(fig_ls[1, 1],
    xlabel="Input Variable",
    ylabel="Lengthscale (log scale)",
    title="Per-Dimension Lengthscales by Species",
    xticks=(1:7, VAR_NAMES),
    xticklabelrotation=pi/4,
    yscale=log10
)

colors = [:blue, :red, :green, :orange, :purple, :brown]
offsets = range(-0.3, 0.3, length=6)

for (i, species) in enumerate(SPECIES)
    d_values = gp_models[species].d
    x_positions = (1:7) .+ offsets[i]
    scatter!(ax_ls, x_positions, d_values, color=colors[i], markersize=12, label=string(species))
end

Legend(fig_ls[1, 2], ax_ls, nbanks=1)
save(joinpath(OUTPUT_DIR, "chap2_lengthscales.png"), fig_ls)
println("Saved: chap2_lengthscales.png")

# --- Figure 2: Parity Plot for Helium ---
println("\nCreating parity plot for He...")

fig_parity = Figure(size=(500, 450))
ax_par = Axis(fig_parity[1, 1],
    xlabel="True Cd",
    ylabel="Predicted Cd",
    title="He: Predicted vs True (RMSPE=$(round(rmspe_values[:He], digits=2))%)",
    aspect=DataAspect()
)

Y_test_He = predictions[:He].Y_test
pred_He = predictions[:He].mean

# Get axis limits
all_vals = vcat(Y_test_He, pred_He)
lims = (minimum(all_vals) - 0.1, maximum(all_vals) + 0.1)

# 1:1 line
lines!(ax_par, [lims[1], lims[2]], [lims[1], lims[2]], color=:red, linewidth=2, linestyle=:dash)

# Scatter points
scatter!(ax_par, Y_test_He, pred_He, color=:blue, markersize=8, alpha=0.7)

xlims!(ax_par, lims...)
ylims!(ax_par, lims...)

save(joinpath(OUTPUT_DIR, "chap2_parity_He.png"), fig_parity)
println("Saved: chap2_parity_He.png")

# --- Figure 3: RMSPE by Species ---
println("\nCreating RMSPE comparison plot...")

fig_rmspe = Figure(size=(600, 400))
ax_rmspe = Axis(fig_rmspe[1, 1],
    xlabel="Species",
    ylabel="RMSPE (%)",
    title="Prediction Accuracy by Species",
    xticks=(1:6, string.(SPECIES))
)

rmspe_vec = [rmspe_values[s] for s in SPECIES]
barplot!(ax_rmspe, 1:6, rmspe_vec, color=colors)

# Add mixture RMSPE as horizontal line
hlines!(ax_rmspe, [rmspe_mixture], color=:black, linewidth=2, linestyle=:dash,
    label="Mixture")

save(joinpath(OUTPUT_DIR, "chap2_species_rmspe.png"), fig_rmspe)
println("Saved: chap2_species_rmspe.png")

# --- Figure 4: Main Effects Analysis ---
println("\nCreating main effects plot...")

# Use He as representative species for main effects
gp_he = gp_models[:He]
baseline = fill(0.5, 7)  # Center of normalized input space

n_me = 100
x_me = range(0.0, 1.0, length=n_me)

# Compute main effects for each input
me = Matrix{Float64}(undef, n_me, 7)

for j in 1:7
    # Create prediction matrix with baseline values
    XX_me = repeat(baseline', n_me, 1)
    XX_me[:, j] = collect(x_me)

    # Make predictions
    pred_me = pred_gp_sep(gp_he, XX_me; lite=true)
    me[:, j] = pred_me.mean
end

fig_me = Figure(size=(800, 500))
ax_me = Axis(fig_me[1, 1],
    xlabel="Normalized Input [0, 1]",
    ylabel="Cd (He)",
    title="Main Effects Analysis (He)"
)

me_colors = [:blue, :red, :green, :orange, :purple, :brown, :gray]

for j in 1:7
    lines!(ax_me, collect(x_me), me[:, j], color=me_colors[j], linewidth=2, label=VAR_NAMES[j])
end

Legend(fig_me[1, 2], ax_me, nbanks=1)
save(joinpath(OUTPUT_DIR, "chap2_main_effects.png"), fig_me)
println("Saved: chap2_main_effects.png")

# --- Figure 5: Mixture Parity Plot ---
println("\nCreating mixture parity plot...")

fig_mix = Figure(size=(500, 450))
ax_mix = Axis(fig_mix[1, 1],
    xlabel="True Cd (mixture)",
    ylabel="Predicted Cd (mixture)",
    title="Atmospheric Mixture: Predicted vs True (RMSPE=$(round(rmspe_mixture, digits=2))%)",
    aspect=DataAspect()
)

# Get axis limits
mix_vals = vcat(Cd_mix_true, Cd_mix_pred)
mix_lims = (minimum(mix_vals) - 0.05, maximum(mix_vals) + 0.05)

# 1:1 line
lines!(ax_mix, [mix_lims[1], mix_lims[2]], [mix_lims[1], mix_lims[2]],
    color=:red, linewidth=2, linestyle=:dash)

# Scatter points
scatter!(ax_mix, Cd_mix_true, Cd_mix_pred, color=:darkgreen, markersize=8, alpha=0.7)

xlims!(ax_mix, mix_lims...)
ylims!(ax_mix, mix_lims...)

save(joinpath(OUTPUT_DIR, "chap2_parity_mixture.png"), fig_mix)
println("Saved: chap2_parity_mixture.png")

# ============================================================================
# PART 7: Summary and Metrics
# ============================================================================

println("\n" * "="^70)
println("Chapter 2 Satellite Drag Example Complete!")
println("="^70)

println("\nGenerated files:")
println("  - chap2_lengthscales.png: Per-dimension lengthscales by species")
println("  - chap2_parity_He.png: Predicted vs true for Helium")
println("  - chap2_species_rmspe.png: RMSPE comparison across species")
println("  - chap2_main_effects.png: Main effects/sensitivity for He")
println("  - chap2_parity_mixture.png: Atmospheric mixture predictions")

println("\n--- RMSPE Summary ---")
for species in SPECIES
    println("  $species: $(round(rmspe_values[species], digits=3))%")
end
println("  Mixture: $(round(rmspe_mixture, digits=3))%")

println("\n--- Lengthscale Summary (by species) ---")
println("  " * rpad("Species", 8) * join([rpad(v, 10) for v in VAR_NAMES]))
for species in SPECIES
    d = gp_models[species].d
    vals = join([rpad(round(d[j], sigdigits=3), 10) for j in 1:7])
    println("  " * rpad(string(species), 8) * vals)
end

println("\n--- Key Findings ---")
# Find most and least important inputs (across all species)
avg_d = zeros(7)
for species in SPECIES
    avg_d .+= gp_models[species].d
end
avg_d ./= length(SPECIES)

# Smaller lengthscale = more important (steeper changes)
sorted_idx = sortperm(avg_d)
println("  Most influential inputs (smallest lengthscales):")
for i in 1:3
    j = sorted_idx[i]
    println("    $(VAR_NAMES[j]): avg lengthscale = $(round(avg_d[j], sigdigits=3))")
end

println("  Least influential inputs (largest lengthscales):")
for i in 5:7
    j = sorted_idx[i]
    println("    $(VAR_NAMES[j]): avg lengthscale = $(round(avg_d[j], sigdigits=3))")
end

println("\n  The mixture RMSPE ($(round(rmspe_mixture, digits=2))%) is typically lower than")
println("  individual species RMSPE due to averaging effects.")
