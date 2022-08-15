### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 6f66fc66-4066-490e-abd7-8ebb2a605bdb
begin
	# enable Table of Contents:
	using PlutoUI
	PlutoUI.TableOfContents(;depth=5)
end

# ╔═╡ 85be4fe0-8dc3-43af-83a4-48bfc0c24628
using CairoMakie, LaTeXStrings

# ╔═╡ 7d7ec467-3c22-4c8a-9ea1-a10e1ee0179b
using Logging

# ╔═╡ 24a47883-c281-4aa1-859f-714a3788a2d3
using Parameters: @unpack

# ╔═╡ e51f949d-6483-45aa-a18a-95d8245b63aa
using Dictionaries# for sorted Dictionaries

# ╔═╡ 89b9850a-99db-44cd-9fa5-d44b4bcdde89
using Parameters: @with_kw 	# convenient type definitions with defaults

# ╔═╡ badb3fe0-9739-4412-a3f1-e1a84d0d5ef4
using JuMP

# ╔═╡ 1b791760-5241-463e-94d1-4ee50bd7e4dc
using OSQP

# ╔═╡ d4a90042-f522-42ff-8260-7965591278c0
md"# Preparations"

# ╔═╡ 2995a784-a402-4c08-9bed-bd6c7d5fdfae
import LinearAlgebra: norm, qr, givens, Hermitian, inv, cholesky

# ╔═╡ b80eec63-10aa-4e77-a657-e0e12239dc15
import LinearAlgebra.I as eye

# ╔═╡ 52499a91-400b-4371-a7a1-d6d924579c48
import ForwardDiff as AD

# ╔═╡ 3c5381fb-0944-4b92-9837-5dddfe4bce11
import FiniteDiff as FD

# ╔═╡ a40db56a-8466-4aa1-95d5-7560e663098a
begin
	BLUE = Makie.RGBf(0, 32/255, 91/255)
	GRAY = Makie.RGBf(85/255, 85/255, 85/255)
	LIGHTGRAY = Makie.RGBf(199/255, 201/255, 199/255)
	RED = Makie.RGBf(215/255, 51/255, 103/255)
	GREEN = Makie.RGBf(164/255, 196/255, 36/255)
	CYAN = Makie.RGBf(24/255, 176/255, 226/255)
	ORANGE = Makie.RGBf(242/255, 149/255, 18/255)
	CASSIS = Makie.RGBf(169/255, 57/255, 131/255)
	LIGHTBLUE = Makie.RGBf(0, 127/255, 185/255)
	nothing
end

# ╔═╡ 68ada2bc-17e0-11ed-3f7b-414dc56ffd57
md"# Multiobjective Trust Region Filter Method for Nonlinear Constraints Using Inexact Gradients"

# ╔═╡ 8c701b7c-65e4-4dd2-9a79-b85b8f7aeae0
md"## About this Notebook"

# ╔═╡ 5280e000-9bae-48b6-a2c6-d73a720a20b9
md"This notebook provides a simple implementation of the algorithm described in our article [^1].
It is aimed at multiobjective problems of the form
```math
	\begin{aligned}
		&\min_{x \in ℝ^n}
		\begin{bmatrix}f_1(x)\\ \vdots \\f_K(x)\end{bmatrix},
		&\text{s.t.}\\ 
		&h_1(x) = 0, …, h_M(x) = 0, \\
		&g_1(x) \le 0, …, g_P(x) \le 0,
	\end{aligned}
```
where the functions can be nonlinear but must be differentiable.
However, the derivatives are not needed explicitly as we can build **fully-linear** models instead and use their gradients for a descent step calculation.
"

# ╔═╡ 1d6c67d5-2b31-4159-8b58-3bb5dc916f8b
md"""
!!! note
    The code provided here is kept as simple as possible. In contrast to the packages [Morbit](https://github.com/manuelbb-upb/Morbit.jl) or its upcoming successor [Compromise](https://github.com/manuelbb-upb/Compromise.jl), many features and checks are missing.
"""

# ╔═╡ 4cfe8b89-0693-4e58-aec0-8612d6d1e1f0
md"## Code"

# ╔═╡ e460c56c-879f-450c-a29a-b7eb8fbf04b7
md"""### Problem Setup
For this notebook, we use a very simple problem structure:
Minimization objectives, as well as nonlinear constraint functions must be provided as ``n``-variate real-valued `Function`s.
If they are meant to be modelled with gradient information, we use automatic differentiaton to obtain it.
We additionally allow the specification of linear constraints (which are passed to the inner solver directly) in the form ``A_i x \le b_i`` and ``A_e x = b_e`` and ``lb \le x \le ub``.
"""

# ╔═╡ b7510970-17e0-425b-b908-133015431191
@with_kw struct MOP
	num_vars :: Int
	lb :: Vector{Float64} = fill(-Inf, num_vars)
	ub :: Vector{Float64} = fill(Inf, num_vars)
	objectives :: Vector{Function} = []
	nl_eq_constraints :: Vector{Function} = []
	nl_ineq_constraints :: Vector{Function} = []
	A_eq :: Matrix{Float64} = Matrix{Float64}(undef, 0, num_vars)
	b_eq :: Vector{Float64} = Vector{Float64}(undef, 0)
	A_ineq :: Matrix{Float64} = Matrix{Float64}(undef, 0, num_vars)
	b_ineq :: Vector{Float64} = Vector{Float64}(undef, 0)

	@assert length(lb) == length(ub) == num_vars
	@assert size(A_eq, 2) == num_vars
	@assert size(A_ineq, 2) == num_vars
	@assert size(A_eq, 1) == length(b_eq)
	@assert size(A_ineq, 1) == length(b_ineq)
end

# ╔═╡ 26a945e3-eb9f-406c-aa0a-6c5cfd9181b8
Base.broadcastable(mop::MOP) = Ref(mop)	# pass `mop` down in broadcasted function calls

# ╔═╡ f219963d-2978-4ae2-acc9-2425d7f8ecec
md"### Problem Evaluation"

# ╔═╡ 5e7ec552-d98f-40fe-9976-618e9b1706dc
md"It is now easy to define some convenience functions for an `MOP` object:"

# ╔═╡ bf20e460-e5f4-4925-8e37-8aed02abdaa3
begin
	# Reading properties of an MOP:
	num_vars(mop) = mop.num_vars
	num_objectives(mop) = length(mop.objectives)
	num_nl_eq_constraints(mop) = length(mop.nl_eq_constraints)
	num_nl_ineq_constraints(mop) = length(mop.nl_ineq_constraints)
	num_lin_eq_constraints(mop) = size(mop.A_eq, 1)
	num_lin_ineq_constraints(mop) = size(mop.A_ineq, 1)
end

# ╔═╡ 8cf1aac5-92f9-42e3-8090-c01d85b72398
md"Evaluation is also straightforward. We rewrite the linear constraints to conform to the form ``Ax - b \leqq 0``."

# ╔═╡ 0513e905-f174-40f7-a667-5ba155dccc5a
begin
	# Evaluation of an MOP at vector `x`
	function eval_objectives(mop, x)
		return reduce(vcat, func(x) for func in mop.objectives; init = Float64[])
	end
	function eval_nl_eq_constraints(mop, x)
		return reduce(vcat, func(x) for func in mop.nl_eq_constraints; init = Float64[])
	end
	function eval_nl_ineq_constraints(mop, x)
		return reduce(vcat, func(x) for func in mop.nl_ineq_constraints; init = Float64[])
	end
	function eval_lin_eq_constraints(mop, x)
		return mop.A_eq * x - mop.b_eq
	end
	function eval_lin_ineq_constraints(mop, x)
		return mop.A_ineq * x - mop.b_ineq
	end

	function eval_all(mop, x)
		return (
			eval_objectives(mop, x),
			eval_nl_eq_constraints(mop, x),
			eval_nl_ineq_constraints(mop, x),
			eval_lin_eq_constraints(mop, x),
			eval_lin_ineq_constraints(mop, x)
		)				
	end
end

# ╔═╡ e2a1619b-9dfc-4f92-bf04-4de5a660506b
md"In our article we use the maximum constraint violation ``θ``:"

# ╔═╡ 5b963584-3568-4710-ab13-2136d8a768fb
begin
	function max_constraint_violation(nl_eq_vals, lin_eq_vals, nl_ineq_vals, lin_ineq_vals)
		return max(
			maximum(abs.(nl_eq_vals); init = 0.0),
			maximum(abs.(lin_eq_vals); init = 0.0),
			maximum(nl_ineq_vals; init = 0.0),
			maximum(lin_ineq_vals; init = 0.0)
		)
	end

	function max_constraint_violation(mop, x)
		return max_constraint_violation(
			eval_nl_eq_constraints(mop, x),
			eval_lin_eq_constraints(mop, x),
			eval_nl_ineq_constraints(mop,x),
			eval_lin_ineq_constraints(mop,x)
		)
	end
end

# ╔═╡ 541008ba-7cf5-43ac-b98f-e4e4233e936f
md"""
#### Result Type

To make our work a little bit more convenient, we store the iterates as `Result`s.
We can generate these automatically from a site `x`.
"""

# ╔═╡ e401ef02-6b83-4231-ba5e-50d59b1aad0d
@with_kw struct Result
	"Evaluation input vector."
	x :: Vector{Float64} = []
	"Objective value vector."
	fx :: Vector{Float64} = []
	"Nonlinear equality constraint value vector."
	hx :: Vector{Float64} = []
	"Nonlinear inequality constraint value vector."
	gx :: Vector{Float64} = []
	"Linear equality constraint residual vector."
	res_eq :: Vector{Float64} = []
	"Linear inequality constraint residual vector."
	res_ineq :: Vector{Float64} = []
	"Maximum constraint violation"
	θ :: Float64 = max_constraint_violation(hx, res_eq, gx, res_ineq)
end

# ╔═╡ ae98e9e4-c4a2-4b21-8755-cea1ce0d28cf
function make_result(x, mop)
	fx, hx, gx, res_eq, res_ineq = eval_all(mop, x)
	## Iteration dependent variables:
	θ = max_constraint_violation(hx, res_eq, gx, res_ineq)	
	return Result(;
		x = copy(x),
		fx, hx, gx, res_eq, res_ineq, θ
	)
end

# ╔═╡ 5e4824ac-e1b6-4d05-92f7-a116e537d111
md"### Model Construction"

# ╔═╡ 5c18ffe8-a6ae-4031-9c46-05006d6dddf0
md"""
Surrogate models are used to obtain an approximate descent direction.
They should be MiMo functions for the seperate function classes (objectives, inequality constraints & equality constraints).
Some model types are only suited for scalar-valued approximation.
We then model each of the functions in the corresponding array of an `MOP` and concatenate the results.
Some other model types (Radial Basis Functions (RBFs) for example) profit from modelling vector-valued functions directly. We concatenate the `MOP` functions *prior* to model construction.

We thus delegates work according to the requested model type which is implied by some `ConfigType <: ModelConfig`.
"""

# ╔═╡ 8508fdf0-df11-4cec-bda6-6e7e19249dd9
abstract type ModelConfig end

# ╔═╡ 95493635-10a8-49a9-96e4-529e09999837
md"For each possible model type we then have to implement 
`build_models(cfg::ConfigType, mop, database, iterate, Δ)`."

# ╔═╡ fcc68c1b-b0e9-41cd-845a-fa9d3fde33af
md"The `database` is just a vector of prior `Result`s. `iterate` is also a `Result`."

# ╔═╡ 3b4fe747-64b0-4b0a-9672-3f19724a1324
function build_models(cfg::T, mop, algo_config, database, iterate, Δ) where T
	error("No model construction algorithm is implemented for type $(T).")
end

# ╔═╡ 15dc4fb7-622a-41b3-a44e-d40c8837abfe
"Return `true` if the models should be updated whenever the trust region radius changes."
depends_on_radius(cfg) = false

# ╔═╡ 8fb3dd12-01ba-47ec-b65a-6aca6bbd9d20
md"#### No Models / Exact Evaluation"

# ╔═╡ 87533b4e-7eed-4fa7-b9e6-3e2aa7bffd0d
struct ExactConfig <: ModelConfig end

# ╔═╡ ba73a960-2cfd-49b6-ac7c-fb40eb678652
md"To use no models at all, we simply have to vertically stack the scalar evaluations."

# ╔═╡ 1d365a8b-89f2-48d9-a214-c7ea416dfa98
function build_models(cfg::ExactConfig, mop, algo_config, database, iterate, Δ) where T
	objf_v = x -> reduce( vcat, func(x) for func = mop.objectives; init = Float64[])
	ineq_v = x -> reduce( vcat, func(x) for func = mop.nl_ineq_constraints; init = Float64[])
	eq_v = x -> reduce( vcat, func(x) for func = mop.nl_eq_constraints; init = Float64[])
	return objf_v, ineq_v, eq_v
end

# ╔═╡ 0c7bffec-3a02-4009-9e24-34113c4d9910
md"#### Taylor Polynomial Models"

# ╔═╡ fc089ee6-c8e5-494e-a9ce-d7ef8eed0c80
md"Taylor Polynomial Models are possibly the easiest to built.
They require derivative information which we obtain via automatic differentiation (`AD`) or finite differences (`FD`)."

# ╔═╡ 7f5aa652-c938-4a5b-91e1-2d2abac84ffe
md"We allow models of degree 1 or degree 2:"

# ╔═╡ d37ca69d-8198-46e0-b63b-dc324315e27e
@with_kw struct TaylorConfig <: ModelConfig
	deg :: Int = 2
	diff :: Symbol = :AD
	@assert 1 <= deg <= 2
end	

# ╔═╡ 4aa512b9-4597-4087-a20b-5a0f39b04f70
md" A model of degree 1 has the form 
```math
f(x_0) + ∇f(x_0)^T (x - x_0)
```
whilst a second degree model is
```math
f(x_0) + ∇f(x_0)^T (x - x_0) + \frac{1}{2} (x-x_0)^T Hf(x_0) (x-x_0).
```
"

# ╔═╡ f5c076bd-62d6-4daa-af94-0501e615ad8f
function make_taylor_model(::Val{:AD}, func, x0, fx0, deg)
	dfx0 = AD.gradient(func, x0)
	if deg == 1
		return x -> fx0 + (x .- x0)'dfx0
	else
		Hx0 = AD.hessian(func, x0)
		return function (x)
			h =(x .- x0 ) 
			return fx0 + dfx0'h + 0.5*h'Hx0*h 
		end
	end
end

# ╔═╡ 86480276-b64f-4706-a011-120669514e89
function make_taylor_model(::Val{:FD}, func, x0, fx0, deg)
	dfx0 = FD.finite_difference_gradient(func, x0)
	if deg == 1
		return x -> fx0 + (x .- x0)'dfx0
	else
		Hx0 = FD.finite_difference_hessian(func, x0)
		return function (x)
			h =(x .- x0 ) 
			return fx0 + dfx0'h + 0.5*h'Hx0*h 
		end
	end
end

# ╔═╡ 08cf28fb-e332-45d0-915a-c07a7488da73
function build_models(cfg::TaylorConfig, mop, algo_config, database, iterate, Δ)
	x = iterate.x

	# build scalar valued models for all functions in `mop`:
	diff = Val(cfg.diff)
	objective_models = [make_taylor_model(diff, func, x, iterate.fx[i], cfg.deg) for (i,func) in enumerate(mop.objectives)]
	nl_eq_constraint_models = [make_taylor_model(diff, func, x, iterate.hx[i], cfg.deg) for (i,func) in enumerate(mop.nl_eq_constraints)]
	nl_ineq_constraint_models = [make_taylor_model(diff, func, x, iterate.gx[i], cfg.deg) for (i,func) in enumerate(mop.nl_ineq_constraints)]

	# now stack them for vector-valued functions
	# (I know that this is not optimal. E.g., (x-x0) is calculated multiple times,
	# But for this demonstration we don't care much about efficiency.)
	objective_vec_mod = x -> reduce(vcat, mod(x) for mod in objective_models; init = Float64[])
	nl_eq_constraint_vec_mod = x -> reduce(vcat, mod(x) for mod in nl_eq_constraint_models; init = Float64[])
	nl_ineq_constraint_vec_mod = x -> reduce(vcat, mod(x) for mod in nl_ineq_constraint_models; init = Float64[])
	
	return (objective_vec_mod, nl_eq_constraint_vec_mod, nl_ineq_constraint_vec_mod)
end

# ╔═╡ deb16981-33a8-4f8b-ab0d-b4ac745b056c
md"#### Radial Basis Function Models"

# ╔═╡ baad6c8b-c181-42d9-892e-739bf23624ad
import RadialBasisFunctionModels as RBF

# ╔═╡ 065a78a6-6b12-408e-acc9-4386d82bbe59
@with_kw struct RbfConfig <: ModelConfig
	poly_deg :: Int = 1
	kernel_func :: Function = Δ -> RBF.Cubic()
	delta_factor :: Float64 = 2.0
	delta_max_factor :: Float64 = 2.0
	piv_factor :: Float64 = 1 / (2*delta_factor)
	piv_cholesky :: Float64 = 1e-7
	max_points :: Int = -1
end

# ╔═╡ 044ed139-5f2c-4720-801a-67726904267c
depends_on_radius( cfg :: RbfConfig ) = true

# ╔═╡ 01526a4d-d916-49ca-a993-f9c34e28454d
"""
    nullify_last_row( R )
Returns matrices ``B`` and ``G`` with ``G⋅R = B``.
``G`` applies givens rotations to make ``R`` 
upper triangular.
``R`` is assumed to be an upper triangular matrix 
augmented by a single row.
"""
function nullify_last_row( R )
	m, n = size( R )
	G = Matrix(eye(m)) # orthogonal transformation matrix
	for j = 1 : min(m-1, n)
		## in each column, take the diagonal as pivot to turn last elem to zero
		g = givens( R[j,j], R[m, j], j, m )[1]
		R = g*R
		G = g*G
	end
	return R, G
end

# ╔═╡ d2e8b51e-9951-4207-8fe2-6d4b53dd416a
function _orthonormal_complement_matrix( Y, p = Inf )
    Q, _ = qr(Y)
    Z = Q[:, size(Y,2) + 1 : end]
    if size(Z,2) > 0
        Z ./= norm.( eachcol(Z), p )'
    end
    return Z
end 

# ╔═╡ 03521fe7-5c6b-4d3a-88fc-12f411009e22
function make_rbf_model( fn, iterate, centers, database, old_site_indices, new_db_indices, φ, cfg )
	labels = Vector{Vector{Float64}}([
		[getfield(iterate, fn),]; 
		[getfield(res, fn) for res in database[old_site_indices]];
		[getfield(res, fn) for res in database[new_db_indices]];
	])
	return RBF.RBFInterpolationModel(centers, labels, φ, cfg.poly_deg)
end

# ╔═╡ c1197240-d173-4bf1-a7a4-edc10cc49684
md"### Filter"

# ╔═╡ f5fc49fc-3ed3-4fb2-a677-56f5dd1ed808
Base.@kwdef struct Filter
	entries :: Vector{Tuple{Float64, Float64}} = [] 	# array of (θ, fx) tuples
	gamma :: Float64 = 0.1               		# envelope factor
end

# ╔═╡ 24bcd634-a199-46d8-a3a1-edfb18d71a3e
function filter_plot!(ax, filter)
	tups = filter.entries
	if isempty(tups)
		return nothing
	end

	# determine axis limits
	γ = filter.gamma
	θ = first.(tups)
	Φ = last.(tups)
	_θ_min, θ_max = extrema(θ)
	_Φ_min, Φ_max = extrema(Φ)
	offset = γ * _θ_min
	θ_min = _θ_min - offset
	Φ_min = _Φ_min - offset
	_θ_w = (θ_max - θ_min)
	_Φ_w = (Φ_max - Φ_min)
	θ_w = _θ_w > 0 ? _θ_w : 0.1
	Φ_w = _Φ_w > 0 ? _Φ_w : 0.1
	θ_lb = θ_min - 0.1 * θ_w
	θ_ub = θ_max + 0.1 * θ_w
	Φ_lb = Φ_min - 0.1 * Φ_w
	Φ_ub = Φ_max + 0.1 * Φ_w
	xlims!(ax, θ_lb, θ_ub )
	ylims!(ax, Φ_lb, Φ_ub )
	
	ax.xlabel = "θ"
	ax.ylabel = "Φ"
	
	for (θj, Φj) in tups
		offset_j = γ * θj
		_θj = θj - offset_j
		_Φj = Φj - offset_j
		verts = [
			_θj _Φj;
			θ_ub _Φj;
			θ_ub Φ_ub;
			_θj Φ_ub;
		]
		faces = [
			1 2 3;
			3 4 1
		]
		mesh!(ax, verts, faces; color = Makie.RGBA(0,0,0,.1))
	end

	scatter!(ax, tups; color = :black)
end

# ╔═╡ 1cd765dc-8739-4dd1-8dd6-0f13d0622332
begin 
	"Return `true` if the tuple `(θ, Φ)` is acceptable for `filter`."
	function is_acceptable( filter, θ, Φ )
		γ = filter.gamma
		for (θj, Φj) in filter.entries
			offset = γ*θj
			if θ > θj - offset && Φ > Φj - offset
				return false
			end
		end
		return true
	end
	"Return `true` if the tuple `(θ, Φ)` is acceptable for `filter`, if it is augmented by `(θ_add, Φ_add)`."
	function is_acceptable( filter, θ, Φ, θ_add, Φ_add )
		γ = filter.gamma
		offset = γ*θ_add
		if θ > θ_add - offset && Φ > Φ_add - offset
			return false
		end
		return is_acceptable(filter, θ, Φ)
	end
end

# ╔═╡ 8926543f-854a-45be-91f5-1598503e7c24
function filter_add!( filter, θ, fx)
	delete_indices = Int[]
	γ = filter.gamma
	_θ = γ * θ
	for (j, tup) = enumerate(filter.entries)
		θj, fj = tup
		if θj >= θ && fj - γ*θj >= fx - _θ
			push!(delete_indices, j)
		end
	end
	deleteat!(filter.entries, delete_indices)
	push!(filter.entries, (θ, fx))
	return nothing
end

# ╔═╡ dd51a28b-f8e3-48dd-94d4-284996b0c558
let
	# setup an example filter
	filter = Filter()
	filter_add!(filter, 1, 10)
	filter_add!(filter, 5, 5)
	filter_add!(filter, 15, 1)

	# define theme for slides:
	presentation_theme = Theme(
		Scatter = (
			markersize = 30,
		),
		Legend = (
			labelsize = 40,
			patchlabelgap = 20,
		),
		Axis = (
			xlabelsize = 40,
			ylabelsize = 40,
			xticklabelsize = 35,
			yticklabelsize = 35,
		)
	)
	
	with_theme( presentation_theme ) do 
		fig = Figure(resolution = (1000, 1000))
		ax = Axis(fig[1,1])
		filter_plot!(ax, filter)
	
		scatter!(ax, [(2.0, 3.0),]; color = GREEN, label="acceptable")
		scatter!(ax, [(10.0, 8.0),]; color = RED, label="inacceptable")
		axislegend(ax)
		fig
	end
end

# ╔═╡ 4aae416a-e005-4b16-b8a5-90bf6360dda3
md"### Main Algorithm"

# ╔═╡ 6846849a-994d-40a4-a84b-7f2004364844
md"#### Configuration Options"

# ╔═╡ 5a07763d-c50e-4dab-affe-e83e7f967d54
@with_kw struct AlgorithmConfig
	# stopping criteria
	max_iter = 100

	"Stop if the trust region radius is reduced to below `stop_delta_min`."
	stop_delta_min = eps(Float64)

	"Stop if the trial point ``xₜ`` is accepted and ``‖xₜ - x‖≤ δ‖x‖``."
	stop_xtol_rel = 1e-5
	"Stop if the trial point ``xₜ`` is accepted and ``‖xₜ - x‖≤ ε``."
	stop_xtol_abs = -Inf
	"Stop if the trial point ``xₜ`` is accepted and ``‖f(xₜ) - f(x)‖≤ δ‖f(x)‖``."
	stop_ftol_rel = 1e-5
	"Stop if the trial point ``xₜ`` is accepted and ``‖f(xₜ) - f(x)‖≤ ε``."
	stop_ftol_abs = -Inf

	"Stop if for the approximate criticality it holds that ``χ̂(x) <= ε`` and for the feasibility that ``θ <= δ``."
	stop_crit_tol_abs = eps(Float64)
	"Stop if for the approximate criticality it holds that ``χ̂(x) <= ε`` and for the feasibility that ``θ <= δ``."
	stop_theta_tol_abs = eps(Float64)
	
	"Stop after the criticality routine has looped `stop_max_crit_loops` times."
	stop_max_crit_loops = 1

	# criticality test thresholds
	eps_crit = 0.1
	eps_theta = 0.1
	crit_B = 1000
	crit_M = 3000
	crit_alpha = 0.5
	
	# initialization
	delta_init = 0.5
	delta_max = 2^5 * delta_init

	# trust region updates
	gamma_shrink_much = 0.1 	# 0.1 is suggested by Fletcher et. al. 
	gamma_shrink = 0.5 			# 0.5 is suggested by Fletcher et. al. 
	gamma_grow = 2.0 			# 2.0 is suggested by Fletcher et. al. 

	# acceptance test 
	strict_acceptance_test = true
	nu_accept = 0.01 			# 1e-2 is suggested by Fletcher et. al. 
	nu_success = 0.9 			# 0.9 is suggested by Fletcher et. al. 
	
	# compatibilty parameters
	c_delta = 0.7 				# 0.7 is suggested by Fletcher et. al. 
	c_mu = 100.0 				# 100 is suggested by Fletcher et. al.
	mu = 0.01 					# 0.01 is suggested by Fletcher et. al.

	# model decrease / constraint violation test
	kappa_theta = 1e-4 			# 1e-4 is suggested by Fletcher et. al. 
	psi_theta = 2.0

	nlopt_restoration_algo = :LN_COBYLA

	# backtracking:
	normalize_grads = false	
	armijo_strict = strict_acceptance_test
	armijo_min_stepsize = eps(Float64)
	armijo_backtrack_factor = .75
	armijo_rhs_factor = 1e-3
end	

# ╔═╡ cd63c696-c877-412e-b2ea-b0b919e26018
md"#### Normal Step Computation"

# ╔═╡ 8ed4cd89-8be9-4c8a-a354-d85f5764f812
md"""
Below we setup a `JuMP` model for solving the normal step problem:
```math
    \begin{aligned}
        &\min \|{n}\|_2^2
        &\text{s.t.}\\
        &n \in \left( \hat{L}^{(k)} - x^{(k)} \right).
	\end{aligned}
```
"""

# ╔═╡ 6d0aec2c-8858-4eab-b4f0-08896b24e851
const QP_OPTIMIZER = OSQP.Optimizer

# ╔═╡ a30efb66-3628-4f7b-9582-b06051149bbd
function compute_normal_step(mop, iterate, mod_objectives, mod_eq_constraints, mod_ineq_constraints)

	@info "Computing a normal step."
	
	x = iterate.x
	
	itrn = JuMP.Model( QP_OPTIMIZER )
	JuMP.set_silent(itrn)
	JuMP.set_optimizer_attribute(itrn, "polish", true)

	@variable(itrn, n[1:num_vars(mop)])
	@objective(itrn, Min, sum(n.^2))

	x_n = x .+ n
	@constraint(itrn, mop.A_eq * x_n .== mop.b_eq)
	@constraint(itrn, mop.A_ineq * x_n .<= mop.b_ineq)

	hx = mod_eq_constraints( x )
	H = AD.jacobian( mod_eq_constraints,  x)
	@constraint(itrn, hx .+ H*n .== 0)

	gx = mod_ineq_constraints( x )
	G = AD.jacobian( mod_ineq_constraints,  x)
	@constraint(itrn, gx .+ G*n .<= 0)

	optimize!(itrn)
	_n = value.(n)
	return _n
end

# ╔═╡ e8325749-7522-459b-9d43-a332baf0bebf
function compatibility_test(mop, algo_config, n, Δ)
	return norm(n, Inf) <= algo_config.c_delta * Δ * min(1, algo_config.c_mu + Δ^algo_config.mu)
end

# ╔═╡ 29e8bcf0-345b-47d2-9e0d-d93ca4eb75ff
md"#### Descent Step Calculation"

# ╔═╡ 77c856a2-f2e8-4335-864c-2dd8b4c13773
md"""
Everything that follows is concernced with solving the descent problem:
```math
\begin{aligned}
&\min_{β, d\in ℝ^n}β&\text{s.t}\\
&\|d\|_∞ \le 1,\\
&\hat{F}^{(k)} ⋅ d \le β, \\
&x^{(k)} + n^{(k)} + d \in \hat{L}^{(k)}.
\end{aligned}
```
"""

# ╔═╡ 0a655b39-01df-4330-8dda-f89c1511d0a5
md"##### Linear Optimization Problem"

# ╔═╡ 366b2ede-a04b-41fd-8a19-f71d0e630658
function compute_descent_step(mop, algo_config, iterate, n, mod_objectives, mod_eq_constraints, mod_ineq_constraints)

	x = iterate.x
	x_n = x .+ n
	
	itrt = JuMP.Model( QP_OPTIMIZER )
	JuMP.set_silent(itrt)
	JuMP.set_optimizer_attribute(itrt, "polish", true)

	@variable(itrt, β)
	@variable(itrt, d[1:num_vars(mop)])
	
	@objective(itrt, Min, β)
	
	# step norm constraint
	@constraint(itrt, -1 .<= d)
	@constraint(itrt, d .<= 1)

	# descent constraint
	F = AD.jacobian( mod_objectives, x_n )
	if algo_config.normalize_grads
		grad_norms = norm.(eachrow(F))
		if !any(iszero.(grad_norms))
			F = F ./ norm.(eachrow(F))
		end
	end
	@constraint(itrt, F*d .<= β)

	s = n .+ d
	x_s = x .+ s
	@constraint(itrt, mop.A_eq * x_s .== mop.b_eq)
	@constraint(itrt, mop.A_ineq * x_s .<= mop.b_ineq)

	#=
	# Note: Evaluating the models at `x_n` is not conformant to the article
	# But it should work nonetheless
	hx = mod_eq_constraints( x_n )
	H = AD.jacobian( mod_eq_constraints,  x_n)
	@constraint(itrt, hx .+ H*d .== 0)

	gx = mod_ineq_constraints( x_n )
	G = AD.jacobian( mod_ineq_constraints,  x_n)
	@constraint(itrt, gx .+ G*d .<= 0)
	=#

	hx = mod_eq_constraints( x )
	H = AD.jacobian( mod_eq_constraints,  x)
	@constraint(itrt, hx .+ H*s .== 0)

	gx = mod_ineq_constraints( x )
	G = AD.jacobian( mod_ineq_constraints,  x)
	@constraint(itrt, gx .+ G*s .<= 0)
	
	optimize!(itrt)

	d = value.(d)
	χ = -value(β)

	if any(isnan.(d)) || isnan(χ)
		return zeros(length(x)), 0.0
	end
	return d, χ
end

# ╔═╡ 7812192f-1e72-4cd8-9b61-ed2e6b7588ef
md"##### Backtracking"

# ╔═╡ 2d865aad-62d3-4711-b2aa-12be5a1959e0
md"""
###### Polytope Intersection
To start backtracking, we sometimes need to solve problems of the form
```math
\operatorname{arg\,max}_{σ\in ℝ} σ \quad\text{  s.t.  }\quad
x + σ ⋅ d ∈ B ∩ L,
```
where ``x\in ℝ^N, d \in ℝ^N,`` and ``B\cap L`` is a polytope.
The problem can be solved using any LP algorithm.

Below, we use an iterative approach with the following logic: \
For each dimension, we determine the interval of allowed step sizes and then intersect these intervals.

For scalars ``x_i, σ, d_i, b_i``, consider the inequality
```math 
x_i + σ ⋅ d_i ≤ b_i.
```
What are possible values for ``σ``?
* If ``d_i=0``:
  - If ``x_i ≤ b_i``, then ``σ\in [-∞, ∞]``.
  - If ``x_i > b_i``, then ``σ\in ∅`` (does not exist!!!).
* Else:
  - If ``x_i = b_i``:
    * If ``d_i < 0``, then ``σ\in [0, ∞]``.
    * If ``d_i > 0``, then ``σ \in [0,0]``.
  - If ``x_i < b_i``:
    * If ``d_i < 0``, then ``σ\in [\underbrace{(b_i-x_i)/d_i}_{<0}, ∞]``.
    * If ``d_i > 0``, then ``σ\in [-∞, (b_i-x_i)/d_i].``
  - If ``x_i > b_i`` (``x`` infeasible):
    * If ``d_i < 0``, then ``σ\in [\underbrace{(b_i-x_i)/d_i}_{>0}, ∞]``.
    * If ``d_i > 0``, then ``σ\in [-∞, (b_i-x_i)/d_i]``.

This decision tree is implemented in `stepsize_interval`:
"""

# ╔═╡ b6ce5fc0-3011-4ca0-87b8-8cb8f33f42e0
function stepsize_interval(
	x :: X, 
	d :: D, 
	b :: B, 
	::Type{T} = Base.promote_type(Float16,X,D,B)
) :: Tuple{T,T} where {X<:Real, D<:Real, B<:Real, T}
	T_Inf = T(Inf)
	T_NaN = T(NaN)
	T_Zero = zero(T)
	if iszero(d)
		if x <= b
			return (-T_Inf, T_Inf)
		else
			return (T_NaN, T_NaN)
		end
	else
		if x == b
			if d < 0
				return (T_Zero, T_Inf)
			else
				return (T_Zero, T_Zero)
			end
		else
			r = T((b-x) / d)
			if d < 0
				return (r, T_Inf)
			else
				return (-T_Inf, r)
			end
		end
	end
end

# ╔═╡ 690aee10-045c-49cc-bd98-5bfc3206a9a5
function stepsize_interval(
	x :: AbstractVector{X}, 
	d :: AbstractVector{D}, 
	b :: AbstractVector{B}, 
	::Type{T} = Base.promote_type(Float16,X,D,B)
) :: Union{Tuple{T,T},Nothing} where {X<:Real, D<:Real, B<:Real, T}
	T_Inf = T(Inf)
	l = -T_Inf
	r = T_Inf
	for (xi, di, bi) = zip(x,d,b)
		li, ri = stepsize_interval(xi, di, bi, T)
		isnan(li) && return (li, ri)
		l = max( l, li )
		r = min( r, ri )
		if l > r
			return nothing
		end
	end
	return (l, r)
end

# ╔═╡ f81657c9-09c0-4689-a2ed-783858349416
function intersect_interval(x, d, b, l, r, T)
	_l, _r = stepsize_interval(x,d,b, T)
	isnan(_l) && return (_l, _r)
	L = max(l, _l)
	R = min(r, _r)
	if L > R
		return (T(NaN), T(NaN))
	end
	return (L,R)
end

# ╔═╡ e1f0ad5f-a601-48af-906c-2190b4c1bd2f
"""
	intersect_linear_constraints(x, d, lb, ub, A_ineq = [], b_ineq = [], A_eq = [], b_eq = [])
Return a tuple ``(σ_-,σ_+)`` of minimum and 
the maximum stepsize ``σ ∈ ℝ`` such that ``x + σd`` 
conforms to the linear constraints ``lb ≤ x+σd ≤ ub`` and ``A(x+σd) - b ≦ 0`` along 
all dimensions.
If there is no such stepsize then `(NaN,NaN)` is returned.
"""
function intersect_linear_constraints( 
	_x :: AbstractVector{X}, 
	d :: AbstractVector{D}, 
	lb :: AbstractVector{LB} = Float32[], 
	ub :: AbstractVector{UB} = Float32[], 
	A_ineq :: AbstractMatrix{AINEQ} = Matrix{Float32}(undef, 0, length(_x)), 
	b_ineq :: AbstractVector{BINEQ} = Float32[],
	A_eq :: AbstractMatrix{AEQ} = Matrix{Float32}(undef, 0, length(_x)), 
	b_eq :: AbstractVector{BEQ} = Float32[], 
) where {X <: Real,D<:Real,LB<:Real,UB<:Real,AEQ<:Real,BEQ<:Real,AINEQ<:Real,BINEQ<:Real}
	
    # TODO can we pass precalculated `Ax` values for `A_eq` and `A_ineq`
    n_vars = length(_x)
    T = Base.promote_type(X,D,LB,UB,AEQ,BEQ,AINEQ,BINEQ)

	T_Inf = T(Inf)
	T_NaN = T(NaN)
	x = T.(_x)
	interval_inf = (-T_Inf, T_Inf)
	interval_nan = (T_NaN, T_NaN)

	# if `d` is zero vector, we can move infinitely in all dimensions and directions
	if iszero(d)
        return interval_inf
    end

	# bounds default to zero:
	_b_ineq = isempty(b_ineq) ? zeros(T, n_vars) : b_ineq
	
	if isempty( A_eq )
		# only inequality constraints
		l, r = interval_inf
		if !isempty(ub)
			l, r = intersect_interval(x, d, ub, l, r, T)
			isnan(l) && return (l, r)
		end

		if !isempty(lb)
			## "lb <= x + σ d" ⇔ "-x + σ(-d) <= -lb"
			l, r = intersect_interval(-x, -d, -lb, l, r, T)
			isnan(l) && return (l, r)
		end
		
		if !isempty(A_ineq)
            ## "A(x+σd) <= b" ⇔ "Ax + σ Ad <= b"
			l, r = intersect_interval(A_ineq*x, A_ineq*d, _b_ineq, l, r, T)
			isnan(l) && return (l, r)
		end
					
		return l, r
	else
		# there are equality constraints
		# they have to be all fullfilled and we loop through them one by one (rows of A_eq)
		N = size(A_eq, 1)
		_b_eq = isempty(b_eq) ? zeros(T, N) : b_eq

		σ = T_NaN
		for i = 1 : N
            # a'(x+ σd) - b = 0 ⇔ σ a'd = -(a'x - b) ⇔ σ = -(a'x -b)/a'd 
            ad = A_eq[i,:]'d
			if !iszero(ad)
				σ_i = - (A_eq[i, :]'x - _b_eq[i]) / ad
			else
                # check for primal feasibility of `x`:
				if !iszero( A_eq[i,:]'x .- _b_veq[i] )
					return interval_nan
				end
			end
			
			if isnan(σ)
				σ = σ_i
			else
				if !(σ_i ≈ σ)
					return interval_nan
				end
			end
		end
		
		if isnan(σ)
			# only way this could happen:
			# ad == 0 for all i && x feasible w.r.t. eq const
			return intersect_linear_constraints(x, d, lb, ub, A_ineq, b_ineq )
		end
			
		# check if x + σd is compatible with the other constraints
		x_trial = x + σ * d
		
		(!isempty(lb) && any(x_trial .< lb )) && return interval_nan
		(!isempty(ub) && any(x_trial .> ub )) && return interval_nan
		(!isempty(A_ineq) && any( A_ineq * x_trial > _b_ineq )) && return interval_nan
		return (σ, σ)
	end
end

# ╔═╡ 7d670e3c-1105-425f-8a31-ffb2dc4d32ab
function build_models(cfg::RbfConfig, mop, algo_config, database, iterate, Δ)
	x = iterate.x
	n_vars = num_vars(mop)
	
	# look for `n_vars` additional, affinely independent points in database
	Δ_1 = cfg.delta_factor * Δ
	lb = max.(mop.lb, x .- Δ_1 )
	ub = min.(mop.ub, x .+ Δ_1 )
	piv_val = cfg.piv_factor * Δ_1

	Y = Matrix{Float64}(undef, n_vars, 0)
	Z = Matrix{Float64}(eye(n_vars))
	old_sites = Vector{Float64}[]
	old_site_indices = Int[]
	for (i,res) in enumerate(database)
		_x = res.x
		if all( lb .<= _x .<= ub )
			ξ = _x - x
			# point is in enlarged trust region
			if norm( Z*(Z'ξ), Inf) > piv_val
				# accept candidate
				Y = hcat(Y, ξ)
				Z = _orthonormal_complement_matrix(Y)
				push!(old_sites, _x)
				push!(old_site_indices, i)
			end
		end
		if size(Y, 2) == n_vars
			break
		end
	end

	# NOTE we skip "Round 2" (Looking for candidates in even larger box)
	# because the models should be fully linear for this notebook

	# sample along improving directions:
	new_sites = Vector{Float64}[]
	for ξ = eachcol(Z)
		σ_1, σ_2 = intersect_linear_constraints(x, ξ, lb, ub)
		σ = abs(σ_1) > abs(σ_2) ? σ_1 : σ_2
		push!(new_sites, x .+ σ*ξ)
	end

	# look for further improving points in database:
	# NOTE: this is pretty much copied from Morbit.
	# ξ no longer is translated into the origin...
	Δ_2 = cfg.delta_max_factor * algo_config.delta_max
	lb = max.(mop.lb, x .- Δ_2 )
	ub = min.(mop.ub, x .+ Δ_2 )
	
	max_points = cfg.max_points < 0 ? ceil(Int, ((n_vars + 1 ) * (n_vars + 2)) / 2) : cfg.max_points
	
	chol_pivot = cfg.piv_cholesky^2

	centers = [[x,]; old_sites; new_sites]
	N = length(centers)
	
	φ = cfg.kernel_func(Δ)
	Φᵀ, Πᵀ, kernels, polys = RBF.get_matrices( 
		φ, centers; 
		poly_deg = cfg.poly_deg 
	)
	Φ = transpose(Φᵀ)
	## Πᵀ is the matrix with ``\dim Π_{n}`` rows and ``N`` columns.
	
	## prepare matrices as described by Wild
	_Q, _R = qr( Matrix{Float64}(Πᵀ) )
	### extract full Q factor		
	dim_Q = size(_Q, 1)
	Q = _Q * Matrix(eye, dim_Q, dim_Q)
	## augment `_R`
	R = [
		_R;
		zeros( dim_Q - size(_R,1), size(_R,2) )
	]
	
	Z = Q[:, N + 1 : end ] ## columns of Z are orthogonal to Πᵀ

	## Note: usually, Z, ZΦZ and L are empty (if `N == n_vars + 1`)
	ZΦZ = Hermitian(Z'Φ*Z)	## make sure, it is really symmetric
	L = cholesky( ZΦZ ).L   ## perform cholesky factorization
	L⁻¹ = inv(L)			## most likely empty at this point

	φ₀ = Φ[1,1]

	for (i,res) in enumerate(database)
		if i in old_site_indices
			continue
		end
		if N >= max_points
			break
		end
		ξ = res.x
		if all( lb .<= ξ .<= ub )

			### apply all RBF kernels
			φξ = kernels( ξ )
		
			### apply polynomial basis system and augment polynomial matrix
			πξ = polys( ξ )
			Rξ = [ 
				R; 
				πξ' 
			]

			### perform Givens rotations to turn last row in Rξ to zeros
			Rξ, G = nullify_last_row(Rξ) 

			### now, from G we can update the other matrices 
			Gᵀ = transpose(G)
			g̃ = Gᵀ[1:(end-1), end]		# last column of transposed matrix
			ĝ = G[end, end]

			Qg̃ = Q*g̃;
			v_ξ = Z'*( Φ*Qg̃ + φξ .* ĝ )
			σ_ξ = Qg̃'*Φ*Qg̃ + (2*ĝ) * φξ'*Qg̃ + ĝ^2*φ₀

			τ_ξ² = σ_ξ - norm( L⁻¹ * v_ξ, 2 )^2 
			## τ_ξ (and hence τ_ξ^2) must be bounded away from zero 
			## for the model to remain fully linear
			if τ_ξ² > chol_pivot
				
				push!(old_sites, ξ)
				push!(old_site_indices, i)
				
				τ_ξ = sqrt(τ_ξ²)

				## zero-pad Q and multiply with Gᵗ
				Q = cat(Q,1; dims=(1,2)) * Gᵀ

				Z = [ 
					Z  						Qg̃;
					zeros(1, size(Z,2)) 	ĝ 
				]
				
				L = [
					L          zeros(size(L,1), 1) ;
					v_ξ'L⁻¹'   τ_ξ 
				]

				L⁻¹ = [
					L⁻¹                zeros(size(L⁻¹,1),1);
					-(v_ξ'L⁻¹'L⁻¹)./τ_ξ   1/τ_ξ 
				]

				R = Rξ

				## finally, augment basis matrices and add new kernel for next iteration
				
				Φ = [ 
					Φ   φξ;
					φξ' φ₀
				]
				push!( kernels, RBF.make_kernel(φ, ξ) )
				
				#Π = [ Π πξ ] #src
				# ZΦZ = [	ZΦZ v_ξ; v_ξ' σ_ξ] #src
				#@show all( diag( L * L⁻¹) .≈ 1 ) #src

				N += 1
			end#if 
		end#for 
	end

	old_len = length(database)
	for _x in new_sites
		res = make_result(_x, mop)
		push!(database, res)
	end
	new_db_indices = collect((old_len + 1) : length(database))
	
	centers = [[x,]; old_sites; new_sites]
	mod_o = make_rbf_model( :fx, iterate, centers, database, old_site_indices, new_db_indices, φ, cfg )
	mod_e = make_rbf_model( :hx, iterate, centers, database, old_site_indices, new_db_indices, φ, cfg )
	mod_i = make_rbf_model( :gx, iterate, centers, database, old_site_indices, new_db_indices, φ, cfg )

	return mod_o, mod_e, mod_i
end

# ╔═╡ 529f9f7f-9a66-4020-aa8b-0d3e63372229
md"##### Armijo-like Backtracking"

# ╔═╡ 8ec19cd2-30fa-4815-8711-8b98afec3a82
begin
	function armijo_test(strict::Val{true}, mx, mx_trial, σ, a, χ)
		rhs = σ * a * χ
		return all( mx .- mx_trial .>= rhs )
	end
	function armijo_test(strict::Val{false}, mx, mx_trial, σ, a, χ)
		return maximum(mx) .- maximum(mx_trial) >= σ * a * χ 
	end
end

# ╔═╡ 18c93e20-2fe3-473f-a206-9b7abf45d6b5
function backtrack( mop, algo_config, iterate, Δ, n, d, χ, mod_objectives, mod_eq_constraints, mod_ineq_constraints )
	
	x = iterate.x
	x_n = x .+ n
	
	norm_d = norm(d, Inf)
	iszero(norm_d) && return norm_d
	
	# determine initial stepsize for normed direction `d/norm_d`
	s = n .+ d
	norm_s = norm(s, Inf)
	σ_init = if norm_s == Δ
		# ``x + s`` lies on the boundary of trust region
		1
	elseif norm_s < Δ
		if norm_d < 1
			# global constraints are binding, we cannot scale `d` up
			1
		else
			# use same constraints as in descent step calculation.
			# `intersect_linear_constraints` expects "Ax <= b".
			# `mop` provides linear constraints in that form
			# models provide "h(x) + H(x)*s ≦ 0" ⇔ "Hs <= -h"
			A_eq = Matrix{Float64}([
				mop.A_eq;
				AD.jacobian( mod_eq_constraints,  x)
			])
			b_eq = Vector{Float64}([
				mop.b_eq;
				-mod_eq_constraints(x)
			])
			A_ineq = Matrix{Float64}([
				mop.A_ineq;
				AD.jacobian( mod_ineq_constraints,  x)
			])
			b_ineq = Vector{Float64}([
				mop.b_ineq;
				-mod_ineq_constraints(x)
			])
			lb = max.( mop.lb, x .- Δ )
			ub = min.( mop.ub, x .+ Δ )
			stepsizes = intersect_linear_constraints(x_n, d, lb, ub, A_ineq, b_ineq, A_eq, b_eq)
			_σ_init = maximum(stepsizes)
			if isnan(_σ_init) || _σ_init < 0
				0
			else
				_σ_init
			end
		end
	elseif norm_s > Δ
		max(0, (Δ - norm(n,Inf))/norm_d ) 	# TODO Think about this
	end

	a = algo_config.armijo_rhs_factor
	b = algo_config.armijo_backtrack_factor
	σ_min = algo_config.armijo_min_stepsize 
    σ = σ_init
	mx = mod_objectives(x_n)
	mx_trial = mod_objectives(x_n + σ * d)
	strict_val = Val(algo_config.armijo_strict)
	while σ >= σ_min && !armijo_test(strict_val, mx, mx_trial, σ, a, χ)
		σ *= b
		mx_trial = mod_objectives(x_n + σ * d)
	end
	return σ
end

# ╔═╡ 7f6c954a-d85f-44a0-839f-6dcf855a8cbd
let
	algo_config = AlgorithmConfig()
	num_vars = 2
	lb = zeros(num_vars)
	ub = ones(num_vars)
	x = [.8, 1.2]
	f1 = x -> sum( (x .- [.25, .5]).^2 )
	f2 = x -> sum( (x .- [.75, .5]).^2 )
	g = x -> sum(x.^2) - 1
	
	mop = MOP(;num_vars, lb, ub)
	
	mod_objectives = x -> [f1(x); f2(x)]
	mod_eq_constraints = x -> Float64[]
	mod_ineq_constraints = x -> [g(x),]
	
	iterate = Result(;x)
	
	n = compute_normal_step(mop, iterate, mod_objectives, mod_eq_constraints, mod_ineq_constraints)
	x_n = x .+ n
	d, χ = compute_descent_step(mop, algo_config, iterate, n, mod_objectives, mod_eq_constraints, mod_ineq_constraints)
	Δ = 0.75
	σ = backtrack( mop, algo_config, iterate, Δ, n, d, χ, mod_objectives, mod_eq_constraints, mod_ineq_constraints )
	d *= σ
	x_s = x_n .+ d

	presentation_theme = Theme(
		Scatter = (
			markersize = 30,
		),
		Legend = (
			labelsize = 40,
			patchlabelgap = 20,
		),
		Axis = (
			xlabelsize = 40,
			ylabelsize = 40,
			xticklabelsize = 35,
			yticklabelsize = 35,
		),
		Arrows = (
			linewidth = 5,
			lengthscale = 0.9f0,
			arrowsize = 30,
			align = :origin,
		),
		Contour = (
			linewidth = 4.0,
			color = GRAY,
		),
		Lines = (
			linewidth = 5.0,
			linestyle = :dash
		)
	)
	with_theme(presentation_theme) do
		fig = Figure(resolution = (1600, 900))
		ax = Axis(fig[1,1])
	
		x_l, x_u = min(lb[1], x[1]) - 0.1, max(x[1], ub[1]) + 0.1
		y_l, y_u = min(lb[2], x[2]) - 0.1, max(x[2], ub[2]) + 0.1
		XX = LinRange(x_l, x_u, 300)
		YY = LinRange(y_l, y_u, 300)
		ZG = [g([xx; yy]) <= 0 ? 1 : 0 for xx = XX, yy = YY]
	
		# linear approximation
		gx = mod_ineq_constraints( x )
		G = AD.jacobian( mod_ineq_constraints,  x)
		_g = ξ -> gx .+ G*(ξ - x) 
		_ZG = [_g([xx; yy])[end] <= 0 ? 1 : 0 for xx = XX, yy = YY]
		image!(ax, XX, YY, _ZG; 
			colormap = [Makie.RGBAf(0,0,0,0),Makie.RGBAf(BLUE,.75)]
		)
		
		# true feasible set
		image!(ax, XX, YY, ZG; 
			transparency = true,
			colormap = [Makie.RGBAf(0,0,0,0),Makie.RGBAf(BLUE,1)]
		)

		F1 = [f1([xx;yy]) for xx = XX, yy = YY]
		contour!(ax, XX, YY, F1; levels = f1(x_n) .* LinRange(0,2,5))
		
		F2 = [f2([xx;yy]) for xx = XX, yy = YY]
		contour!(ax, XX, YY, F2; levels = f2(x_n) .* LinRange(0,2,5))

		# trust region
		lb_t = x .- Δ
		ub_t = x .+ Δ
		lines!(ax, [
			(lb_t[1], lb_t[2]), 
			(ub_t[1], lb_t[2]),
			(ub_t[1], ub_t[2]),
			(lb_t[1], ub_t[2]),
			(lb_t[1], lb_t[2])
		]; color = CYAN)
		
		# normal step
		arrows!(ax, [Point2(x...),], [Point2(n...),]; color = :orange)
	
		# grads
		F = -AD.jacobian( mod_objectives, x_n )
		arrows!(ax, [Point2(x_n...)], [Point2(F[1,:]...)]; 
			label = L"-∇f_1", color = LIGHTGRAY)
		arrows!(ax, [Point2(x_n...)], [Point2(F[2,:]...)]; 
			label = L"-∇f_2", color = LIGHTGRAY)
	
		# descent step
		arrows!(ax, [Point2(x_n...),], [Point2(d...),]; color = RED)
	
		scatter!(ax, Tuple(x); color = :black, label = L"x^{(k)}")
		scatter!(ax, Tuple(x_n); color = ORANGE, label = L"x^{(k)}_n")
		scatter!(ax, Tuple(x_s); color = RED, label = L"x^{(k)}_+")
	
		axislegend(ax; position = :lt)
	
		xlims!(ax, x_l, x_u)
		ylims!(ax, y_l, y_u)
		ax.xlabel = L"x_1"
		ax.ylabel = L"x_2"
		fig
	end
end

# ╔═╡ 7525f4bf-0705-436c-aa97-dbecac5aca1b
md"#### Restoration"

# ╔═╡ b4b4620e-b72e-402b-994b-c6db37199a42
md"""
Restoration of feasibilty is currently done by `NLopt`.
At some point in the future we could like to use our own solver kind of recursively.
"""

# ╔═╡ 587f9a6b-c170-40e7-8b42-e60539cffb60
import NLopt as NL

# ╔═╡ e07d9869-1ca5-486d-9173-192f52d7bd1a
function do_restoration(mop, iterate, n, algo_name = :LN_COBYLA)
	n_vars = num_vars(mop)
	x = iterate.x
	
	opt = NL.Opt(algo_name, n_vars)

	objf = function( r::Vector, grad::Vector )
		if length(grad) > 0
			grad .= AD.gradient( _r -> max_constraint_violation(mop, x .+ _r), r)
		end
		return max_constraint_violation(mop, x .+ r)
	end

	opt.min_objective = objf

	opt.lower_bounds = mop.lb .- x
	opt.upper_bounds = mop.ub .- x
	opt.xtol_rel = 1e-4
	opt.maxeval = 50 * n_vars^2
	opt.stopval = 0

	#r0 = any(isnan.(n)) || any(isinf.(n)) ? zeros(n_vars) : n
	r0 = zeros(n_vars)
	(θ_opt, r_opt, ret) = NL.optimize(opt, r0)
	return θ_opt, r_opt, ret
end

# ╔═╡ 1a5a691e-b6b8-4996-bfdd-34969fc1bfc1
md"#### Criticality Routine"

# ╔═╡ 5a99426d-4e04-4032-b19c-69c5492e3d3b
@enum CRIT_STAT begin
	TOLERANCE_X_EXIT = -3
	CRITICAL_EXIT = -2
	MAX_LOOP_EXIT = -1
	NOT_RUN = 0
	FINISHED = 1
end

# ╔═╡ f8ec520f-f678-448f-a0bd-10cc45262dfc
function criticality_routine(mop, model_config, algo_config, database, iterate, χ, Δ, n, mod_objectives, mod_eq_constraints, mod_ineq_constraints)

	@info "Criticality Routine!"
	crit_stat = FINISHED
	i = 0
	while Δ > algo_config.crit_M * χ
		if i >= algo_config.stop_max_crit_loops
			crit_stat = MAX_LOOP_EXIT
			break
		end
		if Δ <= algo_config.stop_delta_min
			crit_stat = TOLERANCE_X_EXIT
			break
		end
		if χ <= algo_config.stop_crit_tol_abs
			crit_stat = CRITICAL_EXIT
			break
		end
			
		_Δ = algo_config.crit_alpha * Δ
		mod_o, mod_e, mod_i = build_models(model_config, mop, algo_config, database, iterate, Δ)
		_n = compute_normal_step(mop, iterate, mod_o, mod_e, mod_i)
		_n_valid = !(any(isnan.(n))) && !(any(isinf.(n)))
		_n_compatible = _n_valid && compatibility_test(mop, algo_config, _n, _Δ)
		if !_n_compatible
			break
		end
		Δ, n = _Δ, _n
		mod_objectives, mod_eq_constraints, mod_ineq_constraints = mod_o, mod_e, mod_i
		d, χ = compute_descent_step( mop, algo_config, iterate, n, mod_objectives, mod_eq_constraints, mod_ineq_constraints )
		
		i += 1
		
	end
	return χ, Δ, n, mod_objectives, mod_eq_constraints, mod_ineq_constraints, crit_stat	
end

# ╔═╡ d66b7e85-3b70-475f-b634-a9bd58619531
md"#### Optimization Loop"

# ╔═╡ 4e5954bb-4615-46e6-9a20-701afa0175eb
@enum RET_CODE begin
	INFEASIBLE = -1
	CRITICAL = 1
	BUDGET = 2
	TOLERANCE_X = 3
	TOLERANCE_F = 4
end

# ╔═╡ d73b2468-4d9f-47e7-940f-f02bf6901dd8
@enum IT_STAT begin
	RESTORATION = -3
	FILTER_FAIL = -2
	INACCEPTABLE = -1
	INITIALIZATION = 0
	FILTER_ADD = 1
	ACCEPTABLE = 2
	SUCCESSFUL = 3
end

# ╔═╡ a965d55f-f21c-4cab-af43-1ce8bb14b28a
md"## Examples"

# ╔═╡ 0c18e387-e5a0-44a0-b0ea-26c9ad637479
md"""
Below are some tests and examples.
The `optimize` method accepts a keyword argument `logging_callback` 
that we can use to gather information about every iteration.
To this end, I use a `Dictionaries.Dictionary` to store information
in form of an object of type `IterInfo`.
"""

# ╔═╡ 4fee47cb-4702-4dbb-92ae-e928032cfb2c
begin
	struct IterInfo
		iter_index :: Int
		result :: Result
		it_stat :: IT_STAT
		radius :: Float64
	end
	function get_example_logger( info_dict )
		return function (; 
				iter_index = k,
				radius = _Δ,
				current_iterate = iterate,
				#trial_iterate = iterate_t,
				#normal_step = n,
				#descent_step = d,
				#criticality = χ,
				iter_status = _it_stat,
				kwargs...
			)
			set!(info_dict, iter_index, IterInfo(iter_index, current_iterate, iter_status,radius))
		end
	end
end

# ╔═╡ 9fdd78bb-d736-47d2-91cd-9672aed9274c
EXAMPLE_THEME = Theme(
	Figure = (
		resolution = (800, 600),
	),
	Axis = (
		xlabelsize = 40,
		ylabelsize = 40,
		xticklabelsize = 35,
		yticklabelsize = 35,
		titlesize = 40,
	),
	Scatter = (
		markersize = 15,
		color = CYAN,
		cycle = nothing,
		strokewidth = 1,
		strokecolor = :white,
	),
	Lines = (
		linewidth = 4.0,
		linestyle = :solid,
		color = LIGHTGRAY,
		cycle = nothing,
	),
	Contourf = (
		colormap = [BLUE, LIGHTBLUE],
	),
	Legend = (
		labelsize = 35,
		patchlabelgap = 20,
	),
)

# ╔═╡ 9971cc69-fbe7-45da-835f-6dd7eb129d8e
md"### 1D1D: Simple Unconstrained Problem"

# ╔═╡ b7b9c533-b3b8-482e-a615-9be2fee76cbd
md"To verify, that our algorithm does work in principle, we test a very simple problem with only one single objective and no constraints:
"

# ╔═╡ c7c3e8ce-a54e-44a1-8d47-7351f1f127c6
md"""
```math
	\min_{x\in ℝ} x^4
```
"""

# ╔═╡ 6b7b7774-e3ca-4903-8ee3-e9429995b9c6
"Return an object of type `MOP` for the simple **s**ingle **o**bjective example."
function so_problem_mop()
	return MOP(;
		num_vars = 1,
		objectives = [x -> x[end]^4,]
	)
end

# ╔═╡ 28a5ca66-cce7-477a-9b18-606aaa14b195
function plot_so_problem_objective_space(mop, model_config, x0, fin_res, ret, info_dict)
	with_theme(EXAMPLE_THEME) do
		# setup figure
		fig = Figure()
		ax = Axis(fig[1,1])
	
		# filled contour of objective
		XX = LinRange( -8, 8, 300)
		
		f = x -> mop.objectives[1]([x,])[end]
		lines!(ax, XX, f.(XX))
	
		# plot iterates	
		its = [ ( first(inf_d.result.x), first(inf_d.result.fx) ) for inf_d in info_dict ]
		scatter!(ax, its)
		scatter!(ax, ( first(x0), f(first(x0)) ); color = :black, label=L"f(x_0)")
		scatter!(ax, last(its); color = RED, 
			label=latexstring("f(x_{$(length(info_dict))})")
		)

		ax.title = "$(Base.typename(typeof(model_config)).name): Iteration to $(round.(fin_res.x; digits=2))."
		ax.xlabel = L"x"
		ax.ylabel = L"f(x)"

		axislegend(ax; position=:lt)
		return fig
	end
end

# ╔═╡ 665e607b-8c85-4914-ac97-ffa9eedec0e0
md"### 2D1D: Constrained Rosenbrock"

# ╔═╡ f13b5aa6-aa5c-418b-9293-f960cec2fb5b
md"""
Next, we test a more difficult single-objective problem with constraints.
[Wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization) cites a constrained Rosenbrock problem as below:
```math
	\begin{aligned}
	&\min_{x\in ℝ²}~
		(1-x₁)² + 100⋅(x₂ - x₁²)²
	&\text{s.t.}\\
	&-1.5 ≤ x₁ ≤ 1.5, \;
		-0.5 ≤ x₂ ≤ 2.5,\\
	&(x₁ - 1)³-x₂ + 1 ≤ 0, \\
	&x₁ + x₂ -2  ≤ 0,
	\end{aligned}
```
which has its global solution at ``f(1,1) = 0``.
"""

# ╔═╡ 3081e8c0-49e2-413d-a628-58308ed62600
"Return an object of type `MOP` for the *single objective* constrained Rosenbrock problem."
function so_rosenbrock_mop()
	return MOP(;
		num_vars = 2,
		lb = [-1.5; -0.5],
		ub = [1.5, 2.5],
		objectives = [ x -> (1 - x[1])^2 + 100*(x[2] - x[1]^2)^2 ],
		nl_ineq_constraints = [
			x -> (x[1] - 1)^3 - x[2] + 1,
			x -> x[1] + x[2] - 2
		]
	)
end

# ╔═╡ bdc0e9e4-e93a-47ed-a72d-0ea7b71af3ee
function plot_so_rosenbrock_decision_space(mop, model_config, x0, fin_res, ret, info_dict)
	with_theme(EXAMPLE_THEME) do
		# setup figure
		fig = Figure()
		ax = Axis(fig[1,1])
	
		# filled contour of objective
		XX = LinRange( mop.lb[1], mop.ub[1], 300)
		YY = LinRange( mop.lb[2], mop.ub[2], 300)
	
		f = mop.objectives[1]
		F = [f([xx, yy]) for xx=XX, yy=YY]
		contourf!(ax, XX, YY, F; levels = 20)
	
		# mask infeasible area
		g = x -> [func(x) for func in mop.nl_ineq_constraints]
		G = [ all(g([xx,yy]) .<= 0) ? true : false for xx=XX, yy=YY]
		image!(ax, XX, YY, G; colormap=[ORANGE, Makie.RGBAf(0,0,0,0)])
	
		# plot iterates		
		xs = [ Tuple(inf_d.result.x) for inf_d in info_dict ]
		lines!(ax, xs)
		scatter!(ax, xs)
		scatter!(ax, Tuple(x0); color = :black, label=L"x_0")
		scatter!(ax, Tuple(fin_res.x); color = RED, 
			label=latexstring("x_{$(length(info_dict))}")
		)

		ax.title = "$(Base.typename(typeof(model_config)).name): Iteration to $(round.(fin_res.x; digits=2))."
		ax.xlabel = L"x_1"
		ax.ylabel = L"x_2"

		axislegend(ax; position=:lt)
		return fig
	end
end

# ╔═╡ 13cd8e50-5824-41b9-856f-2c81da0bae77
md"""
The Rosenbock function is known te be difficult for local optimizers because its global optimum is in a very shallow parabolic valley.
If we start close to the global optimum, then most of the time, the algorithm converges:
"""

# ╔═╡ e75cd5b9-13c3-48d6-bbe8-25df557d668c
md"However, further away, we probably wont find $[1,1]$."

# ╔═╡ 26b521b4-5cce-4286-8ac9-f3cb0c27967c
function plot_so_rosenbrock_radius(mop, model_config, x0, fin_res, ret, info_dict; it_indices = nothing, do_markers = false)
	with_theme(EXAMPLE_THEME) do
		# setup figure
		fig = Figure()
		ax = Axis(fig[1,1])
	
		if isnothing(it_indices)
			it_indices = collect(keys(info_dict))
		end
		lines!(ax, it_indices, [info_dict[it_ind].radius for it_ind = it_indices])
		make_legend = false
		if do_markers
			# colors:
			cols = Dict(
				RESTORATION => RED,
				FILTER_FAIL => ORANGE,
				INACCEPTABLE => CASSIS,
				INITIALIZATION => :black,
				FILTER_ADD => LIGHTBLUE,
				ACCEPTABLE => BLUE,
				SUCCESSFUL => GREEN,
			)

			for it_class in instances(IT_STAT)
				_it_indices = filter(i -> info_dict[i].it_stat == it_class, it_indices)
				if !isempty(_it_indices)
					scatter!(ax, _it_indices, [info_dict[i].radius for i=_it_indices]; 
						label = string(it_class),
						color = cols[it_class]
					)
					make_legend = true
				end
			end
		end
		ax.title = "$(Base.typename(typeof(model_config)).name): Trust Region Radius."
		ax.xlabel = L"k"
		ax.ylabel = L"Δ"

		if make_legend 
			axislegend()
		end
		return fig
	end
end

# ╔═╡ e43a665f-a8d4-4ba3-bb10-ac7a84d7db9c
md"### Two Parabolas"

# ╔═╡ 86d58d29-120e-4091-ad80-6dda94d7a9dd
md"#### Unconstrained"

# ╔═╡ 2eb7f4f9-d128-4e1b-a264-185be9f7ccaa
md"""
Finally, let's have a look at the “Two Parabola Problem”.
It is a popular test problem for multiobjective optimizers due to its simplicity.
The objectives are to parabolas and the global Pareto set is the line connecting their minima.
The Pareto Front is convex.
"""

# ╔═╡ 42ff0182-8621-4c09-aa72-80cbdc8ca2fc
md"""
```math
\begin{aligned}
	&\min_{x\in ℝ^2}
		\begin{bmatrix}
			(x_1 - 1)^2 + x_2^2 \\
			(x_1 + 1)^2 + x_2^2
		\end{bmatrix}
\end{aligned}
```
"""

# ╔═╡ 61ade99e-6292-4cd0-846a-5dc428bcb50d
function two_parabola_mop()
	return MOP(;
		num_vars = 2,
		objectives = [
			x -> (x[1] - 1)^2 + x[2]^2,
			x -> (x[1] + 1)^2 + x[2]^2
		]
	)
end

# ╔═╡ 6c02cfa7-19ec-4e8b-9cfd-54d71a400f31
function plot_two_parabola_iterations(
	mop, model_config, x0, fin_res, ret, info_dict;
	plot_ps = true,
)
	with_theme(EXAMPLE_THEME) do
		# setup figure
		fig = Figure(resolution=(1000, 720))

		# DECISION SPACE
		ax1 = Axis(fig[1,1])

		# contours of objective
		XX = LinRange( -4, 4, 300)
		YY = XX
		
		f1 = mop.objectives[1]
		F1 = [f1([x,y]) for x=XX, y=YY]
		contour!(ax1, XX, YY, F1)
		
		f2 = mop.objectives[2]
		F2 = [f2([x,y]) for x=XX, y=YY]
		contour!(ax1, XX, YY, F2)

		if plot_ps
			ps_x = LinRange(-1, 1, 100)
			ps_y = [0 for x in ps_x]
			lines!(ax1, ps_x, ps_y; label = "PS", color = BLUE)
		end
	
		# plot iterates	
		xs = Tuple.(id.result.x for id = info_dict)
		lines!(ax1, xs)
		scatter!(ax1, xs)
		scatter!(ax1, first(xs); color = :black, label=L"x_0")
		scatter!(ax1, last(xs); color = RED, 
			label=latexstring("x_{$(length(info_dict))}")
		)

		ax1.xlabel = L"x_1"
		ax1.ylabel = L"x_2"

		# OBJECTIVE SPACE
		ax2 = Axis(fig[1,2])

		# contours of objective
		if plot_ps
			pf = [(f1([x,y]), f2([x,y])) for (x,y) = zip(ps_x,ps_y) ]
			lines!(ax2, pf; label = "PF", color = BLUE)
		end
	
		# plot iterates	
		fxs = Tuple.(id.result.fx for id = info_dict)
		lines!(ax2, fxs)
		scatter!(ax2, fxs)
		scatter!(ax2, first(fxs); color = :black, label=L"f(x_0)")
		scatter!(ax2, last(fxs); color = RED, 
			label=latexstring("f(x_{$(length(info_dict))})")
		)

		ax2.xlabel = L"f_1"
		ax2.yaxisposition = :right
		ax2.ylabel = L"f_2"

		Label(fig[0,:], "$(Base.typename(typeof(model_config)).name): Iteration to $(round.(fin_res.x; digits=2)).", textsize = 35)
	
		
		axislegend(ax1; position=:lt)
		axislegend(ax2; position=:lt)
		return fig
	end
end

# ╔═╡ 663d60fc-8bcd-4284-baa6-f5ab50c71bbf
md"""#### Constrained
We can modify the critical set by introducing constraints.
"""

# ╔═╡ 7e562b08-bcb3-4f01-8cea-6f5a799b7e7e
function two_parabola_mop_c()
	return MOP(;
		num_vars = 2,
		lb = [-3.5, -1],
		ub = [3.5, 3],
		objectives = [
			x -> (x[1] - 1)^2 + x[2]^2,
			x -> (x[1] + 1)^2 + x[2]^2
		],
		nl_ineq_constraints = [
			x -> x[1]^2 + (x[2] - 1.5)^2 - 1
		],
		A_ineq = [1 -1],
		b_ineq = [-2]
	)
end

# ╔═╡ 0e9f4e15-5550-439e-b964-2b84aade660f
function plot_two_parabola_iterations_c(
	mop, model_config, x0, fin_res, ret, info_dict;
	plot_ps = true,
)
	with_theme(EXAMPLE_THEME) do
		# setup figure
		fig = Figure(resolution=(1200, 720))

		# DECISION SPACE
		ax1 = Axis(fig[1,1])

		# plot feasible area
		XX = LinRange( mop.lb[1], mop.ub[1], 300)
		YY = LinRange( mop.lb[2], mop.ub[2], 300)
		_G = [true for x = XX, y = YY]
		
		for g in mop.nl_ineq_constraints
			G = [ all(g([x,y]) .<= 0) ? true : false for x = XX, y = YY]
			image!(ax1, XX, YY, G; 
				colormap=[Makie.RGBAf(0,0,0,0), Makie.RGBAf(GREEN, .35)]
			)
			_G = @. _G && G
		end
		for (i,a) in enumerate(eachrow(mop.A_ineq))
			b = mop.b_ineq[i]
			G = [ a'*[x; y] <= b ? true : false for x = XX, y = YY]
			image!(ax1, XX, YY, G; 
				colormap=[Makie.RGBAf(0,0,0,0), Makie.RGBAf(GREEN, .35)]
			)
			_G = @. _G && G
		end
		_x = [(x,y) for x = XX, y = YY][_G]
		scatter!(ax1, _x;
			strokewidth = 0,
			markersize = 2,
			color=range(GREEN; stop = ORANGE, length = length(_x))
		)
		

		# contours of objective
		f1 = mop.objectives[1]
		F1 = [f1([x,y]) for x=XX, y=YY]
		contour!(ax1, XX, YY, F1)
		
		f2 = mop.objectives[2]
		F2 = [f2([x,y]) for x=XX, y=YY]
		contour!(ax1, XX, YY, F2)

		if plot_ps
			ps_x = LinRange(-1, 1, 100)
			ps_y = [0 for x in ps_x]
			lines!(ax1, ps_x, ps_y; label = "PS", color = BLUE)
		end

		# plot iterates	
		xs = Tuple.(id.result.x for id = info_dict)
		lines!(ax1, xs)
		scatter!(ax1, xs)
		scatter!(ax1, first(xs); color = :black, label=L"x_0")
		scatter!(ax1, last(xs); color = RED, 
			label=latexstring("x_{$(length(info_dict))}")
		)

		ax1.xlabel = L"x_1"
		ax1.ylabel = L"x_2"

		# OBJECTIVE SPACE
		ax2 = Axis(fig[1,2])

		# contours of objective
		if plot_ps
			pf = [(f1([x,y]), f2([x,y])) for (x,y) = zip(ps_x,ps_y) ]
			lines!(ax2, pf; label = "PF", color = BLUE)
		end

		_fx = [(f1(collect(x))[end], f2(collect(x))[end]) for x in _x]
		scatter!(ax2, _fx;
			strokewidth = 0,
			markersize = 2,
			color=range(GREEN; stop = ORANGE, length = length(_x))
		)
		
		# plot iterates	
		fxs = Tuple.(id.result.fx for id = info_dict)
		lines!(ax2, fxs)
		scatter!(ax2, fxs)
		scatter!(ax2, first(fxs); color = :black, label=L"f(x_0)")
		scatter!(ax2, last(fxs); color = RED, 
			label=latexstring("f(x_{$(length(info_dict))})")
		)

		ax2.xlabel = L"f_1"
		ax2.yaxisposition = :right
		ax2.ylabel = L"f_2"

		Label(fig[0,:], "$(Base.typename(typeof(model_config)).name): Iteration to $(round.(fin_res.x; digits=2)).", textsize = 35)
	
		axislegend(ax1; position=:lt)
		axislegend(ax2; position=:lt)
		
		return fig
	end
end

# ╔═╡ a6694cc3-e587-45ff-a8e2-c0ab946df074
md"""
## Footnotes
[^1]: Berkemeier, M., & Peitz, S. (in press). Multiobjective Trust Region Filter Method for Nonlinear Constraints Using Inexact Gradients.
"""

# ╔═╡ 0bb0df31-745f-4cc0-b278-d4d0f02b17a0
md"## Miscellaneous"

# ╔═╡ 19384e46-529f-46af-bc96-bdf8b829bc8e
function vec2str(x, max_entries=typemax(Int),digits=10)
	x_end = min(length(x), max_entries)
	_x = x[1:x_end]
	_, i = findmax(abs.(_x))
	len = length(string(trunc(_x[i]))) + digits + 1
	x_str = "[\n"
	for xi in _x
		x_str *= "\t" * lpad(string(round(xi; digits)), len, " ") * ",\n"
	end
	x_str *= "]"
	return x_str
end

# ╔═╡ 3415f824-105c-472f-9003-e3921b0f58aa
function _optimize(
	mop, x; 
	algo_config = AlgorithmConfig(), 
	model_config = TaylorConfig(),
	logging_callback :: Union{Function, Nothing} = nothing
)

	# Initialization
	Δ = algo_config.delta_init
	μ = algo_config.mu
	filter = Filter()

	# Evaluate `mop`:
	iterate = make_result(x, mop)
	iterate_t = deepcopy(iterate)
	
	database = [iterate,]

	n = fill(NaN, num_vars(mop))
	d = copy(n)
	it_stat = INITIALIZATION

	if logging_callback isa Function
		logging_callback(; 
			iter_index = 0,
			current_iterate = iterate,
			radius = Δ,
			trial_iterate = iterate_t,
			normal_step = n,
			descent_step = d,
			criticality = Inf,
			iter_status = it_stat,
		)
	end

	# Iterate
	k = 1
	while true

		if k >= algo_config.max_iter
			break
		end
				
		@unpack x, fx, hx, gx, res_eq, res_ineq, θ = iterate
		χ = Inf

		if Δ <= algo_config.stop_delta_min
			return iterate, TOLERANCE_X, k, filter, database
		end
		
		@info """
		ITERATION $(k).
		Δ = $(Δ)
		θ = $(θ)
		x = $(vec2str(x))
		fx = $(vec2str(fx))
		"""

		## Construct surrogate models:
		mod_objectives, mod_eq_constraints, mod_ineq_constraints = build_models(model_config, mop, algo_config, database, iterate, Δ)

		if (
			Int(it_stat) >= 0 || it_stat == RESTORATION || depends_on_radius(model_config) || any(isnan.(n))
		)
			if θ > 0
				n = compute_normal_step( mop, iterate, mod_objectives, mod_eq_constraints, mod_ineq_constraints )
			else
				n = zeros(num_vars(mop))
			end
		end

		## check for compatibility:
		n_valid = !(any(isnan.(n))) && !(any(isinf.(n)))
		n_compatible = n_valid &&			
			compatibility_test(mop, algo_config, n, Δ)

		_Δ = Δ
		if !n_compatible
			if it_stat == RESTORATION && n_valid
				## if we already performed restoration, try to increase Δ
				## ‖n‖ ≤ c * Δ * min(1, κ Δ^μ)
				## ‖n‖/c ≤ min(Δ, κ Δ^{1+μ})
				norm_n = norm(n, Inf)
				_Δ_1 = norm_n / algo_config.c_delta
				_Δ_2 = (norm_n / ( algo_config.c_delta * algo_config.c_mu))^(1/(1+μ))
				_Δ = min(_Δ_1, _Δ_2)
				if _Δ <= algo_config.delta_max
					n_compatible = true
				end
			end
		end
		
		if !n_compatible
			@info "Normal step not compatible!"
			## Normal step not compatible => Do restoration
			if it_stat != RESTORATION
				@info "Trying to perform restoration."
				filter_add!(filter, θ,maximum(fx))
				θ_r, d, ret_code  = do_restoration(mop, iterate, n, algo_config.nlopt_restoration_algo)
				succ_ret = ret_code in [
					:SUCCESS,
					:STOPVAL_REACHED,
					:FTOL_REACHED,
					:XTOL_REACHED,
					:MAXEVAL_REACHED,
					:MAXTIME_REACHED,
				]
			else
				succ_ret = false
			end
			
			if !succ_ret
				### could not find restoration step, return unsuccessful
				return iterate, INFEASIBLE, k, filter, database
			end
			@info "Found restoration step $(vec2str(d))"
			## update variables for next iteration
			iterate_t = make_result(x .+ d, mop)
			push!(database, iterate_t)
			_it_stat = RESTORATION

			if logging_callback isa Function
				logging_callback(; 
					iter_index = k,
					radius = Δ,
					current_iterate = iterate,
					trial_iterate = iterate_t,
					normal_step = n,
					descent_step = d,
					criticality = χ,
					iter_status = _it_stat,
				)
			end
			
			it_stat = _it_stat
			iterate = iterate_t
			Δ = _Δ

			k += 1
			continue # proceed to next iteration
		else
			@info "Found compatible normal step $(vec2str(n))"
			@info "Computing descent direction."
			d, χ = compute_descent_step( mop, algo_config, iterate, n, mod_objectives, mod_eq_constraints, mod_ineq_constraints )

			@info "Criticality is $(χ) and descent step is $(vec2str(d))"

			if χ <= algo_config.stop_crit_tol_abs && θ <= algo_config.stop_theta_tol_abs
				return iterate, CRITICAL, k, filter, database
			end

			## Criticality Test
			crit_stat = NOT_RUN
			if θ <= algo_config.eps_theta && χ < algo_config.eps_crit && Δ > algo_config.crit_M * χ
				χ, Δ, n, mod_objectives, mod_eq_constraints, mod_ineq_constraints, crit_stat = criticality_routine(
					mop, model_config, algo_config, 
					database, iterate, χ, Δ, n, 
					mod_objectives, mod_eq_constraints, mod_ineq_constraints
				)
			end
			
			σ = backtrack( mop, algo_config, iterate, Δ, n, d, χ, mod_objectives, mod_eq_constraints, mod_ineq_constraints )
			@info "Trying a descent step with stepsize $(σ) along $(vec2str(d))" 

			# evaluate trial point 
			x_t = x .+ n .+ σ .* d
			iterate_t = make_result( x_t, mop )
			push!(database, iterate_t)
			fx_t, θ_t = iterate_t.fx, iterate_t.θ
			@info """
			θ_t = $(θ_t)
			x_t = $(vec2str(x_t))
			fx_t = $(vec2str(fx_t))
			"""

			_it_stat = INITIALIZATION 	# changed below
			
			_acceptable_for_filter = is_acceptable(filter, θ_t, maximum(fx_t), θ, maximum(fx))
			if _acceptable_for_filter
				mx = mod_objectives(x)
				mx_t = mod_objectives(x_t)
				_model_decrease_constraint_test = if algo_config.strict_acceptance_test
					all( mx .- mx_t .>= algo_config.kappa_theta * θ^algo_config.psi_theta )
				else
					maximum(mx) .- maximum(mx_t) >= algo_config.kappa_theta * θ^algo_config.psi_theta
				end
				rho = if algo_config.strict_acceptance_test
					minimum( (fx .- fx_t) ./ (mx .- mx_t) )
				else
					(maximum(fx) - maximum(fx_t))/(maximum(mx) - maximum(mx_t))
				end
				_sufficient_decrease_test = rho >= algo_config.nu_accept
				if _model_decrease_constraint_test && !_sufficient_decrease_test
					_it_stat = INACCEPTABLE
				end
			else
				_it_stat = FILTER_FAIL
			end

			if _it_stat == FILTER_FAIL || _it_stat == INACCEPTABLE
				_Δ = Δ * algo_config.gamma_shrink_much
			else
				if !_model_decrease_constraint_test
					_it_stat = FILTER_ADD
					filter_add!(filter, θ, maximum(fx))
				end
				_Δ = if !_sufficient_decrease_test
					@assert _it_stat == FILTER_ADD
					Δ * algo_config.gamma_shrink_much
				elseif rho < algo_config.nu_success
					_it_stat = ACCEPTABLE
					Δ * algo_config.gamma_shrink
				else
					_it_stat = SUCCESSFUL
					min( algo_config.delta_max, algo_config.gamma_grow * Δ )
				end
			end
		
			if logging_callback isa Function
				logging_callback(; 
					iter_index = k,
					radius = Δ,
					current_iterate = iterate,
					trial_iterate = iterate_t,
					normal_step = n,
					descent_step = d,
					criticality = χ,
					iter_status = _it_stat,
				)
			end

			@info "Iteration status: $(_it_stat)."
			Δ = _Δ
			it_stat = _it_stat
			k += 1
			
			if Int(it_stat) > 0
				@info "The trial point is accepted!"
				iterate = iterate_t
				
				# trial point as accepted, test relative stopping criteria:
				norm_x = norm( x )
				norm_f = norm( fx )
				change_x = norm( x .- x_t )
				change_f = norm( fx .- fx_t )
				if (
					change_x <= algo_config.stop_xtol_abs || 
					change_x <= algo_config.stop_xtol_rel * norm_x 
				)
					return iterate, TOLERANCE_X, k, filter, database
				end
				if (
					change_f <= algo_config.stop_ftol_abs || 
					change_f <= algo_config.stop_ftol_rel * norm_f
				)
					return iterate, TOLERANCE_F, k, filter, database
				end
			end
		end
	
		if crit_stat == MAX_LOOP_EXIT || crit_stat == CRITICAL_EXIT
			return iterate, CRITICAL, k, filter, database
		elseif crit_stat == TOLERANCE_X_EXIT
			return iterate, TOLERANCE_X, k, filter, database
		end
	end
	return iterate, BUDGET, k, filter, database
end

# ╔═╡ 78eccede-0831-4c76-8454-b9ce028300cf
function optimize(
	mop, x; 
	algo_config = AlgorithmConfig(), 
	model_config = TaylorConfig(),
	logging_callback :: Union{Function, Nothing} = nothing
)
	fin_res, ret_code, num_iters, filter, database = _optimize(mop, x; algo_config, model_config, logging_callback)
	@info "FINISHED OPTIMIZATION AFTER $(num_iters) ITERATIONS."

	return fin_res, ret_code, filter, database
end

# ╔═╡ 8f224886-efe6-4cc6-af9f-682d613b94b4
"""
	do_experiment(mop, x0; model_config=nothing, algo_config=nothing)

Run optimization starting at 2D-Vector `x0`.
"""
function do_experiment(mop, x0; 
	model_config = nothing, algo_config = nothing
)	
		
	if isnothing(model_config)
		model_config = TaylorConfig(; deg=2)
	end
	
	if isnothing(algo_config)
		algo_config = AlgorithmConfig(;
			max_iter = 100,
		)
	end

	# setup the information dict and logging:
	info_dict = Dictionary{Int, IterInfo}()
	logging_callback = get_example_logger(info_dict)

	# call optimization, but log to terminal, not to notebook
	local fin_res, ret
	with_logger( Logging.ConsoleLogger( PlutoRunner.original_stdout ) ) do
		fin_res, ret, _ = optimize(mop, x0;
			model_config, logging_callback, algo_config
		)
	end
	@show ret	
	return mop, model_config, x0, fin_res, ret, info_dict	
end

# ╔═╡ 335a2cca-8f5f-46ce-af9d-e28a3b361986
let
	f1 = x -> 10*sum(x.^2)
	f2 = x -> 10*sum((x.-1).^2)

	mop = MOP(;
		num_vars = 2,
		objectives = [f1, f2]
	)
	mop, model_config, x0, fin_res, ret, info_dict = do_experiment(mop, [.4, 1])

	XX = LinRange(-.5, 1.5, 200)
	YY = XX

	F1 = [ f1([x;y]) for x=XX, y=YY ]
	F2 = [ f2([x;y]) for x=XX, y=YY ]

	fig = Figure(resolution=(400,350))
	ax = Axis3(fig[1,1], azimuth=.6*π)
	
	surface!(ax, XX, YY, F1; color = Makie.RGBAf(BLUE, 0.6))
	surface!(ax, XX, YY, F2; color = Makie.RGBAf(CYAN, 0.6))
	contour!(ax, XX, YY, F1)
	contour!(ax, XX, YY, F2)

	lines!(ax, [(0,0),(1,1)]; color = GRAY, linewidth=5)
	
	scatter!(ax, [(id.result.x[1], id.result.x[2]) for id in info_dict]; color = LIGHTGRAY)
	lines!(ax, [(id.result.x[1], id.result.x[2], id.result.fx[1]) for id in info_dict]; color = BLUE, linewidth = 2)
	lines!(ax, [(id.result.x[1], id.result.x[2], id.result.fx[2]) for id in info_dict]; color = CYAN, linewidth = 2)

	zlims!(ax, 0, 15)

	hidedecorations!(ax)

	fig
end

# ╔═╡ 926eff96-6cf8-411e-a35b-2fd444595219
res_so_problem = do_experiment(so_problem_mop(),[5.0,]; model_config = ExactConfig());

# ╔═╡ c2d5a965-c97a-45ca-8d08-78e32b2ab9f9
plot_so_problem_objective_space(res_so_problem...)

# ╔═╡ bf2b1d24-8352-4991-a473-246fb11ab5c8
"""
	so_rosenbrock_experiment(x0; model_config=nothing, algo_config=nothing)

Run optimization starting at 2D-Vector `x0` and return a Figure that shows the iterations.
"""
function so_rosenbrock_experiment(x0; 
	algo_config=nothing, kwargs...
)	
	# setup the optimization problem
	mop = so_rosenbrock_mop()
	algo_config = if isnothing(algo_config)
		AlgorithmConfig(;max_iter = 1000,)
	else
		algo_config
	end
	return do_experiment(mop, x0; algo_config, kwargs...)
end

# ╔═╡ f188cba6-da11-4652-8287-def1809d6209
res_so_rosenbrock_1 = so_rosenbrock_experiment([0.6, 0.5]; model_config=RbfConfig());

# ╔═╡ a13c1615-874d-4b33-af05-b41d833b44f6
plot_so_rosenbrock_decision_space( res_so_rosenbrock_1... )

# ╔═╡ 1864f087-686b-41df-afe1-8e853b1b3931
res_so_rosenbrock_2 = so_rosenbrock_experiment([-1.45, -0.4]; model_config=RbfConfig());

# ╔═╡ 9ad054f1-f959-4ab4-92c6-be99502a1fe2
plot_so_rosenbrock_decision_space( res_so_rosenbrock_2... )

# ╔═╡ 0eb66c9a-f811-4d90-8949-fd20e8dcfb70
plot_so_rosenbrock_radius(res_so_rosenbrock_2...; it_indices=1:20, do_markers = true)

# ╔═╡ ba4d1b52-e1c9-450a-930f-f8616a92b406
plot_so_rosenbrock_radius(res_so_rosenbrock_2...; do_markers = false)

# ╔═╡ 718fac88-8216-4be6-af8e-1977f04daa7a
res_two_parabola_1 = do_experiment(two_parabola_mop(), [3, 2]; model_config = RbfConfig());

# ╔═╡ 76075f68-de8d-4d2f-b201-93037095e978
plot_two_parabola_iterations(res_two_parabola_1...)

# ╔═╡ db6f88af-38ea-4486-87d4-ae0e0580b95a
res_two_parabola_2 = do_experiment(two_parabola_mop_c(), [3, 2]; model_config = RbfConfig());

# ╔═╡ eff780a2-d21d-4cf1-92c9-a478857c04ae
plot_two_parabola_iterations_c(res_two_parabola_2...)

# ╔═╡ c965104f-bcce-4222-b169-a779e266ec22
res_two_parabola_3 = do_experiment(
	two_parabola_mop_c(), [3, 2]; 
	model_config = TaylorConfig(;deg=2, diff = :AD)
);

# ╔═╡ b7e7bb1e-b73d-4b9e-b7e6-4763f1d4bbbb
plot_two_parabola_iterations_c(res_two_parabola_3...)

# ╔═╡ 63980505-a6a0-455f-82b2-cc2cfd321670
# ╠═╡ disabled = true
#=╠═╡
let
	lb = fill(-4, 2)
	ub = fill(4, 2)
	
	mop = MOP(;
		num_vars = 2,
		lb, ub,	
		objectives = [ 
			x -> sum( (x .+ 1 ).^2 ),
			x -> sum( (x .- 1 ).^2 )
		],
		nl_ineq_constraints = [
			x -> sum( (x .- 2).^2 ) - .5
		]
	)
	x0 = lb .+ (ub .- lb ) .* rand(2)
	#x0 = [ -3.1307670353, 3.8254871275 ]
      
	algo_config = AlgorithmConfig(; 
		max_iter = 100,
	)

	info = Dictionary{Int, IterInfo}()
	logging_callback = get_example_logger(info)

	local fin_res, ret
	with_logger( Logging.ConsoleLogger( PlutoRunner.original_stdout ) ) do
		fin_res, ret, _ = optimize(mop, x0;
			#model_config=RbfConfig(),
			model_config=TaylorConfig(;deg=2),
			algo_config, logging_callback
		)
	end
	@show ret

	fig = Figure()

	its = collect(keys(info))

	ax1 = Axis(fig[1,1])
	lines!(ax1, [(-1, -1), (1,1)])

	xs = [ Tuple(info[it].result.x) for it in its ]
	scatter!(ax1, xs)
	lines!(ax1, xs)
	scatter!(ax1, Tuple(x0); color = :orange)
	scatter!(ax1, Tuple(fin_res.x); color = :red)

	xx = LinRange(mop.lb[1], mop.ub[1], 200)
	yy = LinRange(mop.lb[2], mop.ub[2], 200)
	for cfunc in mop.nl_ineq_constraints
		zz = [ cfunc([x;y]) <= 0 for x = xx, y = yy ]
		image!(xx, yy, zz; colormap=[Makie.RGBA(0,0,0,0), Makie.RGBA(0,0,0,0.2)])
	end
	
	ax2 = Axis(fig[1,2])
	fs = [ Tuple(info[it].result.fx) for it in its ]
	scatter!(ax2, fs)
	lines!(ax2, fs)

	fig
end
  ╠═╡ =#

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
Dictionaries = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Logging = "56ddb016-857b-54e1-b83d-db4d58db5568"
NLopt = "76087f3c-5699-56af-9a33-bf431cd00edd"
OSQP = "ab2f91bb-94b4-55e3-9ba0-7f65df51de79"
Parameters = "d96e819e-fc66-5662-9728-84c9c7592b0a"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
RadialBasisFunctionModels = "48790e7e-73b2-491a-afa5-62818081adcb"

[compat]
CairoMakie = "~0.8.13"
Dictionaries = "~0.3.23"
FiniteDiff = "~2.15.0"
ForwardDiff = "~0.10.32"
JuMP = "~1.1.1"
LaTeXStrings = "~1.3.0"
NLopt = "~0.6.5"
OSQP = "~0.8.0"
Parameters = "~0.12.3"
PlutoUI = "~0.7.39"
RadialBasisFunctionModels = "~0.3.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.3"
manifest_format = "2.0"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.AbstractTrees]]
git-tree-sha1 = "5c0b629df8a5566a06f5fef5100b53ea56e465a0"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.2"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e81c509d2c8e49592413bfb0bb3b08150056c79d"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.1"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "40debc9f72d0511e12d817c7ca06a721b6423ba3"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.17"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Automa]]
deps = ["Printf", "ScanByte", "TranscodingStreams"]
git-tree-sha1 = "d50976f217489ce799e366d9561d56a98a30d7fe"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "0.8.2"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "4c10eee4af024676200bc7752e536f858c6b8f93"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.1"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "d0b3f8b4ad16cb0a2988c6788646a5e6a17b6b1b"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.0.5"

[[deps.CairoMakie]]
deps = ["Base64", "Cairo", "Colors", "FFTW", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "SHA"]
git-tree-sha1 = "387e0102f240244102814cf73fe9fbbad82b9e9e"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.8.13"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRules]]
deps = ["ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics", "StructArrays"]
git-tree-sha1 = "2bebf7552262a9753d3600b719cc7fefdbd7bb21"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.43.2"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "80ca332f6dcb2508adba68f22f551adb2d00a624"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.3"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "2e62a725210ce3c3c2e1a3080190e7ca491f18d7"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.7.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "1fd869cc3875b57347f7027521f561cf46d1fcd8"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.19.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "924cdca592bc16f14d2f7006754a621735280b74"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.1.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "59d00b3139a9de4eb961057eabb65ac6522be954"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.4.0"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Dictionaries]]
deps = ["Indexing", "Random"]
git-tree-sha1 = "aeae0d703e62b18aca622e972500077c64bc04e2"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.3.23"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "28d605d9a0ac17118fe2c5e9ce0fbb76c3ceb120"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.11.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "aafa0665e3db0d3d0890cdc8191ea03dc279b042"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.66"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "5158c2b41018c5f7eb1470d558127ac274eca0c9"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.1"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.Extents]]
git-tree-sha1 = "5e1e4c53fa39afe63a7d356e30452249365fba99"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.1"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "ccd479984c7838684b3ac204b716c89955c76623"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "90630efff0894f8142308e334473eba54c433549"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.5.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "94f5101b96d2d968ace56f7f2db19d0a5f592e28"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.15.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

[[deps.FiniteDiff]]
deps = ["ArrayInterfaceCore", "LinearAlgebra", "Requires", "Setfield", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "5a2cff9b6b77b33b89f3d97a4d367747adce647e"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.15.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "187198a4ed8ccd7b5d99c41b69c679269ea2b2d4"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.32"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "cabd77ab6a6fdff49bfd24af2ebe76e6e018a2b4"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.0.0"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "b5c7fe9cea653443736d264b85466bad8c574f4a"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.9.9"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "73145f1d724b5ee0e90098aec39a65e9697429a6"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.4.2"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "d88b17a38322e153c519f5a9ed8d91e9baa03d8f"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.1"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "fb28b5dc239d0174d7297310ef7b84a11804dfab"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.0.1"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "a7a97895780dab1085a97769316aa348830dc991"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.3"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "53c7e69a6ffeb26bd594f5a1421b889e7219eeaa"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.9.0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "af14a478780ca78d5eb9908b263023096c2b9d64"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.6"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "acf614720ef026d38400b3817614c45882d75500"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.4"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "342f789fd041a55166764c351da1710db97ce0e0"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.6"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "87f7662e03a649cffa2e05bf19c303e168732d3e"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.2+0"

[[deps.Indexing]]
git-tree-sha1 = "ce1566720fd6b19ff3411404d4b977acd4814f9f"
uuid = "313cdc1a-70c2-5d6a-ae34-0150d3930a38"
version = "1.1.1"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "23e651bbb8d00e9971015d0dd306b780edbdb6b9"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.3"

[[deps.IntervalSets]]
deps = ["Dates", "Random", "Statistics"]
git-tree-sha1 = "57af5939800bce15980bddd2426912c4f83012d8"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.1"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "a77b273f1ddec645d1b7c4fd5fb98c8f90ad10a5"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.1"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.JuMP]]
deps = ["Calculus", "DataStructures", "ForwardDiff", "LinearAlgebra", "MathOptInterface", "MutableArithmetics", "NaNMath", "OrderedCollections", "Printf", "SparseArrays", "SpecialFunctions"]
git-tree-sha1 = "534adddf607222b34a0a9bba812248a487ab22b7"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.1.1"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "9816b296736292a80b9a3200eb7fbb57aaa3917a"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.5"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "e7e9184b0bf0158ac4e4aa9daf00041b5909bf1a"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "4.14.0"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg", "TOML"]
git-tree-sha1 = "771bfe376249626d3ca12bcd58ba243d3f961576"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.16+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Lazy]]
deps = ["MacroTools"]
git-tree-sha1 = "1370f8202dac30758f3c345f9909b97f53d87d3f"
uuid = "50d2b5c4-7a5e-59d5-8109-a42b560f39c0"
version = "0.15.1"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "361c2b088575b07946508f135ac556751240091c"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.17"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "e595b205efd49508358f7dc670a940c790204629"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.0.0+0"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "16fa7c2e14aa5b3854bc77ab5f1dbe2cdc488903"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.6.0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Makie]]
deps = ["Animations", "Base64", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Distributions", "DocStringExtensions", "FFMPEG", "FileIO", "FixedPointNumbers", "Formatting", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageIO", "IntervalSets", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MakieCore", "Markdown", "Match", "MathTeXEngine", "Observables", "OffsetArrays", "Packing", "PlotUtils", "PolygonOps", "Printf", "Random", "RelocatableFolders", "Serialization", "Showoff", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "UnicodeFun"]
git-tree-sha1 = "b0323393a7190c9bf5b03af442fc115756df8e59"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.17.13"

[[deps.MakieCore]]
deps = ["Observables"]
git-tree-sha1 = "fbf705d2bdea8fc93f1ae8ca2965d8e03d4ca98c"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.4.0"

[[deps.MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.Match]]
git-tree-sha1 = "1d9bc5c1a6e7ee24effb93f175c9342f9154d97f"
uuid = "7eb4fadd-790c-5f42-8a69-bfa0b872bfbf"
version = "1.2.0"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "Printf", "SparseArrays", "SpecialFunctions", "Test", "Unicode"]
git-tree-sha1 = "e652a21eb0b38849ad84843a50dcbab93313e537"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.6.1"

[[deps.MathProgBase]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9abbe463a1e9fc507f12a69e7f29346c2cdc472c"
uuid = "fdba3010-5040-5b88-9595-932c9decdf73"
version = "0.7.8"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "Test"]
git-tree-sha1 = "114ef48a73aea632b8aebcb84f796afcc510ac7c"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.4.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Memoization]]
deps = ["MacroTools"]
git-tree-sha1 = "55dc27dc3d663900d1d768822528960acadc012a"
uuid = "6fafb56a-5788-4b4e-91ca-c0cea6611c73"
version = "0.1.14"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.MultivariatePolynomials]]
deps = ["ChainRulesCore", "DataStructures", "LinearAlgebra", "MutableArithmetics"]
git-tree-sha1 = "393fc4d82a73c6fe0e2963dd7c882b09257be537"
uuid = "102ac46a-7ee4-5c85-9060-abc95bfdeaa3"
version = "0.4.6"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "4e675d6e9ec02061800d6cfb695812becbd03cdf"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.0.4"

[[deps.NLopt]]
deps = ["MathOptInterface", "MathProgBase", "NLopt_jll"]
git-tree-sha1 = "5a7e32c569200a8a03c3d55d286254b0321cd262"
uuid = "76087f3c-5699-56af-9a33-bf431cd00edd"
version = "0.6.5"

[[deps.NLopt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9b1f15a08f9d00cdb2761dcfa6f453f5d0d6f973"
uuid = "079eb43e-fd8e-5478-9966-2cf3e3edb778"
version = "2.7.1+0"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore"]
git-tree-sha1 = "18efc06f6ec36a8b801b23f076e3c6ac7c3bf153"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OSQP]]
deps = ["Libdl", "LinearAlgebra", "MathOptInterface", "OSQP_jll", "SparseArrays"]
git-tree-sha1 = "3514d0aff03027a9c1b0b312151619c9feec412a"
uuid = "ab2f91bb-94b4-55e3-9ba0-7f65df51de79"
version = "0.8.0"

[[deps.OSQP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d0f73698c33e04e557980a06d75c2d82e3f0eb49"
uuid = "9c4f68bf-6205-5545-a508-2878b064d984"
version = "0.600.200+0"

[[deps.Observables]]
git-tree-sha1 = "dfd8d34871bc3ad08cd16026c1828e271d554db9"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.1"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "1ea784113a6aa054c5ebd95945fa5e52c2f378e7"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.7"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "923319661e9a22712f24596ce81c54fc0366f304"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.1+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e60321e3f2616584ff98f0a4f18d98ae6f89bbb3"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.17+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "e925a64b8585aa9f4e3047b8d2cdc3f0e79fd4e4"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.16"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "1155f6f937fa2b94104162f01fa400e192e4272f"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.4.2"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "03a7a85b76381a3d04c7a1656039197e70eda03d"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.11"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a121dfbba67c94a5bec9dde613c3d0cbcf3a12b"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.50.3+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0044b23da09b5608b4ecacb4e5e6c6332f833a7e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.2"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "a7a7e1a88853564e551e4eba8650f8c38df79b37"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.1.1"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "9888e59493658e476d3073f1ce24348bdc086660"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8d1f54886b9037091edf146b517989fc4a09efec"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.39"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.RadialBasisFunctionModels]]
deps = ["ChainRules", "ForwardDiff", "Lazy", "LinearAlgebra", "MLJModelInterface", "Memoization", "Parameters", "StaticArrays", "StaticPolynomials", "Tables", "ThreadSafeDicts", "Zygote"]
git-tree-sha1 = "84cb167091cd6c61e65b85f4bad80d21bffe4a8e"
uuid = "48790e7e-73b2-491a-afa5-62818081adcb"
version = "0.3.4"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "22c5201127d7b243b9ee1de3b43c408879dff60f"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.3.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.SIMD]]
git-tree-sha1 = "7dbc15af7ed5f751a82bf3ed37757adf76c32402"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.4.1"

[[deps.ScanByte]]
deps = ["Libdl", "SIMD"]
git-tree-sha1 = "2436b15f376005e8790e318329560dcc67188e84"
uuid = "7b38b023-a4d7-4c5e-8d43-3f3097f304eb"
version = "0.3.3"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e5364b687e552d73543cf09e583b944eaffff6c4"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "8fb59825be681d451c246a795117f317ecbcaa28"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.2"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "23368a3313d12a2326ad0035f0db0c0966f438ef"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "66fe9eb253f910fe8cf161953880cfdaef01cdf0"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.0.1"

[[deps.StaticPolynomials]]
deps = ["LinearAlgebra", "MultivariatePolynomials", "StaticArrays"]
git-tree-sha1 = "b3c964e9cad2ac5a519ea3a667c2663e3ed8a4c0"
uuid = "62e018b1-6e46-5407-a5a7-97d4fbcae734"
version = "1.3.5"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "30b9236691858e13f167ce829490a68e1a597782"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.2.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "ec47fb6069c57f1cee2f67541bf8f23415146de7"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.11"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadSafeDicts]]
deps = ["Distributed"]
git-tree-sha1 = "a1a1841ef6bef85f40e5917290f3b950eee341c2"
uuid = "4239201d-c60e-5e0a-9702-85d713665ba7"
version = "0.0.3"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "fcf41697256f2b759de9380a7e8196d6516f0310"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.0"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "58443b63fb7e465a8a7210828c91c08b92132dff"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.14+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "91822d41345b9b9b84babe4debd18dd6ccf45311"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.43"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "78736dab31ae7a53540a6b752efc61f77b304c5b"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.8.6+1"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"
"""

# ╔═╡ Cell order:
# ╟─d4a90042-f522-42ff-8260-7965591278c0
# ╠═6f66fc66-4066-490e-abd7-8ebb2a605bdb
# ╠═85be4fe0-8dc3-43af-83a4-48bfc0c24628
# ╠═7d7ec467-3c22-4c8a-9ea1-a10e1ee0179b
# ╠═24a47883-c281-4aa1-859f-714a3788a2d3
# ╠═2995a784-a402-4c08-9bed-bd6c7d5fdfae
# ╠═b80eec63-10aa-4e77-a657-e0e12239dc15
# ╠═52499a91-400b-4371-a7a1-d6d924579c48
# ╠═3c5381fb-0944-4b92-9837-5dddfe4bce11
# ╠═e51f949d-6483-45aa-a18a-95d8245b63aa
# ╠═a40db56a-8466-4aa1-95d5-7560e663098a
# ╟─68ada2bc-17e0-11ed-3f7b-414dc56ffd57
# ╟─335a2cca-8f5f-46ce-af9d-e28a3b361986
# ╟─8c701b7c-65e4-4dd2-9a79-b85b8f7aeae0
# ╟─5280e000-9bae-48b6-a2c6-d73a720a20b9
# ╟─1d6c67d5-2b31-4159-8b58-3bb5dc916f8b
# ╟─4cfe8b89-0693-4e58-aec0-8612d6d1e1f0
# ╟─e460c56c-879f-450c-a29a-b7eb8fbf04b7
# ╠═89b9850a-99db-44cd-9fa5-d44b4bcdde89
# ╠═b7510970-17e0-425b-b908-133015431191
# ╠═26a945e3-eb9f-406c-aa0a-6c5cfd9181b8
# ╟─f219963d-2978-4ae2-acc9-2425d7f8ecec
# ╟─5e7ec552-d98f-40fe-9976-618e9b1706dc
# ╠═bf20e460-e5f4-4925-8e37-8aed02abdaa3
# ╟─8cf1aac5-92f9-42e3-8090-c01d85b72398
# ╠═0513e905-f174-40f7-a667-5ba155dccc5a
# ╟─e2a1619b-9dfc-4f92-bf04-4de5a660506b
# ╠═5b963584-3568-4710-ab13-2136d8a768fb
# ╟─541008ba-7cf5-43ac-b98f-e4e4233e936f
# ╠═e401ef02-6b83-4231-ba5e-50d59b1aad0d
# ╠═ae98e9e4-c4a2-4b21-8755-cea1ce0d28cf
# ╟─5e4824ac-e1b6-4d05-92f7-a116e537d111
# ╟─5c18ffe8-a6ae-4031-9c46-05006d6dddf0
# ╠═8508fdf0-df11-4cec-bda6-6e7e19249dd9
# ╟─95493635-10a8-49a9-96e4-529e09999837
# ╟─fcc68c1b-b0e9-41cd-845a-fa9d3fde33af
# ╠═3b4fe747-64b0-4b0a-9672-3f19724a1324
# ╟─15dc4fb7-622a-41b3-a44e-d40c8837abfe
# ╟─8fb3dd12-01ba-47ec-b65a-6aca6bbd9d20
# ╠═87533b4e-7eed-4fa7-b9e6-3e2aa7bffd0d
# ╟─ba73a960-2cfd-49b6-ac7c-fb40eb678652
# ╠═1d365a8b-89f2-48d9-a214-c7ea416dfa98
# ╟─0c7bffec-3a02-4009-9e24-34113c4d9910
# ╟─fc089ee6-c8e5-494e-a9ce-d7ef8eed0c80
# ╟─7f5aa652-c938-4a5b-91e1-2d2abac84ffe
# ╠═d37ca69d-8198-46e0-b63b-dc324315e27e
# ╟─4aa512b9-4597-4087-a20b-5a0f39b04f70
# ╠═f5c076bd-62d6-4daa-af94-0501e615ad8f
# ╠═86480276-b64f-4706-a011-120669514e89
# ╠═08cf28fb-e332-45d0-915a-c07a7488da73
# ╟─deb16981-33a8-4f8b-ab0d-b4ac745b056c
# ╠═baad6c8b-c181-42d9-892e-739bf23624ad
# ╠═065a78a6-6b12-408e-acc9-4386d82bbe59
# ╠═044ed139-5f2c-4720-801a-67726904267c
# ╟─01526a4d-d916-49ca-a993-f9c34e28454d
# ╠═d2e8b51e-9951-4207-8fe2-6d4b53dd416a
# ╠═03521fe7-5c6b-4d3a-88fc-12f411009e22
# ╠═7d670e3c-1105-425f-8a31-ffb2dc4d32ab
# ╟─c1197240-d173-4bf1-a7a4-edc10cc49684
# ╠═f5fc49fc-3ed3-4fb2-a677-56f5dd1ed808
# ╟─24bcd634-a199-46d8-a3a1-edfb18d71a3e
# ╟─dd51a28b-f8e3-48dd-94d4-284996b0c558
# ╠═1cd765dc-8739-4dd1-8dd6-0f13d0622332
# ╠═8926543f-854a-45be-91f5-1598503e7c24
# ╟─4aae416a-e005-4b16-b8a5-90bf6360dda3
# ╟─6846849a-994d-40a4-a84b-7f2004364844
# ╠═5a07763d-c50e-4dab-affe-e83e7f967d54
# ╟─cd63c696-c877-412e-b2ea-b0b919e26018
# ╟─8ed4cd89-8be9-4c8a-a354-d85f5764f812
# ╠═badb3fe0-9739-4412-a3f1-e1a84d0d5ef4
# ╠═1b791760-5241-463e-94d1-4ee50bd7e4dc
# ╠═6d0aec2c-8858-4eab-b4f0-08896b24e851
# ╠═a30efb66-3628-4f7b-9582-b06051149bbd
# ╠═e8325749-7522-459b-9d43-a332baf0bebf
# ╟─29e8bcf0-345b-47d2-9e0d-d93ca4eb75ff
# ╟─77c856a2-f2e8-4335-864c-2dd8b4c13773
# ╟─0a655b39-01df-4330-8dda-f89c1511d0a5
# ╠═366b2ede-a04b-41fd-8a19-f71d0e630658
# ╟─7f6c954a-d85f-44a0-839f-6dcf855a8cbd
# ╟─7812192f-1e72-4cd8-9b61-ed2e6b7588ef
# ╟─2d865aad-62d3-4711-b2aa-12be5a1959e0
# ╟─b6ce5fc0-3011-4ca0-87b8-8cb8f33f42e0
# ╟─690aee10-045c-49cc-bd98-5bfc3206a9a5
# ╟─f81657c9-09c0-4689-a2ed-783858349416
# ╟─e1f0ad5f-a601-48af-906c-2190b4c1bd2f
# ╟─529f9f7f-9a66-4020-aa8b-0d3e63372229
# ╠═8ec19cd2-30fa-4815-8711-8b98afec3a82
# ╠═18c93e20-2fe3-473f-a206-9b7abf45d6b5
# ╟─7525f4bf-0705-436c-aa97-dbecac5aca1b
# ╟─b4b4620e-b72e-402b-994b-c6db37199a42
# ╠═587f9a6b-c170-40e7-8b42-e60539cffb60
# ╠═e07d9869-1ca5-486d-9173-192f52d7bd1a
# ╟─1a5a691e-b6b8-4996-bfdd-34969fc1bfc1
# ╠═5a99426d-4e04-4032-b19c-69c5492e3d3b
# ╠═f8ec520f-f678-448f-a0bd-10cc45262dfc
# ╟─d66b7e85-3b70-475f-b634-a9bd58619531
# ╠═4e5954bb-4615-46e6-9a20-701afa0175eb
# ╠═d73b2468-4d9f-47e7-940f-f02bf6901dd8
# ╠═78eccede-0831-4c76-8454-b9ce028300cf
# ╠═3415f824-105c-472f-9003-e3921b0f58aa
# ╟─a965d55f-f21c-4cab-af43-1ce8bb14b28a
# ╟─0c18e387-e5a0-44a0-b0ea-26c9ad637479
# ╠═4fee47cb-4702-4dbb-92ae-e928032cfb2c
# ╟─9fdd78bb-d736-47d2-91cd-9672aed9274c
# ╟─9971cc69-fbe7-45da-835f-6dd7eb129d8e
# ╟─b7b9c533-b3b8-482e-a615-9be2fee76cbd
# ╟─c7c3e8ce-a54e-44a1-8d47-7351f1f127c6
# ╟─6b7b7774-e3ca-4903-8ee3-e9429995b9c6
# ╟─8f224886-efe6-4cc6-af9f-682d613b94b4
# ╟─28a5ca66-cce7-477a-9b18-606aaa14b195
# ╠═926eff96-6cf8-411e-a35b-2fd444595219
# ╠═c2d5a965-c97a-45ca-8d08-78e32b2ab9f9
# ╟─665e607b-8c85-4914-ac97-ffa9eedec0e0
# ╟─f13b5aa6-aa5c-418b-9293-f960cec2fb5b
# ╟─3081e8c0-49e2-413d-a628-58308ed62600
# ╟─bf2b1d24-8352-4991-a473-246fb11ab5c8
# ╟─bdc0e9e4-e93a-47ed-a72d-0ea7b71af3ee
# ╟─13cd8e50-5824-41b9-856f-2c81da0bae77
# ╠═f188cba6-da11-4652-8287-def1809d6209
# ╟─a13c1615-874d-4b33-af05-b41d833b44f6
# ╟─e75cd5b9-13c3-48d6-bbe8-25df557d668c
# ╠═1864f087-686b-41df-afe1-8e853b1b3931
# ╟─9ad054f1-f959-4ab4-92c6-be99502a1fe2
# ╟─26b521b4-5cce-4286-8ac9-f3cb0c27967c
# ╠═0eb66c9a-f811-4d90-8949-fd20e8dcfb70
# ╠═ba4d1b52-e1c9-450a-930f-f8616a92b406
# ╟─e43a665f-a8d4-4ba3-bb10-ac7a84d7db9c
# ╟─86d58d29-120e-4091-ad80-6dda94d7a9dd
# ╟─2eb7f4f9-d128-4e1b-a264-185be9f7ccaa
# ╟─42ff0182-8621-4c09-aa72-80cbdc8ca2fc
# ╟─61ade99e-6292-4cd0-846a-5dc428bcb50d
# ╟─6c02cfa7-19ec-4e8b-9cfd-54d71a400f31
# ╠═718fac88-8216-4be6-af8e-1977f04daa7a
# ╠═76075f68-de8d-4d2f-b201-93037095e978
# ╟─663d60fc-8bcd-4284-baa6-f5ab50c71bbf
# ╠═db6f88af-38ea-4486-87d4-ae0e0580b95a
# ╠═eff780a2-d21d-4cf1-92c9-a478857c04ae
# ╠═7e562b08-bcb3-4f01-8cea-6f5a799b7e7e
# ╠═c965104f-bcce-4222-b169-a779e266ec22
# ╠═b7e7bb1e-b73d-4b9e-b7e6-4763f1d4bbbb
# ╟─0e9f4e15-5550-439e-b964-2b84aade660f
# ╟─a6694cc3-e587-45ff-a8e2-c0ab946df074
# ╟─0bb0df31-745f-4cc0-b278-d4d0f02b17a0
# ╟─19384e46-529f-46af-bc96-bdf8b829bc8e
# ╟─63980505-a6a0-455f-82b2-cc2cfd321670
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
