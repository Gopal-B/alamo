#ifndef NUMERIC_FLUXHANDLER_H
#define NUMERIC_FLUXHANDLER_H

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include "Set/Set.H"
#include "Integrator/ScimitarX.H"
#include "Util/Util.H"
#include "Util/ScimitarX_Util.H"
#include "Model/Fluid/Fluid.H" 
#include "Numeric/Stencil.H" 
#include "Numeric/NumericTypes.H"
#include "Model/Fluid/FiveEquation.H"
#include "Numeric/SolverCapabilities.H"
#include "Numeric/IntegratorVariableAccessLayer.H"
#include "Numeric/SymmetryPreservingRoeAveragingOperations.H"
#include <memory>
#include <cmath>
#include <algorithm>

//forward declaration for ScimitarX
namespace Integrator {
    class ScimitarX;

}

namespace Numeric {

// ClampIndices: Clamps indices to valid bounds (branch-free logic)
AMREX_GPU_DEVICE
inline void ClampIndices(int indices[3], const int lower_bounds[3], const int upper_bounds[3]) {
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        indices[d] = lower_bounds[d] + std::min(std::max(indices[d] - lower_bounds[d], 0), upper_bounds[d] - lower_bounds[d]);
    }
}



// ShiftAndClampIndices: Shifts and clamps indices (branch-free logic)
AMREX_GPU_DEVICE
inline void ShiftAndClampIndices(int indices[3], const int offsets[3],
                                const int lower_bounds[3], const int upper_bounds[3], int direction) {
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        int shifted_index = indices[d] + (d == direction ? offsets[d] : 0);
        indices[d] = lower_bounds[d] + std::min(std::max(shifted_index - lower_bounds[d], 0), upper_bounds[d] - lower_bounds[d]);
    }
}
constexpr int DI5_MAX_NCOMP = 7;

// ========================= DI5 positivity helpers =========================

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool DI5Finite(Set::Scalar x) {
    return std::isfinite(static_cast<double>(x));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Set::Scalar DI5Clamp01(Set::Scalar x) {
    return amrex::max(Set::Scalar(0.0), amrex::min(Set::Scalar(1.0), x));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Set::Scalar DI5Blend(Set::Scalar theta, Set::Scalar hi, Set::Scalar lo) {
    return theta * hi + (Set::Scalar(1.0) - theta) * lo;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool DI5ThermoAdmissible(
    Set::Scalar m1,
    Set::Scalar m2,
    Set::Scalar p,
    Set::Scalar alpha,
    const Model::Fluid::FiveEquation::FluidMixture* mixture_model)
{
    constexpr Set::Scalar eps_m   = Set::Scalar(1.0e-14);
    constexpr Set::Scalar eps_a   = Set::Scalar(1.0e-12);
    constexpr Set::Scalar eps_c2  = Set::Scalar(1.0e-14);

    if (mixture_model == nullptr) return false;
    if (!DI5Finite(m1) || !DI5Finite(m2) || !DI5Finite(p) || !DI5Finite(alpha)) return false;

    if (m1 <= eps_m || m2 <= eps_m) return false;
    if (alpha <= eps_a || alpha >= Set::Scalar(1.0) - eps_a) return false;

    const Set::Scalar rho = m1 + m2;
    if (!DI5Finite(rho) || rho <= eps_m) return false;

    const Set::Scalar gm  = mixture_model->gamma_mix(alpha);
    const Set::Scalar pim = mixture_model->pi_mix(alpha);

    if (!DI5Finite(gm) || !DI5Finite(pim)) return false;

    const Set::Scalar c2 = gm * (p + pim) / rho;


    if (!DI5Finite(c2) || c2 <= eps_c2) return false;


    // constexpr Set::Scalar eps_p = Set::Scalar(1.0e-14);
    // if (p <= eps_p) return false;
    return true;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool DI5FaceStateAdmissible(
    Set::Scalar m1,
    Set::Scalar m2,
    Set::Scalar u,
    Set::Scalar v,
    Set::Scalar w,
    Set::Scalar p,
    Set::Scalar alpha,
    const Model::Fluid::FiveEquation::FluidMixture* mixture_model)
{
    if (!DI5Finite(m1) || !DI5Finite(m2) ||
        !DI5Finite(u)  || !DI5Finite(v)  || !DI5Finite(w) ||
        !DI5Finite(p)  || !DI5Finite(alpha)) {
        return false;
    }

    return DI5ThermoAdmissible(m1, m2, p, alpha, mixture_model);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Set::Scalar DI5ThetaForLowerBound(Set::Scalar q_hi, Set::Scalar q_lo, Set::Scalar qmin)
{
    if (!DI5Finite(q_hi) || !DI5Finite(q_lo)) return Set::Scalar(0.0);
    if (q_hi >= qmin) return Set::Scalar(1.0);
    if (q_lo <= qmin) return Set::Scalar(0.0);

    const Set::Scalar theta = (q_lo - qmin) / (q_lo - q_hi);
    return amrex::max(Set::Scalar(0.0), amrex::min(Set::Scalar(1.0), theta));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Set::Scalar DI5ThetaForUpperBound(Set::Scalar q_hi, Set::Scalar q_lo, Set::Scalar qmax)
{
    if (!DI5Finite(q_hi) || !DI5Finite(q_lo)) return Set::Scalar(0.0);
    if (q_hi <= qmax) return Set::Scalar(1.0);
    if (q_lo >= qmax) return Set::Scalar(0.0);

    const Set::Scalar theta = (qmax - q_lo) / (q_hi - q_lo);
    return amrex::max(Set::Scalar(0.0), amrex::min(Set::Scalar(1.0), theta));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool DI5AdmissibleAtTheta(
    Set::Scalar theta,
    Set::Scalar m1_hi, Set::Scalar m2_hi,
    Set::Scalar u_hi,  Set::Scalar v_hi,  Set::Scalar w_hi,
    Set::Scalar p_hi,  Set::Scalar a_hi,
    Set::Scalar m1_lo, Set::Scalar m2_lo,
    Set::Scalar u_lo,  Set::Scalar v_lo,  Set::Scalar w_lo,
    Set::Scalar p_lo,  Set::Scalar a_lo,
    const Model::Fluid::FiveEquation::FluidMixture* mixture_model)
{
    const Set::Scalar m1 = DI5Blend(theta, m1_hi, m1_lo);
    const Set::Scalar m2 = DI5Blend(theta, m2_hi, m2_lo);
    const Set::Scalar u  = DI5Blend(theta, u_hi,  u_lo);
    const Set::Scalar v  = DI5Blend(theta, v_hi,  v_lo);
    const Set::Scalar w  = DI5Blend(theta, w_hi,  w_lo);
    const Set::Scalar p  = DI5Blend(theta, p_hi,  p_lo);
    const Set::Scalar a  = DI5Blend(theta, a_hi,  a_lo);

    return DI5FaceStateAdmissible(m1, m2, u, v, w, p, a, mixture_model);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Set::Scalar DI5ComputeTheta(
    Set::Scalar m1_hi, Set::Scalar m2_hi,
    Set::Scalar u_hi,  Set::Scalar v_hi,  Set::Scalar w_hi,
    Set::Scalar p_hi,  Set::Scalar a_hi,
    Set::Scalar m1_lo, Set::Scalar m2_lo,
    Set::Scalar u_lo,  Set::Scalar v_lo,  Set::Scalar w_lo,
    Set::Scalar p_lo,  Set::Scalar a_lo,
    const Model::Fluid::FiveEquation::FluidMixture* mixture_model)
{
    constexpr Set::Scalar eps_m = Set::Scalar(1.0e-14);
    constexpr Set::Scalar eps_a = Set::Scalar(1.0e-12);

    // If already admissible, do nothing
    if (DI5FaceStateAdmissible(m1_hi, m2_hi, u_hi, v_hi, w_hi, p_hi, a_hi, mixture_model)) {
        return Set::Scalar(1.0);
    }

    // If any high-order component is NaN/Inf, go straight to first-order fallback
    if (!DI5Finite(m1_hi) || !DI5Finite(m2_hi) ||
        !DI5Finite(u_hi)  || !DI5Finite(v_hi)  || !DI5Finite(w_hi) ||
        !DI5Finite(p_hi)  || !DI5Finite(a_hi)) {
        return Set::Scalar(0.0);
    }

    // First-order state is assumed admissible by construction from VariableBuffer
    Set::Scalar theta = Set::Scalar(1.0);

    // Enforce linear constraints first
    theta = amrex::min(theta, DI5ThetaForLowerBound(m1_hi, m1_lo, eps_m));
    theta = amrex::min(theta, DI5ThetaForLowerBound(m2_hi, m2_lo, eps_m));
    theta = amrex::min(theta, DI5ThetaForLowerBound(a_hi,  a_lo,  eps_a));
    theta = amrex::min(theta, DI5ThetaForUpperBound(a_hi,  a_lo,  Set::Scalar(1.0) - eps_a));

    if (theta <= Set::Scalar(0.0)) return Set::Scalar(0.0);

    // If thermo is already fine at this theta, keep it
    if (DI5AdmissibleAtTheta(theta,
                             m1_hi,m2_hi,u_hi,v_hi,w_hi,p_hi,a_hi,
                             m1_lo,m2_lo,u_lo,v_lo,w_lo,p_lo,a_lo,
                             mixture_model)) {
        return theta;
    }

    // Otherwise do bisection on [0, theta] for the largest admissible theta
    Set::Scalar lo = Set::Scalar(0.0);
    Set::Scalar hi = theta;

    for (int it = 0; it < 24; ++it) {
        const Set::Scalar mid = Set::Scalar(0.5) * (lo + hi);

        if (DI5AdmissibleAtTheta(mid,
                                 m1_hi,m2_hi,u_hi,v_hi,w_hi,p_hi,a_hi,
                                 m1_lo,m2_lo,u_lo,v_lo,w_lo,p_lo,a_lo,
                                 mixture_model)) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    return lo;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void DI5ApplyThetaLimit(
    Set::Scalar& m1, Set::Scalar& m2,
    Set::Scalar& u,  Set::Scalar& v,  Set::Scalar& w,
    Set::Scalar& p,  Set::Scalar& a,
    Set::Scalar theta,
    Set::Scalar m1_lo, Set::Scalar m2_lo,
    Set::Scalar u_lo,  Set::Scalar v_lo,  Set::Scalar w_lo,
    Set::Scalar p_lo,  Set::Scalar a_lo)
{
    if (theta >= Set::Scalar(1.0)) return;

    m1 = DI5Blend(theta, m1, m1_lo);
    m2 = DI5Blend(theta, m2, m2_lo);
    u  = DI5Blend(theta, u,  u_lo);
    v  = DI5Blend(theta, v,  v_lo);
    w  = DI5Blend(theta, w,  w_lo);
    p  = DI5Blend(theta, p,  p_lo);
    a  = DI5Blend(theta, a,  a_lo);
}

// ======================= end DI5 positivity helpers =======================

// Base class for flux reconstruction
template <typename IntegratorDerivedClass>
class FluxReconstruction {
public:

    virtual ~FluxReconstruction() = default;

    // Original method for flux-based reconstruction (for LLF-like methods)
    virtual void ReconstructFluxes(int direction, int lev, IntegratorDerivedClass* solver,
                                const amrex::MultiFab& Flux_Negative, 
                                const amrex::MultiFab& Flux_Positive,
                                amrex::MultiFab& SummedFlux,
                                const std::shared_ptr<GenericVariableAccessor>& variable_accessor,
                                const amrex::MultiFab* AverageStateBuffer = nullptr) const = 0;

    // New method for variable-based reconstruction (for HLLC-like methods)
    virtual void ReconstructVariables(int direction, int lev, IntegratorDerivedClass* solver,
                                    const amrex::MultiFab& VariableBuffer,
                                    amrex::MultiFab& LeftStates, 
                                    amrex::MultiFab& RightStates,
                                    const std::shared_ptr<GenericVariableAccessor>& variable_accessor,
                                    const amrex::MultiFab* AverageStateBuffer = nullptr) const = 0;
};


// Base class for flux methods (e.g., Local Lax-Friedrichs, HLLC)
template <typename IntegratorDerivedClass>
class FluxMethod {
public:
    // Enum to determine reconstruction approach based on method requirements
    enum class ReconstructionTarget {
        Variables,  // Reconstruct variables first (for HLLC-like methods)
        Fluxes      // Reconstruct fluxes first (for LLF-like methods)
    };

    virtual ~FluxMethod() = default;

    // Return whether this method requires reconstructing variables or fluxes first
    virtual ReconstructionTarget getReconstructionTarget() const = 0;

    // Method-specific flux computation implementations
    virtual void SplitFluxes(int direction, int lev, IntegratorDerivedClass* solver, 
                            amrex::MultiFab& VariableBuffer, amrex::MultiFab& CellFluxBuffer, 
                            amrex::MultiFab& Flux_Negative, amrex::MultiFab& Flux_Positive) const = 0;

    // New method for directly computing fluxes from reconstructed states (for HLLC-like methods)
    virtual void ComputeFluxes(int direction, int lev, IntegratorDerivedClass* solver,
                            const amrex::MultiFab& LeftStates, const amrex::MultiFab& RightStates, 
                            amrex::MultiFab& SummedFlux) const = 0;
};

// FluxHandler class for managing reconstruction and flux computation
template <typename IntegratorDerivedClass>
class FluxHandler {
public:
    enum Directions { Xdir = 0, Ydir = 1, Zdir = 2 };

    // Constructor accepts a variable accessor
    FluxHandler(std::shared_ptr<GenericVariableAccessor> accessor) 
        : variable_accessor(accessor) {
        if (!accessor) {
            Util::Warning(INFO, "FluxHandler created with null variable accessor");
        }
    }

    // Set reconstruction method 
    void SetReconstruction(std::shared_ptr<FluxReconstruction<IntegratorDerivedClass>> recon) {
        reconstruction = std::move(recon);
    }

    // Set flux computation method
    void SetFluxMethod(std::shared_ptr<FluxMethod<IntegratorDerivedClass>> method) {
        flux_method = std::move(method);
    }

    // Main method for constructing fluxes
    void ConstructFluxes(int lev, IntegratorDerivedClass* solver) const {
        // Validate reconstruction and flux method are set
        if (!reconstruction || !flux_method) {
            Util::Abort(INFO, "FluxHandler: Reconstruction or flux method is not set.");
        }

        // Check if variable accessor is valid
        if (!variable_accessor) {
            Util::Abort(INFO, "FluxHandler: Variable accessor is not set.");
        }

        // Get solver capabilities for validation
        auto solverCapabilities = variable_accessor->getSolverCapabilities();
        if (!solverCapabilities) {
            Util::Abort(INFO, "FluxHandler: Unable to retrieve solver capabilities.");
        }


        // Process each direction
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            // Choose processing path based on the flux method's reconstruction target
            if (flux_method->getReconstructionTarget() == FluxMethod<IntegratorDerivedClass>::ReconstructionTarget::Variables) {
                // For methods like HLLC that reconstruct variables first
                ProcessVariableBasedReconstruction(dir, lev, solver);
            } else {
                // For methods like LLF that reconstruct fluxes first
                ProcessFluxBasedReconstruction(dir, lev, solver);
            }
        }
    }

    // Get the current variable accessor
    std::shared_ptr<GenericVariableAccessor> GetVariableAccessor() const {
        return variable_accessor;
    }

private:
    // Reconstruction and flux method pointers
    std::shared_ptr<FluxReconstruction<IntegratorDerivedClass>> reconstruction;
    std::shared_ptr<FluxMethod<IntegratorDerivedClass>> flux_method;
    std::shared_ptr<GenericVariableAccessor> variable_accessor;


    
  

    
    // For variable-based reconstruction methods like HLLC
    void ProcessVariableBasedReconstruction(
        int direction, int lev, IntegratorDerivedClass* solver) const;

    // For flux-based reconstruction methods like LLF
    void ProcessFluxBasedReconstruction(
        int direction, int lev, IntegratorDerivedClass* solver) const;
};

// For variable-based reconstruction methods like HLLC
template <typename IntegratorDerivedClass>


void FluxHandler<IntegratorDerivedClass>::ProcessVariableBasedReconstruction(
   

    
    int direction, int lev, IntegratorDerivedClass* solver) const {

    // Validate inputs and components
    if (!reconstruction || !flux_method || !variable_accessor) {
        Util::Abort(INFO, "FluxHandler: Required components not initialized");
    }

    // Get solver capabilities for validation
    auto solverCapabilities = variable_accessor->getSolverCapabilities();
    if (!solverCapabilities) {
        Util::Abort(INFO, "FluxHandler: Unable to retrieve solver capabilities");
    }

    // Get current reconstruction mode
    ReconstructionMode mode = variable_accessor->getCurrentReconstructionMode();

    // Validate method combinations
    auto validationResult = solverCapabilities->validateMethodCombination(
        FluxReconstructionType::WENO,  // Assume WENO, could be parameterized
        FluxScheme::HLLC,             // For this path, we're using HLLC
        TimeSteppingSchemeType::RK3,  // Assume RK3, could be parameterized
        mode
    );

    int ghost_cells = solver->number_of_ghost_cells;


    // 1. Create intermediate buffers for storing cell centered variables
    amrex::MultiFab VariableBuffer = variable_accessor->CreateWorkingBuffer(
        solver->grids[lev], solver->dmap[lev], solver->number_of_components, ghost_cells
    );


    // 2. Transform variables if needed (based on reconstruction mode)
    variable_accessor->CopyVariables(direction, lev, solver, VariableBuffer, 
                                        mode, validationResult);

    // 4. Create a face-centered BoxArray for the flux
    amrex::BoxArray face_ba = solver->grids[lev];
    amrex::IntVect face_type = amrex::IntVect::TheZeroVector();
    face_type[direction] = 1;  // Set face-centered in the specified direction
    face_ba.convert(face_type);


    amrex::MultiFab AverageStateBuffer;

    // 3. If Characteristic Reconstruction, Populate Average States in a buffer
    if ( mode == ReconstructionMode::Characteristic) {
    // 3a. Create intermediate buffers for storing cell centered variables
        AverageStateBuffer = variable_accessor->CreateWorkingBuffer(
        face_ba, solver->dmap[lev], solver->number_of_components, ghost_cells
    );

    // 3b. Compute average state buffers
    variable_accessor->PopulateAverageStates(direction, lev, solver, AverageStateBuffer);  

    }

    amrex::MultiFab LeftStates = variable_accessor->CreateWorkingBuffer(
        face_ba, solver->dmap[lev], solver->number_of_components, ghost_cells);

    amrex::MultiFab RightStates = variable_accessor->CreateWorkingBuffer(
        face_ba, solver->dmap[lev], solver->number_of_components, ghost_cells);

    // 5. Perform reconstruction on variables to get left/right states
    reconstruction->ReconstructVariables(
            direction, lev, solver, 
            VariableBuffer, LeftStates, RightStates, 
            variable_accessor,
            (mode == ReconstructionMode::Characteristic) ? &AverageStateBuffer : nullptr);

        // 5b. DI5 positivity limiter:
    // apply ONLY when reconstructed face state violates the admissible set.
    // Otherwise leave the high-order reconstructed state untouched.
    if (solver->fluid_mixture != nullptr) {
        const auto* mixture_model = solver->fluid_mixture.get();

        const int m1_idx    = solver->variableIndex.M1;
        const int m2_idx    = solver->variableIndex.M2;
        const int momx_idx  = solver->variableIndex.MOMX;
        const int momy_idx  = solver->variableIndex.MOMY;
        const int momz_idx  = solver->variableIndex.MOMZ;
        const int p_idx     = solver->variableIndex.ETOT;   // pressure slot in reconstruction buffer
        const int alpha_idx = solver->variableIndex.ALPHA;

        for (amrex::MFIter mfi(LeftStates, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx_with_ghosts = mfi.grownnodaltilebox(direction, ghost_cells);

            auto const& Var_arr         = VariableBuffer.array(mfi);
            auto const& LeftStates_arr  = LeftStates.array(mfi);
            auto const& RightStates_arr = RightStates.array(mfi);

            amrex::ParallelFor(bx_with_ghosts, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                int index[3];
                int left_index[3];
                int right_index[3];
                int lower_bounds[3];
                int upper_bounds[3];

                index[0] = i; index[1] = j; index[2] = k;
                left_index[0] = i; left_index[1] = j; left_index[2] = k;
                right_index[0] = i; right_index[1] = j; right_index[2] = k;

                lower_bounds[0] = bx_with_ghosts.smallEnd(0);
                lower_bounds[1] = bx_with_ghosts.smallEnd(1);
                lower_bounds[2] = (AMREX_SPACEDIM == 3) ? bx_with_ghosts.smallEnd(2) : 0;

                upper_bounds[0] = bx_with_ghosts.bigEnd(0) - 1;
                upper_bounds[1] = bx_with_ghosts.bigEnd(1) - 1;
                upper_bounds[2] = (AMREX_SPACEDIM == 3) ? bx_with_ghosts.bigEnd(2) - 1 : 0;

                ClampIndices(index, lower_bounds, upper_bounds);

                int left_offset[3]  = {0, 0, 0};
                int right_offset[3] = {0, 0, 0};
                left_offset[direction]  = 0;
                right_offset[direction] = 1;

                ShiftAndClampIndices(left_index,  left_offset,  lower_bounds, upper_bounds, direction);
                ShiftAndClampIndices(right_index, right_offset, lower_bounds, upper_bounds, direction);

                // ---------------- Left face state ----------------
                Set::Scalar Lm1 = LeftStates_arr(index[0], index[1], index[2], m1_idx);
                Set::Scalar Lm2 = LeftStates_arr(index[0], index[1], index[2], m2_idx);
                Set::Scalar Lu  = LeftStates_arr(index[0], index[1], index[2], momx_idx);
#if (AMREX_SPACEDIM >= 2)
                Set::Scalar Lv  = LeftStates_arr(index[0], index[1], index[2], momy_idx);
#else
                Set::Scalar Lv  = Set::Scalar(0.0);
#endif
#if (AMREX_SPACEDIM == 3)
                Set::Scalar Lw  = LeftStates_arr(index[0], index[1], index[2], momz_idx);
#else
                Set::Scalar Lw  = Set::Scalar(0.0);
#endif
                Set::Scalar Lp  = LeftStates_arr(index[0], index[1], index[2], p_idx);
                Set::Scalar La  = LeftStates_arr(index[0], index[1], index[2], alpha_idx);

                const Set::Scalar Lm1_lo = Var_arr(left_index[0], left_index[1], left_index[2], m1_idx);
                const Set::Scalar Lm2_lo = Var_arr(left_index[0], left_index[1], left_index[2], m2_idx);
                const Set::Scalar Lu_lo  = Var_arr(left_index[0], left_index[1], left_index[2], momx_idx);
#if (AMREX_SPACEDIM >= 2)
                const Set::Scalar Lv_lo  = Var_arr(left_index[0], left_index[1], left_index[2], momy_idx);
#else
                const Set::Scalar Lv_lo  = Set::Scalar(0.0);
#endif
#if (AMREX_SPACEDIM == 3)
                const Set::Scalar Lw_lo  = Var_arr(left_index[0], left_index[1], left_index[2], momz_idx);
#else
                const Set::Scalar Lw_lo  = Set::Scalar(0.0);
#endif
                const Set::Scalar Lp_lo  = Var_arr(left_index[0], left_index[1], left_index[2], p_idx);
                const Set::Scalar La_lo  = Var_arr(left_index[0], left_index[1], left_index[2], alpha_idx);

                const Set::Scalar thetaL = DI5ComputeTheta(
                    Lm1, Lm2, Lu, Lv, Lw, Lp, La,
                    Lm1_lo, Lm2_lo, Lu_lo, Lv_lo, Lw_lo, Lp_lo, La_lo,
                    mixture_model
                );

                DI5ApplyThetaLimit(
                    Lm1, Lm2, Lu, Lv, Lw, Lp, La,
                    thetaL,
                    Lm1_lo, Lm2_lo, Lu_lo, Lv_lo, Lw_lo, Lp_lo, La_lo
                );

                LeftStates_arr(index[0], index[1], index[2], m1_idx)    = Lm1;
                LeftStates_arr(index[0], index[1], index[2], m2_idx)    = Lm2;
                LeftStates_arr(index[0], index[1], index[2], momx_idx)  = Lu;
#if (AMREX_SPACEDIM >= 2)
                LeftStates_arr(index[0], index[1], index[2], momy_idx)  = Lv;
#endif
#if (AMREX_SPACEDIM == 3)
                LeftStates_arr(index[0], index[1], index[2], momz_idx)  = Lw;
#endif
                LeftStates_arr(index[0], index[1], index[2], p_idx)     = Lp;
                LeftStates_arr(index[0], index[1], index[2], alpha_idx) = La;

                // ---------------- Right face state ----------------
                Set::Scalar Rm1 = RightStates_arr(index[0], index[1], index[2], m1_idx);
                Set::Scalar Rm2 = RightStates_arr(index[0], index[1], index[2], m2_idx);
                Set::Scalar Ru  = RightStates_arr(index[0], index[1], index[2], momx_idx);
#if (AMREX_SPACEDIM >= 2)
                Set::Scalar Rv  = RightStates_arr(index[0], index[1], index[2], momy_idx);
#else
                Set::Scalar Rv  = Set::Scalar(0.0);
#endif
#if (AMREX_SPACEDIM == 3)
                Set::Scalar Rw  = RightStates_arr(index[0], index[1], index[2], momz_idx);
#else
                Set::Scalar Rw  = Set::Scalar(0.0);
#endif
                Set::Scalar Rp  = RightStates_arr(index[0], index[1], index[2], p_idx);
                Set::Scalar Ra  = RightStates_arr(index[0], index[1], index[2], alpha_idx);

                const Set::Scalar Rm1_lo = Var_arr(right_index[0], right_index[1], right_index[2], m1_idx);
                const Set::Scalar Rm2_lo = Var_arr(right_index[0], right_index[1], right_index[2], m2_idx);
                const Set::Scalar Ru_lo  = Var_arr(right_index[0], right_index[1], right_index[2], momx_idx);
#if (AMREX_SPACEDIM >= 2)
                const Set::Scalar Rv_lo  = Var_arr(right_index[0], right_index[1], right_index[2], momy_idx);
#else
                const Set::Scalar Rv_lo  = Set::Scalar(0.0);
#endif
#if (AMREX_SPACEDIM == 3)
                const Set::Scalar Rw_lo  = Var_arr(right_index[0], right_index[1], right_index[2], momz_idx);
#else
                const Set::Scalar Rw_lo  = Set::Scalar(0.0);
#endif
                const Set::Scalar Rp_lo  = Var_arr(right_index[0], right_index[1], right_index[2], p_idx);
                const Set::Scalar Ra_lo  = Var_arr(right_index[0], right_index[1], right_index[2], alpha_idx);

                const Set::Scalar thetaR = DI5ComputeTheta(
                    Rm1, Rm2, Ru, Rv, Rw, Rp, Ra,
                    Rm1_lo, Rm2_lo, Ru_lo, Rv_lo, Rw_lo, Rp_lo, Ra_lo,
                    mixture_model
                );

                DI5ApplyThetaLimit(
                    Rm1, Rm2, Ru, Rv, Rw, Rp, Ra,
                    thetaR,
                    Rm1_lo, Rm2_lo, Ru_lo, Rv_lo, Rw_lo, Rp_lo, Ra_lo
                );

                RightStates_arr(index[0], index[1], index[2], m1_idx)    = Rm1;
                RightStates_arr(index[0], index[1], index[2], m2_idx)    = Rm2;
                RightStates_arr(index[0], index[1], index[2], momx_idx)  = Ru;
#if (AMREX_SPACEDIM >= 2)
                RightStates_arr(index[0], index[1], index[2], momy_idx)  = Rv;
#endif
#if (AMREX_SPACEDIM == 3)
                RightStates_arr(index[0], index[1], index[2], momz_idx)  = Rw;
#endif
                RightStates_arr(index[0], index[1], index[2], p_idx)     = Rp;
                RightStates_arr(index[0], index[1], index[2], alpha_idx) = Ra;
            });
        }
    }

    // 6. Create MultiFab for the computed flux
    amrex::MultiFab SummedFlux(face_ba, solver->dmap[lev], 
                            solver->number_of_components, ghost_cells);

   
                        
    // 7. Compute fluxes from reconstructed states
    flux_method->ComputeFluxes(direction, lev, solver, 
                            LeftStates, RightStates, SummedFlux);

    // 8. Store fluxes in the appropriate directional flux arrays
    variable_accessor->StoreDirectionalFlux(direction, lev, solver, 
                                        SummedFlux);
}

// For flux-based reconstruction methods like LLF
template <typename IntegratorDerivedClass>
void FluxHandler<IntegratorDerivedClass>::ProcessFluxBasedReconstruction(
    int direction, int lev, IntegratorDerivedClass* solver) const {

    // Validate inputs and components
    if (!reconstruction || !flux_method || !variable_accessor) {
        Util::Abort(INFO, "FluxHandler: Required components not initialized");
    }

    // Get solver capabilities for validation
    auto solverCapabilities = variable_accessor->getSolverCapabilities();
    if (!solverCapabilities) {
        Util::Abort(INFO, "FluxHandler: Unable to retrieve solver capabilities");
    }

    // Get current reconstruction mode
    ReconstructionMode mode = variable_accessor->getCurrentReconstructionMode();

    // Validate method combinations
    auto validationResult = solverCapabilities->validateMethodCombination(
        FluxReconstructionType::WENO,  // Assume WENO, could be parameterized
        FluxScheme::LocalLaxFriedrichs,  // For this path, we're using LLF
        TimeSteppingSchemeType::RK3,   // Assume RK3, could be parameterized
        mode
    );


    int ghost_cells = solver->number_of_ghost_cells;


    // 1. Create intermediate buffers for storing cell centered variables
    amrex::MultiFab VariableBuffer = variable_accessor->CreateWorkingBuffer(
        solver->grids[lev], solver->dmap[lev], solver->number_of_components, ghost_cells
    );

    // 2. Transform variables if needed (based on reconstruction mode)
    variable_accessor->CopyVariables(direction, lev, solver, VariableBuffer, 
                                        mode, validationResult);


    // 4. Create a cell flux buffer for storing computed fluxes
    amrex::MultiFab CellFluxBuffer = variable_accessor->CreateWorkingBuffer(
        solver->grids[lev], solver->dmap[lev], solver->number_of_components, 
        solver->number_of_ghost_cells);

    // 5. Compute and transform fluxes if needed
    variable_accessor->CopyFluxes(direction, lev, solver, CellFluxBuffer, 
                                    mode, validationResult);

    // 6. Create buffers for split fluxes (positive and negative)
    amrex::MultiFab Flux_Negative = variable_accessor->CreateWorkingBuffer(
        solver->grids[lev], solver->dmap[lev], solver->number_of_components, ghost_cells);

    amrex::MultiFab Flux_Positive = variable_accessor->CreateWorkingBuffer(
        solver->grids[lev], solver->dmap[lev], solver->number_of_components, ghost_cells);

    // 7. Split fluxes into positive and negative components
    flux_method->SplitFluxes(direction, lev, solver, VariableBuffer, 
                            CellFluxBuffer, Flux_Negative, Flux_Positive);

    // 8. Create a face-centered BoxArray for the flux
    amrex::BoxArray face_ba = solver->grids[lev];
    amrex::IntVect face_type = amrex::IntVect::TheZeroVector();
    face_type[direction] = 1;  // Set face-centered in the specified direction
    face_ba.convert(face_type);


    amrex::MultiFab AverageStateBuffer;

    // 3. If Characteristic Reconstruction, Populate Average States in a buffer
    if (mode == ReconstructionMode::Characteristic) {
    // 3a. Create intermediate buffers for storing cell centered variables
    AverageStateBuffer = variable_accessor->CreateWorkingBuffer(
        face_ba, solver->dmap[lev], solver->number_of_components, ghost_cells);

    // 3b. Compute average state buffers
    variable_accessor->PopulateAverageStates(direction, lev, solver, AverageStateBuffer);  

    }

    // 9. Create MultiFab for the reconstructed summed flux
    amrex::MultiFab SummedFlux(face_ba, solver->dmap[lev], 
                            solver->number_of_components, ghost_cells);

    // 10. Perform reconstruction on fluxes
    reconstruction->ReconstructFluxes(
            direction, lev, solver, 
            Flux_Negative, Flux_Positive, SummedFlux, 
            variable_accessor,
            (mode == ReconstructionMode::Characteristic) ? &AverageStateBuffer : nullptr);

    // 11. Store fluxes in the appropriate directional flux arrays
    variable_accessor->StoreDirectionalFlux(direction, lev, solver, 
                                        SummedFlux);


}
// First-order upwind reconstruction
template <typename IntegratorDerivedClass>
class FirstOrderReconstruction : public FluxReconstruction<IntegratorDerivedClass> {
public:
    // Implement flux-based reconstruction (for LLF-like methods)
    void ReconstructFluxes(int direction, int /*lev*/, IntegratorDerivedClass* solver,
                        const amrex::MultiFab& Flux_Negative, const amrex::MultiFab& Flux_Positive,
                        amrex::MultiFab& SummedFlux,
                        [[maybe_unused]] const std::shared_ptr<GenericVariableAccessor>& variable_accessor,
                        [[maybe_unused]] const amrex::MultiFab* AverageStateBuffer = nullptr) const override {

        const int nghosts = solver->number_of_ghost_cells; 

        // Loop through face-centered indices for flux reconstruction
        for (amrex::MFIter mfi(SummedFlux, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& face_bx_with_ghosts = mfi.grownnodaltilebox(direction, nghosts);

            auto const& FluxNeg_arr = Flux_Negative.array(mfi);  // Left reconstructed state
            auto const& FluxPos_arr = Flux_Positive.array(mfi);  // Right reconstructed state
            auto const& TotalFlux_arr = SummedFlux.array(mfi);  // Summed flux

            const int num_components = solver->number_of_components;  // Number of components

            amrex::ParallelFor(face_bx_with_ghosts, [=] AMREX_GPU_DEVICE(int iface, int jface, int kface) noexcept {
                // Construct index arrays with correct number of components
                int index[3];
                int left_index[3];
                int right_index[3];
                int lower_bounds[3];
                int upper_bounds[3];

                // Set up indices
                index[0] = iface; index[1] = jface; index[2] = kface;
                left_index[0] = iface; left_index[1] = jface; left_index[2] = kface;
                right_index[0] = iface; right_index[1] = jface; right_index[2] = kface;

                // Define bounds including ghost cells
                lower_bounds[0] = face_bx_with_ghosts.smallEnd(0); 
                lower_bounds[1] = face_bx_with_ghosts.smallEnd(1); 
                lower_bounds[2] = AMREX_SPACEDIM == 3 ? face_bx_with_ghosts.smallEnd(2) : 0;
                upper_bounds[0] = face_bx_with_ghosts.bigEnd(0) - 1;
                upper_bounds[1] = face_bx_with_ghosts.bigEnd(1) - 1;
                upper_bounds[2] = AMREX_SPACEDIM == 3 ? face_bx_with_ghosts.bigEnd(2) - 1 : 0;

                // Clamp the main index
                ClampIndices(index, lower_bounds, upper_bounds);

                // Shift and clamp for left and right indices
                int left_offset[3] = {0, 0, 0};
                int right_offset[3] = {0, 0, 0};
                left_offset[direction] = 0;  // Shift left
                right_offset[direction] = 1;  // Shift right

                ShiftAndClampIndices(left_index, left_offset, lower_bounds, upper_bounds, direction);
                ShiftAndClampIndices(right_index, right_offset, lower_bounds, upper_bounds, direction);

                for (int n = 0; n < num_components; ++n) {
                    const Set::Scalar FL = FluxNeg_arr(left_index[0], left_index[1], left_index[2], n);
                    const Set::Scalar FR = FluxPos_arr(right_index[0], right_index[1], right_index[2], n);  
                    // Assign values to the stencil arrays
                    TotalFlux_arr(index[0], index[1], index[2], n) = FL + FR;
                }
            });
        }
    }

    // Implement variable-based reconstruction (for HLLC-like methods)
    void ReconstructVariables(int direction, int /*lev*/, IntegratorDerivedClass* solver,
                            const amrex::MultiFab& VariableBuffer,
                            amrex::MultiFab& LeftStates, amrex::MultiFab& RightStates,
                        [[maybe_unused]] const std::shared_ptr<GenericVariableAccessor>& variable_accessor,
                        [[maybe_unused]] const amrex::MultiFab* AverageStateBuffer = nullptr) const override {

        const int nghosts = solver->number_of_ghost_cells;

        // Loop through grid cells to reconstruct left/right states at interfaces
        for (amrex::MFIter mfi(VariableBuffer, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx_with_ghosts = mfi.grownnodaltilebox(direction, nghosts);

            auto const& Var_arr = VariableBuffer.array(mfi);
            auto const& LeftStates_arr = LeftStates.array(mfi);
            auto const& RightStates_arr = RightStates.array(mfi);

            const int num_components = solver->number_of_components;

            amrex::ParallelFor(bx_with_ghosts, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                // Construct index arrays
                int index[3];
                int left_index[3];
                int right_index[3];
                int lower_bounds[3];
                int upper_bounds[3];

                // Set up indices
                index[0] = i; index[1] = j; index[2] = k;
                left_index[0] = i; left_index[1] = j; left_index[2] = k;
                right_index[0] = i; right_index[1] = j; right_index[2] = k;

                // Define bounds including ghost cells
                lower_bounds[0] = bx_with_ghosts.smallEnd(0); 
                lower_bounds[1] = bx_with_ghosts.smallEnd(1); 
                lower_bounds[2] = AMREX_SPACEDIM == 3 ? bx_with_ghosts.smallEnd(2) : 0;
                upper_bounds[0] = bx_with_ghosts.bigEnd(0) - 1;
                upper_bounds[1] = bx_with_ghosts.bigEnd(1) - 1;
                upper_bounds[2] = AMREX_SPACEDIM == 3 ? bx_with_ghosts.bigEnd(2) - 1 : 0;

                // Clamp the main index
                ClampIndices(index, lower_bounds, upper_bounds);

                // Shift and clamp for left and right indices
                int left_offset[3] = {0, 0, 0};
                int right_offset[3] = {0, 0, 0};
                left_offset[direction] = 0;  // Shift left
                right_offset[direction] = 1;  // Shift right

                ShiftAndClampIndices(left_index, left_offset, lower_bounds, upper_bounds, direction);
                ShiftAndClampIndices(right_index, right_offset, lower_bounds, upper_bounds, direction);

                for (int n = 0; n < num_components; ++n) {
                    // First-order reconstruction simply uses the cell values directly
                    LeftStates_arr(index[0], index[1], index[2], n) = 
                        Var_arr(left_index[0], left_index[1], left_index[2], n);

                    RightStates_arr(index[0], index[1], index[2], n) = 
                        Var_arr(right_index[0], right_index[1], right_index[2], n);
                }
            });
        }
    }
};

// Local Lax-Friedrichs flux method
template <typename IntegratorDerivedClass>
class LocalLaxFriedrichsMethod : public FluxMethod<IntegratorDerivedClass> {
public:
    using typename FluxMethod<IntegratorDerivedClass>::ReconstructionTarget;

    // Indicate that LLF works with flux-based reconstruction
    ReconstructionTarget getReconstructionTarget() const override {
        return ReconstructionTarget::Fluxes;
    }

    void SplitFluxes(int direction, int lev, IntegratorDerivedClass* solver, amrex::MultiFab& VariableBuffer, amrex::MultiFab& CellFluxBuffer, amrex::MultiFab& Flux_Negative, amrex::MultiFab& Flux_Positive) const override {

        const Set::Scalar gamma = 1.4;  // Ratio of specific heats

        //for (amrex::MFIter mfi(QL_stencil, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        for (amrex::MFIter mfi(Flux_Negative, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {

            const amrex::Box& bx_with_ghosts = mfi.growntilebox();

            // auto const& flux_arr = (direction == FluxHandler<IntegratorDerivedClass>::Xdir) ? solver->XFlux_mf.Patch(lev, mfi) :
            //                         (direction == FluxHandler<IntegratorDerivedClass>::Ydir) ? solver->YFlux_mf.Patch(lev, mfi) :
            //                         solver->ZFlux_mf.Patch(lev, mfi);
            auto const& FluxNeg_arr = Flux_Negative.array(mfi);
            auto const& FluxPos_arr = Flux_Positive.array(mfi);
            auto const& p_arr  = solver->PVec_mf.Patch(lev,mfi);
            auto const& pres_arr  = solver->Pressure_mf.Patch(lev,mfi);
            auto const& q_arr  = VariableBuffer.array(mfi);
            auto const& flux_arr  = CellFluxBuffer.array(mfi);

            // **Direction to velocity and energy mapping**
            int normal = direction;                 // Normal velocity component
            int trans1 = (direction + 1) % AMREX_SPACEDIM;  // First transverse velocity
#if AMREX_SPACEDIM == 3                                                              
            int trans2 = (direction + 2) % AMREX_SPACEDIM;  // Second transverse velocity (3D only)
#endif
            // **Direction to velocity mapping**
            int velocity_component[3] = {
                solver->variableIndex.UVEL,  // X-direction -> UVEL
                solver->variableIndex.VVEL,  // Y-direction -> VVEL
                solver->variableIndex.WVEL   // Z-direction -> WVEL
            };

        amrex::ParallelFor(bx_with_ghosts, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {


            // Construct index arrays with correct number of components
            int index[3];
            int left_index[3];
            int right_index[3];
            int lower_bounds[3];
            int upper_bounds[3];

            // Set up indices
            index[0] = i; index[1] = j; index[2] = k;
            left_index[0] = i; left_index[1] = j; left_index[2] = k;
            right_index[0] = i; right_index[1] = j; right_index[2] = k;

            // Define bounds including ghost cells
            lower_bounds[0] = bx_with_ghosts.smallEnd(0); 
            lower_bounds[1] = bx_with_ghosts.smallEnd(1); 
            lower_bounds[2] = AMREX_SPACEDIM == 3 ? bx_with_ghosts.smallEnd(2) : 0;
            upper_bounds[0] = bx_with_ghosts.bigEnd(0);
            upper_bounds[1] = bx_with_ghosts.bigEnd(1);
            upper_bounds[2] = AMREX_SPACEDIM == 3 ? bx_with_ghosts.bigEnd(2) : 0;

            // Clamp the main index
            ClampIndices(index, lower_bounds, upper_bounds);

            // Shift and clamp for left and right indices
            int left_offset[3] = {0, 0, 0};
            int right_offset[3] = {0, 0, 0};
            left_offset[direction] = 0;  // Shift left
            right_offset[direction] = 1;  // Shift right

            ShiftAndClampIndices(left_index, left_offset, lower_bounds, upper_bounds, direction);
            ShiftAndClampIndices(right_index, right_offset, lower_bounds, upper_bounds, direction);

            // **Left and right primitive variables (QL and QR)**
            Set::Scalar rho_L = p_arr(left_index[0], left_index[1], left_index[2], solver->variableIndex.DENS);
            Set::Scalar rho_R = p_arr(right_index[0], right_index[1], right_index[2], solver->variableIndex.DENS);

            // **Left and right primitive variables (QL and QR)**
            Set::Scalar pressure_L = pres_arr(left_index[0], left_index[1], left_index[2]);
            Set::Scalar pressure_R = pres_arr(right_index[0], right_index[1], right_index[2]);

            Set::Scalar vel_L[3] = {0.0, 0.0, 0.0};
            Set::Scalar vel_R[3] = {0.0, 0.0, 0.0};

            vel_L[normal] = p_arr(left_index[0], left_index[1], left_index[2], velocity_component[normal]);
            vel_R[normal] = p_arr(right_index[0], right_index[1], right_index[2], velocity_component[normal]);

            // **Speed of sound**
            Set::Scalar c_L = std::sqrt(gamma * std::max(pressure_L, 1e-8) / std::max(rho_L, 1e-8));
            Set::Scalar c_R = std::sqrt(gamma * std::max(pressure_R, 1e-8) / std::max(rho_R, 1e-8));

            // **Maximum wave speed (lambda_max)**
            Set::Scalar lambda_max = std::max(std::abs(vel_L[normal]) + c_L, std::abs(vel_R[normal]) + c_R);

            // **Flux arrays**
            int num_components = solver->number_of_components;
            std::vector<Set::Scalar> FL(num_components, 0.0);
            std::vector<Set::Scalar> FR(num_components, 0.0);

            // Mass flux
            FL[solver->variableIndex.DENS] = flux_arr(left_index[0], left_index[1], left_index[2], solver->variableIndex.DENS);
            FR[solver->variableIndex.DENS] = flux_arr(right_index[0], right_index[1], right_index[2], solver->variableIndex.DENS);

            // Momentum fluxes
            FL[solver->variableIndex.UVEL + normal] = flux_arr(left_index[0], left_index[1], left_index[2], solver->variableIndex.UVEL + normal);
            FR[solver->variableIndex.UVEL + normal] = flux_arr(right_index[0], right_index[1], right_index[2], solver->variableIndex.UVEL + normal);

#if AMREX_SPACEDIM >= 2
            FL[solver->variableIndex.UVEL + trans1] = flux_arr(left_index[0], left_index[1], left_index[2], solver->variableIndex.UVEL + trans1);
            FR[solver->variableIndex.UVEL + trans1] = flux_arr(right_index[0], right_index[1], right_index[2], solver->variableIndex.UVEL + trans1);
#endif
#if AMREX_SPACEDIM == 3
            FL[solver->variableIndex.UVEL + trans2] = flux_arr(left_index[0], left_index[1], left_index[2], solver->variableIndex.UVEL + trans2);
            FR[solver->variableIndex.UVEL + trans2] = flux_arr(right_index[0], right_index[1], right_index[2], solver->variableIndex.UVEL + trans2);
#endif

            // Energy flux
            FL[solver->variableIndex.IE] = flux_arr(left_index[0], left_index[1], left_index[2], solver->variableIndex.IE);
            FR[solver->variableIndex.IE] = flux_arr(right_index[0], right_index[1], right_index[2], solver->variableIndex.IE);


            // **Conservative state vectors (left and right)**
            std::vector<Set::Scalar> QL_cons(num_components, 0.0);
            std::vector<Set::Scalar> QR_cons(num_components, 0.0);

            // **Convert primitive variables to conservative variables**
            QL_cons[solver->variableIndex.DENS] = q_arr(left_index[0], left_index[1], left_index[2], solver->variableIndex.DENS); //rho_L;  // Density
            QR_cons[solver->variableIndex.DENS] = q_arr(right_index[0], right_index[1], right_index[2], solver->variableIndex.DENS); //rho_R;

            QL_cons[solver->variableIndex.UVEL + normal] = q_arr(left_index[0], left_index[1], left_index[2], solver->variableIndex.UVEL + normal);//rho_L * vel_L[normal];  // Momentum in normal direction
            QR_cons[solver->variableIndex.UVEL + normal] = q_arr(right_index[0], right_index[1], right_index[2], solver->variableIndex.UVEL + normal); //rho_R * vel_R[normal];

#if AMREX_SPACEDIM >= 2
            QL_cons[solver->variableIndex.UVEL + trans1] = q_arr(left_index[0], left_index[1], left_index[2], solver->variableIndex.UVEL + trans1);//rho_L * vel_L[trans1];  // Momentum in first transverse direction
            QR_cons[solver->variableIndex.UVEL + trans1] = q_arr(right_index[0], right_index[1], right_index[2], solver->variableIndex.UVEL + trans1); //rho_R * vel_R[trans1];
#endif

#if AMREX_SPACEDIM == 3
            QL_cons[solver->variableIndex.UVEL + trans2] = q_arr(left_index[0], left_index[1], left_index[2], solver->variableIndex.UVEL + trans2);//rho_L * vel_L[trans2];  // Momentum in second transverse direction
            QR_cons[solver->variableIndex.UVEL + trans2] = q_arr(right_index[0], right_index[1], right_index[2], solver->variableIndex.UVEL + trans2); //rho_R * vel_R[trans2];
#endif

            QL_cons[solver->variableIndex.IE] = q_arr(left_index[0], left_index[1], left_index[2], solver->variableIndex.IE);//rho_L * E_L;  // Total energy
            QR_cons[solver->variableIndex.IE] = q_arr(right_index[0], right_index[1], right_index[2], solver->variableIndex.IE); //rho_R * E_R;


            // **Local Lax-Friedrichs flux calculation**
            for (int n = 0; n < num_components; ++n) {
            FluxPos_arr(index[0], index[1], index[2], n) =  0.5 * FL[n] + 0.5 * lambda_max * QL_cons[n];
            FluxNeg_arr(index[0], index[1], index[2], n) =  0.5 * FR[n] - 0.5 * lambda_max * QR_cons[n];
            }

        });


        }

    }

    // Implement ComputeFluxes for completeness (though it won't be used for LLF)
    void ComputeFluxes(int /*direction*/, int /*lev*/, IntegratorDerivedClass* /*solver*/,
                    const amrex::MultiFab& /*LeftStates*/, const amrex::MultiFab& /*RightStates*/,
                    amrex::MultiFab& /*SummedFlux*/) const override {
        // This is not the primary pathway for LLF, but provide an implementation
        // for completeness or potential future use
        Util::Warning(INFO, "LLF ComputeFluxes called directly - not optimal path");

        // If needed, a direct implementation could be added here
    }    

};

// HLLC flux method with symmetry-preserving operations
template <typename IntegratorDerivedClass>
class HLLCMethod : public FluxMethod<IntegratorDerivedClass> {
public:

    using typename FluxMethod<IntegratorDerivedClass>::ReconstructionTarget;

    ReconstructionTarget getReconstructionTarget() const override {
        return ReconstructionTarget::Variables;
    }

    void SplitFluxes(int /*direction*/, int /*lev*/, IntegratorDerivedClass* /*solver*/, 
                    amrex::MultiFab& /*VariableBuffer*/, amrex::MultiFab& /*CellFluxBuffer*/,
                    amrex::MultiFab& /*Flux_Negative*/, amrex::MultiFab& /*Flux_Positive*/) const override {
        // Not the primary pathway for HLLC
        Util::Warning(INFO, "HLLC SplitFluxes called - not optimal path");
    }

void ComputeFluxes(int direction, int lev [[maybe_unused]], IntegratorDerivedClass* solver,
            const amrex::MultiFab& LeftStates, const amrex::MultiFab& RightStates,
            amrex::MultiFab& SummedFlux) const override {

    const Set::Scalar gamma = 1.4;  // Ratio of specific heats
    const int nghosts = solver->number_of_ghost_cells;

    // --- MODIFICATION START ---
    // Get the fluid mixture model (will be nullptr if not FiveEquationModel)
    const auto* mixture_model = solver->fluid_mixture.get(); 
    // --- MODIFICATION END ---

    // Loop through the grid
    for (amrex::MFIter mfi(SummedFlux, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& face_bx_with_ghosts = mfi.grownnodaltilebox(direction, nghosts);

        // Get arrays for left/right reconstructed states and flux output
        auto const& Left_arr = LeftStates.array(mfi);
        auto const& Right_arr = RightStates.array(mfi);
        auto const& TotalFlux_arr = SummedFlux.array(mfi);

        // Direction-dependent indices
        int normal = direction;  // Normal velocity component
        int trans1 = (direction + 1) % AMREX_SPACEDIM;  // First transverse velocity
#if AMREX_SPACEDIM == 3
        int trans2 = (direction + 2) % AMREX_SPACEDIM;  // Second transverse velocity (3D only)
#endif
        // Map direction to velocity components
        int velocity_component[3] = {
            solver->variableIndex.UVEL,  // X-direction -> UVEL
            solver->variableIndex.VVEL,  // Y-direction -> VVEL
            solver->variableIndex.WVEL   // Z-direction -> WVEL
        };

        int num_components = solver->number_of_components;

        // --- MODIFICATION START ---
        // Get the face velocity array for storing S_star
        amrex::Array4<Set::Scalar> face_vel_arr;
        
        if (mixture_model != nullptr) {
          
            if (direction == FluxHandler<IntegratorDerivedClass>::Directions::Xdir) {
                face_vel_arr = solver->UFace_mf.Patch(lev, mfi);
            
            } else if (direction == FluxHandler<IntegratorDerivedClass>::Directions::Ydir) {
                face_vel_arr = solver->VFace_mf.Patch(lev, mfi);
            } else {
#if AMREX_SPACEDIM == 3
                face_vel_arr = solver->WFace_mf.Patch(lev, mfi);
#endif
            }
        }

        // Get all 7 indices for the 5-equation model
        const int m1_idx   = solver->variableIndex.M1;
        const int m2_idx   = solver->variableIndex.M2;
        const int momx_idx = solver->variableIndex.MOMX;
        const int momy_idx = solver->variableIndex.MOMY;
        const int momz_idx = solver->variableIndex.MOMZ;
        const int etot_idx = solver->variableIndex.ETOT;
        const int alpha_idx= solver->variableIndex.ALPHA;
        // --- MODIFICATION END ---
     
        // Parallel loop over cells
        amrex::ParallelFor(face_bx_with_ghosts, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            // Setup indices
            int index[3];
            int lower_bounds[3];
            int upper_bounds[3];

            // Initialize index arrays
            index[0] = i; index[1] = j; index[2] = k;

            // Define bounds including ghost cells
            lower_bounds[0] = face_bx_with_ghosts.smallEnd(0); 
            lower_bounds[1] = face_bx_with_ghosts.smallEnd(1); 
            lower_bounds[2] = AMREX_SPACEDIM == 3 ? face_bx_with_ghosts.smallEnd(2) : 0;
            upper_bounds[0] = face_bx_with_ghosts.bigEnd(0) - 1;
            upper_bounds[1] = face_bx_with_ghosts.bigEnd(1) - 1;
            upper_bounds[2] = AMREX_SPACEDIM == 3 ? face_bx_with_ghosts.bigEnd(2) - 1 : 0;

            // Clamp indices to ensure they're within bounds
            ClampIndices(index, lower_bounds, upper_bounds);

            // --- MODIFICATION START ---
            // Check solver type and branch
            // FIXED: Check fluid_mixture pointer instead of SolverType enum
            if (mixture_model != nullptr) {
                
                // --- 5-EQUATION HLLC FLUX ---
           
                const Set::Scalar m1_L     = Left_arr(i, j, k, m1_idx);
                const Set::Scalar m2_L       = Left_arr(i, j, k, m2_idx);
                const Set::Scalar u_L       = Left_arr(i, j, k, momx_idx);
#if AMREX_SPACEDIM >= 2
                const Set::Scalar v_L       = Left_arr(i, j, k, momy_idx);
#else
                const Set::Scalar v_L       = 0.0;
#endif
#if AMREX_SPACEDIM == 3
                const Set::Scalar w_L       = Left_arr(i, j, k, momz_idx);
#else
                const Set::Scalar w_L       = 0.0;
#endif
                const Set::Scalar p_L  = Left_arr(i, j, k, etot_idx);
                const Set::Scalar alpha_L = Left_arr(i, j, k, alpha_idx);
                const Set::Scalar rho_L = m1_L + m2_L;

                const Set::Scalar vel_L_normal = Left_arr(i, j, k, momx_idx + normal);
                const Set::Scalar vel_L_trans1 = (AMREX_SPACEDIM >= 2) ? Left_arr(i, j, k, momx_idx + trans1) : 0.0;
                // FIXED: Add preprocessor guard
#if AMREX_SPACEDIM == 3
                const Set::Scalar vel_L_trans2 = Left_arr(i, j, k, momx_idx + trans2);
#endif

                // 2. Read Reconstructed Primitives (Right)
                const Set::Scalar m1_R     = Right_arr(i, j, k, m1_idx);
                const Set::Scalar m2_R       = Right_arr(i, j, k, m2_idx);
                const Set::Scalar u_R       = Right_arr(i, j, k, momx_idx);
#if AMREX_SPACEDIM >= 2
                const Set::Scalar v_R       = Right_arr(i, j, k, momy_idx);
#else
                const Set::Scalar v_R       = 0.0;
#endif
#if AMREX_SPACEDIM == 3
                const Set::Scalar w_R       = Right_arr(i, j, k, momz_idx);
#else
                const Set::Scalar w_R       = 0.0;
#endif
                const Set::Scalar p_R  = Right_arr(i, j, k, etot_idx);
                const Set::Scalar alpha_R = Right_arr(i, j, k, alpha_idx);
                const Set::Scalar rho_R = m1_R + m2_R;
                
                const Set::Scalar vel_R_normal = Right_arr(i, j, k, momx_idx + normal);
                const Set::Scalar vel_R_trans1 = (AMREX_SPACEDIM >= 2) ? Right_arr(i, j, k, momx_idx + trans1) : 0.0;
                
#if AMREX_SPACEDIM == 3
                const Set::Scalar vel_R_trans2 = Right_arr(i, j, k, momx_idx + trans2);
#endif

                // 3. Compute Conserved States (QL, QR)
                std::vector<Set::Scalar> QL_cons(num_components, 0.0);
                const Set::Scalar ke_L = 0.5 * (u_L*u_L + v_L*v_L + w_L*w_L);
                const Set::Scalar ie_mix_L = mixture_model->mixture_internal_energy(rho_L, p_L, alpha_L);
                QL_cons[m1_idx]   = m1_L;
                QL_cons[m2_idx]   = m2_L;
                QL_cons[momx_idx]  = rho_L * u_L;
#if AMREX_SPACEDIM >= 2
                QL_cons[momy_idx]  = rho_L * v_L;
#endif
#if AMREX_SPACEDIM == 3
                QL_cons[momz_idx]  = rho_L * w_L;
#endif
                QL_cons[etot_idx]  = rho_L * (ie_mix_L + ke_L);
                QL_cons[alpha_idx] = alpha_L; // advect alpha

                std::vector<Set::Scalar> QR_cons(num_components, 0.0);
                const Set::Scalar ke_R = 0.5 * (u_R*u_R + v_R*v_R + w_R*w_R);
                const Set::Scalar ie_mix_R = mixture_model->mixture_internal_energy(rho_R, p_R, alpha_R);
                QR_cons[m1_idx]   = m1_R;
                QR_cons[m2_idx]   = m2_R;
                QR_cons[momx_idx]  = rho_R * u_R;
#if AMREX_SPACEDIM >= 2
                QR_cons[momy_idx]  = rho_R * v_R;
#endif
#if AMREX_SPACEDIM == 3
                QR_cons[momz_idx]  = rho_R * w_R;
#endif
                QR_cons[etot_idx]  = rho_R * (ie_mix_R + ke_R);
                QR_cons[alpha_idx] = alpha_R; // advect alpha

                // 4. Compute Physical Fluxes (FL, FR)
                std::vector<Set::Scalar> FL(num_components, 0.0);
                FL[m1_idx]    = QL_cons[m1_idx] * vel_L_normal;
                FL[m2_idx]    = QL_cons[m2_idx] * vel_L_normal;
                // FL[momx_idx]  = QL_cons[momx_idx] * vel_L_normal;
                FL[momx_idx] = rho_L * u_L * vel_L_normal;
#if AMREX_SPACEDIM >= 2
                // FL[momy_idx]  = QL_cons[momy_idx] * vel_L_normal;
                FL[momy_idx] = rho_L * v_L * vel_L_normal;
#endif
#if AMREX_SPACEDIM == 3
                // FL[momz_idx]  = QL_cons[momz_idx] * vel_L_normal;
                FL[momz_idx] = rho_L * w_L * vel_L_normal;
#endif
                FL[momx_idx + normal] += p_L; // Add pressure term to normal momentum flux
                FL[etot_idx]  = (QL_cons[etot_idx] + p_L) * vel_L_normal;
                FL[alpha_idx] = QL_cons[alpha_idx] * vel_L_normal;

                std::vector<Set::Scalar> FR(num_components, 0.0);
                FR[m1_idx]    = QR_cons[m1_idx] * vel_R_normal;
                FR[m2_idx]    = QR_cons[m2_idx] * vel_R_normal;
                // FR[momx_idx]  = QR_cons[momx_idx] * vel_R_normal;
                FR[momx_idx] = rho_R * u_R * vel_R_normal;
#if AMREX_SPACEDIM >= 2
                // FR[momy_idx]  = QR_cons[momy_idx] * vel_R_normal;
                FR[momy_idx] = rho_R * v_R * vel_R_normal;
#endif
#if AMREX_SPACEDIM == 3
                // FR[momz_idx]  = QR_cons[momz_idx] * vel_R_normal;
                FR[momz_idx] = rho_R * w_R * vel_R_normal;
#endif
                FR[momx_idx + normal] += p_R; // Add pressure term to normal momentum flux
                FR[etot_idx]  = (QR_cons[etot_idx] + p_R) * vel_R_normal;
                FR[alpha_idx] = QR_cons[alpha_idx] * vel_R_normal;

                // 5. Compute HLLC Wave Speeds
                const Set::Scalar c_L = mixture_model->sound_speed_mixture(rho_L, p_L, alpha_L);
                const Set::Scalar c_R = mixture_model->sound_speed_mixture(rho_R, p_R, alpha_R);

                const Set::Scalar SL = amrex::min(vel_L_normal - c_L, vel_R_normal - c_R);
                const Set::Scalar SR = amrex::max(vel_L_normal + c_L, vel_R_normal + c_R);

                const Set::Scalar denom = (rho_L * (SL - vel_L_normal) - rho_R * (SR - vel_R_normal));
                // Get the normal component of momentum for S_star calculation
                const Set::Scalar mom_L_normal_cons = QL_cons[momx_idx + normal];
                const Set::Scalar mom_R_normal_cons = QR_cons[momx_idx + normal];
                
                const Set::Scalar S_star = (p_R - p_L + mom_L_normal_cons * (SL - vel_L_normal) - mom_R_normal_cons * (SR - vel_R_normal)) / (denom + 1e-16);
                

                // ---------- Face/interface velocity u_{i+1/2}
                const Set::Scalar S_minus = amrex::min(Set::Scalar(0.0), SL);  // S^- = min(0, S^L)
                const Set::Scalar S_plus  = amrex::max(Set::Scalar(0.0), SR);  // S^+ = max(0, S^R)

               
                const Set::Scalar sgnSstar = (S_star >= 0.0) ? Set::Scalar(1.0) : Set::Scalar(-1.0);

                const Set::Scalar HL = 0.5 * (1.0 + sgnSstar);  // (1 + sgn(S*))/2
                const Set::Scalar HR = 0.5 * (1.0 - sgnSstar);  // (1 - sgn(S*))/2

               
                const Set::Scalar ratioL = (SL - vel_L_normal) / (SL - S_star + 1e-16);
                const Set::Scalar ratioR = (SR - vel_R_normal) / (SR - S_star + 1e-16);

               
                // u_face = HL * [ uL + S^- ( (SL-uL)/(SL-S*) - 1 ) ] + HR * [ uR + S^+ ( (SR-uR)/(SR-S*) - 1 ) ]
                const Set::Scalar u_face =
                    HL * (vel_L_normal + S_minus * (ratioL - 1.0)) +
                    HR * (vel_R_normal + S_plus  * (ratioR - 1.0));

                // Store for source-term discretization (alpha * div(u))
                face_vel_arr(i, j, k, 0) = u_face;

            
                // 6. Compute Star States
                if (SL >= 0.0) {
                    // Flux is just FL
                    
                    for (int n = 0; n < num_components; ++n) {
                        TotalFlux_arr(index[0], index[1], index[2], n) = FL[n];
                    }
                } else if (SR <= 0.0) {
                    // Flux is just FR
                    

                    for (int n = 0; n < num_components; ++n) {
                        TotalFlux_arr(index[0], index[1], index[2], n) = FR[n];
                    }
                } else {
                    
                
                    std::vector<Set::Scalar> UstarL(num_components, 0.0);
                    std::vector<Set::Scalar> UstarR(num_components, 0.0);
                
                    // Common denominators
                    const Set::Scalar inv_SL_Sstar = 1.0 / (SL - S_star + 1e-16);
                    const Set::Scalar inv_SR_Sstar = 1.0 / (SR - S_star + 1e-16);
                
                   
                    const Set::Scalar factorL = (SL - vel_L_normal) * inv_SL_Sstar;
                
                    // m1*, m2*
                    UstarL[m1_idx] = QL_cons[m1_idx] * factorL;
                    UstarL[m2_idx] = QL_cons[m2_idx] * factorL;
                
                    // momentum: normal = rho* S*, transverse unchanged (carried) times factor
                    const Set::Scalar rho_L_tot = rho_L; // total mixture density already
                    UstarL[momx_idx + normal] = (rho_L_tot * factorL) * S_star;
                
                #if AMREX_SPACEDIM >= 2
                    UstarL[momx_idx + trans1] = QL_cons[momx_idx + trans1] * factorL;
                #endif
                #if AMREX_SPACEDIM == 3
                    UstarL[momx_idx + trans2] = QL_cons[momx_idx + trans2] * factorL;
                #endif
                
                
                    // E*_L = factorL * [ E_L + (S* - u_L) * ( rho_L * S* + P_L/(S_L - u_L) ) ]
                    const Set::Scalar inv_SL_minus_uL = 1.0 / (SL - vel_L_normal + 1e-16);
                    UstarL[etot_idx] =
                        factorL * ( QL_cons[etot_idx]
                                  + (S_star - vel_L_normal) * (rho_L_tot * S_star + p_L * inv_SL_minus_uL) );
                
                    // alpha* (per your python logic: do NOT scale; advect alpha itself)
                    UstarL[alpha_idx] = alpha_L * factorL;
                
                    
                    const Set::Scalar factorR = (SR - vel_R_normal) * inv_SR_Sstar;
                
                    // m1*, m2*
                    UstarR[m1_idx] = QR_cons[m1_idx] * factorR;
                    UstarR[m2_idx] = QR_cons[m2_idx] * factorR;
                
                    // momentum
                    const Set::Scalar rho_R_tot = rho_R;
                    UstarR[momx_idx + normal] = (rho_R_tot * factorR) * S_star;
                
                #if AMREX_SPACEDIM >= 2
                    UstarR[momx_idx + trans1] = QR_cons[momx_idx + trans1] * factorR;
                #endif
                #if AMREX_SPACEDIM == 3
                    UstarR[momx_idx + trans2] = QR_cons[momx_idx + trans2] * factorR;
                #endif
                
                   
                    // E*_R = factorR * [ E_R + (S* - u_R) * ( rho_R * S* + P_R/(S_R - u_R) ) ]
                    const Set::Scalar inv_SR_minus_uR = 1.0 / (SR - vel_R_normal + 1e-16);
                    UstarR[etot_idx] =
                        factorR * ( QR_cons[etot_idx]
                                  + (S_star - vel_R_normal) * (rho_R_tot * S_star + p_R * inv_SR_minus_uR) );
                
                    // alpha*
                    UstarR[alpha_idx] = alpha_R*factorR;
                
                    // 7. Choose Final HLLC Flux (unchanged)
                    if (S_star >= 0.0) {
                        // F = F_L + S_L * (Ustar_L - U_L)
                        for (int n = 0; n < num_components; ++n) {
                            TotalFlux_arr(index[0], index[1], index[2], n) = FL[n] + SL * (UstarL[n] - QL_cons[n]);
                        }
                    } else {
                        // F = F_R + S_R * (Ustar_R - U_R)
                        for (int n = 0; n < num_components; ++n) {
                            TotalFlux_arr(index[0], index[1], index[2], n) = FR[n] + SR * (UstarR[n] - QR_cons[n]);
                        }
                    }
                }


            } else {
            // --- MODIFICATION END ---
            
                // --- ORIGINAL EULER HLLC FLUX ---
                

                // Get LEFT conservative variables from reconstructed state
                Set::Scalar rho_L = Left_arr(index[0], index[1], index[2], solver->variableIndex.DENS);
                rho_L = std::max(rho_L, 1e-8);  // Avoid zero density

                Set::Scalar mom_L_normal = Left_arr(index[0], index[1], index[2], velocity_component[normal]);
                Set::Scalar vel_L_normal = mom_L_normal / rho_L;

                Set::Scalar vel_L[3] = {0.0, 0.0, 0.0};
                vel_L[normal] = vel_L_normal;

#if AMREX_SPACEDIM >= 2    
                Set::Scalar mom_L_trans1 = Left_arr(index[0], index[1], index[2], velocity_component[trans1]);
                vel_L[trans1] = mom_L_trans1 / rho_L;
#endif
#if AMREX_SPACEDIM == 3    
                Set::Scalar mom_L_trans2 = Left_arr(index[0], index[1], index[2], velocity_component[trans2]);
                vel_L[trans2] = mom_L_trans2 / rho_L;
#endif

                Set::Scalar rhoE_L = Left_arr(index[0], index[1], index[2], solver->variableIndex.IE);
                Set::Scalar E_L = rhoE_L / rho_L;

                // Calculate kinetic energy for left state using symmetry-preserving operation
                Set::Scalar KE_L = Numeric::SymmetryPreserving::ConsistentKineticEnergy(
                    rho_L, vel_L[0], vel_L[1], vel_L[2]);

                // Calculate internal energy and pressure for left state
                Set::Scalar e_L = E_L - KE_L / rho_L;
                Set::Scalar pressure_L = (gamma - 1.0) * rho_L * e_L;
                pressure_L = std::max(pressure_L, 1e-8);  // Ensure positive pressure

                // Get RIGHT conservative variables from reconstructed state
                Set::Scalar rho_R = Right_arr(index[0], index[1], index[2], solver->variableIndex.DENS);
                rho_R = std::max(rho_R, 1e-8);  // Avoid zero density

                Set::Scalar mom_R_normal = Right_arr(index[0], index[1], index[2], velocity_component[normal]);
                Set::Scalar vel_R_normal = mom_R_normal / rho_R;

                Set::Scalar vel_R[3] = {0.0, 0.0, 0.0};
                vel_R[normal] = vel_R_normal;

#if AMREX_SPACEDIM >= 2    
                Set::Scalar mom_R_trans1 = Right_arr(index[0], index[1], index[2], velocity_component[trans1]);
                vel_R[trans1] = mom_R_trans1 / rho_R;
#endif
#if AMREX_SPACEDIM == 3    
                Set::Scalar mom_R_trans2 = Right_arr(index[0], index[1], index[2], velocity_component[trans2]);
                vel_R[trans2] = mom_R_trans2 / rho_R;
#endif

                Set::Scalar rhoE_R = Right_arr(index[0], index[1], index[2], solver->variableIndex.IE);
                Set::Scalar E_R = rhoE_R / rho_R;

                // Calculate kinetic energy for right state using symmetry-preserving operation
                Set::Scalar KE_R = Numeric::SymmetryPreserving::ConsistentKineticEnergy(
                    rho_R, vel_R[0], vel_R[1], vel_R[2]);

                // Calculate internal energy and pressure for right state
                Set::Scalar e_R = E_R - KE_R / rho_R;
                Set::Scalar pressure_R = (gamma - 1.0) * rho_R * e_R;
                pressure_R = std::max(pressure_R, 1e-8);  // Ensure positive pressure

                // Sound speeds
                Set::Scalar c_L = std::sqrt(gamma * pressure_L / rho_L);
                Set::Scalar c_R = std::sqrt(gamma * pressure_R / rho_R);

                // Store conservative variables for reconstructed left and right states
                std::vector<Set::Scalar> QL_cons(num_components, 0.0);
                std::vector<Set::Scalar> QR_cons(num_components, 0.0);

                // These are already in conservative form from the reconstruction
                QL_cons[solver->variableIndex.DENS] = rho_L;
                QL_cons[solver->variableIndex.UVEL + normal] = rho_L * vel_L[normal];
#if AMREX_SPACEDIM >= 2
                QL_cons[solver->variableIndex.UVEL + trans1] = rho_L * vel_L[trans1];
#endif
#if AMREX_SPACEDIM == 3
                QL_cons[solver->variableIndex.UVEL + trans2] = rho_L * vel_L[trans2];
#endif
                QL_cons[solver->variableIndex.IE] = rho_L * E_L;

                QR_cons[solver->variableIndex.DENS] = rho_R;
                QR_cons[solver->variableIndex.UVEL + normal] = rho_R * vel_R[normal];
#if AMREX_SPACEDIM >= 2
                QR_cons[solver->variableIndex.UVEL + trans1] = rho_R * vel_R[trans1];
#endif
#if AMREX_SPACEDIM == 3
                QR_cons[solver->variableIndex.UVEL + trans2] = rho_R * vel_R[trans2];
#endif
                QR_cons[solver->variableIndex.IE] = rho_R * E_R;

                // Compute physical fluxes for left and right states
                std::vector<Set::Scalar> FL(num_components, 0.0);
                std::vector<Set::Scalar> FR(num_components, 0.0);

                // Mass flux
                FL[solver->variableIndex.DENS] = rho_L * vel_L[normal];
                FR[solver->variableIndex.DENS] = rho_R * vel_R[normal];

                // Momentum fluxes - use consistent operations
                FL[solver->variableIndex.UVEL + normal] = rho_L * vel_L[normal] * vel_L[normal] + pressure_L;
                FR[solver->variableIndex.UVEL + normal] = rho_R * vel_R[normal] * vel_R[normal] + pressure_R;

#if AMREX_SPACEDIM >= 2
                FL[solver->variableIndex.UVEL + trans1] = rho_L * vel_L[normal] * vel_L[trans1];
                FR[solver->variableIndex.UVEL + trans1] = rho_R * vel_R[normal] * vel_R[trans1];
#endif
#if AMREX_SPACEDIM == 3
                FL[solver->variableIndex.UVEL + trans2] = rho_L * vel_L[normal] * vel_L[trans2];
                FR[solver->variableIndex.UVEL + trans2] = rho_R * vel_R[normal] * vel_R[trans2];
#endif

                // Energy flux - use consistent operations
                FL[solver->variableIndex.IE] = (rho_L * E_L + pressure_L) * vel_L[normal];
                FR[solver->variableIndex.IE] = (rho_R * E_R + pressure_R) * vel_R[normal];

                // Wave speeds using Davis estimates (more robust than Roe averages)
                Set::Scalar SL = std::min(vel_L[normal] - c_L, vel_R[normal] - c_R);
                Set::Scalar SR = std::max(vel_L[normal] + c_L, vel_R[normal] + c_R);

                // Apply entropy fix for transonic rarefactions
                Set::Scalar epsilon = 0.1 * std::min(c_L, c_R); 
                if (std::abs(SR - SL) < epsilon) {
                    SR += epsilon;
                    SL -= epsilon;    
                }   

                // Calculate the middle wave speed (S*) using the symmetry-preserving operation
                Set::Scalar S_star = Numeric::SymmetryPreserving::ConsistentContactWaveSpeed(
                    pressure_R, pressure_L, 
                    rho_L, vel_L[normal], SL, 
                    rho_R, vel_R[normal], SR);

                // Compute star state variables
                std::vector<Set::Scalar> QL_star(num_components, 0.0);
                std::vector<Set::Scalar> QR_star(num_components, 0.0);

                // Density factors for star regions
                Set::Scalar rho_L_star = rho_L * (SL - vel_L[normal]) / (SL - S_star);
                Set::Scalar rho_R_star = rho_R * (SR - vel_R[normal]) / (SR - S_star);

                // Left star state
                QL_star[solver->variableIndex.DENS] = rho_L_star;
                QL_star[solver->variableIndex.UVEL + normal] = rho_L_star * S_star;
#if AMREX_SPACEDIM >= 2
                QL_star[solver->variableIndex.UVEL + trans1] = rho_L_star * vel_L[trans1];
#endif
#if AMREX_SPACEDIM == 3
                QL_star[solver->variableIndex.UVEL + trans2] = rho_L_star * vel_L[trans2];
#endif
                // Use consistent operations for the star state energy
                Set::Scalar pressureTerm_L = pressure_L / (rho_L * (SL - vel_L[normal]));
                Set::Scalar velocityDiff_L = S_star - vel_L[normal];

                QL_star[solver->variableIndex.IE] = rho_L_star * (E_L + velocityDiff_L * (S_star + pressureTerm_L));

                // Right star state
                QR_star[solver->variableIndex.DENS] = rho_R_star;
                QR_star[solver->variableIndex.UVEL + normal] = rho_R_star * S_star;
#if AMREX_SPACEDIM >= 2
                QR_star[solver->variableIndex.UVEL + trans1] = rho_R_star * vel_R[trans1];
#endif
#if AMREX_SPACEDIM == 3
                QR_star[solver->variableIndex.UVEL + trans2] = rho_R_star * vel_R[trans2];
#endif
                // Use consistent operations for the star state energy
                Set::Scalar pressureTerm_R = pressure_R / (rho_R * (SR - vel_R[normal]));
                Set::Scalar velocityDiff_R = S_star - vel_R[normal];

                QR_star[solver->variableIndex.IE] = rho_R_star * (E_R + velocityDiff_R * (S_star + pressureTerm_R));

                // Use a centralized formulation for computing consistent fluxes
                // Precompute star region fluxes
                std::vector<Set::Scalar> FL_star(num_components, 0.0);
                std::vector<Set::Scalar> FR_star(num_components, 0.0);

                for (int n = 0; n < num_components; ++n) {
                    FL_star[n] = FL[n] + SL * (QL_star[n] - QL_cons[n]);
                    FR_star[n] = FR[n] + SR * (QR_star[n] - QR_cons[n]);
                }

                // Define Heaviside-like functions for the wave speeds - outside the loop since they're the same for all components
                Set::Scalar HL = 0.5 * (1.0 + std::copysign(1.0, SL));     // H(SL): 1 if SL >= 0, 0 if SL < 0
                Set::Scalar HStar = 0.5 * (1.0 + std::copysign(1.0, S_star)); // H(S*): 1 if S* >= 0, 0 if S* < 0
                Set::Scalar HR = 0.5 * (1.0 + std::copysign(1.0, SR));     // H(SR): 1 if SR >= 0, 0 if SR < 0

                // Branch-free HLLC flux calculation
                for (int n = 0; n < num_components; ++n) {
                    // Compute full HLLC flux using branch-free formulation
                    Set::Scalar HLLC_flux = 
                        HL * FL[n] +                                  // Case: SL >= 0
                        (1.0 - HL) * HStar * FL_star[n] +            // Case: SL < 0, S* >= 0
                        (1.0 - HStar) * HR * FR_star[n] +            // Case: S* < 0, SR >= 0
                        (1.0 - HR) * FR[n];                          // Case: SR < 0

                    // Store the computed flux
                    TotalFlux_arr(index[0], index[1], index[2], n) = HLLC_flux;
                }

            } // --- MODIFICATION END (end of if/else) ---

        });
    }
}

};



/**
* @class HLLCLowMachMethod
* @brief HLLC flux with reduced dissipation for low Mach regions
* 
* This is an implementation of the HLLC-LM method as described in:
* - Fleischmann, N., Adami, S., & Adams, N. A. (2020). A shock-stable modification 
*   of the HLLC Riemann solver with reduced numerical dissipation. Journal of 
*   Computational Physics, Volume 423, 109762.
* 
* The method reduces acoustic dissipation in low Mach number regions to prevent
* grid-aligned shock instabilities while maintaining accuracy.
*/

// HLLCLowMach flux method with reduced dissipation for improved shock stability
/*template <typename IntegratorDerivedClass>
class HLLCLowMachMethod : public FluxMethod<IntegratorDerivedClass> {
public:
    using typename FluxMethod<IntegratorDerivedClass>::ReconstructionTarget;

    // Use variable-based reconstruction like the base HLLC
    ReconstructionTarget getReconstructionTarget() const override {
        return ReconstructionTarget::Variables;
    }

    void SplitFluxes(int direction, int lev, IntegratorDerivedClass* solver, 
                    amrex::MultiFab& VariableBuffer, amrex::MultiFab& CellFluxBuffer,
                    amrex::MultiFab& Flux_Negative, amrex::MultiFab& Flux_Positive) const override {
        // Not the primary pathway for HLLCLowMach
        Util::Warning(INFO, "HLLCLowMach SplitFluxes called - not optimal path");
    }

    void ComputeFluxes(int direction, int lev, IntegratorDerivedClass* solver,
                    const amrex::MultiFab& LeftStates, const amrex::MultiFab& RightStates,
                    amrex::MultiFab& SummedFlux) const override {

        const Set::Scalar gamma = 1.4;  // Ratio of specific heats
        const int nghosts = solver->number_of_ghost_cells;
        const Set::Scalar Ma_limit = 0.1; // Mach number limit for applying dissipation reduction

        // Loop through the grid
        for (amrex::MFIter mfi(SummedFlux, false); mfi.isValid(); ++mfi) {
            const amrex::Box& face_bx_with_ghosts = mfi.grownnodaltilebox(direction, nghosts);

            // Get arrays for left/right reconstructed states and flux output
            auto const& Left_arr = LeftStates.array(mfi);
            auto const& Right_arr = RightStates.array(mfi);
            auto const& TotalFlux_arr = SummedFlux.array(mfi);

            // Direction-dependent indices
            int normal = direction;  // Normal velocity component
            int trans1 = (direction + 1) % AMREX_SPACEDIM;  // First transverse velocity
#if AMREX_SPACEDIM == 3
            int trans2 = (direction + 2) % AMREX_SPACEDIM;  // Second transverse velocity (3D only)
#endif
            // Map direction to velocity components
            int velocity_component[3] = {
                solver->variableIndex.UVEL,  // X-direction -> UVEL
                solver->variableIndex.VVEL,  // Y-direction -> VVEL
                solver->variableIndex.WVEL   // Z-direction -> WVEL
            };

            int num_components = solver->number_of_components;

            // Parallel loop over cells
            amrex::ParallelFor(face_bx_with_ghosts, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                // Setup indices
                int index[3];
                int lower_bounds[3];
                int upper_bounds[3];

                // Initialize index arrays
                index[0] = i; index[1] = j; index[2] = k;

                // Define bounds including ghost cells
                lower_bounds[0] = face_bx_with_ghosts.smallEnd(0); 
                lower_bounds[1] = face_bx_with_ghosts.smallEnd(1); 
                lower_bounds[2] = AMREX_SPACEDIM == 3 ? face_bx_with_ghosts.smallEnd(2) : 0;
                upper_bounds[0] = face_bx_with_ghosts.bigEnd(0) - 1;
                upper_bounds[1] = face_bx_with_ghosts.bigEnd(1) - 1;
                upper_bounds[2] = AMREX_SPACEDIM == 3 ? face_bx_with_ghosts.bigEnd(2) - 1 : 0;

                // Clamp indices to ensure they're within bounds
                ClampIndices(index, lower_bounds, upper_bounds);

                // Get LEFT conservative variables from reconstructed state
                Set::Scalar rho_L = Left_arr(index[0], index[1], index[2], solver->variableIndex.DENS);
                rho_L = std::max(rho_L, 1e-8);  // Avoid zero density

                Set::Scalar mom_L_normal = Left_arr(index[0], index[1], index[2], velocity_component[normal]);
                Set::Scalar vel_L_normal = mom_L_normal / rho_L;

                Set::Scalar vel_L[3] = {0.0, 0.0, 0.0};
                vel_L[normal] = vel_L_normal;

#if AMREX_SPACEDIM >= 2    
                Set::Scalar mom_L_trans1 = Left_arr(index[0], index[1], index[2], velocity_component[trans1]);
                vel_L[trans1] = mom_L_trans1 / rho_L;
#endif
#if AMREX_SPACEDIM == 3    
                Set::Scalar mom_L_trans2 = Left_arr(index[0], index[1], index[2], velocity_component[trans2]);
                vel_L[trans2] = mom_L_trans2 / rho_L;
#endif

                Set::Scalar rhoE_L = Left_arr(index[0], index[1], index[2], solver->variableIndex.IE);
                Set::Scalar E_L = rhoE_L / rho_L;

                // Calculate kinetic energy for left state using symmetry-preserving operation
                Set::Scalar KE_L = Numeric::SymmetryPreserving::ConsistentKineticEnergy(
                    rho_L, vel_L[0], vel_L[1], vel_L[2]);

                // Calculate internal energy and pressure for left state
                Set::Scalar e_L = E_L - KE_L / rho_L;
                Set::Scalar pressure_L = (gamma - 1.0) * rho_L * e_L;
                pressure_L = std::max(pressure_L, 1e-8);  // Ensure positive pressure

                // Get RIGHT conservative variables from reconstructed state
                Set::Scalar rho_R = Right_arr(index[0], index[1], index[2], solver->variableIndex.DENS);
                rho_R = std::max(rho_R, 1e-8);  // Avoid zero density

                Set::Scalar mom_R_normal = Right_arr(index[0], index[1], index[2], velocity_component[normal]);
                Set::Scalar vel_R_normal = mom_R_normal / rho_R;

                Set::Scalar vel_R[3] = {0.0, 0.0, 0.0};
                vel_R[normal] = vel_R_normal;

#if AMREX_SPACEDIM >= 2    
                Set::Scalar mom_R_trans1 = Right_arr(index[0], index[1], index[2], velocity_component[trans1]);
                vel_R[trans1] = mom_R_trans1 / rho_R;
#endif
#if AMREX_SPACEDIM == 3    
                Set::Scalar mom_R_trans2 = Right_arr(index[0], index[1], index[2], velocity_component[trans2]);
                vel_R[trans2] = mom_R_trans2 / rho_R;
#endif

                Set::Scalar rhoE_R = Right_arr(index[0], index[1], index[2], solver->variableIndex.IE);
                Set::Scalar E_R = rhoE_R / rho_R;

                // Calculate kinetic energy for right state using symmetry-preserving operation
                Set::Scalar KE_R = Numeric::SymmetryPreserving::ConsistentKineticEnergy(
                    rho_R, vel_R[0], vel_R[1], vel_R[2]);

                // Calculate internal energy and pressure for right state
                Set::Scalar e_R = E_R - KE_R / rho_R;
                Set::Scalar pressure_R = (gamma - 1.0) * rho_R * e_R;
                pressure_R = std::max(pressure_R, 1e-8);  // Ensure positive pressure

                // Sound speeds
                Set::Scalar c_L = std::sqrt(gamma * pressure_L / rho_L);
                Set::Scalar c_R = std::sqrt(gamma * pressure_R / rho_R);

                // Store conservative variables for reconstructed left and right states
                std::vector<Set::Scalar> QL_cons(num_components, 0.0);
                std::vector<Set::Scalar> QR_cons(num_components, 0.0);

                // These are already in conservative form from the reconstruction
                QL_cons[solver->variableIndex.DENS] = rho_L;
                QL_cons[solver->variableIndex.UVEL + normal] = rho_L * vel_L[normal];
#if AMREX_SPACEDIM >= 2
                QL_cons[solver->variableIndex.UVEL + trans1] = rho_L * vel_L[trans1];
#endif
#if AMREX_SPACEDIM == 3
                QL_cons[solver->variableIndex.UVEL + trans2] = rho_L * vel_L[trans2];
#endif
                QL_cons[solver->variableIndex.IE] = rho_L * E_L;

                QR_cons[solver->variableIndex.DENS] = rho_R;
                QR_cons[solver->variableIndex.UVEL + normal] = rho_R * vel_R[normal];
#if AMREX_SPACEDIM >= 2
                QR_cons[solver->variableIndex.UVEL + trans1] = rho_R * vel_R[trans1];
#endif
#if AMREX_SPACEDIM == 3
                QR_cons[solver->variableIndex.UVEL + trans2] = rho_R * vel_R[trans2];
#endif
                QR_cons[solver->variableIndex.IE] = rho_R * E_R;

                // Compute physical fluxes for left and right states
                std::vector<Set::Scalar> FL(num_components, 0.0);
                std::vector<Set::Scalar> FR(num_components, 0.0);

                // Mass flux
                FL[solver->variableIndex.DENS] = rho_L * vel_L[normal];
                FR[solver->variableIndex.DENS] = rho_R * vel_R[normal];

                // Momentum fluxes
                FL[solver->variableIndex.UVEL + normal] = rho_L * vel_L[normal] * vel_L[normal] + pressure_L;
                FR[solver->variableIndex.UVEL + normal] = rho_R * vel_R[normal] * vel_R[normal] + pressure_R;

#if AMREX_SPACEDIM >= 2
                FL[solver->variableIndex.UVEL + trans1] = rho_L * vel_L[normal] * vel_L[trans1];
                FR[solver->variableIndex.UVEL + trans1] = rho_R * vel_R[normal] * vel_R[trans1];
#endif
#if AMREX_SPACEDIM == 3
                FL[solver->variableIndex.UVEL + trans2] = rho_L * vel_L[normal] * vel_L[trans2];
                FR[solver->variableIndex.UVEL + trans2] = rho_R * vel_R[normal] * vel_R[trans2];
#endif

                // Energy flux
                FL[solver->variableIndex.IE] = (rho_L * E_L + pressure_L) * vel_L[normal];
                FR[solver->variableIndex.IE] = (rho_R * E_R + pressure_R) * vel_R[normal];

                // Non-modified signal speeds (original HLLC)
                Set::Scalar SL_orig = std::min(vel_L[normal] - c_L, vel_R[normal] - c_R);
                Set::Scalar SR_orig = std::max(vel_L[normal] + c_L, vel_R[normal] + c_R);

                // Local Mach number calculation - use the maximum of left and right
                Set::Scalar Ma_local = std::max(std::abs(vel_L[normal])/c_L, std::abs(vel_R[normal])/c_R);

                // Calculate phi reduction factor based on local Mach number
                // Using the sine function as described in the paper
                Set::Scalar phi = std::sin(std::min(1.0, Ma_local/Ma_limit) * (3.14159265358979323846/2.0));

                // Apply the low Mach number modification to signal speeds as per equation (23)
                Set::Scalar SL = phi * SL_orig;
                Set::Scalar SR = phi * SR_orig;

                // Apply entropy fix for transonic rarefactions
                Set::Scalar epsilon = 0.1 * std::min(c_L, c_R); 
                if (std::abs(SR - SL) < epsilon) {
                    SR += epsilon;
                    SL -= epsilon;    
                }   

                // Calculate the middle wave speed (S*) using symmetry-preserving operation
                Set::Scalar S_star = Numeric::SymmetryPreserving::ConsistentContactWaveSpeed(
                    pressure_R, pressure_L, rho_L, vel_L[normal], SL_orig, rho_R, vel_R[normal], SR_orig);

                // Compute star state variables
                std::vector<Set::Scalar> QL_star(num_components, 0.0);
                std::vector<Set::Scalar> QR_star(num_components, 0.0);

                // Density factors for star regions - use original (non-modified) signal speeds for star state computation
                Set::Scalar rho_L_star = rho_L * (SL_orig - vel_L[normal]) / (SL_orig - S_star);
                Set::Scalar rho_R_star = rho_R * (SR_orig - vel_R[normal]) / (SR_orig - S_star);

                // Left star state
                QL_star[solver->variableIndex.DENS] = rho_L_star;
                QL_star[solver->variableIndex.UVEL + normal] = rho_L_star * S_star;
#if AMREX_SPACEDIM >= 2
                QL_star[solver->variableIndex.UVEL + trans1] = rho_L_star * vel_L[trans1];
#endif
#if AMREX_SPACEDIM == 3
                QL_star[solver->variableIndex.UVEL + trans2] = rho_L_star * vel_L[trans2];
#endif
                QL_star[solver->variableIndex.IE] = rho_L_star * (E_L + (S_star - vel_L[normal]) * 
                                                (S_star + pressure_L / (rho_L * (SL_orig - vel_L[normal]))));

                // Right star state
                QR_star[solver->variableIndex.DENS] = rho_R_star;
                QR_star[solver->variableIndex.UVEL + normal] = rho_R_star * S_star;
#if AMREX_SPACEDIM >= 2
                QR_star[solver->variableIndex.UVEL + trans1] = rho_R_star * vel_R[trans1];
#endif
#if AMREX_SPACEDIM == 3
                QR_star[solver->variableIndex.UVEL + trans2] = rho_R_star * vel_R[trans2];
#endif
                QR_star[solver->variableIndex.IE] = rho_R_star * (E_R + (S_star - vel_R[normal]) * 
                                                (S_star + pressure_R / (rho_R * (SR_orig - vel_R[normal]))));

                // Precompute star region fluxes (using centralized formulation from equation 7)
                std::vector<Set::Scalar> FL_star(num_components, 0.0);
                std::vector<Set::Scalar> FR_star(num_components, 0.0);

                for (int n = 0; n < num_components; ++n) {
                    FL_star[n] = FL[n] + SL * (QL_star[n] - QL_cons[n]);
                    FR_star[n] = FR[n] + SR * (QR_star[n] - QR_cons[n]);
                }

                // Centralized HLLC-LM flux computation as per equation (19)
                std::vector<Set::Scalar> HLLC_flux(num_components, 0.0);
                // Branch-free centralized HLLC-LM flux computation
                for (int n = 0; n < num_components; ++n) {
                    // Create Heaviside-like functions for SL and SR using sign bits
                    Set::Scalar HSL = 0.5 * (1.0 + std::copysign(1.0, SL));  // 1 if SL >= a0, 0 otherwise
                    Set::Scalar HSR = 0.5 * (1.0 + std::copysign(1.0, SR));  // 1 if SR >= 0, 0 otherwise
                    Set::Scalar HStar = 0.5 * (1.0 + std::copysign(1.0, S_star));  // 1 if S* >= 0, 0 otherwise

                    // Calculate star region dissipation contributions
                    Set::Scalar term1 = SL * (QL_star[n] - QL_cons[n]);
                    Set::Scalar term2 = std::abs(S_star) * (2.0 * HStar - 1.0) * (QL_star[n] - QR_star[n]);
                    Set::Scalar term3 = SR * (QR_star[n] - QR_cons[n]);

                    // Use consistent sum for the three dissipation terms
                    Set::Scalar subsonic_dissipation = Numeric::SymmetryPreserving::ConsistentSum3(term1, term2, term3);

                    // Combine fluxes without branching:
                    // If SL >= 0: FL[n]
                    // If SR <= 0: FR[n]
                    // Otherwise: 0.5*(FL[n]+FR[n]) + 0.5*dissipation
                    HLLC_flux[n] = HSL * FL[n] + 
                                    (1.0 - HSR) * FR[n] + 
                                    (1.0 - HSL) * HSR * (0.5 * (FL[n] + FR[n]) + 0.5 * subsonic_dissipation);

                    TotalFlux_arr(index[0], index[1], index[2], n) = HLLC_flux[n];
                }

            });
        }
    }
};*/

// AUSM+up flux method with symmetry-preserving operations
template <typename IntegratorDerivedClass>
class AUSMupMethod : public FluxMethod<IntegratorDerivedClass> {
public:
    
    using typename FluxMethod<IntegratorDerivedClass>::ReconstructionTarget;

    ReconstructionTarget getReconstructionTarget() const override {
        return ReconstructionTarget::Variables;
    }
   
    void SplitFluxes(int /*direction*/, int /*lev*/, IntegratorDerivedClass* /*solver*/, 
                    amrex::MultiFab& /*VariableBuffer*/, amrex::MultiFab& /*CellFluxBuffer*/,
                    amrex::MultiFab& /*Flux_Negative*/, amrex::MultiFab& /*Flux_Positive*/) const override {
        // Not the primary pathway for AUSM+up
        Util::Warning(INFO, "AUSM+up SplitFluxes called - not optimal path");
    }
   
void ComputeFluxes(int direction, int /*lev*/, IntegratorDerivedClass* solver,
            const amrex::MultiFab& LeftStates, const amrex::MultiFab& RightStates,
            amrex::MultiFab& SummedFlux) const override {

    const Set::Scalar gamma = 1.4;  // Ratio of specific heats
    const int nghosts = solver->number_of_ghost_cells;

    //Util::ScimitarX_Util::Debug::SetTargetDebugLocationIndices(100, 25, 1, "true");

    // Loop through the grid
    for (amrex::MFIter mfi(SummedFlux, false); mfi.isValid(); ++mfi) {
        const amrex::Box& face_bx_with_ghosts = mfi.grownnodaltilebox(direction, nghosts);

        // Get arrays for left/right reconstructed states and flux output
        auto const& Left_arr = LeftStates.array(mfi);
        auto const& Right_arr = RightStates.array(mfi);
        auto const& TotalFlux_arr = SummedFlux.array(mfi);
        
        // Direction-dependent indices
        int normal = direction;  // Normal velocity component
        int trans1 = (direction + 1) % AMREX_SPACEDIM;  // First transverse velocity
#if AMREX_SPACEDIM == 3
        int trans2 = (direction + 2) % AMREX_SPACEDIM;  // Second transverse velocity (3D only)
#endif
        // Map direction to velocity components
        int velocity_component[3] = {
            solver->variableIndex.UVEL,  // X-direction -> UVEL
            solver->variableIndex.VVEL,  // Y-direction -> VVEL
            solver->variableIndex.WVEL   // Z-direction -> WVEL
        };

        int num_components = solver->number_of_components;

        // Parallel loop over cells
        amrex::ParallelFor(face_bx_with_ghosts, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            // Setup indices
            int index[3];
            int lower_bounds[3];
            int upper_bounds[3];
            
            // Initialize index arrays
            index[0] = i; index[1] = j; index[2] = k;
            
            // Define bounds including ghost cells
            lower_bounds[0] = face_bx_with_ghosts.smallEnd(0); 
            lower_bounds[1] = face_bx_with_ghosts.smallEnd(1); 
            lower_bounds[2] = AMREX_SPACEDIM == 3 ? face_bx_with_ghosts.smallEnd(2) : 0;
            upper_bounds[0] = face_bx_with_ghosts.bigEnd(0) - 1;
            upper_bounds[1] = face_bx_with_ghosts.bigEnd(1) - 1;
            upper_bounds[2] = AMREX_SPACEDIM == 3 ? face_bx_with_ghosts.bigEnd(2) - 1 : 0;
            
            // Clamp indices to ensure they're within bounds
            ClampIndices(index, lower_bounds, upper_bounds);


            // Get LEFT conservative variables from reconstructed state
            Set::Scalar rho_L = Left_arr(index[0], index[1], index[2], solver->variableIndex.DENS);
            rho_L = std::max(rho_L, 1e-8);  // Avoid zero density
            
            Set::Scalar mom_L_normal = Left_arr(index[0], index[1], index[2], velocity_component[normal]);
            Set::Scalar vel_L_normal = mom_L_normal / rho_L;
            
            Set::Scalar vel_L[3] = {0.0, 0.0, 0.0};
            vel_L[normal] = vel_L_normal;
            
#if AMREX_SPACEDIM >= 2    
            Set::Scalar mom_L_trans1 = Left_arr(index[0], index[1], index[2], velocity_component[trans1]);
            vel_L[trans1] = mom_L_trans1 / rho_L;
#endif
#if AMREX_SPACEDIM == 3    
            Set::Scalar mom_L_trans2 = Left_arr(index[0], index[1], index[2], velocity_component[trans2]);
            vel_L[trans2] = mom_L_trans2 / rho_L;
#endif
            
            Set::Scalar rhoE_L = Left_arr(index[0], index[1], index[2], solver->variableIndex.IE);
            Set::Scalar E_L = rhoE_L / rho_L;
            
            // Calculate kinetic energy for left state using symmetry-preserving operation
            Set::Scalar KE_L = Numeric::SymmetryPreserving::ConsistentKineticEnergy(
                rho_L, vel_L[0], vel_L[1], vel_L[2]);
            
            // Calculate internal energy and pressure for left state
            Set::Scalar e_L = E_L - KE_L / rho_L;
            Set::Scalar pressure_L = (gamma - 1.0) * rho_L * e_L;
            pressure_L = std::max(pressure_L, 1e-8);  // Ensure positive pressure
            
            // Get RIGHT conservative variables from reconstructed state
            Set::Scalar rho_R = Right_arr(index[0], index[1], index[2], solver->variableIndex.DENS);
            rho_R = std::max(rho_R, 1e-8);  // Avoid zero density
            
            Set::Scalar mom_R_normal = Right_arr(index[0], index[1], index[2], velocity_component[normal]);
            Set::Scalar vel_R_normal = mom_R_normal / rho_R;
            
            Set::Scalar vel_R[3] = {0.0, 0.0, 0.0};
            vel_R[normal] = vel_R_normal;
            
#if AMREX_SPACEDIM >= 2    
            Set::Scalar mom_R_trans1 = Right_arr(index[0], index[1], index[2], velocity_component[trans1]);
            vel_R[trans1] = mom_R_trans1 / rho_R;
#endif
#if AMREX_SPACEDIM == 3    
            Set::Scalar mom_R_trans2 = Right_arr(index[0], index[1], index[2], velocity_component[trans2]);
            vel_R[trans2] = mom_R_trans2 / rho_R;
#endif
            
            Set::Scalar rhoE_R = Right_arr(index[0], index[1], index[2], solver->variableIndex.IE);
            Set::Scalar E_R = rhoE_R / rho_R;
            
            // Calculate kinetic energy for right state using symmetry-preserving operation
            Set::Scalar KE_R = Numeric::SymmetryPreserving::ConsistentKineticEnergy(
                rho_R, vel_R[0], vel_R[1], vel_R[2]);
            
            // Calculate internal energy and pressure for right state
            Set::Scalar e_R = E_R - KE_R / rho_R;
            Set::Scalar pressure_R = (gamma - 1.0) * rho_R * e_R;
            pressure_R = std::max(pressure_R, 1e-8);  // Ensure positive pressure
            
            // Sound speeds
            //Set::Scalar c_L = std::sqrt(gamma * pressure_L / rho_L);
            //Set::Scalar c_R = std::sqrt(gamma * pressure_R / rho_R);

        // ######################################
            // Define specific total Enthalpy H_total 
        Set::Scalar H_total_L = E_L + pressure_L/rho_L;  // = Cv.T + (Vel^2)/2 + P/rho = specific Total Enthalpy
        Set::Scalar H_total_R = E_R + pressure_R/rho_R;

        // Define critical speed of sound c_star
        Set::Scalar c_star_L = std::sqrt(2*(gamma-1)*H_total_L/(gamma+1));
        Set::Scalar c_star_R = std::sqrt(2*(gamma-1)*H_total_R/(gamma+1));
                    
        // Define c_L_hat and c_R_hat
        Set::Scalar c_L_hat = std::pow(c_star_L,2)/nonZeroValue(std::max(c_star_L,std::abs(vel_L[normal])));
        Set::Scalar c_R_hat = std::pow(c_star_R,2)/nonZeroValue(std::max(c_star_R,std::abs(vel_R[normal])));

        // Define interface acoustic speed c_interface
        //Set::Scalar c_interface = std::sqrt(c_L*c_R); //simple geometric mean
        Set::Scalar c_interface = std::min(c_L_hat,c_R_hat); // using total enthalpy and critical acoustic speed

        // Define M_L and M_R
        Set::Scalar M_L = vel_L[normal]/nonZeroValue(c_interface);
        Set::Scalar M_R = vel_R[normal]/nonZeroValue(c_interface);

        // Define M_localMean, M_inf  and M0
        Set::Scalar M_localMean = std::sqrt(0.5*(M_L*M_L + M_R*M_R));
        Set::Scalar M_inf = 1.0; //*** has to be user input and SHOULD NOT BE ZERO
        Set::Scalar M0 = std::sqrt(std::min(1.0,std::max(M_localMean*M_localMean, M_inf*M_inf)));

        // Define scaling function fa, interface density rho_interface, sigma and Kp constant terms
        Set::Scalar fa = std::sqrt(std::pow((1.0-M0*M0)*M_localMean , 2) + 4*M0*M0)/(1+M0*M0);
        Set::Scalar rho_interface = 0.5*(rho_L + rho_R);
        const Set::Scalar sigma = 1.0;
        const Set::Scalar Kp = 0.25; // constant for pressure difference (diffusion) term

        // Calculate interface Mach number, M_interface
        Set::Scalar M_pressureDiffusion = -Kp/(fa+1.0e-8)*std::max((1.0-sigma*M_localMean*M_localMean),0.0)*(pressure_R-pressure_L)/(rho_interface*c_interface*c_interface + 1.0e-8);

        Set::Scalar M_interface = massFluxPolynomialFunction(M_L,1.0) + massFluxPolynomialFunction(M_R,-1.0) + M_pressureDiffusion;

        // Calculate the Mass flux...
        Set::Scalar A = 0.5*(1.0+std::copysign(1.0,(M_interface-1e-10)));
        Set::Scalar B = std::abs(1.0-A);  // when A=1: B=0 and vice versa/ then can only be 0 or 1
        Set::Scalar massFlux = c_interface*M_interface*(A * rho_L + B * rho_R);

        // Pressure flux calculations...
        const Set::Scalar Ku = 0.75; // constant for velocity difference (diffusion) term
        Set::Scalar P_velocityDiffusion = -Ku*pressureFluxPolynomialFunction(M_L, 1.0, fa)*pressureFluxPolynomialFunction(M_R, -1.0, fa)*(2*rho_interface)*(fa*c_interface)*(vel_R[normal]-vel_L[normal]);

        Set::Scalar pressureFlux = pressureFluxPolynomialFunction(M_L, 1.0, fa)*pressure_L + pressureFluxPolynomialFunction(M_R, -1.0, fa)*pressure_R + P_velocityDiffusion;

        // Debugging for variables and Fluxes
        /*Util::ScimitarX_Util::Debug::DebugAUSMupFlux(i, j, k, num_components,
                                                rho_L, vel_L[0], vel_L[1], vel_L[2], E_L, pressure_L,
                                                rho_R, vel_R[0], vel_R[1], vel_R[2], E_R, pressure_R,
                                                c_L, c_R, M_L, M_R,
                                                M_interface, massFlux, pressureFlux,
                                                "true"); */

            // Assigning Fluxes
        std::vector<Set::Scalar> AUSMupFluxes(num_components, 0.0);

        Set::Scalar C = 0.5*(1.0+std::copysign(1.0,(massFlux-1e-10)));
        Set::Scalar D = std::abs(1.0 - C);

        AUSMupFluxes[solver->variableIndex.DENS] = massFlux;
        AUSMupFluxes[solver->variableIndex.UVEL + normal] = massFlux * (C*vel_L[normal] + D*vel_R[normal]) +  pressureFlux;
#if AMREX_SPACEDIM >= 2
        AUSMupFluxes[solver->variableIndex.UVEL + trans1] = massFlux * (C*vel_L[trans1] + D*vel_R[trans1]);
#endif
#if AMREX_SPACEDIM == 3
        AUSMupFluxes[solver->variableIndex.UVEL + trans2] = massFlux * (C*vel_L[trans2] + D*vel_R[trans2]);
#endif
        AUSMupFluxes[solver->variableIndex.IE] = massFlux * (C * H_total_L + D * H_total_R);
            
        for (int n = 0; n < num_components; ++n) {
                TotalFlux_arr(index[0], index[1], index[2], n) = AUSMupFluxes[n];
        }
        // ################################################### 
        });
    }
}

private:

// polynomial functions for Mass and Pressure flux calculation:
#include <iostream>

// for mass flux:
Set::Scalar massFluxPolynomialFunction(Set::Scalar M, Set::Scalar sign) const {

    // Define switch variables A and B
    Set::Scalar A = 0.5 * (1.0 + std::copysign(1.0, std::abs(M) - 1.0 + 1e-10));
    Set::Scalar B = std::abs(A-1.0);
    // Hence if A is 1, B will be 0, and vice versa
    
    const Set::Scalar beta = 0.125;
    Set::Scalar result = A*(0.5*(M+sign*std::abs(M))) + sign*B*(0.25*std::pow((M+sign),2) + beta*std::pow((M*M-1.0),2));
    return result;
}

// for pressure flux:
Set::Scalar pressureFluxPolynomialFunction(Set::Scalar M, Set::Scalar sign, Set::Scalar fa) const {

    // Define switch variables A and B
    Set::Scalar A = 0.5 * (1.0 + std::copysign(1.0, std::abs(M)-1.0 + 1e-10));
    Set::Scalar B = std::abs(A - 1.0);
    // Hence if A is 1, B will be 0, and vice versa

    const Set::Scalar alpha = 3.0*(5.0*fa*fa - 4.0)/16.0;
    Set::Scalar result = A*(0.5*(M+sign*std::abs(M)))/(M + 1.0e-10) + B*(0.25*sign*std::pow((M + sign),2)*(2*sign - M) + sign*alpha*M*std::pow((M*M - 1),2));
    return result;
}

// to ensure non-zero value
Set::Scalar nonZeroValue(Set::Scalar value) const {
        Set::Scalar result = value + 1.0e-10;
        return result;
}

};

}  // namespace Numeric

#endif  // NUMERIC_FLUXHANDLER_H
