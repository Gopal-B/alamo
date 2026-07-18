#include "Numeric/FiveEquationVariableAccessor.H"

#include <AMReX_MFIter.H>
#include <AMReX_Array4.H>

#include "Integrator/ScimitarX.H"
#include "Numeric/SymmetryPreservingRoeAveragingOperations.H"
#include "Util/Util.H"

namespace Numeric {
namespace FiveEquation {



std::string FiveEquationCapabilities::getIdentifier() const { return "FiveEquationModel"; }

std::string FiveEquationCapabilities::getDescription() const {
    return "5-Equation Diffuse Interface Model Solver";
}

SolverCapabilities::MethodSupport
FiveEquationCapabilities::supportsFluxReconstruction(FluxReconstructionType method) const {
    if (method == FluxReconstructionType::WENO || method == FluxReconstructionType::FirstOrder)
        return SolverCapabilities::MethodSupport::Supported();
    return SolverCapabilities::MethodSupport::Unsupported();
}

SolverCapabilities::MethodSupport
FiveEquationCapabilities::supportsFluxScheme(FluxScheme scheme) const {
    if (scheme == FluxScheme::HLLC) return SolverCapabilities::MethodSupport::Supported();
    return SolverCapabilities::MethodSupport::Unsupported();
}

SolverCapabilities::MethodSupport
FiveEquationCapabilities::supportsTimeSteppingScheme(TimeSteppingSchemeType scheme) const {
    if (scheme == TimeSteppingSchemeType::RK3) return SolverCapabilities::MethodSupport::Supported();
    return SolverCapabilities::MethodSupport::Unsupported();
}

SolverCapabilities::MethodSupport
FiveEquationCapabilities::supportsReconstructionMode(ReconstructionMode mode) const {
    if (mode == ReconstructionMode::Primitive)      return SolverCapabilities::MethodSupport::Supported();
    if (mode == ReconstructionMode::Characteristic) return SolverCapabilities::MethodSupport::Supported();
    return SolverCapabilities::MethodSupport::Unsupported();
}

SolverCapabilities::MethodSupport
FiveEquationCapabilities::supportsWenoVariant(WenoVariant /*variant*/) const {
    return SolverCapabilities::MethodSupport::Supported();
}

SolverCapabilities::MethodValidationResult
FiveEquationCapabilities::validateMethodCombination(
    FluxReconstructionType /*fr*/,
    FluxScheme fs,
    TimeSteppingSchemeType /*ts*/,
    ReconstructionMode rm,
    WenoVariant /*wv*/
) const {
    MethodValidationResult r;
    r.isValid = true;

    if (fs != FluxScheme::HLLC) {
        r.warnings.push_back("FiveEquationModel is specifically tuned for HLLC.");
    }
    if (rm == ReconstructionMode::Conservative) {
        r.isValid = false;
        r.errors.push_back("Conservative reconstruction is not supported for 5-Equation model.");
    }
    return r;
}

SolverCapabilities::DefaultConfiguration
FiveEquationCapabilities::getDefaultConfiguration() const {
    return {
        FluxReconstructionType::WENO,
        FluxScheme::HLLC,
        TimeSteppingSchemeType::RK3,
        ReconstructionMode::Characteristic,
        WenoVariant::WENOJS5
    };
}

std::shared_ptr<GenericVariableAccessor>
FiveEquationCapabilities::createVariableAccessor(ReconstructionMode mode, int numGhostCells) const {
    return std::make_shared<FiveEquationVariableAccessor>(numGhostCells, mode);
}

// =============================================================
// Variable Accessor
// =============================================================

FiveEquationVariableAccessor::FiveEquationVariableAccessor(int total_ghosts, ReconstructionMode mode)
    : current_mode(mode), total_ghost_cells(total_ghosts) {}

std::shared_ptr<SolverCapabilities>
FiveEquationVariableAccessor::getSolverCapabilities() const {
    return std::make_shared<FiveEquation::FiveEquationCapabilities>();
}

amrex::MultiFab
FiveEquationVariableAccessor::CreateWorkingBuffer(
    const amrex::BoxArray& baseGrids,
    const amrex::DistributionMapping& dm,
    const int num_components,
    const int ghost_cells
) const {
    return amrex::MultiFab(baseGrids, dm, num_components, ghost_cells);
}

int
FiveEquationVariableAccessor::getRequiredGhostCells(
    ReconstructionMode /*mode*/,
    FluxReconstructionType reconstructionType
) const {
    return (reconstructionType == FluxReconstructionType::WENO) ? 4 : 1;
}


static AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void ConsToPrim_DI5(
    const amrex::Array4<const Set::Scalar>& q_arr,
    int i, int j, int k,
    int m1_idx, int m2_idx, int momx_idx, int momy_idx, int momz_idx, int etot_idx, int alpha_idx,
    const Model::Fluid::FiveEquation::FluidMixture* mixture_model,
    Set::Scalar& m1, Set::Scalar& m2,
    Set::Scalar& u, Set::Scalar& v, Set::Scalar& w,
    Set::Scalar& p,
    Set::Scalar& alpha,
    Set::Scalar& rho,
    Set::Scalar& c
) {
    constexpr Set::Scalar tiny        = 1.0e-14;
    constexpr Set::Scalar alpha_floor = 1.0e-12;

    m1    = q_arr(i,j,k,m1_idx);
    m2    = q_arr(i,j,k,m2_idx);
    alpha = q_arr(i,j,k,alpha_idx);

    m1    = amrex::max(m1, tiny);
    m2    = amrex::max(m2, tiny);
    alpha = amrex::min(Set::Scalar(1.0) - alpha_floor,
                       amrex::max(alpha_floor, alpha));

    rho = amrex::max(m1 + m2, tiny);

    const Set::Scalar momx = q_arr(i,j,k,momx_idx);
#if AMREX_SPACEDIM >= 2
    const Set::Scalar momy = q_arr(i,j,k,momy_idx);
#else
    const Set::Scalar momy = 0.0;
#endif
#if AMREX_SPACEDIM == 3
    const Set::Scalar momz = q_arr(i,j,k,momz_idx);
#else
    const Set::Scalar momz = 0.0;
#endif

    u = momx / rho;
#if AMREX_SPACEDIM >= 2
    v = momy / rho;
#else
    v = 0.0;
#endif
#if AMREX_SPACEDIM == 3
    w = momz / rho;
#else
    w = 0.0;
#endif

    const Set::Scalar Etot  = q_arr(i,j,k,etot_idx);
    const Set::Scalar ke    = 0.5 * (u*u + v*v + w*w);
    const Set::Scalar E_spec = Etot / rho;
    Set::Scalar ie_mix = E_spec - ke;
    ie_mix = amrex::max(ie_mix, tiny);

  
    p = mixture_model->closure_pressure(rho, ie_mix, alpha);
    p = amrex::max(p, tiny);

    c = mixture_model->sound_speed_mixture(rho, p, alpha);
    c = amrex::max(c, Set::Scalar(1.0e-12));
}


static AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void BuildLR_DI5_Prim(
    Set::Scalar rho, Set::Scalar c,
    int direction,
    int m1_idx, int m2_idx, int momx_idx, int momy_idx, int momz_idx, int p_idx, int alpha_idx,
    Set::MultiMatrix& L, Set::MultiMatrix& R
) {
    constexpr Set::Scalar tiny = 1.0e-16;
    rho = (rho > tiny) ? rho : tiny;
    c   = (c   > tiny) ? c   : tiny;

    L.setZero();
    R.setZero();

    // Identify normal and tangential velocity component indices in the stored primitive vector
    int in  = momx_idx;
    int it1 = momy_idx;
    int it2 = momz_idx;

    if (direction == 0) {          // X
        in  = momx_idx; it1 = momy_idx; it2 = momz_idx;
    } else if (direction == 1) {   // Y
        in  = momy_idx; it1 = momx_idx; it2 = momz_idx;
    } else {                       // Z
        in  = momz_idx; it1 = momx_idx; it2 = momy_idx;
    }


    R(m1_idx, 0) = 1.0;
    R(m2_idx, 0) = 1.0;
    R(in,     0) = -c / rho;
    R(p_idx,  0) = c * c;

    // m1 wave
    R(m1_idx, 1) = 1.0;

    // m2 wave
    R(m2_idx, 2) = 1.0;

#if AMREX_SPACEDIM >= 2
    R(it1,    3) = 1.0;
#else
    (void)it1;
#endif
#if AMREX_SPACEDIM == 3
    R(it2,    4) = 1.0;
#else
    (void)it2;
#endif

    // alpha wave
    R(alpha_idx, 5) = 1.0;

    // Acoustic+
    R(m1_idx, 6) = 1.0;
    R(m2_idx, 6) = 1.0;
    R(in,     6) = +c / rho;
    R(p_idx,  6) = c * c;

  
    L(0, in)   = -rho / (2.0 * c);
    L(0, p_idx)=  1.0 / (2.0 * c * c);

    // m1
    L(1, m1_idx) = 1.0;
    L(1, p_idx)  = -1.0 / (c * c);

    // m2
    L(2, m2_idx) = 1.0;
    L(2, p_idx)  = -1.0 / (c * c);

    // shear 1
#if AMREX_SPACEDIM >= 2
    L(3, it1) = 1.0;
#endif
    // shear 2
#if AMREX_SPACEDIM == 3
    L(4, it2) = 1.0;
#endif

    // alpha
    L(5, alpha_idx) = 1.0;

    // acoustic+
    L(6, in)   = +rho / (2.0 * c);
    L(6, p_idx)=  1.0 / (2.0 * c * c);
}

void FiveEquationVariableAccessor::CopyVariables(
    int /*direction*/,
    int lev,
    void* solver_void,
    amrex::MultiFab& VariableBuffer,
    ReconstructionMode mode,
    const SolverCapabilities::MethodValidationResult& /*validationResult*/
) const {
    auto solver = static_cast<Integrator::ScimitarX*>(solver_void);

    if (!(mode == ReconstructionMode::Primitive || mode == ReconstructionMode::Characteristic)) {
        Util::Abort(INFO, "FiveEquationVariableAccessor::CopyVariables(): Unsupported ReconstructionMode.");
    }

    const auto* mixture_model = solver->fluid_mixture.get();
    AMREX_ALWAYS_ASSERT(mixture_model != nullptr);

    const int m1_idx    = variableIndices.M1;
    const int m2_idx    = variableIndices.M2;
    const int momx_idx  = variableIndices.MOMX;
    const int momy_idx  = variableIndices.MOMY;
    const int momz_idx  = variableIndices.MOMZ;
    const int p_idx     = variableIndices.ETOT;   // store p in ETOT slot for reconstruction buffer
    const int alpha_idx = variableIndices.ALPHA;

    // Read conserved state from QVec
    for (amrex::MFIter mfi(VariableBuffer, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.growntilebox();

        auto const& q_arr = solver->QVec_mf.Patch(lev, mfi);
        auto out_arr      = VariableBuffer.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            Set::Scalar m1, m2, u, v, w, p, alpha, rho, c;
            ConsToPrim_DI5(
                q_arr, i, j, k,
                m1_idx, m2_idx, momx_idx, momy_idx, momz_idx, variableIndices.ETOT, alpha_idx,
                mixture_model,
                m1, m2, u, v, w, p, alpha, rho, c
            );
            (void)rho; (void)c;

            out_arr(i,j,k,m1_idx)    = m1;
            out_arr(i,j,k,m2_idx)    = m2;
            out_arr(i,j,k,momx_idx)  = u;
#if AMREX_SPACEDIM >= 2
            out_arr(i,j,k,momy_idx)  = v;
#endif
#if AMREX_SPACEDIM == 3
            out_arr(i,j,k,momz_idx)  = w;
#endif
            out_arr(i,j,k,p_idx)     = p;
            out_arr(i,j,k,alpha_idx) = alpha;
        });
    }

    VariableBuffer.FillBoundary();
}

void FiveEquationVariableAccessor::CopyFluxes(
    int direction,
    int lev,
    void* solver_void,
    amrex::MultiFab& CellFluxBuffer,
    ReconstructionMode mode,
    const SolverCapabilities::MethodValidationResult& /*validationResult*/
) const {
    auto solver = static_cast<Integrator::ScimitarX*>(solver_void);

    if (!(mode == ReconstructionMode::Primitive || mode == ReconstructionMode::Characteristic)) {
        Util::Abort(INFO, "FiveEquationVariableAccessor::CopyFluxes(): Unsupported ReconstructionMode.");
    }

    const auto* mixture_model = solver->fluid_mixture.get();
    AMREX_ALWAYS_ASSERT(mixture_model != nullptr);

    const int m1_idx    = variableIndices.M1;
    const int m2_idx    = variableIndices.M2;
    const int momx_idx  = variableIndices.MOMX;
    const int momy_idx  = variableIndices.MOMY;
    const int momz_idx  = variableIndices.MOMZ;
    const int etot_idx  = variableIndices.ETOT;
    const int alpha_idx = variableIndices.ALPHA;

    for (amrex::MFIter mfi(CellFluxBuffer, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.growntilebox();

        auto const& q_arr = solver->QVec_mf.Patch(lev, mfi);
        auto flux_arr     = CellFluxBuffer.array(mfi);

        const int normal = direction;

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            // Extract primitives (for pressure) but keep conserved for flux form
            Set::Scalar m1, m2, u, v, w, p, alpha, rho, c;
            ConsToPrim_DI5(
                q_arr, i, j, k,
                m1_idx, m2_idx, momx_idx, momy_idx, momz_idx, etot_idx, alpha_idx,
                mixture_model,
                m1, m2, u, v, w, p, alpha, rho, c
            );
            (void)c;

            const Set::Scalar momx = q_arr(i,j,k,momx_idx);
#if AMREX_SPACEDIM >= 2
            const Set::Scalar momy = q_arr(i,j,k,momy_idx);
#else
            const Set::Scalar momy = 0.0;
#endif
#if AMREX_SPACEDIM == 3
            const Set::Scalar momz = q_arr(i,j,k,momz_idx);
#else
            const Set::Scalar momz = 0.0;
#endif
            const Set::Scalar Etot = q_arr(i,j,k,etot_idx);

            Set::Scalar un = 0.0;
            if (normal == 0) {
                un = u;
#if AMREX_SPACEDIM >= 2
            } else if (normal == 1) {
                un = v;
#endif
#if AMREX_SPACEDIM == 3
            } else {
                un = w;
#endif
            }

            // Mass fluxes (partial masses)
            flux_arr(i,j,k,m1_idx) = m1 * un;
            flux_arr(i,j,k,m2_idx) = m2 * un;

            // Momentum fluxes
            flux_arr(i,j,k,momx_idx) = momx * un;
#if AMREX_SPACEDIM >= 2
            flux_arr(i,j,k,momy_idx) = momy * un;
#endif
#if AMREX_SPACEDIM == 3
            flux_arr(i,j,k,momz_idx) = momz * un;
#endif
            if (normal == 0) {
                flux_arr(i,j,k,momx_idx) += p;
#if AMREX_SPACEDIM >= 2
            } else if (normal == 1) {
                flux_arr(i,j,k,momy_idx) += p;
#endif
#if AMREX_SPACEDIM == 3
            } else {
                flux_arr(i,j,k,momz_idx) += p;
#endif
            }

            // Energy flux
            flux_arr(i,j,k,etot_idx) = (Etot + p) * un;

            // Alpha advection flux (conservative form used in your code path)
            flux_arr(i,j,k,alpha_idx) = alpha * un;
        });
    }

    CellFluxBuffer.FillBoundary();
}


void FiveEquationVariableAccessor::PopulateAverageStates(
    int direction,
    int lev,
    void* solver_void,
    amrex::MultiFab& AverageStateBuffer
) const {
    auto solver = static_cast<Integrator::ScimitarX*>(solver_void);
    const int nghosts = solver->number_of_ghost_cells;

    const auto* mixture_model = solver->fluid_mixture.get();
    AMREX_ALWAYS_ASSERT(mixture_model != nullptr);

    const int m1_idx    = variableIndices.M1;
    const int m2_idx    = variableIndices.M2;
    const int momx_idx  = variableIndices.MOMX;
    const int momy_idx  = variableIndices.MOMY;
    const int momz_idx  = variableIndices.MOMZ;
    const int p_idx     = variableIndices.ETOT;   // store p_avg here, and c_avg also here? (we store c in ETOT)
    const int alpha_idx = variableIndices.ALPHA;

    for (amrex::MFIter mfi(AverageStateBuffer, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& face_bx = mfi.grownnodaltilebox(direction, nghosts);
        const amrex::Box& cell_bx = solver->QVec_mf[lev]->box(mfi.index());

        auto const& q_arr   = solver->QVec_mf.Patch(lev, mfi);
        auto avg_arr        = AverageStateBuffer.array(mfi);

        const int ilo = cell_bx.smallEnd(0), ihi = cell_bx.bigEnd(0);
        const int jlo = cell_bx.smallEnd(1), jhi = cell_bx.bigEnd(1);
#if AMREX_SPACEDIM == 3
        const int klo = cell_bx.smallEnd(2), khi = cell_bx.bigEnd(2);
#else
        const int klo = 0, khi = 0;
#endif

        amrex::ParallelFor(face_bx, [=] AMREX_GPU_DEVICE(int iface, int jface, int kface) noexcept {
            int iL = iface, jL = jface, kL = kface;
            int iR = iface, jR = jface, kR = kface;

            if (direction == Xdir) {
                iL = iface - 1; iR = iface;
            } else if (direction == Ydir) {
                jL = jface - 1; jR = jface;
            }
#if AMREX_SPACEDIM == 3
            else {
                kL = kface - 1; kR = kface;
            }
#endif

            iL = amrex::max(ilo, amrex::min(ihi, iL));
            iR = amrex::max(ilo, amrex::min(ihi, iR));
            jL = amrex::max(jlo, amrex::min(jhi, jL));
            jR = amrex::max(jlo, amrex::min(jhi, jR));
#if AMREX_SPACEDIM == 3
            kL = amrex::max(klo, amrex::min(khi, kL));
            kR = amrex::max(klo, amrex::min(khi, kR));
#else
            kL = 0; kR = 0;
            (void)klo; (void)khi;
#endif

            Set::Scalar m1L, m2L, uL, vL, wL, pL, aL, rhoL, cL;
            Set::Scalar m1R, m2R, uR, vR, wR, pR, aR, rhoR, cR;

            ConsToPrim_DI5(q_arr, iL,jL,kL, m1_idx,m2_idx,momx_idx,momy_idx,momz_idx, variableIndices.ETOT, alpha_idx,
                           mixture_model, m1L,m2L,uL,vL,wL,pL,aL,rhoL,cL);
            ConsToPrim_DI5(q_arr, iR,jR,kR, m1_idx,m2_idx,momx_idx,momy_idx,momz_idx, variableIndices.ETOT, alpha_idx,
                           mixture_model, m1R,m2R,uR,vR,wR,pR,aR,rhoR,cR);

            const Set::Scalar m1A = 0.5*(m1L + m1R);
            const Set::Scalar m2A = 0.5*(m2L + m2R);
            const Set::Scalar uA  = 0.5*(uL  + uR);
            const Set::Scalar vA  = 0.5*(vL  + vR);
            const Set::Scalar wA  = 0.5*(wL  + wR);
            const Set::Scalar pA  = 0.5*(pL  + pR);
            const Set::Scalar aA  = 0.5*(aL  + aR);
            const Set::Scalar rhoA = amrex::max(m1A + m2A, Set::Scalar(1.0e-14));

            // Recompute c on the averaged state (preferred for frozen Jacobian)
            Set::Scalar cA = mixture_model->sound_speed_mixture(rhoA, amrex::max(pA, Set::Scalar(1.0e-14)), aA);
            cA = amrex::max(cA, Set::Scalar(1.0e-12));

            avg_arr(iface,jface,kface,m1_idx)    = m1A;
            avg_arr(iface,jface,kface,m2_idx)    = m2A;
            avg_arr(iface,jface,kface,momx_idx)  = uA;
#if AMREX_SPACEDIM >= 2
            avg_arr(iface,jface,kface,momy_idx)  = vA;
#endif
#if AMREX_SPACEDIM == 3
            avg_arr(iface,jface,kface,momz_idx)  = wA;
#endif
            // Store c in ETOT slot (AverageStateBuffer is only for characteristic transforms)
            avg_arr(iface,jface,kface,p_idx)     = cA;
            avg_arr(iface,jface,kface,alpha_idx) = aA;
        });
    }

    AverageStateBuffer.FillBoundary();
}


static AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int DI5_NumVars(int spacedim) {
    // 2 partial masses + spacedim velocities + p + alpha
    return 2 + spacedim + 1 + 1;
}

static AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void BuildLR_DI5_Prim_Dynamic(
    int N,
    Set::Scalar rho, Set::Scalar c,
    int direction,
    // slots in your VariableBuffer
    int m1_idx, int m2_idx, int momx_idx, int momy_idx, int momz_idx,
    int p_idx, int alpha_idx,
    Set::MultiMatrix& L, Set::MultiMatrix& R
) {
    constexpr Set::Scalar tiny = 1.0e-16;
    rho = (rho > tiny) ? rho : tiny;
    c   = (c   > tiny) ? c   : tiny;

    L = Set::MultiMatrix(N, N);
    R = Set::MultiMatrix(N, N);
    L.setZero();
    R.setZero();

    // Determine which velocity slots exist in N
    // 1D N=5 -> only momx exists
    // 2D N=6 -> momx,momy exist
    // 3D N=7 -> momx,momy,momz exist
    const bool has_v = (N >= 6);
    const bool has_w = (N >= 7);

    // Pick normal and tangential indices among existing ones
    int in  = momx_idx;
    int it1 = has_v ? momy_idx : -1;
    int it2 = has_w ? momz_idx : -1;

    if (direction == 0) { // X
        in  = momx_idx;
        it1 = has_v ? momy_idx : -1;
        it2 = has_w ? momz_idx : -1;
    } else if (direction == 1) { // Y (only if 2D/3D)
        AMREX_ALWAYS_ASSERT(has_v);
        in  = momy_idx;
        it1 = momx_idx;
        it2 = has_w ? momz_idx : -1;
    } else { // Z (only if 3D)
        AMREX_ALWAYS_ASSERT(has_w);
        in  = momz_idx;
        it1 = momx_idx;
        it2 = momy_idx;
    }

    // Characteristic indices in ξ vector
    // Always:
    const int i_acm = 0;        // acoustic-
    const int i_m1  = 1;
    const int i_m2  = 2;

    int i_s1   = -1;
    int i_s2   = -1;
    int i_al   = -1;
    int i_acp  = -1;

    if (N == 5) {          // 1D
        i_al  = 3;
        i_acp = 4;
    } else if (N == 6) {   // 2D
        i_s1  = 3;
        i_al  = 4;
        i_acp = 5;
    } else {               // 3D (N==7)
        i_s1  = 3;
        i_s2  = 4;
        i_al  = 5;
        i_acp = 6;
    }


    R(in,     i_acm) =  0.5;
    R(p_idx,  i_acm) = -0.5 * rho * c;

    // m1 wave (u_n)
    R(m1_idx, i_m1)  = 1.0;

    // m2 wave (u_n)
    R(m2_idx, i_m2)  = 1.0;

    // shear(s) (u_n)
    if (N >= 6) {
        R(it1, i_s1) = 1.0;
    }
    if (N == 7) {
        R(it2, i_s2) = 1.0;
    }

    // alpha wave (u_n)
    R(alpha_idx, i_al) = 1.0;

    // acoustic+  (u_n + c)
    //   r_+: dm1 = +m1/(2c), dm2 = +m2/(2c), du_n=1/2, dp=+(rho*c)/2
    R(in,     i_acp) =  0.5;
    R(p_idx,  i_acp) =  0.5 * rho * c;

    // ----------------------------
    // Left eigenvectors L = Q^{-1} (rows)
    // ----------------------------

    //   xi_-  = u_n - p/(rho*c)
    //   xi_+  = u_n + p/(rho*c)
    //   xi_m1 = m1 - (m1/(rho*c^2))*p
    //   xi_m2 = m2 - (m2/(rho*c^2))*p
    //   xi_s  = u_t
    //   xi_a  = alpha
    // The m1,m2 pressure couplings are set in the caller using face-averaged m1,m2.

    // acoustic-
    L(i_acm, in)    = 1.0;
    L(i_acm, p_idx) = -1.0 / (rho * c);

    // m1
    L(i_m1, m1_idx) = 1.0;

    // m2
    L(i_m2, m2_idx) = 1.0;

    // shear(s)
    if (N >= 6) {
        L(i_s1, it1) = 1.0;
    }
    if (N == 7) {
        L(i_s2, it2) = 1.0;
    }

    // alpha
    L(i_al, alpha_idx) = 1.0;

    // acoustic+
    L(i_acp, in)    = 1.0;
    L(i_acp, p_idx) =  1.0 / (rho * c);
}

Set::MultiMatrix FiveEquationVariableAccessor::TransformStencilToCharacteristic(
    int /*i*/, int /*j*/, int /*k*/,
    const Set::MultiMatrix& stencil_matrix,
    int direction,
    const Set::MultiVector& avg_state
) const {
    const int npts  = stencil_matrix.rows();
    const int ncomp = stencil_matrix.cols(); // N (5/6/7 depending on dim)

    Set::MultiMatrix char_matrix(npts, ncomp);
    char_matrix.setZero();

    const int m1_idx    = variableIndices.M1;
    const int m2_idx    = variableIndices.M2;
    const int momx_idx  = variableIndices.MOMX;
    const int momy_idx  = variableIndices.MOMY;
    const int momz_idx  = variableIndices.MOMZ;
    const int pslot_idx = variableIndices.ETOT;   // p in VariableBuffer; c in avg_state
    const int alpha_idx = variableIndices.ALPHA;

    // avg_state stores c in ETOT slot (as created by PopulateAverageStates)
    const Set::Scalar rhoA = amrex::max(avg_state(m1_idx) + avg_state(m2_idx), Set::Scalar(1.0e-14));
    const Set::Scalar cA   = amrex::max(avg_state(pslot_idx), Set::Scalar(1.0e-12));

    Set::MultiMatrix L, R;
    BuildLR_DI5_Prim_Dynamic(
        ncomp, rhoA, cA, direction,
        m1_idx, m2_idx, momx_idx, momy_idx, momz_idx, pslot_idx, alpha_idx,
        L, R
    );


    const int i_acm = 0;
    const int i_m1  = 1;
    const int i_m2  = 2;
    const int i_acp = ncomp - 1;

    const Set::Scalar m1A = avg_state(m1_idx);
    const Set::Scalar m2A = avg_state(m2_idx);

    // Right eigenvectors: acoustic columns scale with m1,m2
    R(m1_idx, i_acm) = -m1A / (2.0 * cA);
    R(m2_idx, i_acm) = -m2A / (2.0 * cA);
    R(m1_idx, i_acp) =  m1A / (2.0 * cA);
    R(m2_idx, i_acp) =  m2A / (2.0 * cA);

    // Left eigenvectors: pressure coupling in the advected m1/m2 rows
    L(i_m1, pslot_idx) = -m1A / (rhoA * cA * cA);
    L(i_m2, pslot_idx) = -m2A / (rhoA * cA * cA);

    for (int s = 0; s < npts; ++s) {
        Set::MultiVector W(ncomp);
        W.setZero();

        // Copy physical primitives from stencil_matrix row
        for (int q = 0; q < ncomp; ++q) {
            W(q) = stencil_matrix(s, q);
        }

        // ξ = L * W
        Set::MultiVector xi = Numeric::SymmetryPreserving::ConsistentMatrixVectorMultiply(L, W);

        // Store ξ back into same slots (local convention: ξ vector uses same indexing [0..N-1])
        for (int q = 0; q < ncomp; ++q) {
            char_matrix(s, q) = xi(q);
        }
    }

    return char_matrix;
}

Set::MultiVector FiveEquationVariableAccessor::TransformFromCharacteristic(
    const Set::MultiVector& reconstructed_value,
    int direction,
    const Set::MultiVector& avg_state
) const {
    const int ncomp = reconstructed_value.size(); // N (5/6/7)

    const int m1_idx    = variableIndices.M1;
    const int m2_idx    = variableIndices.M2;
    const int momx_idx  = variableIndices.MOMX;
    const int momy_idx  = variableIndices.MOMY;
    const int momz_idx  = variableIndices.MOMZ;
    const int pslot_idx = variableIndices.ETOT;   // p in VariableBuffer; c in avg_state
    const int alpha_idx = variableIndices.ALPHA;

    const Set::Scalar rhoA = amrex::max(avg_state(m1_idx) + avg_state(m2_idx), Set::Scalar(1.0e-14));
    const Set::Scalar cA   = amrex::max(avg_state(pslot_idx), Set::Scalar(1.0e-12));

    Set::MultiMatrix L, R;
    BuildLR_DI5_Prim_Dynamic(
        ncomp, rhoA, cA, direction,
        m1_idx, m2_idx, momx_idx, momy_idx, momz_idx, pslot_idx, alpha_idx,
        L, R
    );

   
    const int i_acm = 0;
    const int i_m1  = 1;
    const int i_m2  = 2;
    const int i_acp = ncomp - 1;

    const Set::Scalar m1A = avg_state(m1_idx);
    const Set::Scalar m2A = avg_state(m2_idx);

    R(m1_idx, i_acm) = -m1A / (2.0 * cA);
    R(m2_idx, i_acm) = -m2A / (2.0 * cA);
    R(m1_idx, i_acp) =  m1A / (2.0 * cA);
    R(m2_idx, i_acp) =  m2A / (2.0 * cA);

    L(i_m1, pslot_idx) = -m1A / (rhoA * cA * cA);
    L(i_m2, pslot_idx) = -m2A / (rhoA * cA * cA);

    // xi is exactly the reconstructed_value in characteristic space (same N layout)
    Set::MultiVector xi(ncomp);
    xi.setZero();
    for (int q = 0; q < ncomp; ++q) {
        xi(q) = reconstructed_value(q);
    }

    // Recover primitives: W = R * ξ
    Set::MultiVector W = Numeric::SymmetryPreserving::ConsistentMatrixVectorMultiply(R, xi);

    return W; // same primitive slot ordering as VariableBuffer uses
}

void FiveEquationVariableAccessor::StoreDirectionalFlux(
    int direction,
    int lev,
    void* solver_void,
    amrex::MultiFab& SummedFlux
) const {
    auto solver = static_cast<Integrator::ScimitarX*>(solver_void);
    const int nghosts = solver->number_of_ghost_cells;
    const int ncomp   = solver->number_of_components;

    for (amrex::MFIter mfi(SummedFlux, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.grownnodaltilebox(direction, nghosts);
        auto const& src = SummedFlux.array(mfi);

        amrex::Array4<Set::Scalar> dst;
        if (direction == Xdir) {
            dst = solver->XFlux_mf.Patch(lev, mfi);
        } else if (direction == Ydir) {
#if AMREX_SPACEDIM >= 2
            dst = solver->YFlux_mf.Patch(lev, mfi);
#endif
        } else {
#if AMREX_SPACEDIM == 3
            dst = solver->ZFlux_mf.Patch(lev, mfi);
#endif
        }

        if (!dst) continue;

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            for (int n = 0; n < ncomp; ++n) {
                dst(i, j, k, n) = src(i, j, k, n);
            }
        });
    }
}

} // namespace FiveEquation
} // namespace Numeric