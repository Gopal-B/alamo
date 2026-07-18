#ifndef NUMERIC_TIMESTEPPER_H_
#define NUMERIC_TIMESTEPPER_H_

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Box.H>
#include <AMReX_MFIter.H>
#include <AMReX_Array4.H>
#include "Set/Set.H"
#include "Integrator/Integrator.H"
#include "Numeric/NumericFactory.H"
#include "Util/Util.H"
#include "Util/ScimitarX_Util.H"
#include <memory>
#include <string>

namespace Numeric {

// Helper function to compute face indices for a control volume's flux calculation
AMREX_GPU_DEVICE
inline void GetFaceIndices(int i, int j, int k, amrex::IntVect& west_face, amrex::IntVect& east_face,
                            amrex::IntVect& south_face, amrex::IntVect& north_face,
                            amrex::IntVect& bottom_face, amrex::IntVect& top_face) {
    // West and east face indices for the x-direction
    west_face = amrex::IntVect(AMREX_D_DECL(i - 1, j, k));
    east_face = amrex::IntVect(AMREX_D_DECL(i, j, k));

    // South and north face indices for the y-direction
    south_face = amrex::IntVect(AMREX_D_DECL(i, j - 1, k));
    north_face = amrex::IntVect(AMREX_D_DECL(i, j, k));

    // Bottom and top face indices for the z-direction
    bottom_face = amrex::IntVect(AMREX_D_DECL(i, j, k - 1));
    top_face = amrex::IntVect(AMREX_D_DECL(i, j, k));
}

template <typename IntegratorDerivedClass>
class TimeSteppingScheme {
public:
    virtual ~TimeSteppingScheme() = default;

    virtual void ComputeSubStep(int lev, Set::Scalar dt, int stage, IntegratorDerivedClass* solver) = 0;
    
    // Method to return the number of stages for the scheme
    virtual int GetNumberOfStages() const = 0;
};

// Forward Euler time stepping scheme
template <typename IntegratorDerivedClass>
class EulerForwardScheme : public TimeSteppingScheme<IntegratorDerivedClass> {
public:
    int GetNumberOfStages() const override { return 1; }

    void ComputeSubStep(int lev, Set::Scalar dt, int /*stage*/, IntegratorDerivedClass* solver) override {
        
        const Set::Scalar* DX = solver->geom[lev].CellSize();
        
        // [DI MODIFICATION] Check for mixture model
        bool use_source_term = (solver->fluid_mixture != nullptr);
        int alpha_idx = -1;
        if (use_source_term) alpha_idx = solver->variableIndex.ALPHA;

        for (amrex::MFIter mfi(*solver->QVec_mf[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();

            auto const& q_arr = solver->QVec_mf[lev]->array(mfi);
            auto const& q_old_arr = solver->QVec_old_mf[lev]->array(mfi);
            auto const& xflux_arr = solver->XFlux_mf[lev]->array(mfi);
#if (AMREX_SPACEDIM >= 2)
            auto const& yflux_arr = solver->YFlux_mf[lev]->array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
            auto const& zflux_arr = solver->ZFlux_mf[lev]->array(mfi);
#endif

            // [DI MODIFICATION] Access Face Velocities for Source Term
            amrex::Array4<Set::Scalar> u_face_arr, v_face_arr, w_face_arr;
            if (use_source_term) {
                u_face_arr = solver->UFace_mf[lev]->array(mfi);
#if (AMREX_SPACEDIM >= 2)
                v_face_arr = solver->VFace_mf[lev]->array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
                w_face_arr = solver->WFace_mf[lev]->array(mfi);
#endif
            }

            int ncomp = solver->number_of_components;

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                for (int n = 0; n < ncomp; ++n) {
                    Set::Scalar flux_div = 0.0;
                    Set::Scalar xflux_right = 0.0, xflux_left = 0.0;
                    Set::Scalar yflux_top = 0.0, yflux_bottom = 0.0;
#if (AMREX_SPACEDIM == 3)
                    Set::Scalar zflux_front = 0.0, zflux_back = 0.0;
#endif

                    // X-direction flux divergence
                        xflux_right = xflux_arr(i, j, k, n);  // Right face flux
                        xflux_left = xflux_arr(i-1, j, k, n);  // Left face flux
                        flux_div += (xflux_right - xflux_left) / DX[0];

                    // Y-direction flux divergence (if 2D or 3D)
                        yflux_top = yflux_arr(i, j, k, n);  // Top face flux
                        yflux_bottom = yflux_arr(i, j-1, k, n);  // Bottom face flux
                        flux_div += (yflux_top - yflux_bottom) / DX[1];

                    // Z-direction flux divergence (if 3D)
#if (AMREX_SPACEDIM == 3)
                        zflux_front = zflux_arr(i, j, k, n);  // Front face flux
                        zflux_back = zflux_arr(i, j, k-1, n);  // Back face flux
                        flux_div += (zflux_front - zflux_back) / DX[2];
#endif

                    // [DI MODIFICATION] Add Source Term for Alpha Equation
                    // Equation: d(alpha)/dt + div(alpha*u) = alpha * div(u)
                    // Update: alpha_new = alpha_old - dt * (div(Flux) - Source)
                    if (use_source_term && n == alpha_idx) {
                        Set::Scalar div_u = (u_face_arr(i, j, k, 0) - u_face_arr(i-1, j, k, 0)) / DX[0];
#if (AMREX_SPACEDIM >= 2)
                        div_u += (v_face_arr(i, j, k, 0) - v_face_arr(i, j-1, k, 0)) / DX[1];
#endif
#if (AMREX_SPACEDIM == 3)
                        div_u += (w_face_arr(i, j, k, 0) - w_face_arr(i, j, k-1, 0)) / DX[2];
#endif
                        Set::Scalar alpha = q_old_arr(i, j, k, alpha_idx);
                        Set::Scalar source_term = alpha * div_u;
                        
                        flux_div -= source_term;
                    }
                    // --- END DI MODIFICATION ---

                    // Forward Euler update
                    q_arr(i, j, k, n) = q_old_arr(i, j, k, n) - dt * flux_div;
                }
            });
        }
    }
};

// RK3 time stepping scheme
struct RK3Constants {
    static constexpr Set::Scalar a[] = {0.0, 3.0/4.0, 1.0/3.0};
    static constexpr Set::Scalar b[] = {1.0, 1.0/4.0, 2.0/3.0};
};

template <typename IntegratorDerivedClass>
class RK3Scheme : public TimeSteppingScheme<IntegratorDerivedClass> {
public:
    int GetNumberOfStages() const override { return 3; }

    void ComputeSubStep(int lev, Set::Scalar dt, int stage, IntegratorDerivedClass* solver) override {
        
        const Set::Scalar* DX = solver->geom[lev].CellSize();

        // [DI MODIFICATION] Check for mixture model
        bool use_source_term = (solver->fluid_mixture != nullptr);
        int alpha_idx = -1;
        if (use_source_term) alpha_idx = solver->variableIndex.ALPHA;

        for (amrex::MFIter mfi(*solver->QVec_mf[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();

            auto const& q_arr = solver->QVec_mf[lev]->array(mfi);
            auto const& q_old_arr = solver->QVec_old_mf[lev]->array(mfi);
            auto const& xflux_arr = solver->XFlux_mf[lev]->array(mfi);
#if (AMREX_SPACEDIM >= 2)
            auto const& yflux_arr = solver->YFlux_mf[lev]->array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
            auto const& zflux_arr = solver->ZFlux_mf[lev]->array(mfi);
#endif

            // [DI MODIFICATION] Access Face Velocities
            amrex::Array4<Set::Scalar> u_face_arr, v_face_arr, w_face_arr;
            if (use_source_term) {
                u_face_arr = solver->UFace_mf[lev]->array(mfi);
#if (AMREX_SPACEDIM >= 2)
                v_face_arr = solver->VFace_mf[lev]->array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
                w_face_arr = solver->WFace_mf[lev]->array(mfi);
#endif
            }

            int ncomp = solver->number_of_components;

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                for (int n = 0; n < ncomp; ++n) {
                    Set::Scalar flux_div = 0.0;
                    Set::Scalar xflux_right = 0.0, xflux_left = 0.0;
                    Set::Scalar yflux_top = 0.0, yflux_bottom = 0.0;
#if (AMREX_SPACEDIM == 3)
                    Set::Scalar zflux_front = 0.0, zflux_back = 0.0;
#endif

                    // X-direction flux divergence
                        xflux_right = xflux_arr(i, j, k, n);  // Right face flux
                        xflux_left = xflux_arr(i-1, j, k, n);  // Left face flux
                        flux_div += (xflux_right - xflux_left) / DX[0];

                    // Y-direction flux divergence (if 2D or 3D)
                        yflux_top = yflux_arr(i, j, k, n);  // Top face flux
                        yflux_bottom = yflux_arr(i, j-1, k, n);  // Bottom face flux
                        flux_div += (yflux_top - yflux_bottom) / DX[1];

                    // Z-direction flux divergence (if 3D)
#if (AMREX_SPACEDIM == 3)
                        zflux_front = zflux_arr(i, j, k, n);  // Front face flux
                        zflux_back = zflux_arr(i, j, k-1, n);  // Back face flux
                        flux_div += (zflux_front - zflux_back) / DX[2];
#endif

                    // [DI MODIFICATION] Add Source Term for Alpha Equation
                    // Note: For RK3, we use the intermediate 'q_arr' alpha for consistency in stages
                    if (use_source_term && n == alpha_idx) {
                        Set::Scalar div_u = (u_face_arr(i, j, k, 0) - u_face_arr(i-1, j, k, 0)) / DX[0];
#if (AMREX_SPACEDIM >= 2)
                        div_u += (v_face_arr(i, j, k, 0) - v_face_arr(i, j-1, k, 0)) / DX[1];
#endif
#if (AMREX_SPACEDIM == 3)
                        div_u += (w_face_arr(i, j, k, 0) - w_face_arr(i, j, k-1, 0)) / DX[2];
#endif
                        Set::Scalar alpha = q_arr(i, j, k, alpha_idx);
                        Set::Scalar source_term = alpha * div_u;
                        
                        flux_div -= source_term;
                    }   
                    // --- END DI MODIFICATION ---

                    // **Runge-Kutta 3rd order update formula**
                    q_arr(i, j, k, n) = RK3Constants::a[stage] * q_old_arr(i, j, k, n) +\
                                        RK3Constants::b[stage] * q_arr(i, j, k, n) -\
                                        dt * RK3Constants::b[stage] * flux_div;
                }
            });
        }
    }
};

template <typename IntegratorDerivedClass>
class TimeStepper {
public:
    void SetTimeSteppingScheme(std::shared_ptr<TimeSteppingScheme<IntegratorDerivedClass>> scheme) {
        time_stepping_scheme = std::move(scheme);
    }

    void ComputeSubStep(int lev, Set::Scalar dt, int stage, IntegratorDerivedClass* solver) {
        if (!time_stepping_scheme) {
            Util::Abort(INFO, "TimeStepper: No time stepping scheme set.");
        }
        time_stepping_scheme->ComputeSubStep(lev, dt, stage, solver);
    }

    int GetNumberOfStages() const {
        if (!time_stepping_scheme) {
            return 0; // Or handle error appropriately
        }
        return time_stepping_scheme->GetNumberOfStages();
    }

private:
    std::shared_ptr<TimeSteppingScheme<IntegratorDerivedClass>> time_stepping_scheme;
};

} // namespace Numeric

#endif // NUMERIC_TIMESTEPPER_H_