#ifndef MODEL_FLUID_FIVEEQUATION_H_
#define MODEL_FLUID_FIVEEQUATION_H_

#include <AMReX.H>
#include <AMReX_REAL.H>
#include "Set/Set.H"
#include "IO/ParmParse.H"
#include <cmath>

namespace Model {
namespace Fluid {
namespace FiveEquation {

/**
 * @class FluidMixture
 * @brief Two-material stiffened-gas mixture EOS utilities for the
 *     
 */
class FluidMixture {
public:
    explicit FluidMixture(IO::ParmParse& pp) {
        pp.query_required("ic.shock.pvec.gamma1", m_gamma1);
        pp.query_required("ic.shock.pvec.gamma2", m_gamma2);

        // Stiffened-gas pi parameters (Pa). For ideal gas, leave as 0.
        pp.query_default("ic.shock.pvec.pi1", m_pi1, Set::Scalar(0.0));
        pp.query_default("ic.shock.pvec.pi2", m_pi2, Set::Scalar(0.0));

        m_inv_gamma1_minus_1 = Set::Scalar(1.0) / (m_gamma1 - Set::Scalar(1.0));
        m_inv_gamma2_minus_1 = Set::Scalar(1.0) / (m_gamma2 - Set::Scalar(1.0));
    }

    // A(alpha) = sum alpha_k/(gamma_k - 1)
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Set::Scalar InvGammaMixMinusOne(Set::Scalar alpha) const {
        return alpha * m_inv_gamma1_minus_1 + (Set::Scalar(1.0) - alpha) * m_inv_gamma2_minus_1;
    }

    // B(alpha) = sum alpha_k * gamma_k * pi_k / (gamma_k - 1)
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Set::Scalar GammaPiMixTerm(Set::Scalar alpha) const {
        const Set::Scalar term1 = alpha * m_gamma1 * m_pi1 * m_inv_gamma1_minus_1;
        const Set::Scalar term2 = (Set::Scalar(1.0) - alpha) * m_gamma2 * m_pi2 * m_inv_gamma2_minus_1;
        return term1 + term2;
    }

    // Pressure closure: p = (rho*e_mix - B)/A
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Set::Scalar closure_pressure(Set::Scalar rho, Set::Scalar e_mix, Set::Scalar alpha) const {
        constexpr Set::Scalar rho_floor = Set::Scalar(1.0e-12);
        constexpr Set::Scalar eps       = Set::Scalar(1.0e-16);

        rho = amrex::max(rho, rho_floor);

        const Set::Scalar A = InvGammaMixMinusOne(alpha);
        const Set::Scalar B = GammaPiMixTerm(alpha);

        return (rho * e_mix - B) / (A + eps);
    }

    // Mixture internal energy: e_mix = (p*A + B)/rho
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Set::Scalar mixture_internal_energy(Set::Scalar rho, Set::Scalar p, Set::Scalar alpha) const {
        constexpr Set::Scalar rho_floor = Set::Scalar(1.0e-12);
        constexpr Set::Scalar eps       = Set::Scalar(1.0e-16);

        rho = amrex::max(rho, rho_floor);

        const Set::Scalar A = InvGammaMixMinusOne(alpha);
        const Set::Scalar B = GammaPiMixTerm(alpha);

        return (p * A + B) / (rho + eps);
    }

    // Mixture gamma: g_mix = 1 + 1/A
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Set::Scalar gamma_mix(Set::Scalar alpha) const {
        constexpr Set::Scalar eps = Set::Scalar(1.0e-16);
        const Set::Scalar A = InvGammaMixMinusOne(alpha);
        return Set::Scalar(1.0) + Set::Scalar(1.0) / (A + eps);
    }

    // pi_mix consistent with B = g_mix*pi_mix/(g_mix-1)
    // => pi_mix = ((g_mix-1)/g_mix) * (B/A)
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Set::Scalar pi_mix(Set::Scalar alpha) const {
        constexpr Set::Scalar eps = Set::Scalar(1.0e-16);

        const Set::Scalar A = InvGammaMixMinusOne(alpha);
        const Set::Scalar B = GammaPiMixTerm(alpha);
        const Set::Scalar g = Set::Scalar(1.0) + Set::Scalar(1.0) / (A + eps);

        return ((g - Set::Scalar(1.0)) / (g + eps)) * (B / (A + eps));
    }

    // Sound speed: c^2 = g_mix * (p + pi_mix) / rho
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Set::Scalar sound_speed_mixture(Set::Scalar rho, Set::Scalar p, Set::Scalar alpha) const {
        constexpr Set::Scalar rho_floor = Set::Scalar(1.0e-12);
        constexpr Set::Scalar p_floor   = Set::Scalar(1.0e-12);
        constexpr Set::Scalar eps       = Set::Scalar(1.0e-16);

        rho = amrex::max(rho, rho_floor);
        p   = amrex::max(p,   p_floor);

        const Set::Scalar A   = InvGammaMixMinusOne(alpha);
        const Set::Scalar B   = GammaPiMixTerm(alpha);
        const Set::Scalar g   = Set::Scalar(1.0) + Set::Scalar(1.0) / (A + eps);
        const Set::Scalar pim = ((g - Set::Scalar(1.0)) / (g + eps)) * (B / (A + eps));

        const Set::Scalar c2 = g * (p + pim) / rho;
        return std::sqrt(amrex::max(c2, Set::Scalar(1.0e-12)));
    }

private:
    Set::Scalar m_gamma1{0}, m_gamma2{0};
    Set::Scalar m_pi1{0}, m_pi2{0};

    Set::Scalar m_inv_gamma1_minus_1{0};
    Set::Scalar m_inv_gamma2_minus_1{0};
};

} // namespace FiveEquation
} // namespace Fluid
} // namespace Model

#endif // MODEL_FLUID_FIVEEQUATION_H_