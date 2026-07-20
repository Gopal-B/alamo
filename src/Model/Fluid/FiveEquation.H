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

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Set::Scalar phase_sound_speed_sq(
        int phase,
        Set::Scalar rho_k,
        Set::Scalar p
    ) const {
        constexpr Set::Scalar rho_floor = Set::Scalar(1.0e-14);
        constexpr Set::Scalar c2_floor  = Set::Scalar(1.0e-14);

        rho_k = amrex::max(rho_k, rho_floor);

        if (phase == 1) {
            const Set::Scalar c2 =
                m_gamma1 * (p + m_pi1) / rho_k;
            return amrex::max(c2, c2_floor);
        }

        const Set::Scalar c2 =
            m_gamma2 * (p + m_pi2) / rho_k;
        return amrex::max(c2, c2_floor);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Set::Scalar kapila_K(
        Set::Scalar alpha1,
        Set::Scalar m1,
        Set::Scalar m2,
        Set::Scalar p,
        Set::Scalar alpha_floor,
        Set::Scalar denom_floor,
        bool limit_K
    ) const {
        const Set::Scalar a1 =
            amrex::min(Set::Scalar(1.0) - alpha_floor,
                       amrex::max(alpha_floor, alpha1));

        const Set::Scalar a2 =
            amrex::min(Set::Scalar(1.0) - alpha_floor,
                       amrex::max(alpha_floor, Set::Scalar(1.0) - a1));

        const Set::Scalar rho1 =
            amrex::max(m1 / a1, Set::Scalar(1.0e-14));

        const Set::Scalar rho2 =
            amrex::max(m2 / a2, Set::Scalar(1.0e-14));

        const Set::Scalar c1sq =
            phase_sound_speed_sq(1, rho1, p);

        const Set::Scalar c2sq =
            phase_sound_speed_sq(2, rho2, p);

        const Set::Scalar A1 = rho1 * c1sq;
        const Set::Scalar A2 = rho2 * c2sq;

        const Set::Scalar denom = a1 * A2 + a2 * A1;

        if (!std::isfinite(static_cast<double>(denom)) ||
            denom <= denom_floor) {
            return Set::Scalar(0.0);
        }

        Set::Scalar K =
            a1 * a2 * (A2 - A1) / denom;

        if (!std::isfinite(static_cast<double>(K))) {
            return Set::Scalar(0.0);
        }

        if (limit_K) {
            const Set::Scalar K_min = -a1 + alpha_floor;
            const Set::Scalar K_max =  a2 - alpha_floor;
            K = amrex::min(K_max, amrex::max(K_min, K));
        }

        return K;
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