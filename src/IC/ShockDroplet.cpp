#include "IC/IC.H"
#include "IC/ShockDroplet.H"
#include "Set/Set.H"
#include "Util/Util.H"
#include "Util/ScimitarX_Util.H"
#include "Numeric/IntegratorVariableAccessLayer.H"

#include <cmath>
#include <string>

namespace IC {

ShockDroplet::ShockDroplet(
    amrex::Vector<amrex::Geometry>& _geom,
    IO::ParmParse& pp,
    std::string name,
    const Numeric::GenericVariableAccessor::VariableIndices& precomputed_indices)
    : IC(_geom),
      variable_indices(&precomputed_indices),
      requires_variable_indices(true)
{
    initialize(pp, name);
}

ShockDroplet::ShockDroplet(
    amrex::Vector<amrex::Geometry>& _geom,
    IO::ParmParse& pp,
    std::string name)
    : IC(_geom),
      variable_indices(nullptr),
      requires_variable_indices(false)
{
    initialize(pp, name);
}

void ShockDroplet::Add(const int& lev, Set::Field<Set::Scalar>& a_phi, Set::Scalar time)
{
    AddConstant(lev, a_phi, time);
}

void ShockDroplet::initialize(IO::ParmParse& pp, const std::string& name)
{
    mf_name = name;

    Util::Message(INFO, "Initializing ShockDroplet IC with mf_name = " + mf_name);

    // This IC is intended for the FiveEquation primitive field and the pressure field.
    if (mf_name.find("pvec") != std::string::npos) {
        if (!requires_variable_indices || variable_indices == nullptr) {
            Util::Abort(INFO, "ShockDroplet pvec initialization requires valid variable indices.");
        }
        if (variable_indices->ALPHA == -1 || variable_indices->M1 == -1 ||
            variable_indices->M2 == -1 || variable_indices->ETOT == -1) {
            Util::Abort(INFO, "ShockDroplet pvec initialization currently supports only the FiveEquation model.");
        }
    }

    // Domain defaults, useful for optional y/z driver extents.
    Set::Scalar domain_xlo = geom[0].ProbLo()[0];
    Set::Scalar domain_xhi = geom[0].ProbHi()[0];
#if AMREX_SPACEDIM >= 2
    Set::Scalar domain_ylo = geom[0].ProbLo()[1];
    Set::Scalar domain_yhi = geom[0].ProbHi()[1];
#endif
#if AMREX_SPACEDIM == 3
    Set::Scalar domain_zlo = geom[0].ProbLo()[2];
    Set::Scalar domain_zhi = geom[0].ProbHi()[2];
#endif

    // -----------------------------
    // Geometry parsing
    // -----------------------------
    // Driver region: support both x_driver_lo/x_driver_hi and x_hp_min/x_hp_max names.
    x_driver_lo = domain_xlo;
    x_driver_hi = domain_xlo;
    if (!pp.query("ic.shockdroplet.geometry.x_driver_lo", x_driver_lo)) {
        pp.query_required("ic.shockdroplet.geometry.x_hp_min", x_driver_lo);
    }
    if (!pp.query("ic.shockdroplet.geometry.x_driver_hi", x_driver_hi)) {
        pp.query_required("ic.shockdroplet.geometry.x_hp_max", x_driver_hi);
    }

#if AMREX_SPACEDIM >= 2
    y_driver_lo = domain_ylo;
    y_driver_hi = domain_yhi;
    pp.query("ic.shockdroplet.geometry.y_driver_lo", y_driver_lo);
    pp.query("ic.shockdroplet.geometry.y_driver_hi", y_driver_hi);
    pp.query("ic.shockdroplet.geometry.y_hp_min", y_driver_lo);
    pp.query("ic.shockdroplet.geometry.y_hp_max", y_driver_hi);
#endif

#if AMREX_SPACEDIM == 3
    z_driver_lo = domain_zlo;
    z_driver_hi = domain_zhi;
    pp.query("ic.shockdroplet.geometry.z_driver_lo", z_driver_lo);
    pp.query("ic.shockdroplet.geometry.z_driver_hi", z_driver_hi);
#endif

    droplet_center_x = 0.5 * (domain_xlo + domain_xhi);
    pp.query_required("ic.shockdroplet.geometry.droplet_center_x", droplet_center_x);
#if AMREX_SPACEDIM >= 2
    droplet_center_y = 0.5 * (domain_ylo + domain_yhi);
    pp.query_required("ic.shockdroplet.geometry.droplet_center_y", droplet_center_y);
#endif
#if AMREX_SPACEDIM == 3
    droplet_center_z = 0.5 * (domain_zlo + domain_zhi);
    pp.query_required("ic.shockdroplet.geometry.droplet_center_z", droplet_center_z);
#endif

    droplet_radius = -1.0;
    if (!pp.query("ic.shockdroplet.geometry.droplet_radius", droplet_radius)) {
        pp.query_required("ic.shockdroplet.geometry.radius", droplet_radius);
    }

    interface_thickness = 0.0;
    pp.query("ic.shockdroplet.geometry.interface_thickness", interface_thickness);
    if (interface_thickness < 0.0) interface_thickness = -interface_thickness;

    phase1_is_liquid = true;
    pp.query("ic.shockdroplet.phase1_is_liquid", phase1_is_liquid);

    if (x_driver_hi <= x_driver_lo) {
        Util::Abort(INFO, "ShockDroplet: x_driver_hi must be greater than x_driver_lo.");
    }
#if AMREX_SPACEDIM >= 2
    if (y_driver_hi <= y_driver_lo) {
        Util::Abort(INFO, "ShockDroplet: y_driver_hi must be greater than y_driver_lo.");
    }
#endif
#if AMREX_SPACEDIM == 3
    if (z_driver_hi <= z_driver_lo) {
        Util::Abort(INFO, "ShockDroplet: z_driver_hi must be greater than z_driver_lo.");
    }
#endif
    if (droplet_radius <= 0.0) {
        Util::Abort(INFO, "ShockDroplet: droplet_radius must be positive.");
    }

    // -----------------------------
    // State parsing
    // -----------------------------
    // Ambient gas
    ambient_rho_g = 1.0;
    if (!pp.query("ic.shockdroplet.ambient.rho_g", ambient_rho_g)) {
        pp.query_required("ic.shockdroplet.ambient.rho", ambient_rho_g);
    }
    pp.query_required("ic.shockdroplet.ambient.p", ambient_p);
    pp.query_default("ic.shockdroplet.ambient.u", ambient_u, Set::Scalar(0.0));
#if AMREX_SPACEDIM >= 2
    pp.query_default("ic.shockdroplet.ambient.v", ambient_v, Set::Scalar(0.0));
#endif
#if AMREX_SPACEDIM == 3
    pp.query_default("ic.shockdroplet.ambient.w", ambient_w, Set::Scalar(0.0));
#endif

    // Driver gas
    driver_rho_g = 1.0;
    if (!pp.query("ic.shockdroplet.driver.rho_g", driver_rho_g)) {
        pp.query_required("ic.shockdroplet.driver.rho", driver_rho_g);
    }
    pp.query_required("ic.shockdroplet.driver.p", driver_p);
    pp.query_default("ic.shockdroplet.driver.u", driver_u, Set::Scalar(0.0));
#if AMREX_SPACEDIM >= 2
    pp.query_default("ic.shockdroplet.driver.v", driver_v, Set::Scalar(0.0));
#endif
#if AMREX_SPACEDIM == 3
    pp.query_default("ic.shockdroplet.driver.w", driver_w, Set::Scalar(0.0));
#endif

    // Liquid
    liquid_rho_l = 1000.0;
    if (!pp.query("ic.shockdroplet.liquid.rho_l", liquid_rho_l)) {
        pp.query_required("ic.shockdroplet.liquid.rho", liquid_rho_l);
    }
    pp.query_required("ic.shockdroplet.liquid.p", liquid_p);
    pp.query_default("ic.shockdroplet.liquid.u", liquid_u, Set::Scalar(0.0));
#if AMREX_SPACEDIM >= 2
    pp.query_default("ic.shockdroplet.liquid.v", liquid_v, Set::Scalar(0.0));
#endif
#if AMREX_SPACEDIM == 3
    pp.query_default("ic.shockdroplet.liquid.w", liquid_w, Set::Scalar(0.0));
#endif

    Util::Message(INFO, "ShockDroplet geometry parsed:");
    Util::Message(INFO, "  driver x-range = [" + std::to_string(x_driver_lo) + ", " + std::to_string(x_driver_hi) + "]");
#if AMREX_SPACEDIM >= 2
    Util::Message(INFO, "  driver y-range = [" + std::to_string(y_driver_lo) + ", " + std::to_string(y_driver_hi) + "]");
#endif
#if AMREX_SPACEDIM == 3
    Util::Message(INFO, "  driver z-range = [" + std::to_string(z_driver_lo) + ", " + std::to_string(z_driver_hi) + "]");
#endif
    Util::Message(INFO, "  droplet radius = " + std::to_string(droplet_radius));
    Util::Message(INFO, "  interface thickness = " + std::to_string(interface_thickness));
    Util::Message(INFO, std::string("  phase1_is_liquid = ") + (phase1_is_liquid ? "true" : "false"));
}

void ShockDroplet::AddConstant(const int& lev, Set::Field<Set::Scalar>& a_phi, Set::Scalar)
{
    const int ncomp = a_phi[lev]->nComp();

    // Variable indices (FiveEquation only for pvec)
    const int m1_idx   = (requires_variable_indices && variable_indices) ? variable_indices->M1   : -1;
    const int m2_idx   = (requires_variable_indices && variable_indices) ? variable_indices->M2   : -1;
    const int momx_idx = (requires_variable_indices && variable_indices) ? variable_indices->MOMX : -1;
#if AMREX_SPACEDIM >= 2
    const int momy_idx = (requires_variable_indices && variable_indices) ? variable_indices->MOMY : -1;
#endif
#if AMREX_SPACEDIM == 3
    const int momz_idx = (requires_variable_indices && variable_indices) ? variable_indices->MOMZ : -1;
#endif
    const int etot_idx = (requires_variable_indices && variable_indices) ? variable_indices->ETOT : -1;
    const int alpha_idx= (requires_variable_indices && variable_indices) ? variable_indices->ALPHA: -1;

    // Copy member data to local POD values for GPU lambdas.
    const bool is_pressure_mf = (mf_name == "ic.shockdroplet.pressure");

    const Set::Scalar xdrv_lo = x_driver_lo;
    const Set::Scalar xdrv_hi = x_driver_hi;
#if AMREX_SPACEDIM >= 2
    const Set::Scalar ydrv_lo = y_driver_lo;
    const Set::Scalar ydrv_hi = y_driver_hi;
#endif
#if AMREX_SPACEDIM == 3
    const Set::Scalar zdrv_lo = z_driver_lo;
    const Set::Scalar zdrv_hi = z_driver_hi;
#endif

    const Set::Scalar xc = droplet_center_x;
#if AMREX_SPACEDIM >= 2
    const Set::Scalar yc = droplet_center_y;
#endif
#if AMREX_SPACEDIM == 3
    const Set::Scalar zc = droplet_center_z;
#endif
    const Set::Scalar R  = droplet_radius;
    const Set::Scalar eps = interface_thickness;
    const bool phase1_liq = phase1_is_liquid;

    const Set::Scalar rho_g_amb = ambient_rho_g;
    const Set::Scalar p_amb     = ambient_p;
    const Set::Scalar u_amb     = ambient_u;
#if AMREX_SPACEDIM >= 2
    const Set::Scalar v_amb     = ambient_v;
#endif
#if AMREX_SPACEDIM == 3
    const Set::Scalar w_amb     = ambient_w;
#endif

    const Set::Scalar rho_g_drv = driver_rho_g;
    const Set::Scalar p_drv     = driver_p;
    const Set::Scalar u_drv     = driver_u;
#if AMREX_SPACEDIM >= 2
    const Set::Scalar v_drv     = driver_v;
#endif
#if AMREX_SPACEDIM == 3
    const Set::Scalar w_drv     = driver_w;
#endif

    const Set::Scalar rho_l     = liquid_rho_l;
    const Set::Scalar p_l       = liquid_p;
    const Set::Scalar u_l       = liquid_u;
#if AMREX_SPACEDIM >= 2
    const Set::Scalar v_l       = liquid_v;
#endif
#if AMREX_SPACEDIM == 3
    const Set::Scalar w_l       = liquid_w;
#endif

    for (amrex::MFIter mfi(*a_phi[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box bx = mfi.growntilebox();
        amrex::IndexType type = a_phi[lev]->ixType();
        amrex::Array4<Set::Scalar> const& phi = a_phi[lev]->array(mfi);

        for (int n = 0; n < ncomp; ++n) {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Set::Vector x = Set::Position(i, j, k, geom[lev], type);

                const Set::Scalar xpos = x(0);
#if AMREX_SPACEDIM >= 2
                const Set::Scalar ypos = x(1);
#else
                const Set::Scalar ypos = 0.0;
#endif
#if AMREX_SPACEDIM == 3
                const Set::Scalar zpos = x(2);
#else
                const Set::Scalar zpos = 0.0;
#endif

                bool in_driver = (xpos >= xdrv_lo && xpos <= xdrv_hi);
#if AMREX_SPACEDIM >= 2
                in_driver = in_driver && (ypos >= ydrv_lo && ypos <= ydrv_hi);
#endif
#if AMREX_SPACEDIM == 3
                in_driver = in_driver && (zpos >= zdrv_lo && zpos <= zdrv_hi);
#endif

                const Set::Scalar dx = xpos - xc;
#if AMREX_SPACEDIM >= 2
                const Set::Scalar dy = ypos - yc;
#else
                const Set::Scalar dy = 0.0;
#endif
#if AMREX_SPACEDIM == 3
                const Set::Scalar dz = zpos - zc;
#else
                const Set::Scalar dz = 0.0;
#endif

                const Set::Scalar r = std::sqrt(dx*dx + dy*dy + dz*dz);

                Set::Scalar alpha_liq = 0.0;
                if (eps > 0.0) {
                    alpha_liq = 0.5 * (Set::Scalar(1.0) - std::tanh((r - R) / eps));
                    alpha_liq = amrex::min(amrex::max(alpha_liq, Set::Scalar(0.0)), Set::Scalar(1.0));
                } else {
                    alpha_liq = (r <= R) ? Set::Scalar(1.0) : Set::Scalar(0.0);
                }

                // Droplet gets priority over driver gas if there is any overlap.
                const bool liquid_dominant = (alpha_liq >= Set::Scalar(0.5));
                const bool use_driver_gas = in_driver && !liquid_dominant;

                const Set::Scalar rho_g = use_driver_gas ? rho_g_drv : rho_g_amb;
                const Set::Scalar p_g   = use_driver_gas ? p_drv     : p_amb;
                const Set::Scalar u_g   = use_driver_gas ? u_drv     : u_amb;
#if AMREX_SPACEDIM >= 2
                const Set::Scalar v_g   = use_driver_gas ? v_drv     : v_amb;
#endif
#if AMREX_SPACEDIM == 3
                const Set::Scalar w_g   = use_driver_gas ? w_drv     : w_amb;
#endif

                // Shared velocity field for the 5-equation model.
                // If interface_thickness == 0, this reduces to region-wise constants.
                const Set::Scalar u0 = alpha_liq * u_l + (Set::Scalar(1.0) - alpha_liq) * u_g;
#if AMREX_SPACEDIM >= 2
                const Set::Scalar v0 = alpha_liq * v_l + (Set::Scalar(1.0) - alpha_liq) * v_g;
#endif
#if AMREX_SPACEDIM == 3
                const Set::Scalar w0 = alpha_liq * w_l + (Set::Scalar(1.0) - alpha_liq) * w_g;
#endif

                // Pressure field. For smoothed interfaces, blend linearly.
                const Set::Scalar p0 = (eps > 0.0)
                    ? (alpha_liq * p_l + (Set::Scalar(1.0) - alpha_liq) * p_g)
                    : ((r <= R) ? p_l : p_g);

                if (is_pressure_mf) {
                    phi(i, j, k, n) = p0;
                    return;
                }

                // pvec path: only FiveEquation is supported here.
                Set::Scalar rho1 = 0.0;
                Set::Scalar rho2 = 0.0;
                Set::Scalar alpha_store = 0.0;

                if (phase1_liq) {
                    rho1 = rho_l;
                    rho2 = rho_g;
                    alpha_store = alpha_liq;
                } else {
                    rho1 = rho_g;
                    rho2 = rho_l;
                    alpha_store = Set::Scalar(1.0) - alpha_liq;
                }

                if (n == m1_idx) {
                    phi(i, j, k, n) = rho1;
                } else if (n == m2_idx) {
                    phi(i, j, k, n) = rho2;
                } else if (n == momx_idx) {
                    phi(i, j, k, n) = u0;
#if AMREX_SPACEDIM >= 2
                } else if (n == momy_idx) {
                    phi(i, j, k, n) = v0;
#endif
#if AMREX_SPACEDIM == 3
                } else if (n == momz_idx) {
                    phi(i, j, k, n) = w0;
#endif
                } else if (n == etot_idx) {
                    // Placeholder. ScimitarX::Initialize() later recomputes
                    // consistent mixture internal energy from pressure + alpha.
                    phi(i, j, k, n) = Set::Scalar(0.0);
                } else if (n == alpha_idx) {
                    phi(i, j, k, n) = alpha_store;
                }
            });
        }
    }
}

} // namespace IC
