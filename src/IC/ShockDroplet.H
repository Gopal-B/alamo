#ifndef IC_SHOCKDROPLET_H_
#define IC_SHOCKDROPLET_H_

#include "Set/Set.H"
#include "IC/IC.H"
#include "IO/ParmParse.H"
#include "Util/ScimitarX_Util.H"
#include "Numeric/IntegratorVariableAccessLayer.H"

namespace IC {

class ShockDroplet : public IC<Set::Scalar> {
public:
    // Constructor for pvec initialization (requires variable indices)
    ShockDroplet(amrex::Vector<amrex::Geometry>& _geom,
                 IO::ParmParse& pp,
                 std::string name,
                 const Numeric::GenericVariableAccessor::VariableIndices& precomputed_indices);

    // Constructor for pressure initialization
    ShockDroplet(amrex::Vector<amrex::Geometry>& _geom,
                 IO::ParmParse& pp,
                 std::string name);

    virtual void Add(const int& lev, Set::Field<Set::Scalar>& a_phi, Set::Scalar time) override;

private:
    void initialize(IO::ParmParse& pp, const std::string& name);
    void AddConstant(const int& lev, Set::Field<Set::Scalar>& a_phi, Set::Scalar time);

    std::string mf_name;

    const Numeric::GenericVariableAccessor::VariableIndices* variable_indices;
    bool requires_variable_indices;

    // Geometry: rectangular high-pressure gas driver region
    Set::Scalar x_driver_lo;
    Set::Scalar x_driver_hi;
#if AMREX_SPACEDIM >= 2
    Set::Scalar y_driver_lo;
    Set::Scalar y_driver_hi;
#endif
#if AMREX_SPACEDIM == 3
    Set::Scalar z_driver_lo;
    Set::Scalar z_driver_hi;
#endif

    // Geometry: circular/spherical droplet region
    Set::Scalar droplet_center_x;
#if AMREX_SPACEDIM >= 2
    Set::Scalar droplet_center_y;
#endif
#if AMREX_SPACEDIM == 3
    Set::Scalar droplet_center_z;
#endif
    Set::Scalar droplet_radius;

    // Optional diffuse-interface thickness for alpha smoothing.
    // If <= 0, a sharp 0/1 initialization is used.
    Set::Scalar interface_thickness;

    // Phase convention.
    // true  -> phase-1 = liquid, alpha = liquid volume fraction
    // false -> phase-1 = gas,    alpha = gas volume fraction
    bool phase1_is_liquid;

    // Ambient gas state
    Set::Scalar ambient_rho_g;
    Set::Scalar ambient_p;
    Set::Scalar ambient_u;
#if AMREX_SPACEDIM >= 2
    Set::Scalar ambient_v;
#endif
#if AMREX_SPACEDIM == 3
    Set::Scalar ambient_w;
#endif

    // Driver/high-pressure gas state
    Set::Scalar driver_rho_g;
    Set::Scalar driver_p;
    Set::Scalar driver_u;
#if AMREX_SPACEDIM >= 2
    Set::Scalar driver_v;
#endif
#if AMREX_SPACEDIM == 3
    Set::Scalar driver_w;
#endif

    // Liquid droplet state
    Set::Scalar liquid_rho_l;
    Set::Scalar liquid_p;
    Set::Scalar liquid_u;
#if AMREX_SPACEDIM >= 2
    Set::Scalar liquid_v;
#endif
#if AMREX_SPACEDIM == 3
    Set::Scalar liquid_w;
#endif
};

} // namespace IC

#endif // IC_SHOCKDROPLET_H_
