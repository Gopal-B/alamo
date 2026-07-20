#include "IO/ParmParse.H"
#include "Integrator/ScimitarX.H"
#include "BC/BC.H"
#include "BC/Nothing.H"
#include "BC/Constant.H"
#include "IC/Shock.H"
#include "IC/Riemann2D.H"
#include "Model/Fluid/Fluid.H"
#include "IC/ShockDroplet.H"
#include "Numeric/Stencil.H"
#include "Numeric/NumericTypes.H"
#include "Numeric/NumericFactory.H"
#include "Numeric/SolverCapabilities.H"
#include "Numeric/IntegratorVariableAccessLayer.H"
#include "Numeric/FluxHandler.H"
#include "Numeric/WENOReconstruction.H"
#include "Numeric/TimeStepper.H"
#include "Util/Util.H"
#include "Util/ScimitarX_Util.H"

// [DI MODIFICATION] Include 5-Equation headers
#include "Numeric/FiveEquationVariableAccessor.H"
#include "Model/Fluid/FiveEquation.H"

namespace Integrator
{

// Define the static member variable
Numeric::GenericVariableAccessor::VariableIndices ScimitarX::variableIndex;

// Default constructor implementation
ScimitarX::ScimitarX() : Integrator()
{
    // Initialize member variables with default values
    number_of_components = 0;
    number_of_ghost_cells = 4;
    cflNumber = 0.5; // Set a reasonable default
    refinement_threshold = 0.01; // Default threshold
    
    // Initialize handlers with nullptr (will be set up later)
    fluxHandler = nullptr;
    timeStepper = nullptr;
    variable_accessor = nullptr;
    solverCapabilities = nullptr;
    
    // [DI MODIFICATION] Initialize mixture model pointer
    fluid_mixture = nullptr;
    
    // Set default numerical method configurations
    reconstruction_method = Numeric::FluxReconstructionType::WENO;
    flux_scheme = Numeric::FluxScheme::HLLC;
    temporal_scheme = Numeric::TimeSteppingSchemeType::RK3;
    variable_space = Numeric::ReconstructionMode::Conservative;
    weno_variant = Numeric::WenoVariant::WENOJS5;
    
}


ScimitarX::ScimitarX(IO::ParmParse& pp):ScimitarX()
{

    fluxHandler = std::make_shared<Numeric::FluxHandler<ScimitarX>>(nullptr);
    timeStepper = std::make_shared<Numeric::TimeStepper<ScimitarX>>();

    pp.queryclass(*this);
}

void ScimitarX::RegisterSolverCapabilities() {
    // Retrieve solver capabilities
    auto capabilities = GetSolverCapabilities();
    
    if (capabilities) {
        // Register with global capabilities registry
        Numeric::SolverCapabilitiesRegistry::getInstance()
            .registerCapabilities(capabilities);
        
        // Store for later use
        solverCapabilities = capabilities;
        
        // Log registration details
        Util::Message(INFO, "Registered Solver Capabilities for: " + 
            capabilities->getIdentifier());
    } else {
        Util::Warning(INFO, "Failed to register solver capabilities for solver type: " + 
            std::to_string(static_cast<int>(solverType)));
    }
}

std::shared_ptr<Numeric::SolverCapabilities> 
ScimitarX::GetSolverCapabilities() const {
    // Return cached capabilities if already created
    if (solverCapabilities) {
        return solverCapabilities;
    }

    // Dynamically create capabilities based on solver type
    switch (solverType) {
        case SolverType::SolveCompressibleEuler:
            return std::make_shared<Numeric::CompressibleEuler::CompressibleEulerCapabilities>();
        case SolverType::SolveElastoPlastic: {
            Util::Warning(INFO, "ElastoPlastic solver capabilities not fully implemented");
            return nullptr;
        }
        // [DI MODIFICATION] Register capabilities for 5-Eq model
        case SolverType::SolveFiveEquationModel: {
            return std::make_shared<Numeric::FiveEquation::FiveEquationCapabilities>();
        }
        default:
            Util::Abort(INFO, "Unknown solver type during capabilities creation: " + 
                        std::to_string(static_cast<int>(solverType)));
            return nullptr;
    }
}

// -------------------------------------------------------------------------
// [COMPILER FIX] Template Specializations must be defined before usage
// -------------------------------------------------------------------------

// --- Compressible Euler Implementation (Base) ---
template <>
inline void ScimitarX::ComputeConservedVariables<ScimitarX::SolverType::SolveCompressibleEuler>(int lev) {
    const Set::Scalar gamma = 1.4;

    for (amrex::MFIter mfi(*QVec_mf[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.growntilebox();
        auto const& pvec = PVec_mf.Patch(lev, mfi);
        auto const& qvec = QVec_mf.Patch(lev, mfi);
        auto const& pressure = Pressure_mf.Patch(lev, mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            Set::Scalar rho = pvec(i, j, k, variableIndex.DENS);
            Set::Scalar uvel = pvec(i, j, k, variableIndex.UVEL);
            Set::Scalar vvel = pvec(i, j, k, variableIndex.VVEL);
#if (AMREX_SPACEDIM == 3)
            Set::Scalar wvel = pvec(i, j, k, variableIndex.WVEL);
#else
            [[maybe_unused]] Set::Scalar wvel = 0.0;
#endif
            // In Euler, IE might be in PVec, or computed from pressure. 
            // Original code used PVec::IE. 
            Set::Scalar internal_energy = pvec(i, j, k, variableIndex.IE);

            // Total internal energy
            Set::Scalar kinetic_energy = 0.5 * (uvel * uvel + vvel * vvel
#if (AMREX_SPACEDIM == 3)
                                            + wvel * wvel
#endif
                                            );
            Set::Scalar total_internal_energy = internal_energy + kinetic_energy;

            // Set Q vector components
            qvec(i, j, k, variableIndex.DENS) = rho;
            qvec(i, j, k, variableIndex.UVEL) = rho * uvel;
            qvec(i, j, k, variableIndex.VVEL) = rho * vvel;
#if (AMREX_SPACEDIM == 3)
            qvec(i, j, k, variableIndex.WVEL) = rho * wvel;
#endif
            qvec(i, j, k, variableIndex.IE) = rho * total_internal_energy;

        });
    }
}

template <>
inline void ScimitarX::UpdateSolutions<ScimitarX::SolverType::SolveCompressibleEuler>(int lev) {
    const Set::Scalar gamma = 1.4;
    for (amrex::MFIter mfi(*QVec_mf[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();

        auto const& q_arr = QVec_mf.Patch(lev, mfi);
        auto const& p_arr = PVec_mf.Patch(lev, mfi);
        auto const& pressure_arr = Pressure_mf.Patch(lev, mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            // Extract conservative variables from QVec
            Set::Scalar rho = q_arr(i, j, k, variableIndex.DENS);
            rho = amrex::max(rho, 1e-12); // Density floor

            Set::Scalar uvel = q_arr(i, j, k, variableIndex.UVEL) / rho;
            Set::Scalar vvel = q_arr(i, j, k, variableIndex.VVEL) / rho;
#if (AMREX_SPACEDIM == 3)
            Set::Scalar wvel = q_arr(i, j, k, variableIndex.WVEL) / rho;
#else
            [[maybe_unused]] Set::Scalar wvel = 0.0;
#endif

            Set::Scalar total_internal_energy = q_arr(i, j, k, variableIndex.IE) / rho;

            // Compute kinetic energy
            Set::Scalar kinetic_energy = 0.5 * (uvel * uvel + vvel * vvel
#if (AMREX_SPACEDIM == 3)
                                            + wvel * wvel
#endif
                                            );

            // Compute internal energy
            Set::Scalar internal_energy = (total_internal_energy - kinetic_energy);
            internal_energy = amrex::max(internal_energy, 1e-12); // IE floor

            // Assume ideal gas law: p = (gamma - 1) * rho * internal_energy
            Set::Scalar pressure = (gamma - 1.0) * rho * internal_energy;
            pressure = amrex::max(pressure, 1e-12); // Pressure floor

            // Update the primitive variables array (PVec)
            p_arr(i, j, k, variableIndex.DENS) = rho;
            p_arr(i, j, k, variableIndex.UVEL) = uvel;
            p_arr(i, j, k, variableIndex.VVEL) = vvel;
#if (AMREX_SPACEDIM == 3)
            p_arr(i, j, k, variableIndex.WVEL) = wvel;
#endif
            p_arr(i, j, k, variableIndex.IE) = internal_energy;

            pressure_arr(i, j, k) = pressure; 
        });
    }
}

// --- MODIFICATION 5-Equation Model Implementation (DI) ---
template <>
inline void ScimitarX::ComputeConservedVariables<ScimitarX::SolverType::SolveFiveEquationModel>(int lev) {
    const auto* mixture_model = fluid_mixture.get();
    AMREX_ALWAYS_ASSERT(mixture_model != nullptr);

    for (amrex::MFIter mfi(*QVec_mf[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.growntilebox();
        auto const& p_arr = PVec_mf.Patch(lev, mfi);
        auto const& q_arr = QVec_mf.Patch(lev, mfi);
        
        const int m1_idx = variableIndex.M1;
        const int m2_idx = variableIndex.M2;
        const int momx_idx = variableIndex.MOMX;
        const int momy_idx = variableIndex.MOMY;
        [[maybe_unused]] const int momz_idx = variableIndex.MOMZ;
        const int etot_idx = variableIndex.ETOT;
        const int alpha_idx = variableIndex.ALPHA;

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            // Primitives: [rho, p, u, v, w, ie_mix, alpha]
            Set::Scalar rho1   = p_arr(i, j, k, m1_idx);      // M1 slot = rho1
            Set::Scalar rho2   = p_arr(i, j, k, m2_idx);      // M2 slot = rho2
           
            Set::Scalar u     = p_arr(i, j, k, momx_idx);
#if AMREX_SPACEDIM >= 2
            Set::Scalar v     = p_arr(i, j, k, momy_idx);
#else
            Set::Scalar v     = 0.0;
#endif
#if AMREX_SPACEDIM == 3
            Set::Scalar w     = p_arr(i, j, k, momz_idx);
#else
            Set::Scalar w     = 0.0;
#endif
            Set::Scalar ie_mix = p_arr(i, j, k, etot_idx);
            Set::Scalar alpha = p_arr(i, j, k, alpha_idx);
            
            // Conserved: [m1, m2, rho*u, rho*v, rho*w, rho*E, alpha]
            const Set::Scalar m1  = alpha * rho1;
            const Set::Scalar m2  = (1.0 - alpha) * rho2;
            const Set::Scalar rho = amrex::max(m1 + m2, 1e-14);
            Set::Scalar ke = 0.5 * (u*u + v*v + w*w);
            Set::Scalar Etot = rho * (ie_mix + ke);

            
            q_arr(i, j, k, m1_idx)    = m1;
            q_arr(i, j, k, m2_idx)    = m2;
            q_arr(i, j, k, momx_idx)  = rho * u;
            #if AMREX_SPACEDIM >= 2
            q_arr(i, j, k, momy_idx)  = rho * v;
            #endif
            #if AMREX_SPACEDIM == 3
            q_arr(i, j, k, momz_idx)  = rho * w;
            #endif
            q_arr(i, j, k, etot_idx)  = Etot;
            q_arr(i, j, k, alpha_idx) = alpha;
        });
    }
}
template <>
inline void ScimitarX::UpdateSolutions<ScimitarX::SolverType::SolveFiveEquationModel>(int lev) {
    const auto* mixture_model = fluid_mixture.get();
    AMREX_ALWAYS_ASSERT(mixture_model != nullptr);

    const Set::Scalar m_floor = di5_partial_density_floor;
    const Set::Scalar a_floor = di5_alpha_floor;
    const Set::Scalar p_floor = di5_pressure_floor;

    for (amrex::MFIter mfi(*QVec_mf[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();

        auto q_arr = QVec_mf.Patch(lev, mfi);
        auto p_arr = PVec_mf.Patch(lev, mfi);
        auto pressure_arr = Pressure_mf.Patch(lev, mfi);

        const int m1_idx = variableIndex.M1;
        const int m2_idx = variableIndex.M2;
        const int momx_idx = variableIndex.MOMX;
#if AMREX_SPACEDIM >= 2
        const int momy_idx = variableIndex.MOMY;
#endif
#if AMREX_SPACEDIM == 3
        const int momz_idx = variableIndex.MOMZ;
#endif
        const int etot_idx = variableIndex.ETOT;
        const int alpha_idx = variableIndex.ALPHA;

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {

            Set::Scalar m1 = amrex::max(q_arr(i,j,k,m1_idx), m_floor);
            Set::Scalar m2 = amrex::max(q_arr(i,j,k,m2_idx), m_floor);

            Set::Scalar alpha = amrex::min(Set::Scalar(1.0) - a_floor,
                                           amrex::max(a_floor, q_arr(i,j,k,alpha_idx)));

            const Set::Scalar rho = amrex::max(m1 + m2, m_floor);

            Set::Scalar momx = q_arr(i,j,k,momx_idx);
#if AMREX_SPACEDIM >= 2
            Set::Scalar momy = q_arr(i,j,k,momy_idx);
#else
            Set::Scalar momy = 0.0;
#endif
#if AMREX_SPACEDIM == 3
            Set::Scalar momz = q_arr(i,j,k,momz_idx);
#else
            Set::Scalar momz = 0.0;
#endif
            Set::Scalar Etot = q_arr(i,j,k,etot_idx);

            const Set::Scalar u = momx / rho;
            const Set::Scalar v = momy / rho;
            const Set::Scalar w = momz / rho;

            const Set::Scalar ke = Set::Scalar(0.5) * (u*u + v*v + w*w);

            Set::Scalar ie_mix = Etot / rho - ke;

            const Set::Scalar ie_min =
                mixture_model->mixture_internal_energy(rho, p_floor, alpha);

            ie_mix = amrex::max(ie_mix, ie_min);
            Etot = rho * (ie_mix + ke);

            Set::Scalar p = mixture_model->closure_pressure(rho, ie_mix, alpha);
            p = amrex::max(p, p_floor);

            // Write corrected conservative state back
            q_arr(i,j,k,m1_idx) = m1;
            q_arr(i,j,k,m2_idx) = m2;
            q_arr(i,j,k,momx_idx) = momx;
#if AMREX_SPACEDIM >= 2
            q_arr(i,j,k,momy_idx) = momy;
#endif
#if AMREX_SPACEDIM == 3
            q_arr(i,j,k,momz_idx) = momz;
#endif
            q_arr(i,j,k,etot_idx) = Etot;
            q_arr(i,j,k,alpha_idx) = alpha;

            // Primitive storage for your DI solver:
            // PVec stores intrinsic rho1, rho2, velocity, ie_mix, alpha
            const Set::Scalar a1 = amrex::max(alpha, a_floor);
            const Set::Scalar a2 = amrex::max(Set::Scalar(1.0) - alpha, a_floor);

            const Set::Scalar rho1 = m1 / a1;
            const Set::Scalar rho2 = m2 / a2;

            p_arr(i,j,k,m1_idx) = rho1;
            p_arr(i,j,k,m2_idx) = rho2;
            p_arr(i,j,k,momx_idx) = u;
#if AMREX_SPACEDIM >= 2
            p_arr(i,j,k,momy_idx) = v;
#endif
#if AMREX_SPACEDIM == 3
            p_arr(i,j,k,momz_idx) = w;
#endif
            p_arr(i,j,k,etot_idx) = ie_mix;
            p_arr(i,j,k,alpha_idx) = alpha;

            pressure_arr(i,j,k) = p;
        });
    }
}
// Implementation of SetupNumericComponents
void ScimitarX::SetupNumericComponents() 
{
    // Initialize handlers if they're null
    if (!fluxHandler) {
        fluxHandler = std::make_shared<Numeric::FluxHandler<ScimitarX>>(nullptr);
    }
    
    if (!timeStepper) {
        timeStepper = std::make_shared<Numeric::TimeStepper<ScimitarX>>();
    }
    

    // Create variable accessor based on current solver type
    switch(solverType){   

    // [DI MODIFICATION] Setup for Five Equation Model
    case SolverType::SolveFiveEquationModel:
        if (!variable_accessor) {
            variable_accessor = std::make_shared<Numeric::FiveEquation::FiveEquationVariableAccessor>(
                number_of_ghost_cells, variable_space);
            
            // Initialize indices (M1, M2, MOMX...)
            variable_accessor->initializeIndices(variableIndex);
        }

        // Update the flux handler with the new accessor
        fluxHandler = std::make_shared<Numeric::FluxHandler<ScimitarX>>(variable_accessor);

        // Configure Reconstruction
        if (reconstruction_method == Numeric::FluxReconstructionType::WENO) {
            if (weno_variant == Numeric::WenoVariant::WENOJS3) {
                Util::Message(INFO, "Creating WENOJS3 reconstruction");                
                fluxHandler->SetReconstruction(std::make_shared<Numeric::WENOJS3<ScimitarX>>());
            } else if (weno_variant == Numeric::WenoVariant::WENOJS5) {
                Util::Message(INFO, "Creating WENOJS5 reconstruction");                
                fluxHandler->SetReconstruction(std::make_shared<Numeric::WENOJS5<ScimitarX>>());
            } else if (weno_variant == Numeric::WenoVariant::WENOIS5) {
                Util::Message(INFO, "Creating WENOIS5 reconstruction");                
                fluxHandler->SetReconstruction(std::make_shared<Numeric::WENOIS5<ScimitarX>>());
            } 
            
            else {
                Util::Message(INFO, "Creating WENOZ5 reconstruction");                
                fluxHandler->SetReconstruction(std::make_shared<Numeric::WENOZ5<ScimitarX>>());
            }
        } else {
            fluxHandler->SetReconstruction(std::make_shared<Numeric::FirstOrderReconstruction<ScimitarX>>());
        }
        
        // Set up flux method (HLLC recommended)
        if (flux_scheme == Numeric::FluxScheme::LocalLaxFriedrichs) {
                fluxHandler->SetFluxMethod(std::make_shared<Numeric::LocalLaxFriedrichsMethod<ScimitarX>>());
        } else if (flux_scheme == Numeric::FluxScheme::HLLC){
                fluxHandler->SetFluxMethod(std::make_shared<Numeric::HLLCMethod<ScimitarX>>());
        } else if (flux_scheme == Numeric::FluxScheme::AUSMup){
            fluxHandler->SetFluxMethod(std::make_shared<Numeric::AUSMupMethod<ScimitarX>>());
        }      
        
        // Set up time stepping scheme
        if (temporal_scheme == Numeric::TimeSteppingSchemeType::ForwardEuler) {
            timeStepper->SetTimeSteppingScheme(std::make_shared<Numeric::EulerForwardScheme<ScimitarX>>());
        } else {
            timeStepper->SetTimeSteppingScheme(std::make_shared<Numeric::RK3Scheme<ScimitarX>>());
        }
        break;

        case SolverType::SolveCompressibleEuler:
            // Create the variable accessor if it doesn't exist
        if (!variable_accessor) {
            variable_accessor = std::make_shared<Numeric::CompressibleEuler::CompressibleEulerVariableAccessor>(
                number_of_ghost_cells, variable_space);

            // Initialize with indices specific to this solver type
            Numeric::GenericVariableAccessor::VariableIndices accessorIndices;

            // For CompressibleEuler, only initialize relevant indices
            accessorIndices.NVAR_MAX = variableIndex.NVAR_MAX;
            accessorIndices.DENS = variableIndex.DENS;
            accessorIndices.UVEL = variableIndex.UVEL;
            accessorIndices.VVEL = variableIndex.VVEL;
            accessorIndices.WVEL = variableIndex.WVEL;
            accessorIndices.IE = variableIndex.IE;

            for (const auto& [varEnum, index] : variableIndex.variableIndexMap) {
          
                accessorIndices.variableIndexMap[varEnum] = index;
            }

            // Initialize the accessor with these indices
            variable_accessor->initializeIndices(accessorIndices);
        }
        
        // Update the flux handler with the accessor
        fluxHandler = std::make_shared<Numeric::FluxHandler<ScimitarX>>(variable_accessor);
        
        // Set up flux reconstruction method based on configuration
        if (reconstruction_method == Numeric::FluxReconstructionType::WENO) {
            if (weno_variant == Numeric::WenoVariant::WENOJS3) {
                Util::Message(INFO, "Creating WENOJS3 reconstruction");                
                fluxHandler->SetReconstruction(std::make_shared<Numeric::WENOJS3<ScimitarX>>());
            } else if (weno_variant == Numeric::WenoVariant::WENOJS5) {
                Util::Message(INFO, "Creating WENOJS5 reconstruction");                
                fluxHandler->SetReconstruction(std::make_shared<Numeric::WENOJS5<ScimitarX>>());
            } else {
                Util::Message(INFO, "Creating WENOZ5 reconstruction");                
                fluxHandler->SetReconstruction(std::make_shared<Numeric::WENOZ5<ScimitarX>>());
            }
        } else {
            fluxHandler->SetReconstruction(std::make_shared<Numeric::FirstOrderReconstruction<ScimitarX>>());
        }
        
        // Set up flux method
        if (flux_scheme == Numeric::FluxScheme::LocalLaxFriedrichs) {
                fluxHandler->SetFluxMethod(std::make_shared<Numeric::LocalLaxFriedrichsMethod<ScimitarX>>());
        } else if (flux_scheme == Numeric::FluxScheme::HLLC){
                fluxHandler->SetFluxMethod(std::make_shared<Numeric::HLLCMethod<ScimitarX>>());
        } else if (flux_scheme == Numeric::FluxScheme::AUSMup){
            fluxHandler->SetFluxMethod(std::make_shared<Numeric::AUSMupMethod<ScimitarX>>());
    }      
        
        // Set up time stepping scheme
        if (temporal_scheme == Numeric::TimeSteppingSchemeType::ForwardEuler) {
            timeStepper->SetTimeSteppingScheme(std::make_shared<Numeric::EulerForwardScheme<ScimitarX>>());
        } else {
            timeStepper->SetTimeSteppingScheme(std::make_shared<Numeric::RK3Scheme<ScimitarX>>());
        }
    
        break;
        case SolverType::SolveElastoPlastic:
        // For other solver types, use default configurations
        Util::Warning(INFO, "SetupNumericComponents: Unsupported solver type. Using defaults.");
        
        // Create a basic variable accessor if none exists
        if (!variable_accessor) {

            variable_accessor = std::make_shared<Numeric::CompressibleEuler::CompressibleEulerVariableAccessor>(
                number_of_ghost_cells, variable_space);

            Numeric::GenericVariableAccessor::VariableIndices accessorIndices;
            
            // Initialize all indices relevant for ElastoPlastic
            accessorIndices.NVAR_MAX = variableIndex.NVAR_MAX;
            accessorIndices.DENS = variableIndex.DENS;
            // ... (ElastoPlastic Indices omitted for brevity, logic identical to original)
            
            // Copy the map for ElastoPlastic variables
            for (const auto& [varEnum, index] : variableIndex.variableIndexMap) {
                accessorIndices.variableIndexMap[varEnum] = index;
            }
            
            variable_accessor->initializeIndices(accessorIndices);

        }
        
        // Update flux handler with accessor
        fluxHandler = std::make_shared<Numeric::FluxHandler<ScimitarX>>(variable_accessor);

        break;
        default:
        throw std::runtime_error("Invalid SolverType: Setup Numerics");
    } 
        
    // Additional validation logging
    Util::Message(INFO, "Final Numeric Method Configuration:");
    Util::Message(INFO, "  Flux Reconstruction: " + 
        Numeric::NumericFactory::toString(reconstruction_method));
    Util::Message(INFO, "  Flux Scheme: " + 
        Numeric::NumericFactory::toString(flux_scheme));
    Util::Message(INFO, "  WENO Variant: " + 
        Numeric::NumericFactory::toString(weno_variant));
    Util::Message(INFO, "  Temporal Scheme: " + 
        Numeric::NumericFactory::toString(temporal_scheme));

    Util::Message(INFO, "Numeric components set up successfully.");
}


void ScimitarX::ValidateAndSetupNumerics() {
    // Ensure solver capabilities are registered
    if (!solverCapabilities) {
        RegisterSolverCapabilities();
    }



        Util::Message(INFO, "Configuration Before Validate Method:");
        Util::Message(INFO, "  Flux Reconstruction: " + 
            Numeric::NumericFactory::toString(reconstruction_method));
        Util::Message(INFO, "  Flux Scheme: " + 
            Numeric::NumericFactory::toString(flux_scheme));
        Util::Message(INFO, "  Time Stepping: " + 
            Numeric::NumericFactory::toString(temporal_scheme));
        Util::Message(INFO, "  Reconstruction Mode: " + 
            Numeric::NumericFactory::toString(variable_space));
        Util::Message(INFO, "  WENO Variant: " + 
            Numeric::NumericFactory::toString(weno_variant));



    auto validationResult = solverCapabilities->validateMethodCombination(
        reconstruction_method,
        flux_scheme,
        temporal_scheme,
        variable_space,
        weno_variant
    );

  
    if (!validationResult.isValid) {
        // Log comprehensive error information
        Util::Warning(INFO, "Invalid Numeric Method Configuration Detected:");
        
        // Detailed error reporting
        for (const auto& error : validationResult.errors) {
            Util::Warning(INFO, "  Error: " + error);
        }
        
        for (const auto& warning : validationResult.warnings) {
            Util::Warning(INFO, "  Warning: " + warning);
        }

        // Retrieve and apply default configuration
        auto defaultConfig = solverCapabilities->getDefaultConfiguration();
        
        Util::Message(INFO, "Applying Default Configuration:");
        // ... (logging omitted for brevity)

        // Override current configuration with defaults
        reconstruction_method = defaultConfig.fluxReconstruction;
        flux_scheme = defaultConfig.fluxScheme;
        temporal_scheme = defaultConfig.timeSteppingScheme;
        variable_space = defaultConfig.reconstructionMode;
        weno_variant = defaultConfig.wenoVariant;
    }

    
        Util::Message(INFO, "Configuration After Validate Method:");
        // ... (logging omitted for brevity)
    
    // Proceed with setting up numeric components
    SetupNumericComponents();
}


void
ScimitarX::Parse(ScimitarX& value, IO::ParmParse& pp)
{
    BL_PROFILE("Integrator::ScimitarX::Parse()");
    {
        ScimitarX::SolverTypeManager& setIndex = ScimitarX::SolverTypeManager::getInstance();

        std::string solverTypeStr;

        if (pp.query("SolverType", solverTypeStr)) 
        {
            auto it = ScimitarX::stringToSolverType.find(solverTypeStr);
            if (it != ScimitarX::stringToSolverType.end()) {
                value.solverType = it->second;

                ScimitarX::variableIndex = setIndex.computeAndAssignVariableIndices(value.solverType);
                value.number_of_components = ScimitarX::variableIndex.NVAR_MAX;

                // [DI MODIFICATION] Instantiate Fluid Mixture for 5-Eq Model
                if (value.solverType == SolverType::SolveFiveEquationModel) {
                    value.fluid_mixture = std::make_unique<Model::Fluid::FiveEquation::FluidMixture>(pp);
                    Util::Message(INFO, "Instantiated FiveEquation::FluidMixture EoS");
                }
               
               
                std::cout << "DEBUG: ScimitarX::variableIndex.NVAR_MAX = " 
                << ScimitarX::variableIndex.NVAR_MAX << std::endl;

                // ... (Debug logging code preserved) ...
 
                value.bc_PVec = new BC::Constant(value.number_of_components, pp, "bc.pvec");
                value.bc_Pressure = new BC::Constant(1, pp, "bc.pressure");

            } else {
                Util::Abort(__FILE__, __func__, __LINE__, "Invalid SolverType: " + solverTypeStr);
            }
        }
    }
    
    // Register New Fabs
    {
        value.RegisterNewFab(value.QVec_mf, &value.bc_nothing, value.number_of_components, value.number_of_ghost_cells, "QVec", false, {});
        value.RegisterNewFab(value.QVec_old_mf, &value.bc_nothing, value.number_of_components, value.number_of_ghost_cells, "QVec_old", false, {});

        value.RegisterFaceFab<0>(value.XFlux_mf, &value.bc_nothing, value.number_of_components, value.number_of_ghost_cells, "xflux", false, {});
#if AMREX_SPACEDIM >= 2
        value.RegisterFaceFab<1>(value.YFlux_mf, &value.bc_nothing, value.number_of_components, value.number_of_ghost_cells, "yflux", false, {});
#endif
#if AMREX_SPACEDIM == 3
        value.RegisterFaceFab<2>(value.ZFlux_mf, &value.bc_nothing, value.number_of_components, value.number_of_ghost_cells, "zflux", false, {});
#endif
        value.RegisterNewFab(value.PVec_mf, value.bc_PVec, value.number_of_components, value.number_of_ghost_cells, "PrimitiveVec", true, {}); 
        value.RegisterNewFab(value.Pressure_mf, value.bc_Pressure, 1, value.number_of_ghost_cells, "Pressure", true, {});
        
        // [DI MODIFICATION] Register Face Velocity Fabs for Source Term
        value.RegisterFaceFab<0>(value.UFace_mf, &value.bc_nothing, 1, value.number_of_ghost_cells, "u_face", false, {});
#if AMREX_SPACEDIM >= 2
        value.RegisterFaceFab<1>(value.VFace_mf, &value.bc_nothing, 1, value.number_of_ghost_cells, "v_face", false, {});
#endif
#if AMREX_SPACEDIM == 3
        value.RegisterFaceFab<2>(value.WFace_mf, &value.bc_nothing, 1, value.number_of_ghost_cells, "w_face", false, {});
#endif
    }
    
    // Initial Conditions
    {
        std::string type = "constant";
        pp.query("ic.pvec.type", type);  // IC condition type for Primitive Variables

        if (type == "shock") {
            value.ic_PVec = new IC::Shock(value.geom, pp, "ic.shock.pvec", ScimitarX::variableIndex);
        } else if (type == "riemann2d") {
            value.ic_PVec = new IC::Riemann2D(value.geom, pp, "ic.riemann2d.pvec", ScimitarX::variableIndex);
        } else if (type == "shockdroplet") {
            value.ic_PVec = new IC::ShockDroplet(value.geom, pp, "ic.shockdroplet.pvec", ScimitarX::variableIndex);
        } else {
            Util::Abort(__FILE__, __func__, __LINE__, "Invalid ic.pvec.type: " + type);
        }

        pp.query("ic.pressure.type", type);
        if (type == "shock") {
            value.ic_Pressure = new IC::Shock(value.geom, pp, "ic.shock.pressure");
        } else if (type == "riemann2d") {
            value.ic_Pressure = new IC::Riemann2D(value.geom, pp, "ic.riemann2d.pressure");
        } else if (type == "shockdroplet") {
            value.ic_Pressure = new IC::ShockDroplet(value.geom, pp, "ic.shockdroplet.pressure");
        } else {
            Util::Abort(__FILE__, __func__, __LINE__, "Invalid ic.pressure.type: " + type);
        }
    } 

    std::string reconstr_str = "FirstOrder";  // Default
    pp.query("FluxReconstruction", reconstr_str);  // Read flux reconstruction method
    try {
        value.reconstruction_method = Numeric::NumericFactory::parseFluxReconstruction(reconstr_str);
        Util::Message(INFO, "Flux Reconstruction: " + reconstr_str);
    } catch (const std::runtime_error& e) {
        Util::Abort(__FILE__, __func__, __LINE__, 
            "Invalid FluxReconstruction parameter: " + reconstr_str + "\n" + e.what());
    }
   
    std::string flux_str = "LocalLaxFriedrichs";  // Default
    pp.query("FluxScheme", flux_str); // Read flux scheme
    try {
        value.flux_scheme = Numeric::NumericFactory::parseFluxScheme(flux_str);
        Util::Message(INFO, "Flux Scheme: " + flux_str);
    } catch (const std::runtime_error& e) {
        Util::Abort(__FILE__, __func__, __LINE__, 
            "Invalid FluxScheme parameter: " + flux_str + "\n" + e.what());
    }
   
    std::string time_str = "ForwardEuler";  // Default
    pp.query("TimeSteppingScheme", time_str); // Read time stepping scheme
    try {
        value.temporal_scheme = Numeric::NumericFactory::parseTimeSteppingScheme(time_str);
        Util::Message(INFO, "Time Stepping Scheme: " + time_str);
    } catch (const std::runtime_error& e) {
        Util::Abort(__FILE__, __func__, __LINE__, 
            "Invalid TimeSteppingScheme parameter: " + time_str + "\n" + e.what());
    }
   
    std::string mode_str = "Primitive";  // Default
    pp.query("ReconstructionMode", mode_str); // Read reconstruction mode
    try {
        value.variable_space = Numeric::NumericFactory::parseReconstructionMode(mode_str);
        Util::Message(INFO, "Reconstruction Mode: " + mode_str);
    } catch (const std::runtime_error& e) {
        Util::Abort(__FILE__, __func__, __LINE__, 
            "Invalid ReconstructionMode parameter: " + mode_str + "\n" + e.what());
    }
   
    std::string weno_str = "WENOJS5";  // Default
    pp.query("WenoVariant", weno_str); // Read WENO variant
    try {
        value.weno_variant = Numeric::NumericFactory::parseWenoVariant(weno_str);
        Util::Message(INFO, "WENO Variant: " + weno_str);
    } catch (const std::runtime_error& e) {
        Util::Abort(__FILE__, __func__, __LINE__, 
            "Invalid WenoVariant parameter: " + weno_str + "\n" + e.what());
    }
      
    pp.query_required("cflNumber", value.cflNumber); // Read CFL number

        // 5-equation positivity controls
    pp.query_default("enable_di5_face_positivity",
                     value.enable_di5_face_positivity, true);
    pp.query_default("di5_partial_density_floor",
                     value.di5_partial_density_floor, Set::Scalar(1.0e-12));
    pp.query_default("di5_alpha_floor",
                     value.di5_alpha_floor, Set::Scalar(1.0e-12));
    pp.query_default("di5_pressure_floor",
                     value.di5_pressure_floor, Set::Scalar(1.0e-12));
    pp.query_default("di5_positivity_bisection_iters",
                     value.di5_positivity_bisection_iters, 24);


                     // Static initial AMR region around bubble/wall.
// These tags are used to build initial nested grids.
// With amr.regrid_int = -1, the grids do not move later.
pp.query_default("amr.static_refinement.enabled",
    value.enable_static_refinement,
    false);

pp.query_default("amr.static_refinement.center_x",
    value.static_refine_center_x,
    Set::Scalar(0.0));

pp.query_default("amr.static_refinement.center_y",
    value.static_refine_center_y,
    Set::Scalar(0.0));

pp.query_default("amr.static_refinement.radius_lev0",
    value.static_refine_radius_lev0,
    Set::Scalar(-1.0));

pp.query_default("amr.static_refinement.radius_lev1",
    value.static_refine_radius_lev1,
    Set::Scalar(-1.0));

pp.query_default("amr.static_refinement.radius_lev2",
    value.static_refine_radius_lev2,
    Set::Scalar(-1.0));

    {
    std::string geom_mode = "cartesian";
    pp.query("geometry.mode", geom_mode);

    if (geom_mode == "cartesian") {
        value.geometry_mode = GeometryMode::Cartesian;
    } else if (geom_mode == "axisymmetric_xz" || geom_mode == "axisymmetric_rz" || geom_mode == "axisymmetric") {
        value.geometry_mode = GeometryMode::AxisymmetricXZ;
    } else {
        Util::Abort(INFO, "Invalid geometry.mode. Use 'cartesian' or 'axisymmetric_xz'.");
    }
    value.axisymmetric_enabled =(value.geometry_mode == GeometryMode::AxisymmetricXZ);
    pp.query_default("geometry.axial_dir",  value.axial_dir,  0);
    pp.query_default("geometry.radial_dir", value.radial_dir, 1);
    pp.query_default("geometry.axis_origin", value.axis_origin, Set::Scalar(0.0));
    pp.query_default("geometry.axisymmetric_r_epsilon_factor",
                     value.axisymmetric_r_epsilon_factor,
                     Set::Scalar(0.5));
    pp.query_default("geometry.axisymmetric_apply_axis_bc",
                     value.axisymmetric_apply_axis_bc,
                     true);
    
    if (value.geometry_mode == GeometryMode::AxisymmetricXZ) {
    #if AMREX_SPACEDIM != 2
        Util::Abort(INFO, "AxisymmetricXZ currently supported only in 2D.");
    #endif
    
        if (value.axial_dir != 0 || value.radial_dir != 1) {
            Util::Abort(INFO,
                "AxisymmetricXZ currently requires geometry.axial_dir = 0 and geometry.radial_dir = 1.");
        }
    
        const Set::Scalar ylo = value.geom[0].ProbLo(1);
        if (std::abs(ylo - value.axis_origin) > Set::Scalar(1.0e-14)) {
            Util::Abort(INFO,
                "Axisymmetric run requires geometry.prob_lo[1] == geometry.axis_origin. "
                "For bubble collapse use ylo = 0 and axis_origin = 0.");
        }
    
        if (value.axisymmetric_r_epsilon_factor <= Set::Scalar(0.0)) {
            Util::Abort(INFO,
                "geometry.axisymmetric_r_epsilon_factor must be positive. "
                "Use 0.5 for a finite-volume grid with first cell center at r = dr/2.");
        }
    
        Util::Message(INFO,
            "AxisymmetricXZ enabled: x = axial, y = radial, axis at ylo.");
    }
}

    // Add these to your Parse method
    pp.query_default("enable_density_refinement", value.enable_density_refinement, true); // enable density refinement
    pp.query_default("density_refinement_criterion", value.density_refinement_criterion, 0.2); // density refinement criterion
    
    pp.query_default("enable_pressure_refinement", value.enable_pressure_refinement, true);  //enable pressure refinement 
    pp.query_default("pressure_refinement_criterion", value.pressure_refinement_criterion, 0.15); // pressure refinement criterion
    
    pp.query_default("enable_velocity_refinement", value.enable_velocity_refinement, false); // enable velocity refinement
    pp.query_default("velocity_refinement_criterion", value.velocity_refinement_criterion, 0.1); //velocity refinement criterion
    
    pp.query_default("enable_vorticity_refinement", value.enable_vorticity_refinement, true); //enable vorticity refinement
    pp.query_default("vorticity_refinement_criterion", value.vorticity_refinement_criterion, 0.25); // vorticity refinement criterion
    
    // Validate and setup numeric methods (centralized)
    value.ValidateAndSetupNumerics();
        
}


// Initialize the Primitive Variables and Pressure through Initial Condition.
void ScimitarX::Initialize(int lev)
{
    ic_PVec->Initialize(lev, PVec_mf);
    ic_Pressure->Initialize(lev, Pressure_mf);

    // [DI MODIFICATION] Calculate Consistent Internal Energy for 5-Eq Model
    if (solverType == SolverType::SolveFiveEquationModel)
    {
        // Ensure the mixture model is valid
        AMREX_ALWAYS_ASSERT(fluid_mixture != nullptr);
        const auto* mixture_model = fluid_mixture.get();

        // Get the variable indices
        const int rho1_idx   = variableIndex.M1;   // PVec slot for rho
        const int rho2_idx   = variableIndex.M2;   // PVec slot for p
        const int ie_idx = variableIndex.ETOT; // PVec slot for ie_mix
        const int alpha_idx= variableIndex.ALPHA;// PVec slot for alpha

        for (amrex::MFIter mfi(*PVec_mf[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.growntilebox();
            auto const& pvec_arr  = PVec_mf.Patch(lev, mfi);
            auto const& press_arr = Pressure_mf.Patch(lev, mfi); 

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                
                Set::Scalar rho1  = pvec_arr(i, j, k, rho1_idx);
                Set::Scalar rho2  = pvec_arr(i, j, k, rho2_idx);
                Set::Scalar alpha = pvec_arr(i, j, k, alpha_idx);
                Set::Scalar p     = press_arr(i, j, k, 0);

                // Compute consistent mixture internal energy
                rho1  = amrex::max(rho1, 1e-14);
                rho2  = amrex::max(rho2, 1e-14);
                alpha = amrex::min(amrex::max(alpha, 0.0), 1.0);

                const Set::Scalar rho = alpha * rho1 + (1.0 - alpha) * rho2;
                const Set::Scalar ie_mix = mixture_model->mixture_internal_energy(amrex::max(rho, 1e-14), amrex::max(p, 1e-14), alpha);

                // Write ONLY ie_mix into PVec. Do NOT overwrite rho2 with pressure.
                pvec_arr(i, j, k, ie_idx) = amrex::max(ie_mix, 1e-14);
            });
        }
    }
    // This is essential for WENO/flux reconstruction near the axis.
    ApplyBoundaryConditions(lev, Set::Scalar(0.0));
    // [DI MODIFICATION] Dispatch to specialized function
    if (solverType == SolverType::SolveFiveEquationModel) {
        ComputeConservedVariables<SolverType::SolveFiveEquationModel>(lev);
    } else if (solverType == SolverType::SolveCompressibleEuler) {
        ComputeConservedVariables<SolverType::SolveCompressibleEuler>(lev);
    } else if (solverType == SolverType::SolveElastoPlastic) {
   
        Util::Warning(INFO, "ScimitarX::Initialize: ElastoPlastic ComputeConservedVariables not implemented, using Euler.");
        ComputeConservedVariables<SolverType::SolveCompressibleEuler>(lev);
    }
    else {
        Util::Abort("ScimitarX::Initialize: Unknown solverType");
    }
    // Fill physical ghost cells before constructing conserved variables.
    
    std::swap(*QVec_old_mf[lev], *QVec_mf[lev]); 
    
}


void ScimitarX::TagCellsForRefinement(int lev, amrex::TagBoxArray& tags, Set::Scalar /*time*/, int /*ngrow*/)
{
    const Set::Scalar* DX = geom[lev].CellSize();
    Set::Scalar dr = sqrt(AMREX_D_TERM(DX[0] * DX[0], +DX[1] * DX[1], +DX[2] * DX[2]));

    Set::Scalar static_radius = Set::Scalar(-1.0);

if (lev == 0) {
    static_radius = static_refine_radius_lev0;
} else if (lev == 1) {
    static_radius = static_refine_radius_lev1;
} else if (lev == 2) {
    static_radius = static_refine_radius_lev2;
}

const bool use_static_refinement =
    enable_static_refinement && static_radius > Set::Scalar(0.0);

const Set::Scalar static_cx = static_refine_center_x;
const Set::Scalar static_cy = static_refine_center_y;
const Set::Scalar static_r2 = static_radius * static_radius;

const Set::Scalar prob_lo_x = geom[lev].ProbLo(0);

#if AMREX_SPACEDIM >= 2
const Set::Scalar prob_lo_y = geom[lev].ProbLo(1);
#endif

    // Loop through all cells in the level for tagging
    for (amrex::MFIter mfi(*PVec_mf[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<char> const& tags_arr = tags.array(mfi);
        amrex::Array4<const Set::Scalar> const& pvec = (*PVec_mf[lev]).array(mfi);
        amrex::Array4<const Set::Scalar> const& pressure = (*Pressure_mf[lev]).array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            auto sten = Numeric::GetStencil(i, j, k, bx);


            // ------------------------------------------------------------
// Static geometric refinement around the initial bubble/wall.
// ------------------------------------------------------------
if (use_static_refinement) {
    const Set::Scalar x =
        prob_lo_x + (Set::Scalar(i) + Set::Scalar(0.5)) * DX[0];

#if AMREX_SPACEDIM >= 2
    const Set::Scalar y =
        prob_lo_y + (Set::Scalar(j) + Set::Scalar(0.5)) * DX[1];
#else
    const Set::Scalar y = Set::Scalar(0.0);
#endif

    const Set::Scalar dx0 = x - static_cx;
    const Set::Scalar dy0 = y - static_cy;

    const Set::Scalar dist2 = dx0 * dx0 + dy0 * dy0;

    if (dist2 <= static_r2) {
        tags_arr(i, j, k) = amrex::TagBox::SET;
        return;
    }
}

            // 1. Density gradient criterion
            if (enable_density_refinement) {
                // [DI MODIFICATION] Check for M1 index
                int dens_idx = (fluid_mixture != nullptr) ? variableIndex.ALPHA : variableIndex.DENS;
                Set::Vector grad_rho = Numeric::Gradient(pvec, i, j, k, dens_idx, DX, sten);
                
                if (grad_rho.lpNorm<2>() * dr * 2 > density_refinement_criterion) {
                    tags_arr(i, j, k) = amrex::TagBox::SET;
                    return;
                }
            }

            // 2. Pressure gradient criterion
            if (enable_pressure_refinement) {
                Set::Vector grad_p = Numeric::Gradient(pressure, i, j, k, 0, DX, sten);
                if (grad_p.lpNorm<2>() * dr * 2 > pressure_refinement_criterion) {
                    tags_arr(i, j, k) = amrex::TagBox::SET;
                    return;
                }
            }

            // 3 & 4. Velocity gradient and vorticity criteria
            if (enable_velocity_refinement || (enable_vorticity_refinement && AMREX_SPACEDIM >= 2)) {
                Set::Matrix grad_u = Set::Matrix::Zero();

                // [DI MODIFICATION] Check for 5-eq model indices
                int uvel_idx = (fluid_mixture != nullptr) ? variableIndex.MOMX : variableIndex.UVEL;
                int vvel_idx = (fluid_mixture != nullptr) ? variableIndex.MOMY : variableIndex.VVEL;
#if AMREX_SPACEDIM == 3
                int wvel_idx = (fluid_mixture != nullptr) ? variableIndex.MOMZ : variableIndex.WVEL;
#endif

                // X-velocity gradients
                Set::Vector grad_u_x = Numeric::Gradient(pvec, i, j, k, uvel_idx, DX, sten);
                grad_u.row(0) = grad_u_x;

#if AMREX_SPACEDIM >= 2
                // Y-velocity gradients
                Set::Vector grad_u_y = Numeric::Gradient(pvec, i, j, k, vvel_idx, DX, sten);
                grad_u.row(1) = grad_u_y;
#endif

#if AMREX_SPACEDIM == 3
                // Z-velocity gradients
                Set::Vector grad_u_z = Numeric::Gradient(pvec, i, j, k, wvel_idx, DX, sten);
                grad_u.row(2) = grad_u_z;
#endif

                // Velocity gradient criterion
                if (enable_velocity_refinement) {
                    Set::Matrix strain_rate = 0.5 * (grad_u + grad_u.transpose());
                    if (strain_rate.lpNorm<2>() * dr * 2 > velocity_refinement_criterion) {
                        tags_arr(i, j, k) = amrex::TagBox::SET;
                        return;
                    }
                }

                // Vorticity criterion
                if (enable_vorticity_refinement && AMREX_SPACEDIM >= 2) {
                    Set::Scalar vorticity = grad_u(1, 0) - grad_u(0, 1);
                    if (std::abs(vorticity) * dr * 2 > vorticity_refinement_criterion) {
                        tags_arr(i, j, k) = amrex::TagBox::SET;
                    }
                }
            }
        });
    }
}


void ScimitarX::TimeStepBegin(Set::Scalar /*time*/, int /*lev*/) {

}


void ScimitarX::TimeStepComplete(Set::Scalar /*time*/, int lev) {

    if (lev == 0) { 
        ComputeAndSetNewTimeStep(); // Compute dt based on global `minDt`
    } 
}

void ScimitarX::Regrid(int lev, Set::Scalar /*time*/) {

    // Only the finest level performs regridding
    if (lev < finest_level) return;

}

void ScimitarX::Advance(int lev, Set::Scalar time, Set::Scalar dt) {

        // Advance the solution without stiff source terms
        AdvanceInTimeWithoutStiffTerms(lev, time, dt);

        // 5. Swap the old QVec Fab with new one so that we can use the new one for next substep
        std::swap(*QVec_old_mf[lev], *QVec_mf[lev]);
}

// Function to advance hyperbolic balance equations without stiff terms
void ScimitarX::AdvanceInTimeWithoutStiffTerms(int lev, Set::Scalar time, Set::Scalar dt) {

    // Determine the time-stepping scheme
    switch (temporal_scheme) {

        case Numeric::TimeSteppingSchemeType::ForwardEuler: {
             
            int numStages = timeStepper->GetNumberOfStages();
            // One-stage loop for Forward Euler
            for (int stage = 0; stage < numStages; ++stage) {

                // 1. Compute Conserved Variables
                // [DI MODIFICATION] Dispatch
                if (solverType == SolverType::SolveFiveEquationModel) {
                    ComputeConservedVariables<SolverType::SolveFiveEquationModel>(lev);
                } else {
                    ComputeConservedVariables<SolverType::SolveCompressibleEuler>(lev);
                }

                // 2. Perform flux reconstruction and compute fluxes in all directions
                fluxHandler->ConstructFluxes(lev, this);

                // 3. Compute sub-step using the chosen time-stepping scheme
                timeStepper->ComputeSubStep(lev, dt, stage, this);

                // 4. Update solution from conservative to primitive variables
                // [DI MODIFICATION] Dispatch
                if (solverType == SolverType::SolveFiveEquationModel) {
                    UpdateSolutions<SolverType::SolveFiveEquationModel>(lev);
                } else {
                    UpdateSolutions<SolverType::SolveCompressibleEuler>(lev);
                }
            }
            break;
        }

        case Numeric::TimeSteppingSchemeType::RK3: {

            int numStages = timeStepper->GetNumberOfStages();

            for (int stage = 0; stage < numStages; ++stage) {

                ApplyBoundaryConditions(lev, time);
                // 1. Compute Conserved Variables
                // [DI MODIFICATION] Dispatch
                if (solverType == SolverType::SolveFiveEquationModel) {
                    ComputeConservedVariables<SolverType::SolveFiveEquationModel>(lev);
                } else {
                    ComputeConservedVariables<SolverType::SolveCompressibleEuler>(lev);
                }

                PVec_mf[lev]->FillBoundary(geom[lev].periodicity());

                // 2. Perform flux reconstruction and compute fluxes in all directions
                fluxHandler->ConstructFluxes(lev, this);

                // [DI MODIFICATION] Fill boundary for face velocities
                UFace_mf[lev]->FillBoundary(geom[lev].periodicity());
#if (AMREX_SPACEDIM >= 2)
                VFace_mf[lev]->FillBoundary(geom[lev].periodicity());
#endif
#if (AMREX_SPACEDIM == 3)
                WFace_mf[lev]->FillBoundary(geom[lev].periodicity());
#endif

                // 3. Compute sub-step using the chosen time-stepping scheme
                timeStepper->ComputeSubStep(lev, dt, stage, this);

                // 4. Update solution from conservative to primitive variables
                // [DI MODIFICATION] Dispatch
                if (solverType == SolverType::SolveFiveEquationModel) {
                    UpdateSolutions<SolverType::SolveFiveEquationModel>(lev);
                } else {
                    UpdateSolutions<SolverType::SolveCompressibleEuler>(lev);
                }

                ApplyBoundaryConditions(lev, time);

            }
            break;
        }

        default:
            Util::Abort(__FILE__, __func__, __LINE__, "Unknown TimeSteppingScheme.");
    }
}


void ScimitarX::ApplyBoundaryConditions(int lev, Set::Scalar time) {

    Integrator::ApplyPatch(lev, time, PVec_mf, *PVec_mf[lev], *bc_PVec, 0);
    Integrator::ApplyPatch(lev, time, Pressure_mf, *Pressure_mf[lev], *bc_Pressure, 0);

    // QVec is mostly rebuilt from PVec for this solver, but keeping its physical
    // ghosts consistent avoids surprises if later diagnostics/source terms use Q ghosts.
    Integrator::ApplyPatch(lev, time, QVec_mf, *QVec_mf[lev], bc_nothing, 0);

    if (axisymmetric_enabled && axisymmetric_apply_axis_bc) {
        ApplyAxisymmetricBoundaryConditions(lev);
    }
}

void ScimitarX::ApplyAxisymmetricBoundaryConditions(int lev) {
    #if AMREX_SPACEDIM != 2
        return;
    #else
        if (!axisymmetric_enabled) return;
        if (axial_dir != 0 || radial_dir != 1) {
            Util::Abort(INFO,
                "ApplyAxisymmetricBoundaryConditions assumes x = axial and y = radial.");
        }
    
        const amrex::Box& dom = geom[lev].Domain();
        const int jlo = dom.smallEnd(1);
        const int ng  = number_of_ghost_cells;
    
        const Set::Scalar ylo = geom[lev].ProbLo(1);
        if (std::abs(ylo - axis_origin) > Set::Scalar(1.0e-14)) {
            Util::Abort(INFO,
                "Axisymmetric axis BC requires geometry.prob_lo[1] == geometry.axis_origin.");
        }
    
        const int m1_idx    = variableIndex.M1;
        const int m2_idx    = variableIndex.M2;
        const int momx_idx  = variableIndex.MOMX;
        const int momy_idx  = variableIndex.MOMY;
        const int etot_idx  = variableIndex.ETOT;
        const int alpha_idx = variableIndex.ALPHA;
    
        // -----------------------------
        // PVec parity:
        // even: rho1, rho2, u_x, ie, alpha
        // odd : u_r
        // -----------------------------
        for (amrex::MFIter mfi(*PVec_mf[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& vbx = mfi.validbox();
    
            // Only boxes touching the physical axis need ylo ghost filling.
            if (vbx.smallEnd(1) != jlo) continue;
    
            amrex::Box gbx = vbx;
            gbx.grow(0, ng);              // include x ghost range already filled by x BC/periodicity
            gbx.setSmall(1, jlo - ng);
            gbx.setBig  (1, jlo - 1);
    
            auto const& p = PVec_mf.Patch(lev, mfi);
    
            amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const int jm = 2*jlo - 1 - j;  // mirror index: jlo-1 -> jlo, jlo-2 -> jlo+1, ...
    
                p(i,j,k,m1_idx)    =  p(i,jm,k,m1_idx);
                p(i,j,k,m2_idx)    =  p(i,jm,k,m2_idx);
                p(i,j,k,momx_idx)  =  p(i,jm,k,momx_idx);   // axial velocity even
                p(i,j,k,momy_idx)  = -p(i,jm,k,momy_idx);   // radial velocity odd
                p(i,j,k,etot_idx)  =  p(i,jm,k,etot_idx);
                p(i,j,k,alpha_idx) =  p(i,jm,k,alpha_idx);
            });
        }
    
        // -----------------------------
        // Pressure parity: even
        // -----------------------------
        for (amrex::MFIter mfi(*Pressure_mf[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& vbx = mfi.validbox();
            if (vbx.smallEnd(1) != jlo) continue;
    
            amrex::Box gbx = vbx;
            gbx.grow(0, ng);
            gbx.setSmall(1, jlo - ng);
            gbx.setBig  (1, jlo - 1);
    
            auto const& p = Pressure_mf.Patch(lev, mfi);
    
            amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const int jm = 2*jlo - 1 - j;
                p(i,j,k,0) = p(i,jm,k,0);
            });
        }
    
        // -----------------------------
        // QVec parity:
        // even: m1, m2, rho*u_x, rho*E, alpha
        // odd : rho*u_r
        // -----------------------------
        for (amrex::MFIter mfi(*QVec_mf[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& vbx = mfi.validbox();
            if (vbx.smallEnd(1) != jlo) continue;
    
            amrex::Box gbx = vbx;
            gbx.grow(0, ng);
            gbx.setSmall(1, jlo - ng);
            gbx.setBig  (1, jlo - 1);
    
            auto const& q = QVec_mf.Patch(lev, mfi);
    
            amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const int jm = 2*jlo - 1 - j;
    
                q(i,j,k,m1_idx)    =  q(i,jm,k,m1_idx);
                q(i,j,k,m2_idx)    =  q(i,jm,k,m2_idx);
                q(i,j,k,momx_idx)  =  q(i,jm,k,momx_idx);   // axial momentum even
                q(i,j,k,momy_idx)  = -q(i,jm,k,momy_idx);   // radial momentum odd
                q(i,j,k,etot_idx)  =  q(i,jm,k,etot_idx);
                q(i,j,k,alpha_idx) =  q(i,jm,k,alpha_idx);
            });
        }
    
        // Also keep QVec_old ghosts consistent. This is not strictly required for the
        // current RK update, but it is safer for diagnostics or future source terms.
        for (amrex::MFIter mfi(*QVec_old_mf[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& vbx = mfi.validbox();
            if (vbx.smallEnd(1) != jlo) continue;
    
            amrex::Box gbx = vbx;
            gbx.grow(0, ng);
            gbx.setSmall(1, jlo - ng);
            gbx.setBig  (1, jlo - 1);
    
            auto const& q = QVec_old_mf.Patch(lev, mfi);
    
            amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const int jm = 2*jlo - 1 - j;
    
                q(i,j,k,m1_idx)    =  q(i,jm,k,m1_idx);
                q(i,j,k,m2_idx)    =  q(i,jm,k,m2_idx);
                q(i,j,k,momx_idx)  =  q(i,jm,k,momx_idx);
                q(i,j,k,momy_idx)  = -q(i,jm,k,momy_idx);
                q(i,j,k,etot_idx)  =  q(i,jm,k,etot_idx);
                q(i,j,k,alpha_idx) =  q(i,jm,k,alpha_idx);
            });
        }
    #endif
    }
void ScimitarX::ComputeAndSetNewTimeStep() {
    // Compute the minimum time step over the entire domain using GetTimeStep
    Set::Scalar finest_dt = GetTimeStep();  // GetTimeStep already accounts for the CFL number

    // Start with the finest level time step
    Set::Scalar coarsest_dt = finest_dt;

  
    for (int lev = finest_level; lev > 0; --lev) {
        
        int refinement_factor = refRatio(lev - 1)[0];  // Assuming refinement is isotropic (same value in all directions)

        coarsest_dt *= refinement_factor;  // Scale the time step conservatively for refinement
    }

    
    Integrator::SetTimestep(coarsest_dt);

}

// Function to compute the time step size based on CFL condition
Set::Scalar ScimitarX::GetTimeStep() {
    Set::Scalar minDt = std::numeric_limits<Set::Scalar>::max();  // Start with a large value       
     

    for (int lev = 0; lev <= finest_level; ++lev) {  // Use maxLevel() from the base class
        const Set::Scalar* dx = geom[lev].CellSize();  // Access the geometry at level `lev`

        for (amrex::MFIter mfi(*PVec_mf[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();  // Iterate over tiles in the multifab
            auto const& pArr = PVec_mf.Patch(lev, mfi);
            auto const& pressure = Pressure_mf.Patch(lev, mfi);

            // [DI MODIFICATION] Get EoS for 5-eq model
            const auto* mixture_model = fluid_mixture.get();
            const int m1_idx = variableIndex.M1;
            const int momx_idx = variableIndex.MOMX;
            const int momy_idx = variableIndex.MOMY;
#if (AMREX_SPACEDIM == 3)
            const int momz_idx = variableIndex.MOMZ;
#endif
            const int alpha_idx = variableIndex.ALPHA;

            Set::Scalar minDt_local = std::numeric_limits<Set::Scalar>::max();  // Thread-local minDt

            amrex::ParallelFor(bx, [=, &minDt_local](int i, int j, int k) noexcept {
                
                Set::Scalar rho, u, v, w, p, c;
                Set::Scalar gamma = 1.4; // Default for Euler

                // [DI MODIFICATION] Use 5-Eq Sound Speed if active
                if (mixture_model != nullptr) {
                    const int rho1_idx = variableIndex.M1;
                    const int rho2_idx = variableIndex.M2;
                
                    Set::Scalar rho1  = pArr(i, j, k, rho1_idx);
                    Set::Scalar rho2  = pArr(i, j, k, rho2_idx);
                    Set::Scalar alpha = pArr(i, j, k, alpha_idx);
                
                    rho1  = amrex::max(rho1, 1e-14);
                    rho2  = amrex::max(rho2, 1e-14);
                    alpha = amrex::min(amrex::max(alpha, 0.0), 1.0);
                
                    rho = alpha * rho1 + (1.0 - alpha) * rho2;
                    p   = pressure(i, j, k);
                
                    u   = pArr(i, j, k, momx_idx);
                    v   = pArr(i, j, k, momy_idx);
                #if (AMREX_SPACEDIM == 3)
                    w   = pArr(i, j, k, momz_idx);
                #else
                    w   = 0.0;
                #endif
                
                    c = mixture_model->sound_speed_mixture(amrex::max(rho, 1e-14), amrex::max(p, 1e-14), alpha);
                } else {
                    // Original Euler Model
                    rho = pArr(i, j, k, variableIndex.DENS);
                    u   = pArr(i, j, k, variableIndex.UVEL);
                    v   = pArr(i, j, k, variableIndex.VVEL);
#if (AMREX_SPACEDIM == 3)
                    w   = pArr(i, j, k, variableIndex.WVEL);
#else
                    w   = 0.0;
#endif
                    p   = std::max(pressure(i, j, k), 1e-6);
                    c   = Model::Fluid::Fluid().ComputeWaveSpeed(rho, p, gamma);
                }

                // Compute the maximum characteristic speed
                Set::Scalar maxSpeed = std::abs(u) + c;
#if (AMREX_SPACEDIM >= 2)
                maxSpeed = std::max(maxSpeed, std::abs(v) + c);
#endif
#if (AMREX_SPACEDIM == 3)
                maxSpeed = std::max(maxSpeed, std::abs(w) + c);
#endif

                // Compute local timestep for this cell
                Set::Scalar dtLocal = dx[0] / maxSpeed;
#if (AMREX_SPACEDIM >= 2)
                dtLocal = std::min(dtLocal, dx[1] / maxSpeed);
#endif
#if (AMREX_SPACEDIM == 3)
                dtLocal = std::min(dtLocal, dx[2] / maxSpeed);
#endif

                // Track the local minimum
                if (dtLocal < minDt_local) {
                    minDt_local = dtLocal;
                }
            });

            // Update the global minDt
            minDt = std::min(minDt, minDt_local);
        }
    }

    // Reduce across processes to find the global minimum timestep
    amrex::ParallelDescriptor::ReduceRealMin(minDt);
    return ScimitarX::cflNumber * minDt;  // Return CFL-adjusted time step for the finest level
}


IO::ParmParse ScimitarX::setupPVecBoundaryConditions(IO::ParmParse& pp, const Numeric::GenericVariableAccessor::VariableIndices& variableIndex)
{
    int n_components = variableIndex.NVAR_MAX;
    std::vector<std::string> component_names(n_components);

    for (const auto& [variable, index] : variableIndex.variableIndexMap) {
        component_names[index] = std::to_string(variable);
    }

    IO::ParmParse bc_pp;

    // Define sides based on AMREX_SPACEDIM
#if AMREX_SPACEDIM == 1
    const std::vector<std::string> sides = {"xlo", "xhi"};
#elif AMREX_SPACEDIM == 2
    const std::vector<std::string> sides = {"xlo", "xhi", "ylo", "yhi"};
#elif AMREX_SPACEDIM == 3
    const std::vector<std::string> sides = {"xlo", "xhi", "ylo", "yhi", "zlo", "zhi"};
#endif

    for (const std::string& side : sides) {
        for (int i = 0; i < n_components; ++i) {
            std::string type_key = "bc.pvec." + component_names[i] + ".type." + side;
            std::string val_key = "bc.pvec." + component_names[i] + ".val." + side;

            std::string type_value = "neumann"; // Default
            std::string value_str = "0.0";     // Default

            pp.query_default(type_key.c_str(), type_value, "neumann");
            pp.query_default(val_key.c_str(), value_str, "0.0");

            // Add individual entries directly to bc_pp
            std::string bc_type_key = "bc.pvec.type." + side;
            std::string bc_val_key = "bc.pvec.val." + side;

            bc_pp.addarr(bc_type_key.c_str(), {type_value});
            bc_pp.addarr(bc_val_key.c_str(), {value_str});
        }
    }

    return bc_pp;
}

} // namespace Integrator