#ifndef INTEGRATOR_SCIMITARX_H // Include guards
#define INTEGRATOR_SCIMITARX_H // 

// Standard library includes
#include <iostream>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <vector>
#include <stdexcept>
#include <memory>

#include "Integrator/Integrator.H"
#include "IC/ShockDroplet.H"
#include "IO/ParmParse.H"
#include "BC/BC.H"
#include "BC/Nothing.H"
#include "IC/IC.H"
#include "IC/Sphere.H"
#include "IC/Constant.H"
#include "IC/Shock.H"
#include "IC/Laminate.H"
#include "IC/Expression.H"
#include "Util/ScimitarX_Util.H"
#include "Numeric/NumericTypes.H"
#include "Numeric/NumericFactory.H"
#include "Numeric/SolverCapabilities.H"
#include "Numeric/IntegratorVariableAccessLayer.H"
#include "Numeric/FluxHandler.H"
#include "Numeric/WENOReconstruction.H"
#include "Numeric/TimeStepper.H"

// [DI MODIFICATION] Include the new EoS model
#include "Model/Fluid/FiveEquation.H"

namespace Numeric {
    template<typename IntegratorDerivedClass> class FluxHandler;
    template<typename IntegratorDerivedClass> class FluxMethod;
    template<typename IntegratorDerivedClass> class FluxReconstruction;
    template<typename IntegratorDerivedClass> class FirstOrderReconstruction;
    template<typename IntegratorDerivedClass, typename VariantTraits> class WENOReconstruction;
    template<typename IntegratorDerivedClass> class LocalLaxFriedrichsMethod;
    template<typename IntegratorDerivedClass> class HLLCMethod;
    template<typename IntegratorDerivedClass> class AUSMupMethod;
    template<typename IntegratorDerivedClass> class TimeSteppingScheme;
    template<typename IntegratorDerivedClass> class EulerForwardScheme;
    template<typename IntegratorDerivedClass> class RK3Scheme;
    
    // [DI MODIFICATION] Forward declaration for 5-Eq accessor
    namespace FiveEquation {
        class FiveEquationVariableAccessor;
    }
}

namespace Integrator
{
class ScimitarX : virtual public Integrator
{
public:
    // Enum for Solver Types
    enum class SolverType {
        SolveCompressibleEuler,
        SolveElastoPlastic,
        SolveFiveEquationModel // [DI MODIFICATION] Added 5-Eq Model
    };

    // Enum for Euler Field Variables
    enum class EulerFieldVariables {
        DENS,   // Density
        UVEL,   // U-velocity
        VVEL,   // V-velocity
        WVEL,   // W-velocity
        IE      // Specific Internal Energy
    };

    // Enum for ElatoPlastic Field Variables
    enum class ElastoPlasticFieldVariables {
        DENS,         // Density
        UVEL,         // U-velocity
        VVEL,         // V-velocity
        WVEL,         // W-velocity
        IE,           // Specific Internal Energy    
        SXX,          //Sxx scalar
        SXY,          //Sxy scalar
        SYY,          //Syy scalar
        SXZ,          //Sxz scalar
        SYZ,          //Syz scalar
        SZZ,          //Szz scalar
        EPSBAR,       //Effective Plastic Strain
        IE_ELASTIC    //Elastic energy for temperature calculations
    };

    // [DI MODIFICATION] Enum for Five Equation Field Variables
    enum class FiveEquationFieldVariables {
        M1,     // m1 = alpha * rho1 (partial density 1)
        M2,     // m2 = (1-alpha) * rho2 (partial density 2)
        MOMX,   // rho * u
        MOMY,   // rho * v
        MOMZ,   // rho * w
        ETOT,   // rho * E (Total Energy)
        ALPHA   // Volume fraction
    };

    
    static const std::map<std::string, SolverType> stringToSolverType;

    template <typename Enum>
    static constexpr int toInt(Enum e) noexcept {
        return static_cast<int>(e);
    }

   
    class SolverTypeManager {
    public:

        static SolverTypeManager& getInstance() {
            static SolverTypeManager instance; // Thread-safe in C++11 and later
            return instance;
        }



    template<typename EnumType>
    const std::vector<EnumType>& getVariablesForType(SolverType solverType) const {
        switch (solverType) {
            case SolverType::SolveCompressibleEuler:
            
                return reinterpret_cast<const std::vector<EnumType>&>(getEulerVariables());
            case SolverType::SolveElastoPlastic:
                return reinterpret_cast<const std::vector<EnumType>&>(getElastoPlasticVariables());
            // [DI MODIFICATION] Return 5-Eq variables
            case SolverType::SolveFiveEquationModel:
                return reinterpret_cast<const std::vector<EnumType>&>(getFiveEquationVariables());
            default:
                throw std::runtime_error("Invalid SolverType: No variables mapped.");
            }
        }

    Numeric::GenericVariableAccessor::VariableIndices computeAndAssignVariableIndices(SolverType& solverType) {
        Numeric::GenericVariableAccessor::VariableIndices result;
        
        switch (solverType) {
            case SolverType::SolveCompressibleEuler:
                computeEulerIndices(result);
                break;
                
            case SolverType::SolveElastoPlastic:
                computeElastoPlasticIndices(result);
                break;
            // [DI MODIFICATION] Compute indices for 5-Eq
            case SolverType::SolveFiveEquationModel:
                computeFiveEquationIndices(result);
                break;    
            default:
                throw std::runtime_error("Unsupported solver type for variable indexing");
        }

        return result;
    }

    private:

    SolverTypeManager() {
        // Initialize the vectors of variables for each solver type
        initializeEulerVariables();
        initializeElastoPlasticVariables();
        // [DI MODIFICATION] Initialize 5-Eq vars
        initializeFiveEquationVariables();
    }

    SolverTypeManager(const SolverTypeManager&) = delete; // Prevent copy
    SolverTypeManager& operator=(const SolverTypeManager&) = delete; // Prevent assignment

    void initializeEulerVariables() {
        eulerVariables = {
            EulerFieldVariables::DENS,
            EulerFieldVariables::UVEL,
#if AMREX_SPACEDIM >= 2
            EulerFieldVariables::VVEL,
#endif
#if AMREX_SPACEDIM == 3
            EulerFieldVariables::WVEL,
#endif
            EulerFieldVariables::IE
        };
    }

    void initializeElastoPlasticVariables() {
        elastoPlasticVariables = {
            ElastoPlasticFieldVariables::DENS,
            ElastoPlasticFieldVariables::UVEL,
#if AMREX_SPACEDIM >= 2
            ElastoPlasticFieldVariables::VVEL,
#endif
#if AMREX_SPACEDIM == 3
            ElastoPlasticFieldVariables::WVEL,
#endif
            ElastoPlasticFieldVariables::IE,
            ElastoPlasticFieldVariables::SXX,
            ElastoPlasticFieldVariables::SYY,
            ElastoPlasticFieldVariables::SZZ,
            ElastoPlasticFieldVariables::SXY,
#if AMREX_SPACEDIM == 3
            ElastoPlasticFieldVariables::SXZ,
            ElastoPlasticFieldVariables::SYZ,
#endif
            ElastoPlasticFieldVariables::EPSBAR,
            ElastoPlasticFieldVariables::IE_ELASTIC
        };
    }

    // [DI MODIFICATION] Initialize 5-Eq variables
    void initializeFiveEquationVariables() {
        fiveEquationVariables = {
            FiveEquationFieldVariables::M1,
            FiveEquationFieldVariables::M2,
            FiveEquationFieldVariables::MOMX,
#if AMREX_SPACEDIM >= 2
            FiveEquationFieldVariables::MOMY,
#endif
#if AMREX_SPACEDIM == 3
            FiveEquationFieldVariables::MOMZ,
#endif
            FiveEquationFieldVariables::ETOT,
            FiveEquationFieldVariables::ALPHA
        };
    }

    void computeEulerIndices(Numeric::GenericVariableAccessor::VariableIndices& result) {
        std::set<int> allVariables;
        
        // Add all Euler variables to the set
        for (const auto& var : eulerVariables) {
            allVariables.insert(toInt(var));
        }

        // Assign indices dynamically
        int index = 0;
        for (int var : allVariables) {
            result.variableIndexMap[var] = index++;
        }

        // Set up specific indices
        result.NVAR_MAX = index;
        result.DENS = result.variableIndexMap[toInt(EulerFieldVariables::DENS)];
        result.UVEL = result.variableIndexMap[toInt(EulerFieldVariables::UVEL)];
#if AMREX_SPACEDIM >= 2
        result.VVEL = result.variableIndexMap[toInt(EulerFieldVariables::VVEL)];
#endif
#if AMREX_SPACEDIM == 3
        result.WVEL = result.variableIndexMap[toInt(EulerFieldVariables::WVEL)];
#endif
        result.IE = result.variableIndexMap[toInt(EulerFieldVariables::IE)];
    }


    void computeElastoPlasticIndices(Numeric::GenericVariableAccessor::VariableIndices& result) {
        std::set<int> allVariables;
        
        // Add all ElastoPlastic variables to the set
        for (const auto& var : elastoPlasticVariables) {
            allVariables.insert(toInt(var));
        }

        // Assign indices dynamically
        int index = 0;
        for (int var : allVariables) {
            result.variableIndexMap[var] = index++;
        }

        // Set up specific indices
        result.NVAR_MAX = index;
        result.DENS = result.variableIndexMap[toInt(ElastoPlasticFieldVariables::DENS)];
        result.UVEL = result.variableIndexMap[toInt(ElastoPlasticFieldVariables::UVEL)];
#if AMREX_SPACEDIM >= 2
        result.VVEL = result.variableIndexMap[toInt(ElastoPlasticFieldVariables::VVEL)];
#endif
#if AMREX_SPACEDIM == 3
        result.WVEL = result.variableIndexMap[toInt(ElastoPlasticFieldVariables::WVEL)];
#endif
        result.IE = result.variableIndexMap[toInt(ElastoPlasticFieldVariables::IE)];
        
        // ElastoPlastic specific variables
        result.SXX = result.variableIndexMap[toInt(ElastoPlasticFieldVariables::SXX)];
        result.SYY = result.variableIndexMap[toInt(ElastoPlasticFieldVariables::SYY)];
        result.SZZ = result.variableIndexMap[toInt(ElastoPlasticFieldVariables::SZZ)];
        result.SXY = result.variableIndexMap[toInt(ElastoPlasticFieldVariables::SXY)];
#if AMREX_SPACEDIM == 3
        result.SXZ = result.variableIndexMap[toInt(ElastoPlasticFieldVariables::SXZ)];
        result.SYZ = result.variableIndexMap[toInt(ElastoPlasticFieldVariables::SYZ)];
#endif
        result.EPSBAR = result.variableIndexMap[toInt(ElastoPlasticFieldVariables::EPSBAR)];
        result.IE_ELASTIC = result.variableIndexMap[toInt(ElastoPlasticFieldVariables::IE_ELASTIC)];
    }
    
    // [DI MODIFICATION] Compute 5-Eq indices
    void computeFiveEquationIndices(Numeric::GenericVariableAccessor::VariableIndices& result) {
        std::set<int> allVariables;
        
        // Add all Five Equation variables to the set
        for (const auto& var : fiveEquationVariables) {
            allVariables.insert(toInt(var));
        }

        // Assign indices dynamically
        int index = 0;
        for (int var : allVariables) {
            result.variableIndexMap[var] = index++;
        }

        // Set up specific indices
        result.NVAR_MAX = index;
        result.M1 = result.variableIndexMap[toInt(FiveEquationFieldVariables::M1)];
        result.M2 = result.variableIndexMap[toInt(FiveEquationFieldVariables::M2)];
        result.MOMX = result.variableIndexMap[toInt(FiveEquationFieldVariables::MOMX)];
#if AMREX_SPACEDIM >= 2
        result.MOMY = result.variableIndexMap[toInt(FiveEquationFieldVariables::MOMY)];
#endif
#if AMREX_SPACEDIM == 3
        result.MOMZ = result.variableIndexMap[toInt(FiveEquationFieldVariables::MOMZ)];
#endif
        result.ETOT = result.variableIndexMap[toInt(FiveEquationFieldVariables::ETOT)];
        result.ALPHA = result.variableIndexMap[toInt(FiveEquationFieldVariables::ALPHA)];
    }

    const std::vector<EulerFieldVariables>& getEulerVariables() const {
        return eulerVariables;
    }

    const std::vector<ElastoPlasticFieldVariables>& getElastoPlasticVariables() const {
        return elastoPlasticVariables;
    }

    // [DI MODIFICATION] Getter for five equation variables
    const std::vector<FiveEquationFieldVariables>& getFiveEquationVariables() const {
        return fiveEquationVariables;
    }

    // Member variables
    std::vector<EulerFieldVariables> eulerVariables;
    std::vector<ElastoPlasticFieldVariables> elastoPlasticVariables;
    // [DI MODIFICATION] Member variable for five equation enums
    std::vector<FiveEquationFieldVariables> fiveEquationVariables;

    };

    static Numeric::GenericVariableAccessor::VariableIndices variableIndex; //variableIndex is now a static member of ScimitarX 

    static IO::ParmParse setupPVecBoundaryConditions(IO::ParmParse& pp, const Numeric::GenericVariableAccessor::VariableIndices& variableIndex);

    // Constructors and destructor
    ScimitarX();
    ScimitarX(IO::ParmParse& pp);
    static void Parse(ScimitarX& value, IO::ParmParse& pp);
    virtual ~ScimitarX()
    {
        delete bc_PVec;
        delete bc_Pressure;

        delete ic_PVec;
        delete ic_Pressure;

    }

    void RegisterSolverCapabilities();
protected:
    // Required override methods from Integrator base class
    void Initialize(int lev) override;
    void TimeStepBegin(Set::Scalar time, int iter) override;
    void TimeStepComplete(Set::Scalar time, int lev) override;
    void Advance(int lev, Set::Scalar time, Set::Scalar dt) override;
    void TagCellsForRefinement(int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow) override;
    void Regrid(int lev, Set::Scalar time) override;

    // ScimitarX specific methods
    void AdvanceInTimeWithoutStiffTerms(int lev, Set::Scalar time, Set::Scalar dt);
    Set::Scalar GetTimeStep();
    void ComputeAndSetNewTimeStep();
    void ApplyBoundaryConditions(int lev, Set::Scalar time);

    // [DI MODIFICATION] Templates for solver specialization
    template <SolverType solverType> void ComputeConservedVariables(int lev);
    template <SolverType solverType> void UpdateSolutions(int lev);
 
    std::shared_ptr<Numeric::SolverCapabilities> GetSolverCapabilities() const;

protected:
    // Field data structures
    Set::Field<Set::Scalar> QVec_mf;          // Conservative variables (current timestep)
    Set::Field<Set::Scalar> QVec_old_mf;      // Conservative variables (previous timestep)
    Set::Field<Set::Scalar> XFlux_mf;         // X-dir Flux Vector
    Set::Field<Set::Scalar> YFlux_mf;         // Y-dir Flux Vector
    Set::Field<Set::Scalar> ZFlux_mf;         // Z-dir Flux Vector
    Set::Field<Set::Scalar> PVec_mf;          // Primitive variables
    Set::Field<Set::Scalar> Pressure_mf;      // Pressure variable
    
    // [DI MODIFICATION] Face-centered velocity for source term
    Set::Field<Set::Scalar> UFace_mf;         // X-dir Face Velocity (u_face / S_star)
    Set::Field<Set::Scalar> VFace_mf;         // Y-dir Face Velocity
    Set::Field<Set::Scalar> WFace_mf;         // Z-dir Face Velocity

    // Core solver parameters
    int number_of_components;                 // Number of components
    int number_of_ghost_cells;            // Number of ghost cells
    SolverType solverType;                    // Current solver type
    Set::Scalar cflNumber;                    // CFL number (read from input)

    Set::Scalar refinement_threshold;         // Threshold for AMR refinement

        // 5-equation positivity controls
    bool enable_di5_face_positivity = true;
    Set::Scalar di5_partial_density_floor = 1.0e-12;
    Set::Scalar di5_alpha_floor = 1.0e-12;
    Set::Scalar di5_pressure_floor = 1.0e-12;
    int di5_positivity_bisection_iters = 24;

    // Refinement threshold values
    Set::Scalar density_refinement_criterion = 0.01;
    Set::Scalar velocity_refinement_criterion = 0.01;
    Set::Scalar pressure_refinement_criterion = 0.01;
    Set::Scalar vorticity_refinement_criterion = 0.01;
    
    // Enable/disable flags for each refinement criterion
    bool enable_density_refinement = false;
    bool enable_velocity_refinement = false;
    bool enable_pressure_refinement = false;
    bool enable_vorticity_refinement = false;
    
    // Numeric handlers (using shared_ptr for better lifetime management)
    std::shared_ptr<Numeric::FluxHandler<ScimitarX>> fluxHandler;
    std::shared_ptr<Numeric::TimeStepper<ScimitarX>> timeStepper;
    std::shared_ptr<Numeric::GenericVariableAccessor> variable_accessor;
    std::shared_ptr<Numeric::SolverCapabilities> solverCapabilities;

    // [DI MODIFICATION] Mixture Model
    std::unique_ptr<Model::Fluid::FiveEquation::FluidMixture> fluid_mixture;

    // Initial and boundary conditions
    IC::IC<Set::Scalar>* ic_PVec = nullptr;            // Object used to initialize Primitive Vector
    IC::IC<Set::Scalar>* ic_Pressure = nullptr;        // Object used to initialize Pressure
    BC::BC<Set::Scalar>* bc_PVec = nullptr;            // Boundary Condition for Primitive Vector
    BC::BC<Set::Scalar>* bc_Pressure = nullptr;        // Boundary Condition for Pressure
    BC::Nothing bc_nothing;                            // Default "do-nothing" boundary condition

    // Numeric method configuration 
    Numeric::FluxReconstructionType reconstruction_method;
    Numeric::FluxScheme flux_scheme;
    Numeric::TimeSteppingSchemeType temporal_scheme;
    Numeric::ReconstructionMode variable_space;
    Numeric::WenoVariant weno_variant;

private:
    // Helper method to setup numeric components based on solverType
    void SetupNumericComponents();
    
    // Helper to validate and configure numeric methods
    void ValidateAndSetupNumerics();

    // Friend declarations for Numeric namespace access
    template <typename IntegratorDerivedClass>
    friend class Numeric::FluxReconstruction;

    template <typename IntegratorDerivedClass>
    friend class Numeric::FluxMethod;    

    template <typename IntegratorDerivedClass>
    friend class Numeric::FluxHandler;    

    template <typename IntegratorDerivedClass>
    friend class Numeric::FirstOrderReconstruction;

    template <typename IntegratorDerivedClass, typename VariantTraits>
    friend class Numeric::WENOReconstruction;

    template <typename IntegratorDerivedClass>
    friend class Numeric::LocalLaxFriedrichsMethod;

    template <typename IntegratorDerivedClass>
    friend class Numeric::HLLCMethod;
    
    template <typename IntegratorDerivedClass>
    friend class Numeric::AUSMupMethod;
    
    template <typename IntegratorDerivedClass>
    friend class Numeric::TimeSteppingScheme;

    template <typename IntegratorDerivedClass>
    friend class Numeric::EulerForwardScheme;

    template <typename IntegratorDerivedClass>
    friend class Numeric::RK3Scheme; 
 
    friend class Numeric::CompressibleEuler::CompressibleEulerVariableAccessor;
    
    // [DI MODIFICATION] Friend access
    friend class Numeric::FiveEquation::FiveEquationVariableAccessor;
            
    
}; //ScimitarX Class

// Static member initialization for string to solver type mapping
inline const std::map<std::string, ScimitarX::SolverType> ScimitarX::stringToSolverType = {
    {"SolveCompressibleEuler", ScimitarX::SolverType::SolveCompressibleEuler},
    {"SolveElastoPlastic", ScimitarX::SolverType::SolveElastoPlastic},
    {"SolveFiveEquationModel", ScimitarX::SolverType::SolveFiveEquationModel} // [DI MODIFICATION]
};

} // namespace Integrator
#endif // INTEGRATOR_SCIMITARX_H