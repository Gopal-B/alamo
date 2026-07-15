#ifndef NUMERIC_FIVE_EQUATION_VARIABLE_ACCESSOR_H
#define NUMERIC_FIVE_EQUATION_VARIABLE_ACCESSOR_H

#include "Numeric/IntegratorVariableAccessLayer.H"

namespace Numeric {
namespace FiveEquation {

class FiveEquationCapabilities : public SolverCapabilities {
public:
    ~FiveEquationCapabilities() override = default;

    std::string getIdentifier() const override;
    std::string getDescription() const override;

    MethodSupport supportsFluxReconstruction(FluxReconstructionType method) const override;
    MethodSupport supportsFluxScheme(FluxScheme scheme) const override;
    MethodSupport supportsTimeSteppingScheme(TimeSteppingSchemeType scheme) const override;
    MethodSupport supportsReconstructionMode(ReconstructionMode mode) const override;
    MethodSupport supportsWenoVariant(WenoVariant variant) const override;

    MethodValidationResult validateMethodCombination(
        FluxReconstructionType fluxReconstruction,
        FluxScheme fluxScheme,
        TimeSteppingSchemeType timeSteppingScheme,
        ReconstructionMode reconstructionMode,
        WenoVariant wenoVariant
    ) const override;

    DefaultConfiguration getDefaultConfiguration() const override;

    std::shared_ptr<GenericVariableAccessor> createVariableAccessor(
        ReconstructionMode mode,
        int numGhostCells
    ) const override;
};

class FiveEquationVariableAccessor : public GenericVariableAccessor {
public:
    FiveEquationVariableAccessor(int total_ghosts, ReconstructionMode mode);
    ~FiveEquationVariableAccessor() override = default;

    std::shared_ptr<SolverCapabilities> getSolverCapabilities() const override;

    amrex::MultiFab CreateWorkingBuffer(
        const amrex::BoxArray& baseGrids,
        const amrex::DistributionMapping& dm,
        const int num_components,
        const int ghost_cells
    ) const override;

    int getRequiredGhostCells(
        ReconstructionMode mode,
        FluxReconstructionType reconstructionType
    ) const override;

    void CopyVariables(
        int direction,
        int lev,
        void* solver_void,
        amrex::MultiFab& VariableBuffer,
        ReconstructionMode mode,
        const SolverCapabilities::MethodValidationResult& validationResult
    ) const override;

    void CopyFluxes(
        int direction,
        int lev,
        void* solver_void,
        amrex::MultiFab& CellFluxBuffer,
        ReconstructionMode mode,
        const SolverCapabilities::MethodValidationResult& validationResult
    ) const override;

    void PopulateAverageStates(
        int direction,
        int lev,
        void* solver_void,
        amrex::MultiFab& AverageStateBuffer
    ) const override;

    Set::MultiMatrix TransformStencilToCharacteristic(
        int i, int j, int k,
        const Set::MultiMatrix& stencil_matrix,
        int direction,
        const Set::MultiVector& avg_state
    ) const override;

    Set::MultiVector TransformFromCharacteristic(
        const Set::MultiVector& reconstructed_value,
        int direction,
        const Set::MultiVector& avg_state
    ) const override;

    void StoreDirectionalFlux(
        int direction,
        int lev,
        void* solver_void,
        amrex::MultiFab& SummedFlux
    ) const override;

    ReconstructionMode getCurrentReconstructionMode() const override { return current_mode; }

private:
    ReconstructionMode current_mode;
    int total_ghost_cells;

    enum Directions { Xdir = 0, Ydir = 1, Zdir = 2 };
};

} // namespace FiveEquation
} // namespace Numeric

#endif
