# SCPGrainElastic Hydro Debug Context

Branch: `bugfix/find-SCPGrainElastic-Break`
Last stable commit: `d1942071b` ("Restore m0 mass source term; revert failed velocity-recompute attempt")
Reference "known-good" commit (pre-PR274 architecture, used for diffing): `d7cf9f681b164cdbf218bc395ffb0dcdc97ae13e`
(An earlier, WRONG reference commit that was mistakenly used for a while: `77b72ee7aa572f877c21476c9805f991e3e88047` — it's from a divergent multi-species branch, do not use it.)

## Background

`SCPGrainElastic` is a regression test combining `Flame` (phase-field combustion), `Hydro` (compressible
Riemann-solver fluid), and an elastic solver, coupled via a diffuse `eta` phase field (eta=1 fluid, eta=0 solid).
After merging PR274 ("Major overhaul of Hydro") into this branch, the test began crashing almost immediately
(`hi rho=0, ...deltav1 -nan` in the Roe solver). A long debugging session (see full history in prior
conversation transcript) found and fixed several real bugs, in order:

1. `solidrho(i,j,k)=m0(i,j,k)` conflation in `Flame::UpdateFluxes` (src/Integrator/Flame.cpp) — solid density
   field was being overwritten with the mass-flux source term instead of the actual solid density formula.
2. `CPG::ComputeT`/`ComputeE` (src/Model/Gas/EOS/CPG.cpp) divided by raw `density` unguarded, producing
   inf/nan near eta=0. Fixed with a local `+small` (1E-8) regularization on all such divisions. This is a
   permanent, confirmed-good fix.
3. `Hydro::Mix()` (src/Integrator/Hydro.cpp) was using the raw *mixed* density paired with fluid-only velocity
   when reconstructing conserved fluid quantities, which made density near solid boundaries freeze toward
   `rho_solid` instead of evolving. Fixed by properly de-mixing:
   `rho_fluid = (rho(i,j,k) - rho_solid(i,j,k)*(1.0-eta))/(eta+small)` before use. Mix() runs every step
   (not once), confirmed necessary for stability.
4. `Hydro::RHS()`'s flux computation had accumulated ad-hoc `(eta<cutoff) ? Flux() : Solve()` guards (added
   during this debugging session) meant to avoid computing fluxes in near-solid cells. These guards actually
   *caused* a "vacuum wall" artifact — spurious growing velocity in cells adjacent to the solid boundary,
   because a hard-zeroed flux has no back-pressure. Removed entirely; restored the reference commit's
   unconditional `flux_xlo = riemannsolver->Solve(state_xlo_fluid, state_x_fluid, gas, molef, i,j,k,0,small) * eta;`
   (and similarly for xhi/ylo/yhi). This matches `d7cf9f681`.
5. `mdot0 = -m0(i,j,k)*grad_eta_mag;` — sign must be negative (matches reference). It drifted to positive at
   one point during experimentation and was caught/fixed.
6. `Pdot0` (momentum source in RHS) must be `Set::Vector::Zero()`, NOT `mdot0*u0` — an earlier fix based on
   the wrong reference commit (#77b72ee7a) had set this incorrectly; reverted once the correct reference
   (d7cf9f681) was consulted.
7. `hydro.cutoff=0.1` added to `tests/SCPGrainElastic/input` — threshold below which `Hydro::Advance()`'s
   post-integration clamp treats a cell as pure solid (`rho_new=rho_solid`, `M_new=M_solid`, `E_new=E_solid`
   if `eta<cutoff`).

With all of the above, the test is now stable and runs 11000+ timesteps without crashing (previously crashed
almost immediately). Density evolves/diffuses correctly in the fluid region. This is the current committed
state (`d1942071b`).

## Currently disabled / incomplete physics

To reach stability, two physical source terms are still **zeroed out** in `Flame::UpdateFluxes`
(src/Integrator/Flame.cpp), where the local variable `Set::Vector u0;` (NOT the same as `Hydro::u0_mf`/
`u0_patch` field, confusingly similarly named) is used to compute `solidM = solidrho*u0`:

```cpp
Set::Vector u0;
u0(0) = 0.0;   // should eventually be: hydro.u0_ap*phi + hydro.u0_htpb*(1.0-phi)  (regression velocity)
u0(1) = 0.0;
```

Restoring `u0(0)` to the physical regression-velocity formula (matching the reference commit) causes a
**severe density blow-up** (density values reaching -17133, 27155, -9765, etc.) even though velocity itself
stayed exactly 0 the whole time it was blowing up — i.e. the blow-up happened through some other coupling
path, not directly through velocity feedback. This was reverted; root cause NOT found.

`u0_patch(i,j,k,0/1)` (the actual `Hydro::u0_mf` field consumed by `Hydro::RHS`'s interface terms) is also
currently hardcoded to `0.0, 0.0` in Flame.cpp, with the physical formula left commented out nearby as a
reminder of what it should eventually be.

The `m0` mass-flux source term IS currently active/re-enabled and appears to work (source term nonzero,
density does change), though the user had noted at one point "Source(i,j,k,0) is positive but it doesn't
look like density is being added" — this was WHILE other things were broken; may or may not still be an
issue now that Mix() and the flux guards are fixed. Not reconfirmed since the last several fixes.

## Open, unresolved problem: `velocity_mf` reads as exactly 0.0 everywhere

This is the main blocking issue and the most likely candidate for the Opus planning session to solve.

**Symptom**: In plotfile output, `velocity_mf` (the `v` field) is exactly `0.0` in every cell, at every
timestep, even though `momentum_mf` is genuinely nonzero and evolving correctly, and density is evolving.

**Root cause (identified, not yet fixed correctly)**: `amrex::TimeIntegrator` is used in `Hydro::Advance()`
via aliased MultiFabs (`density_old_mf`/`density_mf` etc. built with `amrex::MakeType::make_alias`).
`RHS()` is registered via `set_rhs()` and is called once per Runge-Kutta stage, receiving **stage-local**
intermediate state buffers — not the final, fully-integrated post-`.advance()` state. `velocity_mf` is only
ever written inside `RHS()`'s T/P/velocity computation loop:

```cpp
v(i,j,k,0) = Mx_fluid/(density_fluid + small);
```

Since `RHS()` never runs again after the final stage/final state is assembled by the TimeIntegrator, the
`velocity_mf` field is left holding whatever it was set to during the LAST stage evaluation, not recomputed
from the actual final `momentum_mf`/`density_mf` after `.advance()` completes. It's not clear why this reads
as *exactly* 0 rather than a slightly-stale-but-nonzero value — this inconsistency itself may be a clue
worth investigating (e.g. is `v`/velocity_mf possibly a completely different, never-written field vs. the
one used for vorticity/dt calc? Or is the stage buffer for v not aliased/registered the way density/momentum
are, so it's writing to a scratch copy that gets discarded?).

**The reference commit (d7cf9f681) has the same architecture and the same limitation** — it doesn't solve
this either (may be a pre-existing issue in that codebase, or velocity may simply not have been used for
anything critical in that version). So there is no known-good precedent to copy verbatim; this needs fresh
analysis of the `amrex::TimeIntegrator` API and how `Hydro::Advance()`/`Hydro::RHS()` are wired up.

### Three failed fix attempts (all reverted, do not repeat blindly)

All three attempts tried adding a velocity-recompute pass in `Hydro::Advance()`'s post-integration loop
("APPLY CUTOFFS AND DO DYNAMIC TIMESTEP CALCULATION" section), i.e. recomputing `v = M_new/rho_new` from the
final clamped/integrated state, AFTER the TimeIntegrator's `.advance()` call completes. All three made
stability strictly WORSE than baseline (crashing earlier each time), with velocity feeding runaway values
through what appear to be viscous/diffusive source terms elsewhere in RHS:

1. **Naive recompute** in the existing `mfi.validbox()` loop: crashed at step 35 with
   `u ~ 1e27–1e55` at cell `(i=18, j=0)`.
2. **Split into two loops**: `growntilebox()` for velocity recompute + separate `validbox()` pass for
   vorticity, hypothesizing the vorticity gradient stencil was reading stale ghost cells. Produced the
   *identical* crash (same step, same numbers, same cell) — this disproved the ghost-cell staleness theory.
3. **Floor instead of `+small`**: changed `/(density_fluid+small)` to `/std::max(density_fluid,small)` in
   both the new recompute pass and in `RHS()`. Crashed even *earlier* (step 15) with even *larger* magnitude
   (`1.86e+199`) at cell `(i=16, j=0)`.

Given the pattern (recomputing velocity from final state makes things worse, not better, and the failure
mode is always a runaway blowup near a low-eta/near-boundary cell), the likely real problem is NOT simply
"where/when is velocity written" but something about how velocity (once made non-stale) interacts badly with
a boundary/cutoff/viscous term elsewhere — possibly the same class of issue as the `u0(0)` density blow-up
above. It's plausible both open issues (stale velocity_mf, and u0(0)-regression-velocity blow-up) share a
common root cause in how velocity is computed/used near eta≈cutoff cells.

## Key files

- `src/Integrator/Flame.cpp` / `Flame.H` — `UpdateFluxes()`, couples Flame's eta/phi fields to Hydro's
  `m0`, `u0_patch` (prescribed velocity), `solid.density`, `solid.momentum` fields. Has a persistent
  `deta_dt_mf` field (added this session, replaces a local var).
- `src/Integrator/Hydro.cpp` / `Hydro.H` — `Mix()`, `Advance()`, `RHS()`. Core fluid solver, TimeIntegrator
  wiring, de-mixing formulas, cutoff clamp, flux computation.
- `src/Model/Gas/EOS/CPG.cpp` — calorically-perfect-gas EOS, has the `+small` regularization fix (keep this,
  it's confirmed good).
- `src/Solver/Local/Riemann/Roe.H`, `HLLE.H` — Riemann solvers, `Solve(stateL, stateR, gas, molef, i,j,k,
  direction, small)` signature (post-PR274; gas-model-aware, differs from old ideal-gas-only signature).
- `tests/SCPGrainElastic/input` — test config; `hydro.cutoff=0.1`, `hydro.lagrange=1.0`,
  `hydro.solid.energy.ic.expression.region0="5.0"`, wall BCs
  `hydro.momentum.bc.constant.type.ylo/yhi = neumann dirichlet`.

## Useful tools/workflow established this session

- No visualization tooling available; built a hand-written Python plotfile parser
  (`struct.unpack('<Nd', ...)` on the binary `Cell_D_*` files, little-endian, component-major layout) to
  directly inspect density/velocity/momentum/eta/pressure across space/time. Worth recreating if needed.
- Rebuild after every C++ change: `make -j8 ./bin/alamo` from the alamo root directory.
- Run test: `./bin/alamo-2d-g++ ./tests/SCPGrainElastic/input` (run in background for long stop_times,
  check `tests/SCPGrainElastic/output/NNNNNcell` directories for progress / crash point).
- Diff against the known-good reference commit rather than guessing:
  `git show d7cf9f681b164cdbf218bc395ffb0dcdc97ae13e:src/Integrator/Hydro.cpp` etc.

## Guidance for the planning session

- Prefer minimal, reference-matched changes over speculative rewrites — compare against `d7cf9f681` first,
  understand *why* it differs, before editing large sections of Hydro.cpp/Flame.cpp.
- Do not repeat the three failed velocity-recompute attempts above verbatim.
- After any fix, must rebuild and actually RUN the test (not just compile) to validate — prior "fixes" that
  looked plausible caused earlier/worse crashes.
- Only commit when explicitly asked by the user.

## Update (2026-07-08 planning session): mechanism found, fix attempts 4 and 5 also failed

Root-caused the `velocity_mf` always-reads-0 mechanism precisely (see below), fixed one confirmed unrelated
bug, and tried two theoretically-motivated fixes for the real instability — both failed. Current committed
state (`e80b30955`) has the one confirmed fix; `u0`/`solidM`/`u0_patch` remain zeroed in `Flame.cpp`.

**Confirmed bug #1 (fixed, committed in `e80b30955`)**: `Hydro::Advance()` swaps
`density_old_mf<->density_mf` (and momentum/energy) *before* calling `Mix()`. Right after the swap,
`density_old_mf` holds the correct just-completed state, while `density_mf` is stale scratch from *two*
steps ago, about to be overwritten. `Mix()` was reading its pre-mix density (`gas.ComputeLocalFractions`,
`gas.ComputeD`, and the `rho_fluid` de-mix formula) from `density_mf` — i.e. from two-step-old data — then
copying that corrupted result back into `density_old_mf`, feeding the TimeIntegrator a corrupted starting
state every step. **Fixed** by reading from `density_old_mf` (`rho_old` in the code) instead. Verified: with
sources still off, produces the same physically-sane result as before (density in [0.03, 2.0], no blow-up,
3000+ steps). This bug is real but was NOT the cause of the u0-blow-up or the velocity=0 symptom.

**The `velocity_mf`-reads-0 mechanism, fully traced (this is now understood, not guessed)**:
- Confirmed this test uses `integration.type = ForwardEuler` (single-stage) — the "RK-stage-drift"
  hypothesis from an earlier planning pass is **wrong**; there are no multiple stages for drift to occur
  between. A fix built on that hypothesis (freezing the `etadot*(field-field_solid)/(eta+small)` terms into
  a once-per-step field instead of recomputing them in `RHS()`) was implemented and tested: it produced
  **bit-identical** output to before, confirming it's a mathematical no-op for this integrator. Reverted.
- The real loop: `Mix()` rebuilds `momentum_mf` every step as
  `M = (rho_fluid*v)*eta + M_solid*(1-eta)`, using whatever `velocity_mf` currently holds. `velocity_mf` is
  *only* ever written inside `RHS()`, computed from the `rho`/`M` it's handed for that call — which for
  Forward Euler is `solution_old`, i.e. the state at the *start* of the step (which was just set by `Mix()`
  using the *previous* `v`). So: `Mix()` writes `M` from `v`; `RHS()` decodes `v` back out of that same `M`
  — a closed loop, `v` never incorporates the real `M_new = M_old + dt*RHS(...)` that the TimeIntegrator
  computes afterward. That real update is genuinely applied to `momentum_mf`/`density_mf` each step (and is
  what's plotted), but is thrown away again at the *start* of the next step when `Mix()` re-stamps `M` from
  the stale `v`.
- This directly explains the u0-blow-up: the viscous drag (`Ldot0`) and Lagrange no-penetration terms in
  `RHS()` use `(u - u0)`. With `u` pinned near 0 by the above loop, `(u-u0) ≈ -u0` is a **constant, undamped
  forcing** every step (no physical relaxation as real velocity should rise to meet `u0`), instead of
  decaying — explaining unbounded growth precisely when `u0` is nonzero, and stability when it's exactly 0
  (since then `(u-u0)=(0-0)=0`, no forcing at all).

**Fix attempt 4 (failed, reverted)**: Recompute `velocity_mf` from the just-integrated `M_new`/`rho_new`
right after `timeintegrator.advance()` in `Advance()` (two-pass: `growntilebox()` pass to recompute `v`
including ghosts, since `density_mf`/`momentum_mf` ghosts are valid post `set_post_stage_action`; then the
existing `validbox()` cutoff/vorticity/dt pass reads the fresh `v`). This is conceptually the same idea as
the three previously-failed attempts (see above), but built on the precise mechanistic understanding above
rather than trial and error. **Result: identical failure mode** — crashed via the Roe solver
(`hi rho=..., deltav1 ...`) at a boundary cell (`i≈16, j=0`) within ~10-18 steps, reproduced even with
`u0=0` (source-off baseline), which is *worse* than the every-step-Mix baseline (stable 11000+ steps).
Four attempts at "recompute v post-advance" (3 prior + this one) have now failed identically. This is strong
evidence the recompute-after-advance approach is fundamentally destabilizing in this discretization, not an
implementation bug — do not retry this exact direction without a substantially different idea.

**Fix attempt 5 (failed, reverted)**: Break the circularity from the other end — stop `Mix()` from
re-stamping `M`/`rho`/`E` every step. Restored the (already-present but dead) `mixed[lev]` gate
(`if (managed) mixed[lev] = true;` at the end of `Mix()`), so it only runs once, at IC time, letting the
TimeIntegrator's `RHS()` (fluxes + viscous terms + the live `etadot` interface-transport terms) evolve the
mixed state on its own thereafter — closer to how the reference commit `d7cf9f681` avoided this same loop,
but with `etadot` kept live (required, since the interface genuinely moves; the reference sidestepped this
by hard-zeroing `etadot`, which is not an option here). **Result: also failed, and faster than attempt 4** —
crashed at step 17 via the same Roe-solver blow-up, again reproduced with sources off. So per-step `Mix()`
re-stamping, despite discarding real momentum dynamics, appears to be acting as a **necessary numerical
stabilizer** for the current flux/viscous scheme at this mesh/timestep — removing it (by any means tried so
far) makes the interface region diverge *faster*, not slower.

**Where this leaves things**: the every-step-Mix()-with-live-v-circularity is, empirically, the only stable
configuration found so far (stable 11000+ steps with sources off; the confirmed Mix() buffer fix doesn't
change this). But it structurally cannot support the `u0` momentum-injection source, because that same
circularity prevents `(u-u0)` from ever relaxing. Two independent ways of breaking the circularity (recompute
after advance; stop Mix from re-stamping) both made the *baseline* (sourceless) case crash much faster than
before, which suggests the every-step `Mix()` re-stamp isn't merely irrelevant scaffolding — it's currently
propping up an otherwise-unstable flux/viscous discretization near the interface. A real fix likely needs to
either (a) identify and fix whatever makes the raw flux+viscous RHS unstable on its own (so `Mix()` doesn't
need to babysit it every step), or (b) find a way to update `v` from real dynamics *without* removing
`Mix()`'s stabilizing effect — e.g. blend/relax `v` toward the recomputed value with a damping factor rather
than a hard overwrite, or investigate whether the viscous/Lagrange terms themselves (`Ldot0`, the
`(p==s&&q==r)` symmetric term added post-PR274, `mu = gas.dynamic_viscosity(...)` vs the reference's constant
`mu`) are the actual unstable ingredient once `v` is allowed to move. Not yet attempted this session.
