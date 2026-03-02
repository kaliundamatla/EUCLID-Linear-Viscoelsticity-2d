"""
Run the inverse problem to identify viscoelastic Prony series parameters.

Reads FE dataset from standard_FE_dataset/, solves the inverse problem,
and writes results + plots to Postprocessing/final_outputs/.

Usage:
    cd inverse_problem
    python run_experiment.py
"""

from pathlib import Path
from inverse_problem import InverseProblem
from core.boundary import BottomForceBC, TopBottomForce


def main():
    # ========== CONFIGURATION ==========
    EXPERIMENT_NUMBER = 800           # 800 = synthetic rectangle (coarse mesh)
                                      # 900 = preprocessed real PA6 data
    N_MAXWELL_SHEAR = 150             # Number of shear Prony terms (nG)
    N_MAXWELL_BULK = 150              # Number of bulk Prony terms (nK)
    TAU_MIN = 1.0                     # Min relaxation time [s]
    TAU_MAX = 600.0                   # Max relaxation time [s]
    LAMBDA_INTERIOR = 0.0             # Interior equation weight (0=off, 1=on)
    LAMBDA_BOUNDARY = 1.0             # Boundary equation weight
    BOUNDARY_CONDITION = 'TopBottomForce'  # 'TopBottomForce' or 'BottomForceBC'
    APPLY_CLUSTERING = True
    CLUSTERING_RANGE = 0.3            # 30% relative distance threshold
    CREATE_PLOTS = True

    # Select boundary condition
    if BOUNDARY_CONDITION == 'TopBottomForce':
        boundary_condition = TopBottomForce()
        bc_description = "Forces on top and bottom (F[0]=top_y, F[1]=bottom_y)"
    elif BOUNDARY_CONDITION == 'BottomForceBC':
        boundary_condition = BottomForceBC()
        bc_description = "Force only on bottom (F[1]=bottom_y), top free"
    else:
        raise ValueError(f"Unknown boundary condition type: {BOUNDARY_CONDITION}")

    # ========== PRINT CONFIGURATION ==========
    print("="*70)
    print(f"EXPERIMENT {EXPERIMENT_NUMBER} - INVERSE PROBLEM")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Boundary condition: {BOUNDARY_CONDITION}")
    print(f"    {bc_description}")
    print(f"  nG = {N_MAXWELL_SHEAR}, nK = {N_MAXWELL_BULK}")
    print(f"  tau range: [{TAU_MIN}, {TAU_MAX}] seconds")
    print(f"  lambda_interior = {LAMBDA_INTERIOR}, lambda_boundary = {LAMBDA_BOUNDARY}")
    print(f"  Clustering: {'Enabled' if APPLY_CLUSTERING else 'Disabled'}")
    if APPLY_CLUSTERING:
        print(f"    Clustering range: {CLUSTERING_RANGE} (relative distance)")
    print("="*70)

    # ========== STEP 1: CREATE INVERSE PROBLEM ==========
    print("\n[1/3] Creating inverse problem...")

    # Data path: standard_FE_dataset/ contains both synthetic and real preprocessed data
    data_path = Path(__file__).parent.parent / "standard_FE_dataset"
    print(f"  Data source: {data_path / str(EXPERIMENT_NUMBER)}")

    problem = InverseProblem(
        experiment_number=EXPERIMENT_NUMBER,
        data_path=data_path,
        n_maxwell_shear=N_MAXWELL_SHEAR,
        n_maxwell_bulk=N_MAXWELL_BULK,
        tau_min=TAU_MIN,
        tau_max=TAU_MAX,
        element_type="Triangle3Node",
        boundary_condition=boundary_condition,
        lambda_interior=LAMBDA_INTERIOR,
        lambda_boundary=LAMBDA_BOUNDARY
    )

    print("Inverse problem created")
    print(f"  Nodes: {problem.exp_data.n_nodes}")
    print(f"  Elements: {problem.mesh.n_elements}")
    print(f"  Timesteps: {problem.exp_data.n_timesteps}")

    # ========== PREPARE OUTPUT PATH ==========
    output_dir_name = (
        f"{EXPERIMENT_NUMBER}_nG{N_MAXWELL_SHEAR}_nK{N_MAXWELL_BULK}_"
        f"tau{int(TAU_MIN)}-{int(TAU_MAX)}_"
        f"li{LAMBDA_INTERIOR}_lb{LAMBDA_BOUNDARY}_raw"
    )
    output_path = Path("./Postprocessing/final_outputs") / output_dir_name

    # ========== STEP 2: RUN IDENTIFICATION ==========
    print("\n[2/3] Running identification pipeline...")

    parameters = problem.run(
        create_plots=CREATE_PLOTS,
        apply_clustering=APPLY_CLUSTERING,
        clustering_range=CLUSTERING_RANGE,
        output_path=output_path
    )

    print("Identification complete")

    # ========== STEP 3: SAVE RESULTS ==========
    print("\n[3/3] Saving results...")
    problem.save_results(output_path)
    print(f"Results saved to: {output_path}")

    # ========== PRINT SUMMARY ==========
    print("\n" + "="*70)
    print("IDENTIFICATION SUMMARY")
    print("="*70)

    tau_G_nz, G_nz = parameters.get_nonzero_G()
    tau_K_nz, K_nz = parameters.get_nonzero_K()

    print(f"\nMaterial Parameters:")
    print(f"  G_inf: {parameters.G_inf:>12.2f} MPa")
    print(f"  K_inf: {parameters.K_inf:>12.2f} MPa")
    print(f"  G(0):  {parameters.total_G:>12.2f} MPa (instantaneous)")
    print(f"  K(0):  {parameters.total_K:>12.2f} MPa (instantaneous)")

    print(f"\nProny Series:")
    print(f"  Deviatoric (G):  {len(G_nz)} non-zero terms")
    if len(G_nz) > 0:
        print(f"    tau range: [{tau_G_nz.min():.2e}, {tau_G_nz.max():.2e}] s")
        print(f"    Moduli:    [{G_nz.min():.2e}, {G_nz.max():.2e}] MPa")

    print(f"  Volumetric (K):  {len(K_nz)} non-zero terms")
    if len(K_nz) > 0:
        print(f"    tau range: [{tau_K_nz.min():.2e}, {tau_K_nz.max():.2e}] s")
        print(f"    Moduli:    [{K_nz.min():.2e}, {K_nz.max():.2e}] MPa")

    print(f"\nOutput Files:")
    print(f"  Directory: {output_path}")
    print(f"  Results:   {output_path / 'results.npz'}")
    print(f"  Figures:   {output_path / 'figures'}/*.png")

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
