#!/bin/bash
set +e

# Same 13 checkpoint paths as run_e1_all.sh, on purpose -- reusing the exact
# manifest (not re-deriving it) is what guarantees E1's geometry and E2's
# AUROC describe the same checkpoint for every nominal (rung, seed) row,
# which is exactly what open_questions.md Q3 flags as unverified for
# eval_ood_benchmarks.py's own directory-based resolution. Run this from
# paper-3/, same as run_e1_all.sh.

run () {
    ckpt=$1
    rung=$2
    seed=$3

    echo "===================================================="
    echo "$(date)  $rung seed=$seed"

    python scripts/extract_auroc_e2.py \
        --checkpoint "$ckpt" \
        --method csg \
        --rung "$rung" \
        --seed "$seed" \
        --metadata_csv ../data/master_metadata_lesion_only_soft.csv
}

# runA_grl
run ../checkpoints/csg_lite/runA_grl_s42/best-38.ckpt runA_grl 42
run ../checkpoints/csg_lite/runA_grl_s52/best-37.ckpt runA_grl 52
run ../checkpoints/csg_lite/runA_grl_s62/best-34.ckpt runA_grl 62
run ../checkpoints/csg_lite/runA_grl_s72/best-35.ckpt runA_grl 72
run ../checkpoints/csg_lite/runA_grl_s82/best-23.ckpt runA_grl 82

# runB_orth1
run ../checkpoints/csg_lite/runB_orth1_s42/best-36.ckpt runB_orth1 42
run ../checkpoints/csg_lite/runB_orth1_s52/best-32.ckpt runB_orth1 52
run ../checkpoints/csg_lite/runB_orth1_s62/best-34.ckpt runB_orth1 62
run ../checkpoints/csg_lite/runB_orth1_s72/best-35.ckpt runB_orth1 72
run ../checkpoints/csg_lite/runB_orth1_s82/best-37.ckpt runB_orth1 82

# runB
run ../checkpoints/csg_lite/runB_s42/best-31.ckpt runB 42
run ../checkpoints/csg_lite/runB_s52/best-28.ckpt runB 52
run ../checkpoints/csg_lite/runB_s62/best-28.ckpt runB 62

echo
echo "================== FINISHED =================="
date
