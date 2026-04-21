

# Step 1: Scan all problems across all verifier models and identify two subsets:
#   - false_problems.npz: problems where every verifier model has at least one False label
#   - true_problems.npz:  problems where every verifier model has at least one True label
# (--gt-label is unused in this branch but required by argparse)
python rq2_step1.py \
            --problem-select 'True' \
            --gt-label 'True'


# Step 2: For each verifier model, collect per-problem answer predictions on the
# true-label subset (gt_label=True), and save to results_correct/verifier_generator_gtTrue.npy
python rq2_step1.py \
            --ans-select 'True' \
            --gt-label 'True'


# Step 3: Same as Step 2 but for the false-label subset (gt_label=False),
# saving to results_correct/verifier_generator_gtFalse.npy
python rq2_step1.py \
            --ans-select 'True' \
            --gt-label 'False'


# Step 4: Using verifier_generator_gtFalse.npy, sample one answer per problem
# run_times times and compute true-negative rates (correct = pred==2).
# Saves results_correct/verifier_generator_gtlabel_False_dict.npy
python rq2_step2.py \
            --gt-label 'False'


# Step 5: Same as Step 4 but for the true-label subset, computing true-positive rates
# (correct = pred==1). Saves results_correct/verifier_generator_gtlabel_True_dict.npy
python rq2_step2.py \
            --gt-label 'True'
