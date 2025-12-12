#!/bin/bash
# ============================================================================
# Launch DeBERTa HPO Studies with Updated Settings
# ============================================================================
# Launches 2 HPO studies for DeBERTa-v3-base:
#   1. WITHOUT augmentation (baseline)
#   2. WITH augmentation (searches aug params)
#
# All studies use:
#   - 100 epochs per trial
#   - Early stopping patience: 20
#   - torch.compile: disabled (optimal for HPO)
#   - 2000 trials with HyperbandPruner
#   - 5-fold cross-validation per trial
# ============================================================================

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}  DeBERTa HPO Launch - Updated Settings Applied${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo -e "Configuration:"
echo -e "  • Epochs: ${GREEN}100${NC} (with early stopping patience: ${GREEN}20${NC})"
echo -e "  • torch.compile: ${YELLOW}disabled${NC} (optimal for HPO)"
echo -e "  • Trials: ${GREEN}2000${NC} per study"
echo -e "  • Pruner: ${GREEN}HyperbandPruner${NC} (reduction_factor=4, bootstrap=30)"
echo -e "  • K-folds: ${GREEN}5${NC}"
echo ""

# ============================================================================
# Study 1: DeBERTa Base WITHOUT Augmentation
# ============================================================================
echo -e "${GREEN}[1/2] Launching: deberta_base_no_aug${NC}"
echo "  Model: microsoft/deberta-v3-base"
echo "  Augmentation: DISABLED"
echo "  Study: deberta_base_no_aug"
echo ""

nohup python -m criteria_bge_hpo.cli hpo \
    model=deberta_nli \
    hpo=pc_ce \
    hpo.study_name=deberta_base_no_aug \
    augmentation.enable=false \
    experiment_name=deberta_base_no_aug_hpo \
    > hpo_deberta_base_no_aug.log 2>&1 &

PID1=$!
echo -e "  ✓ Started with PID: ${GREEN}${PID1}${NC}"
echo -e "  ✓ Logs: ${YELLOW}hpo_deberta_base_no_aug.log${NC}"
echo ""

sleep 5  # Wait for first study to initialize database

# ============================================================================
# Study 2: DeBERTa Base WITH Augmentation
# ============================================================================
echo -e "${GREEN}[2/2] Launching: deberta_base_aug${NC}"
echo "  Model: microsoft/deberta-v3-base"
echo "  Augmentation: ENABLED (HPO searches aug_prob, aug_method)"
echo "  Study: deberta_base_aug"
echo ""

nohup python -m criteria_bge_hpo.cli hpo \
    model=deberta_nli \
    hpo=pc_ce \
    hpo.study_name=deberta_base_aug \
    augmentation.enable=true \
    experiment_name=deberta_base_aug_hpo \
    > hpo_deberta_base_aug.log 2>&1 &

PID2=$!
echo -e "  ✓ Started with PID: ${GREEN}${PID2}${NC}"
echo -e "  ✓ Logs: ${YELLOW}hpo_deberta_base_aug.log${NC}"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo -e "${BLUE}============================================================================${NC}"
echo -e "${GREEN}✓ All HPO studies launched successfully${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo "Running Studies:"
echo -e "  1. ${GREEN}deberta_base_no_aug${NC}  (PID: ${PID1}) → hpo_deberta_base_no_aug.log"
echo -e "  2. ${GREEN}deberta_base_aug${NC}      (PID: ${PID2}) → hpo_deberta_base_aug.log"
echo ""
echo "Monitor Progress:"
echo -e "  ${YELLOW}# Real-time logs${NC}"
echo "  tail -f hpo_deberta_base_no_aug.log"
echo "  tail -f hpo_deberta_base_aug.log"
echo ""
echo -e "  ${YELLOW}# Check running processes${NC}"
echo "  ps aux | grep criteria_bge_hpo.cli"
echo ""
echo -e "  ${YELLOW}# Check GPU usage${NC}"
echo "  nvidia-smi"
echo ""
echo -e "  ${YELLOW}# Check Optuna database${NC}"
echo "  sqlite3 optuna.db \"SELECT study_name, COUNT(*) as trials FROM studies JOIN trials USING(study_id) GROUP BY study_name;\""
echo ""
echo -e "  ${YELLOW}# Check study progress (requires tools/monitor_hpo.py modification)${NC}"
echo "  python3 tools/monitor_hpo.py  # Edit study_name first"
echo ""
echo "Stop Studies:"
echo "  kill ${PID1}  # Stop deberta_base_no_aug"
echo "  kill ${PID2}  # Stop deberta_base_aug"
echo ""
echo -e "${BLUE}Expected Runtime:${NC}"
echo "  • ~4-6 hours per trial (avg with pruning)"
echo "  • ~500-800 hours total per study (20-33 days)"
echo "  • Both studies run in parallel (same wall-clock time)"
echo ""
echo -e "${YELLOW}⚠  WARNING: Multi-GPU conflict if both run simultaneously!${NC}"
echo "  • Both studies will compete for GPU 0"
echo "  • Consider running sequentially or using CUDA_VISIBLE_DEVICES"
echo ""
echo "Run Sequentially (alternative):"
echo "  # Comment out Study 2 above, or manually launch after Study 1 completes"
echo ""
